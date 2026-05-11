"""
train_v4.py — trains RadarPoseHeatmapUNet (U-Net + soft-argmax).

Notes specific to heatmap-based pose models:
  - Loss is on coordinates (after soft-argmax) — same target shape as other models
  - Auxiliary heatmap loss is supported but disabled by default since the
    bbox-normalised targets are coords, not Gaussian heatmaps. Enable
    `use_heatmap_aux=True` only if you generate Gaussian heatmap targets first.
  - Lower LR than train_v2: U-Net's many skip connections produce larger
    gradients and benefit from a steadier optimiser.
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import MMVRRadarPoseDataset
from model import RadarPoseHeatmapUNet, WingLoss


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_radar, batch_keypoints in loader:
            batch_radar = batch_radar.to(device)
            batch_keypoints = batch_keypoints.to(device)
            preds = model(batch_radar)
            loss = criterion(preds, batch_keypoints)
            total_loss += loss.item()
    return total_loss / len(loader)


def main() -> None:
    root_dir = "../P1"
    batch_size = 8           # smaller: U-Net memory footprint is larger
    learning_rate = 3e-4     # conservative for the deep encoder-decoder
    num_epochs = 80
    train_ratio = 0.8
    random_seed = 42
    patience = 12
    grad_clip = 1.0
    weight_decay = 1e-4
    base_ch = 32             # backbone width — try 48 if you want more capacity

    set_seeds(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = MMVRRadarPoseDataset(root_dir=root_dir)
    print("Total samples:", len(dataset))

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print("Train samples:", len(train_dataset))
    print("Val samples:  ", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = RadarPoseHeatmapUNet(num_keypoints=17, base_ch=base_ch).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # WingLoss on the soft-argmax coordinates — same contract as other train scripts
    criterion = WingLoss(w=10.0, eps=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Warmup (5 epochs) + cosine decay
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, num_epochs - warmup)
        return max(1e-6 / learning_rate, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    output_dir = Path("checkpoints_v4")
    output_dir.mkdir(exist_ok=True)
    num_batches = len(train_loader)
    train_losses: list[float] = []
    val_losses: list[float] = []
    lr_history: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for i, (batch_radar, batch_keypoints) in enumerate(train_loader):
            batch_radar = batch_radar.to(device)
            batch_keypoints = batch_keypoints.to(device)

            preds = model(batch_radar)
            loss = criterion(preds, batch_keypoints)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_total += loss.item()

            print(
                f"  Epoch {epoch+1:03d}/{num_epochs} | "
                f"Batch {i+1}/{num_batches} | "
                f"Loss: {loss.item():.6f}",
                end="\r", flush=True,
            )

        print()

        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train: {avg_train_loss:.6f} | "
            f"Val: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        lr_history.append(current_lr)
        _ep = list(range(1, len(train_losses) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training Curves (v4)", fontsize=13, fontweight="bold")
        axes[0].plot(_ep, train_losses, label="Train", color="steelblue")
        axes[0].plot(_ep, val_losses,   label="Val",   color="darkorange")
        axes[0].set_title("Loss per Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(_ep, lr_history, color="mediumseagreen")
        axes[1].set_title("Learning Rate")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model -> {save_path}")
        else:
            epochs_no_improve += 1

        config = {
            "model": "RadarPoseHeatmapUNet",
            "root_dir": root_dir,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "train_ratio": train_ratio,
            "random_seed": random_seed,
            "patience": patience,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "base_ch": base_ch,
            "best_val_loss": best_val_loss,
            "last_epoch": epoch + 1,
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
