"""
train_v5.py — trains RadarPoseDETR (CNN encoder + transformer decoder with
learnable joint queries that cross-attend to spatial features).

Notes specific to DETR-style pose models:
  - Joint queries are randomly initialised — they need many epochs to specialise
  - Cross-attention can be unstable in early epochs without warmup
  - Lower LR + longer warmup vs train_v3 because the decoder has both
    self-attention AND cross-attention per layer (more attention surface)
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_directml
from torch.utils.data import DataLoader, random_split

from dataset import MMVRRadarPoseDataset
from model import RadarPoseDETR, WingLoss


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pass  # DirectML has no GPU seed API


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
    batch_size = 64
    learning_rate = 2e-4     # lower than v3: cross-attention + self-attention layers
    num_epochs = 100         # joint queries take many epochs to specialise
    train_ratio = 0.8
    random_seed = 42
    patience = 20            # significant patience: queries refine slowly
    grad_clip = 0.5
    weight_decay = 5e-5

    d_model = 256
    nhead = 8
    num_decoder_layers = 4

    set_seeds(random_seed)

    device = torch_directml.device(0) if torch_directml.device_count() > 0 else torch.device("cpu")
    print("Using device:", device)

    dataset = MMVRRadarPoseDataset(root_dir=root_dir)
    print("Total samples:", len(dataset))

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print("Train samples:", len(train_dataset))
    print("Val samples:  ", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    model = RadarPoseDETR(
        num_keypoints=17,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = WingLoss(w=10.0, eps=2.0)

    # Two parameter groups: backbone gets a lower LR than the transformer decoder
    # Standard DETR practice — the encoder has more iterations of supervision
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": other_params,    "lr": learning_rate},
        ],
        weight_decay=weight_decay,
    )

    # Long warmup (10 epochs) — joint queries need gentle ramp-up to find features
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, num_epochs - warmup)
        return max(1e-6 / learning_rate, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    output_dir = Path("checkpoints_v5")
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
        # Two LRs in this optimiser — log the decoder one
        current_lr = optimizer.param_groups[1]["lr"]

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
        fig.suptitle("Training Curves (v5)", fontsize=13, fontweight="bold")
        axes[0].plot(_ep, train_losses, label="Train", color="steelblue")
        axes[0].plot(_ep, val_losses,   label="Val",   color="darkorange")
        axes[0].set_title("Loss per Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(_ep, lr_history, color="mediumseagreen")
        axes[1].set_title("Learning Rate (decoder)")
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
            "model": "RadarPoseDETR",
            "root_dir": root_dir,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "train_ratio": train_ratio,
            "random_seed": random_seed,
            "patience": patience,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "d_model": d_model,
            "nhead": nhead,
            "num_decoder_layers": num_decoder_layers,
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
