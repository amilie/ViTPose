import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import MMVRRadarPoseDataset
from model import SimpleRadarPoseCNN, WingLoss


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


def main():
    root_dir = "../P2_02"
    batch_size = 16          # increased: wider model benefits from larger batches
    learning_rate = 5e-4     # lowered: per-joint heads need finer initial steps
    num_epochs = 60
    train_ratio = 0.8
    random_seed = 42
    patience = 10
    grad_clip = 1.0
    weight_decay = 1e-4

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = SimpleRadarPoseCNN(num_keypoints=17).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # WingLoss: penalises small errors more heavily than MSE — better for keypoint accuracy
    criterion = WingLoss(w=10.0, eps=2.0)

    # AdamW: decoupled weight decay works better with the per-joint head structure
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Warmup then cosine decay: prevents instability in early epochs with per-joint heads
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, num_epochs - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for batch_radar, batch_keypoints in train_loader:
            batch_radar = batch_radar.to(device)
            batch_keypoints = batch_keypoints.to(device)

            preds = model(batch_radar)
            loss = criterion(preds, batch_keypoints)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_total += loss.item()

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model -> {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement.")
                break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")

    config = {
        "model": "SimpleRadarPoseCNN",
        "root_dir": root_dir,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "train_ratio": train_ratio,
        "random_seed": random_seed,
        "patience": patience,
        "grad_clip": grad_clip,
        "weight_decay": weight_decay,
        "best_val_loss": best_val_loss,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
