"""
smoke_train.py — runs both models for 5 epochs and prints a side-by-side summary.

Usage:
    python3 smoke_train.py
    python3 smoke_train.py --epochs 5 --root ../P2_02
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import MMVRRadarPoseDataset
from model import SimpleRadarPoseCNN, RadarPoseResTransformer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item()
    return total / len(loader)


def train_one_model(
    model: torch.nn.Module,
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    grad_clip: float,
    weight_decay: float,
    output_dir: Path,
) -> dict:
    """Train for num_epochs, return per-epoch history and timing."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = []
    best_val = float("inf")
    t_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}   Device: {device}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        train_total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_total += loss.item()

        avg_train = train_total / len(train_loader)
        avg_val = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), output_dir / f"smoke_{name}_best.pth")

        history.append({"epoch": epoch + 1, "train": avg_train, "val": avg_val})
        print(f"  Epoch {epoch+1:02d}/{num_epochs} | train {avg_train:.6f} | val {avg_val:.6f} | lr {current_lr:.2e}")

    elapsed = time.perf_counter() - t_start
    return {"name": name, "history": history, "best_val": best_val, "elapsed_s": elapsed,
            "params": params}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--root", type=str, default="../P2_02")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = MMVRRadarPoseDataset(root_dir=args.root)
    print(f"Dataset: {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Both models share the same loaders so the split is identical
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    output_dir = Path("checkpoints_smoke")
    output_dir.mkdir(exist_ok=True)

    results = []

    # --- SimpleRadarPoseCNN ---
    set_seeds(args.seed)  # reset before each model so init is comparable
    cnn = SimpleRadarPoseCNN(num_keypoints=17).to(device)
    results.append(train_one_model(
        model=cnn, name="SimpleRadarPoseCNN",
        train_loader=train_loader, val_loader=val_loader,
        device=device, num_epochs=args.epochs,
        lr=1e-3, grad_clip=1.0, weight_decay=1e-4,
        output_dir=output_dir,
    ))

    # --- RadarPoseResTransformer ---
    set_seeds(args.seed)
    rtr = RadarPoseResTransformer(num_keypoints=17, d_model=128, nhead=4, num_transformer_layers=2).to(device)
    results.append(train_one_model(
        model=rtr, name="RadarPoseResTransformer",
        train_loader=train_loader, val_loader=val_loader,
        device=device, num_epochs=args.epochs,
        lr=5e-4, grad_clip=0.5, weight_decay=1e-4,
        output_dir=output_dir,
    ))

    # --- Side-by-side summary ---
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST SUMMARY  ({args.epochs} epochs)")
    print(f"{'='*60}")
    print(f"  {'Model':<28} {'Params':>10}  {'Best Val':>10}  {'Time':>8}  {'ms/epoch':>10}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")
    for r in results:
        ms_per_epoch = r["elapsed_s"] / args.epochs * 1000
        print(
            f"  {r['name']:<28}  {r['params']:>10,}  {r['best_val']:>10.6f}"
            f"  {r['elapsed_s']:>6.1f}s  {ms_per_epoch:>8.0f}ms"
        )

    # Per-epoch val loss table
    print(f"\n  Val loss per epoch:")
    header = f"  {'Epoch':>5}" + "".join(f"  {r['name'][:20]:>20}" for r in results)
    print(header)
    for ep in range(args.epochs):
        row = f"  {ep+1:>5}"
        for r in results:
            row += f"  {r['history'][ep]['val']:>20.6f}"
        print(row)

    # Save results
    with open(output_dir / "smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {output_dir / 'smoke_results.json'}")
    print(f"  Best checkpoints saved to {output_dir}/")


if __name__ == "__main__":
    main()
