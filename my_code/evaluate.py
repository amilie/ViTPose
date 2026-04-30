"""
evaluate.py — shared evaluation utilities for all train scripts.

Phase 1 metrics:
  - Loss value
  - % improvement from epoch 1 baseline
  - PCK@0.2 (Percentage of Correct Keypoints)
  - MAE per joint (Mean Absolute Error — average distance off per joint)
  - 4-panel training plot saved to disk after a run
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path


# COCO 17-keypoint names, in order.
# These match the index order in your (17, 2) keypoint tensors.
JOINT_NAMES = [
    "nose",
    "left_eye",   "right_eye",
    "left_ear",   "right_ear",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]


def compute_pck(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.2):
    """
    PCK — Percentage of Correct Keypoints.

    A joint prediction counts as 'correct' if it lands within `threshold`
    of the ground truth (in normalized [0,1] coordinate space).

    Think of it like: "within 20% of the person's bounding box width/height."

    Args:
        preds:     (B, 17, 2) predicted keypoints, normalized coords
        targets:   (B, 17, 2) ground truth keypoints, normalized coords
        threshold: max allowed distance to count as correct (default 0.2)

    Returns:
        overall_pck:   float — fraction of ALL joints correct, e.g. 0.73 = 73%
        per_joint_pck: list of 17 floats — correctness rate for each joint
    """
    # Euclidean distance between predicted and true position for each joint
    # preds - targets → (B, 17, 2), then norm across the 2 (x,y) → (B, 17)
    dist = torch.norm(preds - targets, dim=-1)      # (B, 17)

    # 1.0 where correct (within threshold), 0.0 where not
    correct = (dist < threshold).float()            # (B, 17)

    # Average across all samples AND all joints → one number
    overall_pck = correct.mean().item()

    # Average across samples only → one number per joint
    per_joint_pck = correct.mean(dim=0).tolist()    # list of 17

    return overall_pck, per_joint_pck


def compute_per_joint_error(preds: torch.Tensor, targets: torch.Tensor):
    """
    Mean Absolute Error per joint (using Euclidean distance).

    This is the average "how far off were you?" for each joint,
    in normalized [0,1] coordinate units.

    Args:
        preds:   (B, 17, 2)
        targets: (B, 17, 2)

    Returns:
        overall_mae:   float — average error across all joints and samples
        per_joint_mae: list of 17 floats — average error per joint
    """
    dist = torch.norm(preds - targets, dim=-1)      # (B, 17)
    overall_mae = dist.mean().item()
    per_joint_mae = dist.mean(dim=0).tolist()       # list of 17
    return overall_mae, per_joint_mae


def run_eval(model, loader, criterion, device) -> dict:
    """
    Run the model on a DataLoader (no gradient updates) and return all metrics.

    Use this after each epoch to measure how well the model is doing on
    a held-out validation set (or train set for smoke tests like train_v1).

    Returns a dict with keys:
        loss           — average loss over the loader
        pck            — overall PCK@0.2 (float, 0–1)
        per_joint_pck  — list of 17 PCK values
        mae            — overall mean joint error (float)
        per_joint_mae  — list of 17 MAE values
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_radar, batch_keypoints in loader:
            batch_radar = batch_radar.to(device)
            batch_keypoints = batch_keypoints.to(device)

            preds = model(batch_radar)
            loss = criterion(preds, batch_keypoints)

            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(batch_keypoints.cpu())

    avg_loss = total_loss / len(loader)

    # Stack all batches into one big tensor for metric computation
    all_preds = torch.cat(all_preds, dim=0)         # (N, 17, 2)
    all_targets = torch.cat(all_targets, dim=0)     # (N, 17, 2)

    overall_pck, per_joint_pck = compute_pck(all_preds, all_targets)
    overall_mae, per_joint_mae = compute_per_joint_error(all_preds, all_targets)

    return {
        "loss": avg_loss,
        "pck": overall_pck,
        "per_joint_pck": per_joint_pck,
        "mae": overall_mae,
        "per_joint_mae": per_joint_mae,
    }


class EpochLogger:
    """
    Tracks and prints metrics each epoch.

    Remembers the loss from epoch 1 so it can report % improvement
    at every subsequent epoch — a simple way to see "how much have
    we learned since the start?"

    Usage:
        logger = EpochLogger()
        for epoch in range(num_epochs):
            metrics = run_eval(model, loader, criterion, device)
            logger.log(epoch + 1, metrics)

        logger.print_joint_summary(metrics)   # detailed per-joint table at end
    """

    def __init__(self) -> None:
        self._baseline_loss = None  # set on first call to log()

        # History lists — one entry appended per epoch, used for plotting
        self.epochs: list[int]   = []
        self.losses: list[float] = []
        self.improvements: list[float] = []   # % vs epoch 1
        self.maes: list[float]   = []
        self.pcks: list[float]   = []         # stored as 0–100 for readability

    def log(self, epoch: int, metrics: dict) -> None:
        """Print one-line summary for this epoch and store values for plotting."""
        loss = metrics["loss"]
        pck  = metrics["pck"]
        mae  = metrics["mae"]

        # Record the very first loss as our baseline
        if self._baseline_loss is None:
            self._baseline_loss = loss

        # % improvement: positive = better, negative = worse
        if self._baseline_loss > 0:
            improvement = (self._baseline_loss - loss) / self._baseline_loss * 100
        else:
            improvement = 0.0

        # Store for plotting
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.improvements.append(improvement)
        self.maes.append(mae)
        self.pcks.append(pck * 100)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.6f} | "
            f"vs Epoch 1: {improvement:+.1f}% | "
            f"PCK@0.2: {pck * 100:.1f}% | "
            f"MAE: {mae:.4f}"
        )

    def print_joint_summary(self, metrics: dict) -> None:
        """
        Print a full per-joint breakdown table.
        Call this at the end of training to see which joints are hard.

        Example output:
            Per-joint results (final epoch):
              nose          PCK: 91.2%  MAE: 0.0312  ████████████████████
              left_wrist    PCK: 54.1%  MAE: 0.1204  ██████████░░░░░░░░░░
        """
        print("\nPer-joint results (final epoch):")
        print(f"  {'joint':<16} {'PCK@0.2':>8}  {'MAE':>7}  bar")
        print(f"  {'-'*16} {'-'*8}  {'-'*7}  ---")

        for name, pck_j, mae_j in zip(
            JOINT_NAMES,
            metrics["per_joint_pck"],
            metrics["per_joint_mae"],
        ):
            filled = int(pck_j * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {name:<16} {pck_j * 100:7.1f}%  {mae_j:.4f}   {bar}")

    def save_plot(self, output_path: str = "training_curves.png") -> None:
        """
        Save a 4-panel plot of training history to disk.

        The four panels are:
          Top-left:     Loss per epoch
          Top-right:    % improvement vs epoch 1
          Bottom-left:  MAE (mean joint error) per epoch
          Bottom-right: PCK@0.2 (% correct keypoints) per epoch

        Args:
            output_path: where to save the PNG, e.g. "checkpoints/training_curves.png"

        Call this once after your training loop finishes:
            logger.save_plot("checkpoints/training_curves.png")
        """
        if not self.epochs:
            print("No data to plot — call logger.log() at least once first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training History", fontsize=14, fontweight="bold")

        # --- Top-left: Loss ---
        ax = axes[0, 0]
        ax.plot(self.epochs, self.losses, color="steelblue", linewidth=2)
        ax.set_title("Loss per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # --- Top-right: % improvement vs epoch 1 ---
        ax = axes[0, 1]
        ax.plot(self.epochs, self.improvements, color="darkorange", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)  # baseline at 0%
        ax.set_title("Improvement vs Epoch 1 (%)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("% Improvement")
        ax.grid(True, alpha=0.3)

        # --- Bottom-left: MAE ---
        ax = axes[1, 0]
        ax.plot(self.epochs, self.maes, color="mediumseagreen", linewidth=2)
        ax.set_title("Mean Joint Error (MAE) per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (normalized coords)")
        ax.grid(True, alpha=0.3)

        # --- Bottom-right: PCK@0.2 ---
        ax = axes[1, 1]
        ax.plot(self.epochs, self.pcks, color="mediumpurple", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_title("PCK@0.2 per Epoch (%)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("% Correct Keypoints")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Make sure the output directory exists before saving
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # free memory — important if training many models
        print(f"Training curves saved to: {output_path}")
