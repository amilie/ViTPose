import torch
from torch.utils.data import DataLoader, Subset

from dataset import MMVRRadarPoseDataset
from model import SimpleRadarPoseCNN
from evaluate import run_eval, EpochLogger


def main():
    root_dir = "../P2_02"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load full dataset, use first 8 samples as a smoke test
    full_dataset = MMVRRadarPoseDataset(root_dir=root_dir)
    small_dataset = Subset(full_dataset, list(range(3)))
    loader = DataLoader(full_dataset, batch_size=4, shuffle=True)

    model = SimpleRadarPoseCNN().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    logger = EpochLogger()

    for epoch in range(num_epochs):
        # --- Training step ---
        model.train()
        for batch_radar, batch_keypoints in loader:
            batch_radar = batch_radar.to(device)
            batch_keypoints = batch_keypoints.to(device)

            preds = model(batch_radar)
            loss = criterion(preds, batch_keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Evaluation + logging ---
        metrics = run_eval(model, loader, criterion, device)
        logger.log(epoch + 1, metrics)

    # Print the per-joint breakdown once at the end
    logger.print_joint_summary(metrics)

    # Save the 4-panel training plot
    logger.save_plot("checkpoints_v1/training_curves.png")


if __name__ == "__main__":
    main()
