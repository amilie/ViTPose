import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import MMVRRadarPoseDataset
from model import SimpleRadarPoseCNN
from evaluate import run_eval, EpochLogger


def main():
    root_dir = "../P1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_dataset = MMVRRadarPoseDataset(root_dir=root_dir)
    loader = DataLoader(full_dataset, batch_size=4, shuffle=True)

    model = SimpleRadarPoseCNN().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 47
    logger = EpochLogger()

    output_dir = Path("checkpoints_v1")
    output_dir.mkdir(exist_ok=True)                                                                                                                                                                                                                                                               
    best_loss = float("inf")
    num_batches = len(loader)

    for epoch in range(num_epochs):
        model.train()
        for i, (batch_radar, batch_keypoints) in enumerate(loader):
            batch_radar = batch_radar.to(device)
            batch_keypoints = batch_keypoints.to(device)

            preds = model(batch_radar)
            loss = criterion(preds, batch_keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"  Epoch {epoch+1:03d}/{num_epochs} | "
                f"Batch {i+1}/{num_batches} | "
                f"Loss: {loss.item():.6f}",
                end="\r", flush=True,
            )

        print()

        metrics = run_eval(model, loader, criterion, device)
        logger.log(epoch + 1, metrics)

        torch.save(model.state_dict(), output_dir / "latest.pth")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  Saved best model (loss: {best_loss:.6f})")

        logger.save_plot(str(output_dir / "training_curves.png"))

    logger.print_joint_summary(metrics)


if __name__ == "__main__":
    main()
