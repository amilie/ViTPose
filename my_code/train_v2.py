from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import MMVRRadarPoseDataset
from model import SimpleRadarPoseCNN


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
    root_dir = "../../P2_02"
    batch_size = 8
    learning_rate = 1e-3
    num_epochs = 30
    train_ratio = 0.8
    random_seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = MMVRRadarPoseDataset(root_dir=root_dir)
    print("Total samples:", len(dataset))

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleRadarPoseCNN().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
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
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved new best model to {save_path}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()