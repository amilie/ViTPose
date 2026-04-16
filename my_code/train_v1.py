import torch
from torch.utils.data import DataLoader, Subset

from dataset import MMVRRadarPoseDataset
from model import SimpleRadarPoseCNN


def main():
    root_dir = "../../P2_02"

    # Load full dataset
    full_dataset = MMVRRadarPoseDataset(root_dir=root_dir)

    # Sanity-check subset: first 8 samples
    small_dataset = Subset(full_dataset, list(range(8)))

    loader = DataLoader(small_dataset, batch_size=4, shuffle=True)

    model = SimpleRadarPoseCNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_radar, batch_keypoints in loader:
            preds = model(batch_radar)
            loss = criterion(preds, batch_keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()