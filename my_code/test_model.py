import torch
from torch.utils.data import DataLoader

from dataset import MMVRRadarPoseDataset
from model import SimpleRadarPoseCNN


def main():
    dataset = MMVRRadarPoseDataset(root_dir="../../P2_02")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch_radar, batch_keypoints = next(iter(loader))

    model = SimpleRadarPoseCNN()
    preds = model(batch_radar)

    criterion = torch.nn.MSELoss()
    loss = criterion(preds, batch_keypoints)

    print("Input shape:     ", batch_radar.shape)
    print("Target shape:    ", batch_keypoints.shape)
    print("Prediction shape:", preds.shape)
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()