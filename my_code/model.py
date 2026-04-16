import torch
import torch.nn as nn


class SimpleRadarPoseCNN(nn.Module):
    """
    Simple CNN for radar-to-pose regression.

    Input:
        (batch_size, 2, 256, 128)

    Output:
        (batch_size, 17, 2)
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # -> (16, 128, 64)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # -> (32, 64, 32)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # -> (64, 32, 16)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # -> (128, 16, 8)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 34),  # 17 keypoints * 2 coordinates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        x = x.view(-1, 17, 2)
        return x