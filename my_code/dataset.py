from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MMVRRadarPoseDataset(Dataset):
    """
    PyTorch Dataset for MMVR radar -> pose learning.

    Each sample returns:
        radar_tensor: FloatTensor of shape (2, 256, 128)
        keypoints_tensor: FloatTensor of shape (17, 2)
    """

    def __init__(self, root_dir: str, normalize_keypoints: bool = True) -> None:
        """
        Args:
            root_dir: Path to the extracted dataset root, e.g. ".../P2_02"
            normalize_keypoints: Whether to normalize keypoints using bbox_i
        """
        self.root_dir = Path(root_dir)
        self.normalize_keypoints = normalize_keypoints
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(f"No valid samples found under {self.root_dir}")

    def _collect_samples(self) -> List[Tuple[Path, Path, Path]]:
        """
        Find all triplets of:
            *_radar.npz
            *_pose.npz
            *_bbox.npz

        Returns:
            List of tuples: (radar_path, pose_path, bbox_path)
        """
        samples: List[Tuple[Path, Path, Path]] = []

        for radar_path in sorted(self.root_dir.rglob("*_radar.npz")):
            stem = radar_path.name.replace("_radar.npz", "")
            folder = radar_path.parent

            pose_path = folder / f"{stem}_pose.npz"
            bbox_path = folder / f"{stem}_bbox.npz"

            if pose_path.exists() and bbox_path.exists():
                samples.append((radar_path, pose_path, bbox_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        radar_path, pose_path, bbox_path = self.samples[idx]

        # Load radar
        radar_data = np.load(radar_path)
        hm_hori = radar_data["hm_hori"].astype(np.float32)
        hm_vert = radar_data["hm_vert"].astype(np.float32)

        # Stack into (2, 256, 128)
        radar = np.stack([hm_hori, hm_vert], axis=0)

        # Optional radar normalization
        radar = self._normalize_radar(radar)

        # Load pose
        pose_data = np.load(pose_path)
        kp = pose_data["kp"].astype(np.float32)

        # For now, only use the first person in the frame
        # Shape becomes (17, 3)
        kp = kp[0]

        # Keep only x, y
        keypoints = kp[:, :2]

        # Normalize keypoints using image bbox
        if self.normalize_keypoints:
            bbox_data = np.load(bbox_path)
            bbox_i = bbox_data["bbox_i"].astype(np.float32)[0]  # (5,)
            keypoints = self._normalize_keypoints_with_bbox(keypoints, bbox_i)

        radar_tensor = torch.from_numpy(radar)
        keypoints_tensor = torch.from_numpy(keypoints)

        return radar_tensor, keypoints_tensor

    def _normalize_radar(self, radar: np.ndarray) -> np.ndarray:
        """
        Normalize each radar channel independently to [0, 1].
        """
        radar = radar.copy()

        for c in range(radar.shape[0]):
            channel = radar[c]
            min_val = channel.min()
            max_val = channel.max()

            if max_val > min_val:
                radar[c] = (channel - min_val) / (max_val - min_val)
            else:
                radar[c] = np.zeros_like(channel)

        return radar

    def _normalize_keypoints_with_bbox(
        self, keypoints: np.ndarray, bbox_i: np.ndarray
    ) -> np.ndarray:
        """
        Normalize x,y keypoints relative to bbox_i = [x1, y1, x2, y2, conf]
        """
        x1, y1, x2, y2, _ = bbox_i
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox with non-positive size: {bbox_i}")

        kp_norm = keypoints.copy()
        kp_norm[:, 0] = (kp_norm[:, 0] - x1) / w
        kp_norm[:, 1] = (kp_norm[:, 1] - y1) / h

        return kp_norm