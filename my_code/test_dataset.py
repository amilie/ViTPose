from torch.utils.data import DataLoader
from dataset import MMVRRadarPoseDataset

dataset = MMVRRadarPoseDataset(root_dir="../../P2_02")
print("Number of samples:", len(dataset))

radar, keypoints = dataset[0]
print("Radar shape:", radar.shape)
print("Keypoints shape:", keypoints.shape)
print("Keypoints:", keypoints)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch_radar, batch_keypoints = next(iter(loader))
print("Batch radar shape:", batch_radar.shape)
print("Batch keypoints shape:", batch_keypoints.shape)