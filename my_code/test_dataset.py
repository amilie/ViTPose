from torch.utils.data import DataLoader
from dataset import MMVRRadarPoseDataset


def main():
    dataset = MMVRRadarPoseDataset(root_dir="../P2_02")
    print("Number of samples:", len(dataset))

    # Added: assert dataset is non-empty before proceeding
    assert len(dataset) > 0, "Dataset is empty"

    radar, keypoints = dataset[0]
    print("Radar shape:", radar.shape)
    print("Keypoints shape:", keypoints.shape)
    print("Keypoints:", keypoints)

    # Added: validate output shapes match expected dimensions
    assert radar.shape == (2, 256, 128), f"Unexpected radar shape: {radar.shape}"
    assert keypoints.shape == (17, 2), f"Unexpected keypoints shape: {keypoints.shape}"

    # Added: validate normalized values are in a reasonable range
    assert radar.min() >= 0.0 and radar.max() <= 1.0, "Radar values outside [0, 1]"
    print("Shape and range assertions passed.")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_radar, batch_keypoints = next(iter(loader))
    print("Batch radar shape:", batch_radar.shape)
    print("Batch keypoints shape:", batch_keypoints.shape)

    # Added: validate batch dimensions
    assert batch_radar.shape == (4, 2, 256, 128), f"Unexpected batch radar shape: {batch_radar.shape}"
    assert batch_keypoints.shape == (4, 17, 2), f"Unexpected batch keypoints shape: {batch_keypoints.shape}"
    print("Batch shape assertions passed.")


# Added: wrap in main() so this file can be imported without side effects
if __name__ == "__main__":
    main()
