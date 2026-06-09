"""
Main file for loading and processing the ReplicaPano dataset
author: YuanDong
"""

import json
import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
from torchvision import transforms
from depth_anything_3.utils.geometry import affine_inverse
from training.nova3r.data.dataset_utils import read_image_cv2

from training.nova3r.data.datasets.replica_utils.igibson_utils import ReplicaPanoScene
from training.nova3r.data.base_dataset import BaseDataset
import cv2
import os
import torch
from torch.utils.data._utils.collate import default_collate


class Front3DDataset(BaseDataset):
    """
    ReplicaPano Dataset implementation for loading 360-degree panoramic scenes.

    This dataset loads ReplicaPano scenes from pickle files and processes them
    to provide image, depth, and camera parameter data for training.
    """

    def __init__(
        self,
        common_conf,
        data_root,
        split="train",
        samples_list_path=None,
    ):
        """
        Initialize the ReplicaPano dataset.

        Args:
            common_conf: Common configuration from BaseDataset
            data_root: Root directory containing pickle files or scene data
            split: Dataset split ('train', 'val', 'test'). Default: 'train'
            scenes_list_path: Path to a file containing a list of specific scenes to load. If None, load all scenes.
        """
        super().__init__(common_conf)
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.data_root = Path(data_root)
        self.split = split

        self.sequence_list = []

        with open(samples_list_path, "r") as f:
            self.sample_list = json.load(f)
        for scene in self.sample_list:
            self.sequence_list.append(scene)

        self.data_store = {}
        self._load_metadata()

        self.img_norm = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.sample_list)

    def _load_metadata(self):
        """
        Load available scene pickle files from data_root.

        Args:
            scenes_list: Optional list of specific scene names to load
        """
        for scene in self.sequence_list:
            entry = {}
            data_path = self.data_root / scene

            entry["world_points_path"] = f"{data_path}_full.ply"

            self.data_store[scene] = entry

    def resize_image(self, image, interpolation=cv2.INTER_LANCZOS4):
        """
        Resize image to the target img_size.
        """
        return cv2.resize(image, (self.img_size, self.img_size), interpolation=interpolation)

    def __getitem__(self, index):
        seq_name = self.sample_list[index]
        return self.get_data(seq_name=seq_name)

    def get_data(
        self,
        seq_name: str = None,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            id (int): Specific ID to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        data_entry = self.data_store[seq_name]
        scene_pcd = o3d.io.read_point_cloud(data_entry["world_points_path"])
        # from y up to y down
        T = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        scene_pcd.transform(T)
        cam_points = np.asarray(scene_pcd.points)
        # self.save_debug_points(cam_points, filename=f"3d_front_cam_points.ply")
        point_masks = np.ones(cam_points.shape[0], dtype=bool)  # Assuming all points are valid for now

        intrinsics = torch.eye(3).repeat(6, 1, 1)  # Placeholder intrinsics (identity matrix)
        images = [torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32) for _ in range(6)]  # Placeholder images (black)
        normalized_extrinsics = torch.eye(4).repeat(6, 1, 1)
        original_sizes = [
            np.array([self.img_size, self.img_size], dtype=np.int64) for _ in range(6)
        ]  # Placeholder original sizes (same as target size)
        subseq_ids = np.zeros(6, dtype=np.int64)

        set_name = "3DFront"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "id": "0",
            "subseq_ids": subseq_ids,
            "frame_num": len(normalized_extrinsics),
            "images": images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            # "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        return batch

    @staticmethod
    def save_debug_points(points, output_dir="debug_points", filename="points.ply"):
        """Saves points array to a PLY file for debugging"""
        import os
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)
        if not isinstance(points, np.ndarray):
            pts_np = np.asarray(points.points)
        else:
            pts_np = points
        pts_np = pts_np.reshape(-1, 3)

        ply_path = os.path.join(output_dir, filename)
        header = f"ply\nformat ascii 1.0\nelement vertex {len(pts_np)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"

        with open(ply_path, "w") as f:
            f.write(header)
            for p in pts_np:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        print(f"Saved {ply_path}")

    def dynamic_pad_collate_fn(self, batch):
        """
        Collates variable-sized point clouds by dynamically finding the max
        point count within this specific batch, padding them to the front,
        and packaging the rest of the metadata.
        """
        # 1. Find the maximum number of points present ONLY in this batch
        max_pts_in_batch = max([item["cam_points"].shape[0] for item in batch])
        batch_size = len(batch)

        # 2. Allocate uniform tensors for the point data
        # padded_world_pts = torch.zeros((batch_size, max_pts_in_batch, 3), dtype=torch.float32)
        padded_cam_pts = torch.zeros((batch_size, max_pts_in_batch, 3), dtype=torch.float32)
        point_masks = torch.zeros((batch_size, max_pts_in_batch), dtype=torch.bool)
        valid_counts = torch.zeros(batch_size, dtype=torch.long)

        # 3. Populate tensors (this naturally places valid data at the front)
        for idx, item in enumerate(batch):
            # w_pts = torch.as_tensor(item["world_points"], dtype=torch.float32)
            c_pts = torch.as_tensor(item["cam_points"], dtype=torch.float32)
            num_pts = c_pts.shape[0]

            # padded_world_pts[idx, :num_pts, :] = w_pts
            padded_cam_pts[idx, :num_pts, :] = c_pts
            point_masks[idx, :num_pts] = True
            valid_counts[idx] = num_pts

        # 4. Handle all other keys (images, matrices, names) smoothly
        collated_batch = {}
        for key in batch[0].keys():
            if key in ["cam_points", "point_masks"]:  # "world_points"
                continue  # We already handled these manually above

            if key in ["seq_name", "id"]:
                # Keep strings/IDs as simple lists
                collated_batch[key] = [item[key] for item in batch]
            else:
                # Let PyTorch handle uniform items like images, extrinsics, and intrinsics
                collated_batch[key] = default_collate([item[key] for item in batch])

        # Add our freshly padded point tensors back into the final dictionary
        # collated_batch["world_points"] = padded_world_pts
        collated_batch["cam_points"] = padded_cam_pts
        collated_batch["point_masks"] = point_masks
        collated_batch["valid_counts"] = valid_counts  # Crucial for your GPU FPS kernel!

        return collated_batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Front3DDataset")
    parser.add_argument("--data_root", type=str, default="/tmp/datasets/3DFront/3D-FRONT-TEST-SCENE/", help="Root directory of the dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train/val/test)")
    args = parser.parse_args()

    conf = {
        "img_size": 518,
        "patch_size": 16,
        "aug_scale": {
            "scales": [0.5, 1.0, 1.5],
        },
        "rescale": True,
        "rescale_aug": True,
        "landscape_check": True,
        "allow_duplicate_img": False,
    }
    from omegaconf import OmegaConf

    common_conf = OmegaConf.create(conf)  # Assuming OmegaConf is defined elsewhere
    dataset = Front3DDataset(
        common_conf=common_conf, data_root=args.data_root, split=args.split, samples_list_path="3D-Front/midi_test_room_ids.json"
    )

    print(f"Dataset length: {len(dataset)}")
    for scene in dataset:
        print(f"Scene: {scene['seq_name']}, ID: {scene['id']}, Frame Num: {scene['frame_num']}")
        print(f"Image shapes: {[img.shape for img in scene['images']]}")
        print(f"Extrinsics shapes: {[ext.shape for ext in scene['extrinsics']]}")
        print(f"Intrinsics shapes: {[int.shape for int in scene['intrinsics']]}")
        print(f"Cam points shape: {scene['cam_points'].shape}")
        print(f"Point masks shape: {scene['point_masks'].shape}")
