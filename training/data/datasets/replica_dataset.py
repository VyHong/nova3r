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
from training.data.dataset_utils import read_image_cv2
from training.data.datasets.replica_utils.igibson_utils import ReplicaPanoScene
from training.data.base_dataset import BaseDataset
import cv2
import os
import torch
from torch.utils.data._utils.collate import default_collate


class ReplicaPanoDataset(BaseDataset):
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
        scenes_list_path=None,
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
        if scenes_list_path is None:
            self._load_scenes_list()
            self.scenes_list = [f"{scene} {i:05}" for i in range(100) for scene in self.sequence_list]
            # save scenes_list to a json file for future use
            with open(f"{split}_list.json", "w") as f:
                json.dump(self.scenes_list, f, indent=4)

        else:
            with open(scenes_list_path, "r") as f:
                self.scenes_list = json.load(f)
            for scene in self.scenes_list:
                self.sequence_list.append(f"{scene.split(' ')[0]}")

        self.data_store = {}
        self._load_metadata()

        self.img_norm = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.scenes_list)

    def _load_scenes_list(self):
        """
        Load available scene pickle files from data_root.

        Args:
            scenes_list: Optional list of specific scene names to load
        """
        if not self.data_root.exists():
            raise ValueError(f"Data root directory does not exist: {self.data_root}")

        for scene_folder in self.data_root.iterdir():
            if scene_folder.is_dir():
                self.sequence_list.append(scene_folder.name)

    def _load_metadata(self):
        """
        Load available scene pickle files from data_root.

        Args:
            scenes_list: Optional list of specific scene names to load
        """
        for scene in self.sequence_list:
            sequence_metadata = {}
            for seq_entry_folder in Path(self.data_root / f"{scene}/{scene}/Scene_Info").iterdir():
                seq_entry_metadata = {}

                if scene.startswith("large_apartment"):
                    seq_entry_metadata["world_points_path"] = f"{self.data_root}/{scene}/{scene}/{scene[:-3]}cropped.ply"
                elif scene.startswith("hotel"):
                    if int(seq_entry_folder.name) < 18:
                        seq_entry_metadata["world_points_path"] = f"{self.data_root}/{scene}/{scene}/{scene[:-3]}0_cropped.ply"
                    else:
                        seq_entry_metadata["world_points_path"] = f"{self.data_root}/{scene}/{scene}/{scene[:-3]}1_cropped.ply"
                else:
                    seq_entry_metadata["world_points_path"] = f"{self.data_root}/{scene}/{scene}/{scene[:-3]}aligned.ply"
                seq_entry_metadata["seq_entry_folder"] = f"{seq_entry_folder}"
                pkl_file = seq_entry_folder / "data.pkl"
                seq_entry_metadata["pkl_path"] = pkl_file
                for subsequence_folder in seq_entry_folder.iterdir():
                    if subsequence_folder.is_dir():
                        seq_entry_metadata["subsequence_images"] = []
                        for subsequence_file in subsequence_folder.iterdir():
                            if subsequence_file.suffix == ".jpg":
                                seq_entry_metadata["subsequence_images"].append(subsequence_file)
                            if subsequence_file.suffix == ".json":
                                seq_entry_metadata["camera_data"] = subsequence_file

                sequence_metadata[seq_entry_folder.name] = seq_entry_metadata
            self.data_store[scene] = sequence_metadata

    def resize_image(self, image, interpolation=cv2.INTER_LANCZOS4):
        """
        Resize image to the target img_size.
        """
        return cv2.resize(image, (self.img_size, self.img_size), interpolation=interpolation)

    def get_data(
        self,
        seq_name: str = None,
        id: int = None,
        seq_index: int = None,
        img_per_seq: int = 1,
        subseq_ids: list = None,
        aspect_ratio: float = 1.0,
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
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        if subseq_ids is None:
            subseq_ids = np.arange(6)
            if self.split == "train":
                subseq_ids = np.random.choice(subseq_ids, len(subseq_ids), replace=self.allow_duplicate_img)

        metadata = self.data_store[seq_name]

        if id is None:
            ids = np.random.choice(len(metadata), img_per_seq, replace=self.allow_duplicate_img)
        if isinstance(id, str):
            ids = [id]

        annos = [metadata[i] for i in ids]

        if subseq_ids is None:
            subseq_ids = np.arange(len(annos[0]["subsequence_images"]))

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        for anno in annos:

            replica_scene = ReplicaPanoScene.from_pickle(anno["pkl_path"])
            scene_pcd = o3d.io.read_point_cloud(anno["world_points_path"])

            with open(anno["camera_data"], "r") as f:
                camera_data = json.load(f)

            for i, subseq_id in enumerate(subseq_ids):
                filepath = anno["subsequence_images"][subseq_id]
                image_path = os.path.join(self.data_root, filepath)
                image = read_image_cv2(image_path)
                orig_size_hw = image.shape[:2]

                image = self.resize_image(image, cv2.INTER_LANCZOS4)
                new_size_hw = image.shape[:2]

                image = self.img_norm(image)
                original_size = np.array(image.shape[1:])

                subseq_intrinsics = np.array(camera_data[f"{subseq_id:04d}"]["intrinsics"], dtype=np.float32)

                # Correctly set up the intrinsics to account for the resize operation
                subseq_intrinsics[0, :] *= new_size_hw[1] / float(orig_size_hw[1])  # scale fx, cx
                subseq_intrinsics[1, :] *= new_size_hw[0] / float(orig_size_hw[0])  # scale fy, cy

                subseq_w2c = np.array(camera_data[f"{subseq_id:04d}"]["extrinsics"])
                T_to_colmap = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

                scene_w2c = replica_scene.transform_3d.camera["world2cam3d"]
                colmap_scene_w2c = T_to_colmap @ scene_w2c @ T_to_colmap.T

                image_extrinsics = subseq_w2c @ colmap_scene_w2c

                if i == 0:

                    colmap_points = scene_pcd.transform(T_to_colmap)
                    cam_points = colmap_points.transform(image_extrinsics)

                    # world_points = np.asarray(colmap_points.points)
                    cam_points = np.asarray(cam_points.points)
                    point_masks = np.ones(len(cam_points), dtype=bool)
                    # self.save_debug_points(world_points, output_dir="debug_points", filename=f"{seq_name}_{id}_world_points.ply")
                    # self.save_debug_points(cam_points, output_dir="debug_points", filename=f"{seq_name}_{id}_da3_cam_points.ply")
                images.append(image)
                original_sizes.append(original_size)
                # extrinsics in w2c
                extrinsics.append(torch.from_numpy(image_extrinsics).float())
                intrinsics.append(subseq_intrinsics)

        # We add a dummy batch dimension because _normalize_extrinsics expects it like (B, N, 4, 4)
        ex_t_batched = torch.stack(extrinsics).unsqueeze(0)
        normalized_extrinsics = self._normalize_extrinsics(ex_t_batched).squeeze(0)

        # Save normalized extrinsics as JSON for debugging / downstream use
        # norm_ex_np = normalized_extrinsics.cpu().numpy().tolist()
        # save_path = os.path.join("debug_points", f"{seq_name}_{str(id)}_normalized_extrinsics.json")
        # with open(save_path, "w") as jf:
        #     json.dump(norm_ex_np, jf, indent=4)

        intrinsics = torch.from_numpy(np.array(intrinsics))
        set_name = "replica_pano"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "id": id,
            "subseq_ids": subseq_ids,
            "frame_num": len(extrinsics),
            "images": images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            # "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        return batch

    def __getitem__(self, index):
        seq_name, id = self.scenes_list[index].split(" ")
        return self.get_data(seq_name=seq_name, id=id)

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def transform_points_post(self, points, transform_matrix):
        """
        Transforms points using post-multiplication (Point @ Matrix).
        Assumes points are shape (N, 3).
        """
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])
        result_homo = points_homo @ transform_matrix
        return result_homo[:, :3]

    def transform_points_pre(self, points, transform_matrix):
        """
        Transforms points using pre-multiplication (Matrix @ Point).
        Assumes points are shape (N, 3).
        """
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])
        result_homo = (transform_matrix @ points_homo.T).T
        return result_homo[:, :3]

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
    parser = argparse.ArgumentParser(description="Test ReplicaPanoDataset")
    parser.add_argument("--data_root", type=str, default="/tmp/datasets/replica_pano", help="Root directory of the dataset")
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
    dataset = ReplicaPanoDataset(
        common_conf=common_conf,
        data_root=args.data_root,
        split=args.split,
        # scenes_list_path="/workspaces/projects/nova3r/data/replica_pano/train_list.json"
    )

    print(f"Dataset length: {len(dataset)}")

    for scene in dataset:
        print(f"Scene: {scene['seq_name']}, ID: {scene['id']}, Frame Num: {scene['frame_num']}")
        print(f"Image shapes: {[img.shape for img in scene['images']]}")
        print(f"Extrinsics shapes: {[ext.shape for ext in scene['extrinsics']]}")
        print(f"Intrinsics shapes: {[int.shape for int in scene['intrinsics']]}")
        print(f"Cam points shape: {scene['cam_points'].shape}")
        print(f"World points shape: {scene['world_points'].shape}")
        print(f"Point masks shape: {scene['point_masks'].shape}")
