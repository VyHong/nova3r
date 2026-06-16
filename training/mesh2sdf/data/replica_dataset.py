"""
Main file for loading and processing the ReplicaPano dataset
author: YuanDong
"""

import json
import os
import argparse
import cv2
import numpy as np
from collections import OrderedDict  # For LRU Cache implementation
from pathlib import Path
import open3d as o3d
from torchvision import transforms
from depth_anything_3.utils.geometry import affine_inverse
from training.nova3r.data.dataset_utils import read_image_cv2
from training.nova3r.data.datasets.replica_utils.igibson_utils import ReplicaPanoScene
from training.nova3r.data.base_dataset import BaseDataset
import torch
from torch.utils.data._utils.collate import default_collate
import trimesh

from nova3r.heads.hunyuan_model.surface_loaders import SharpEdgeSurfaceLoader


class ReplicaPanoSDFDataset(BaseDataset):
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
        format="pointcloud",
        use_lru_cache=False,
    ):
        """
        Initialize the ReplicaPano dataset.

        Args:
            common_conf: Common configuration from BaseDataset
            data_root: Root directory containing pickle files or scene data
            split: Dataset split ('train', 'val', 'test'). Default: 'train'
            sample_list_path: Path to a file containing a list of specific samples to load. If None, load all samples.
            use_lru_cache: If True, caches cam_points dynamically to avoid redundant computations.
        """
        super().__init__(common_conf)
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.data_root = Path(data_root)
        self.sdf_data_root = "/mnt/home/vyhong/projects/nova3r/datasets/ReplicaPano/sdf"
        self.split = split
        self.format = format

        self.sequence_list = []
        if samples_list_path is None:
            self._load_sequence_list()
            self.samples_list = [f"{scene} {i:05}" for i in range(100) for scene in self.sequence_list]
            # save samples_list to a json file for future use
            with open(f"{split}_list.json", "w") as f:
                json.dump(self.samples_list, f, indent=4)

        else:
            with open(samples_list_path, "r") as f:
                self.samples_list = json.load(f)
            for scene in self.samples_list:
                self.sequence_list.append(f"{scene.split(' ')[0]}")

        self.data_store = {}
        self._load_metadata()

        self.img_norm = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.pc_size = common_conf.pc_size
        self.pc_sharpedge_size = common_conf.pc_sharpedge_size
        self.return_normal = common_conf.return_normal
        self.sharpedge_label = common_conf.sharpedge_label

        self.sdf_size = common_conf.sdf_size
        self.sdf_near_size = common_conf.sdf_near_size
        self.sdf_sharpedge_size = common_conf.sdf_sharpedge_size
        self.sdf_label = common_conf.sdf_label

    def __len__(self):
        return len(self.samples_list)

    def _load_sequence_list(self):
        """
        Load available scene pickle files from data_root.
        """
        if not self.data_root.exists():
            raise ValueError(f"Data root directory does not exist: {self.data_root}")

        for scene_folder in self.data_root.iterdir():
            if scene_folder.is_dir():
                self.sequence_list.append(scene_folder.name)

    def _load_metadata(self):
        """
        Load available scene pickle files from data_root.
        """
        for scene in self.sequence_list:
            sequence_metadata = {}
            for seq_entry_folder in Path(self.data_root / f"{scene}/{scene}/Scene_Info").iterdir():
                seq_entry_metadata = {}

                if scene.startswith("large_apartment"):
                    seq_entry_metadata["surface"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}cropped_surface.npz"
                    seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}cropped_sdf.npz"
                elif scene.startswith("hotel"):
                    if int(seq_entry_folder.name) < 18:
                        seq_entry_metadata["surface"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}0_cropped_surface.npz"
                        seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}0_cropped_sdf.npz"
                    else:
                        seq_entry_metadata["surface"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}1_cropped_surface.npz"
                        seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}1_cropped_sdf.npz"
                else:
                    # only adapt for room2 for now
                    seq_entry_metadata["surface"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}aligned_surface.npz"
                    seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}aligned_sdf.npz"

                    # seq_entry_metadata["surface"] = "/mnt/home/vyhong/projects/nova3r/datasets/hunyuan/00a4cff37043361068376104a292f5b44b5eacbd174651553b6a7ae35647a2a6_surface.npz"
                    # seq_entry_metadata["geo_points"] = "/mnt/home/vyhong/projects/nova3r/datasets/hunyuan/00a4cff37043361068376104a292f5b44b5eacbd174651553b6a7ae35647a2a6_sdf.npz"
                    # seq_entry_metadata["surface"] = "/mnt/home/vyhong/projects/nova3r/datasets/hunyuan/room_2_512_384_2m_surface.npz"
                    # seq_entry_metadata["geo_points"] = "/mnt/home/vyhong/projects/nova3r/datasets/hunyuan/room_2_512_384_2m_sdf.npz"

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

    def load_surface_points(self, rng, random_surface, sharpedge_surface):
        surface_normal = []
        if self.pc_size > 0:
            ind = rng.choice(random_surface.shape[0], self.pc_size, replace=False)
            random_surface = random_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size, 1))
                random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
            surface_normal.append(random_surface)

        if self.pc_sharpedge_size > 0:
            ind_sharpedge = rng.choice(sharpedge_surface.shape[0], self.pc_sharpedge_size, replace=False)
            sharpedge_surface = sharpedge_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_sharpedge_size, 1))
                sharpedge_surface = np.concatenate((sharpedge_surface, sharpedge_label), axis=1)
            surface_normal.append(sharpedge_surface)

        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size

        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
        return surface

    def load_sdf_points(
        self,
        rng,
        vol_points,
        vol_label,
        near_points,
        near_label,
        sharpedge_points,
        sharpedge_label,
    ):
        sdf_points = []
        if self.sdf_size > 0:
            ind = rng.choice(vol_points.shape[0], self.sdf_size, replace=False)
            vol_sdf_points = np.concatenate((vol_points[ind], vol_label[ind][:, None]), axis=1)
            if self.sdf_label:
                sdf_label = np.zeros((self.sdf_size, 1))
                vol_sdf_points = np.concatenate((vol_sdf_points, sdf_label), axis=1)
            sdf_points.append(vol_sdf_points)

        if self.sdf_near_size > 0:
            ind_near = rng.choice(near_points.shape[0], self.sdf_near_size, replace=False)
            near_sdf_points = np.concatenate((near_points[ind_near], near_label[ind_near][:, None]), axis=1)
            if self.sdf_label:
                sdf_label = np.ones((self.sdf_near_size, 1))
                near_sdf_points = np.concatenate((near_sdf_points, sdf_label), axis=1)
            sdf_points.append(near_sdf_points)

        if self.sdf_sharpedge_size > 0:
            ind_sharpedge = rng.choice(sharpedge_points.shape[0], self.sdf_sharpedge_size, replace=False)
            sharpedge_sdf_points = np.concatenate((sharpedge_points[ind_sharpedge], sharpedge_label[ind_sharpedge][:, None]), axis=1)
            if self.sdf_label:
                sdf_label = np.ones((self.sdf_sharpedge_size, 1)) * 2
                sharpedge_sdf_points = np.concatenate((sharpedge_sdf_points, sdf_label), axis=1)
            sdf_points.append(sharpedge_sdf_points)

        sdf_points = np.concatenate(sdf_points, axis=0)
        sdf_points = torch.FloatTensor(sdf_points)
        return sdf_points

    def get_data(
        self,
        seq_name: str = None,
        id: int = None,
        seq_index: int = None,
        img_per_seq: int = 1,
        subseq_ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        if subseq_ids is None:
            subseq_ids = np.arange(6)
            if self.split == "for da3 in training":
                subseq_ids = np.random.choice(subseq_ids, len(subseq_ids), replace=self.allow_duplicate_img)

        metadata = self.data_store[seq_name]

        if id is None:
            ids = np.random.choice(len(metadata), img_per_seq, replace=self.allow_duplicate_img)
        if isinstance(id, str):
            ids = [id]

        annos = [metadata[i] for i in ids]

        images, extrinsics, intrinsics, original_sizes = [], [], [], []

        for anno in annos:
            replica_scene = ReplicaPanoScene.from_pickle(anno["pkl_path"])

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
                subseq_intrinsics[0, :] *= new_size_hw[1] / float(orig_size_hw[1])
                subseq_intrinsics[1, :] *= new_size_hw[0] / float(orig_size_hw[0])

                subseq_w2c = np.array(camera_data[f"{subseq_id:04d}"]["extrinsics"])
                T_to_colmap = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

                scene_w2c = replica_scene.transform_3d.camera["world2cam3d"]
                colmap_scene_w2c = T_to_colmap @ scene_w2c @ T_to_colmap.T
                image_extrinsics = subseq_w2c @ colmap_scene_w2c

                if i == 0:

                    rng = np.random.default_rng()
                    surface = np.load(anno["surface"], allow_pickle=True)
                    geo_points = np.load(anno["geo_points"], allow_pickle=True)

                    total_transform = image_extrinsics @ T_to_colmap
                    scale = surface["scale"]
                    centroid = surface["centroid"]

                    transformed_centroid = total_transform[:3, :3] @ centroid
                    total_transform[:3, 3] += transformed_centroid
                    total_transform[:3, 3] /= scale

                    surface = self.load_surface_points(rng, random_surface=surface["random_surface"], sharpedge_surface=surface["sharp_surface"])
                    geo_points = self.load_sdf_points(
                        rng,
                        vol_points=geo_points["vol_points"],
                        vol_label=geo_points["vol_label"],
                        near_points=geo_points["random_near_points"],
                        near_label=geo_points["random_near_label"],
                        sharpedge_points=geo_points["sharp_near_points"],
                        sharpedge_label=geo_points["sharp_near_label"],
                    )
                    total_transform = torch.from_numpy(total_transform).float()
                    surface[:, :3] = (total_transform[:3, :3] @ surface[:, :3].T).T + total_transform[:3, 3]
                    surface[:, 3:6] = (total_transform[:3, :3] @ surface[:, 3:6].T).T
                    geo_points[:, :3] = (total_transform[:3, :3] @ geo_points[:, :3].T).T + total_transform[:3, 3]

                images.append(image)
                original_sizes.append(original_size)
                extrinsics.append(torch.from_numpy(image_extrinsics).float())
                intrinsics.append(subseq_intrinsics)

        ex_t_batched = torch.stack(extrinsics).unsqueeze(0)
        normalized_extrinsics = self._normalize_extrinsics(ex_t_batched).squeeze(0)

        del replica_scene
        intrinsics = torch.from_numpy(np.array(intrinsics))
        set_name = "replica_pano"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "images": images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "surface": surface,
            "geo_points": geo_points,
        }

        return batch

    def __getitem__(self, index):
        seq_name, id = self.samples_list[index].split(" ")
        return self.get_data(seq_name=seq_name, id=id)

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
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

    def dynamic_pad_collate_fn(self, batch):
        return default_collate(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ReplicaPanoDataset")
    parser.add_argument("--data_root", type=str, default="/tmp/datasets/replica_pano", help="Root directory of the dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train/val/test)")
    args = parser.parse_args()

    conf = {
        "img_size": 518,
        "patch_size": 16,
        "aug_scale": {"scales": [0.5, 1.0, 1.5]},
        "rescale": True,
        "rescale_aug": True,
        "landscape_check": True,
        "allow_duplicate_img": False,
    }
    from omegaconf import OmegaConf

    common_conf = OmegaConf.create(conf)

    # Example turning LRU off
    dataset = ReplicaPanoSDFDataset(
        common_conf=common_conf,
        data_root=args.data_root,
        split=args.split,
        samples_list_path="data/replica_pano/o_train_list.json",
        format="mesh",
        use_lru_cache=False,  # Switch this to True/False as needed
    )

    import cProfile
    import pstats

    def profile_dataset():
        for i in range(20):
            _ = dataset[i]

    profiler = cProfile.Profile()
    profiler.enable()

    profile_dataset()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(30)
