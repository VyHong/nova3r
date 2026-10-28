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
from torchvision import transforms
from training.nova3r.data.dataset_utils import read_image_cv2
from training.nova3r.data.datasets.replica_utils.igibson_utils import ReplicaPanoScene
import torch
import trimesh

from nova3r.heads.hunyuan_model.surface_loaders import SharpEdgeSurfaceLoader
from training.mesh2sdf.data.sdf_dataset_mixin import SDFDatasetMixin
from nova3r.utils.batch_debug import save_batch_debug


class ReplicaPanoSDFDataset(SDFDatasetMixin):
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
        format="mesh",
        load_conditioning_latent=False,
        conditioning_latent_root=None,
        subseq_ids=None,
        load_visible_geometry_latent=False,
        visible_geometry_latent_root=None,
    ):
        """
        Initialize the ReplicaPano dataset.

        Args:
            common_conf: Common configuration shared by the SDF datasets
            data_root: Root directory containing pickle files or scene data
            split: Dataset split ('train', 'val', 'test'). Default: 'train'
            sample_list_path: Path to a file containing a list of specific samples to load. If None, load all samples.
            load_conditioning_latent: Load a precomputed VAE conditioning latent
                for each panorama.
            conditioning_latent_root: Root containing
                ``<scene>/<sequence>.pt`` latent files.
        """
        super().__init__(common_conf)
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.data_root = Path(data_root)
        self.sdf_data_root = "/mnt/home/vyhong/projects/nova3r/datasets/ReplicaPano/udf_no_norm"
        self.split = split
        self.format = format
        self.configure_subseq_ids(subseq_ids)
        self.configure_conditioning_latents(
            load_conditioning_latent, conditioning_latent_root
        )
        self.configure_visible_geometry_latents(
            load_visible_geometry_latent, visible_geometry_latent_root
        )

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
            self.sequence_list = list(dict.fromkeys(scene.split(" ")[0] for scene in self.samples_list))

        self.data_store = {}
        self._load_metadata()

        self.img_norm = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.pc_size = common_conf.pc_size
        self.pc_sharpedge_size = common_conf.pc_sharpedge_size
        self.pointcloud_sample = getattr(common_conf, "pointcloud_sample", True)
        self.return_normal = common_conf.return_normal
        self.sharpedge_label = common_conf.sharpedge_label

        self.sdf_size = common_conf.sdf_size
        self.sdf_near_size = common_conf.sdf_near_size
        self.sdf_sharpedge_size = common_conf.sdf_sharpedge_size
        self.sdf_label = common_conf.sdf_label
        self.debug_vis_dir = "debug_points/replica_udf_vis"

    def _pointcloud_rng(self, seq_name, id, first_subseq_id):
        if self.split != "train":
            return self.deterministic_rng(type(self).__name__, self.split, seq_name, id, int(first_subseq_id))
        if self.pointcloud_sample:
            return np.random.default_rng()

        return self.deterministic_rng(type(self).__name__, self.split, seq_name, id, int(first_subseq_id))

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
                    seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}cropped_udf.npz"
                elif scene.startswith("hotel"):
                    if int(seq_entry_folder.name) < 18:
                        seq_entry_metadata["surface"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}0_cropped_surface.npz"
                        seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}0_cropped_udf.npz"
                    else:
                        seq_entry_metadata["surface"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}1_cropped_surface.npz"
                        seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}1_cropped_udf.npz"
                else:
                    seq_entry_metadata["surface"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}aligned_surface.npz"
                    seq_entry_metadata["geo_points"] = f"{self.sdf_data_root}/{scene}/{scene}/{scene[:-3]}aligned_udf.npz"

                seq_entry_metadata["seq_entry_folder"] = f"{seq_entry_folder}"
                pkl_file = seq_entry_folder / "data.pkl"
                seq_entry_metadata["pkl_path"] = pkl_file
                for subsequence_folder in sorted(seq_entry_folder.iterdir(), key=lambda path: path.name):
                    if subsequence_folder.is_dir():
                        seq_entry_metadata["subsequence_images"] = []
                        for subsequence_file in sorted(subsequence_folder.iterdir(), key=lambda path: path.name):
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

    @staticmethod
    def _intrinsics_to_resized_pixels(intrinsics, orig_size_hw, new_size_hw):
        intrinsics = intrinsics.copy()
        uses_normalized_intrinsics = (
            abs(intrinsics[0, 0]) <= 2.0 and abs(intrinsics[1, 1]) <= 2.0 and abs(intrinsics[0, 2]) <= 2.0 and abs(intrinsics[1, 2]) <= 2.0
        )
        if uses_normalized_intrinsics:
            intrinsics[0, :] *= float(orig_size_hw[1])
            intrinsics[1, :] *= float(orig_size_hw[0])

        intrinsics[0, :] *= new_size_hw[1] / float(orig_size_hw[1])
        intrinsics[1, :] *= new_size_hw[0] / float(orig_size_hw[0])
        return intrinsics

    def get_data(
        self,
        seq_name: str = None,
        id: int = None,
        seq_index: int = None,
        img_per_seq: int = 1,
        subseq_ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        subseq_ids = self.select_subseq_ids(subseq_ids)

        metadata = self.data_store[seq_name]

        ids = [f"{int(id):05d}"]

        annos = [metadata[i] for i in ids]

        images, extrinsics, intrinsics, original_sizes = [], [], [], []
        camera_translation_scale = None

        for anno in annos:
            replica_scene = ReplicaPanoScene.from_pickle(anno["pkl_path"])

            with open(anno["camera_data"], "r") as f:
                camera_data = json.load(f)

            for subseq_order, subseq_id in enumerate(subseq_ids):
                filepath = anno["subsequence_images"][subseq_id]
                image_path = os.path.join(self.data_root, filepath)
                image = read_image_cv2(image_path)
                orig_size_hw = image.shape[:2]
                image = self.resize_image(image, cv2.INTER_LANCZOS4)
                new_size_hw = image.shape[:2]

                image = self.img_norm(image)
                original_size = np.array(image.shape[1:])

                subseq_intrinsics = np.array(camera_data[f"{subseq_id:04d}"]["intrinsics"], dtype=np.float32)
                subseq_intrinsics = self._intrinsics_to_resized_pixels(
                    subseq_intrinsics,
                    orig_size_hw=orig_size_hw,
                    new_size_hw=new_size_hw,
                )

                subseq_w2c = np.array(camera_data[f"{subseq_id:04d}"]["extrinsics"])
                T_to_colmap = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

                scene_w2c = replica_scene.transform_3d.camera["world2cam3d"]
                colmap_scene_w2c = T_to_colmap @ scene_w2c @ T_to_colmap.T
                image_extrinsics = subseq_w2c @ colmap_scene_w2c

                if subseq_order == 0:

                    rng = self._pointcloud_rng(seq_name, id, subseq_ids[0])
                    surface = np.load(anno["surface"], allow_pickle=True)
                    geo_points = np.load(anno["geo_points"], allow_pickle=True)

                    total_transform = image_extrinsics @ T_to_colmap

                    total_transform = torch.from_numpy(total_transform).float()

                    surface = self.load_surface_points(
                        rng,
                        random_surface=surface["random_surface"],
                        sharpedge_surface=surface["sharp_surface"],
                        **self.surface_rgb_kwargs(surface),
                    )
                    mesh_bounds = self.load_mesh_bounds(geo_points)
                    geo_points = self.load_sdf_points(
                        rng,
                        vol_points=geo_points["vol_points"],
                        vol_label=geo_points["vol_label"],
                        near_points=geo_points["random_near_points"],
                        near_label=geo_points["random_near_label"],
                        sharpedge_points=geo_points["sharp_near_points"],
                        sharpedge_label=geo_points["sharp_near_label"],
                        **self.sdf_rgb_kwargs(geo_points),
                    )
                    geo_points = self.add_exact_surface_points(
                        rng, surface, geo_points
                    )
                    surface[:, :3] = (total_transform[:3, :3] @ surface[:, :3].T).T + total_transform[:3, 3]
                    surface[:, 3:6] = (total_transform[:3, :3] @ surface[:, 3:6].T).T
                    geo_points[:, :3] = (total_transform[:3, :3] @ geo_points[:, :3].T).T + total_transform[:3, 3]

                    mesh_bounds = mesh_bounds @ total_transform[:3, :3].T + total_transform[:3, 3]

                    camera_translation_scale = 1

                images.append(image)
                original_sizes.append(original_size)
                extrinsics.append(torch.from_numpy(image_extrinsics).float())
                intrinsics.append(subseq_intrinsics)

        ex_t_batched = torch.stack(extrinsics).unsqueeze(0)
        normalized_extrinsics = self._normalize_extrinsics(
            ex_t_batched,
            translation_scale=camera_translation_scale,
        ).squeeze(0)

        del replica_scene
        intrinsics = torch.from_numpy(np.array(intrinsics))
        set_name = "replica_pano"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "id": id,
            "subseq_ids": torch.as_tensor(subseq_ids, dtype=torch.long),
            "images": images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "surface": surface,
            "geo_points": geo_points,
            "mesh_bounds_corners": mesh_bounds,
        }
        if self.load_conditioning_latent:
            batch["conditioning_latents"] = self._load_conditioning_latent(seq_name, ids)
        if self.load_visible_geometry_latent:
            batch["visible_geometry_latents"] = self._load_visible_geometry_latent(seq_name, ids)
        # save_batch_debug(batch, self.debug_vis_dir, batched=False)
        return batch

    def __getitem__(self, index):
        seq_name, id = self.samples_list[index].split(" ")
        return self.get_data(seq_name=seq_name, id=id)


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
