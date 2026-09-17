"""3D-FRONT surface and SDF dataset for Mesh2SDF training."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from training.mesh2sdf.data.sdf_dataset_mixin import SDFDatasetMixin
from nova3r.utils.batch_debug import save_batch_debug


class Front3DSDFDataset(SDFDatasetMixin):
    """Load precomputed 3D-FRONT surface and signed-distance samples."""

    def __init__(
        self,
        common_conf,
        data_root,
        split="train",
        samples_list_path=None,
        sdf_data_root=None,
        augment_translation=False,
        augment_rotation=False,
        augment_horizontal_rotation=False,
        distance_suffix="udf",
    ):
        super().__init__(common_conf)

        self.data_root = Path(data_root)
        self.sdf_data_root = Path(sdf_data_root) if sdf_data_root else self.data_root
        self.split = split
        self.augment_translation = augment_translation
        self.augment_rotation = augment_rotation or augment_horizontal_rotation
        self.augment_horizontal_rotation = augment_horizontal_rotation
        self.distance_suffix = str(distance_suffix).lower()
        if self.distance_suffix not in {"sdf", "udf"}:
            raise ValueError("distance_suffix must be 'sdf' or 'udf'.")

        if samples_list_path is None:
            self.samples_list = self._discover_samples()
        else:
            with open(samples_list_path, "r") as file:
                self.samples_list = json.load(file)

        self.data_store = {}
        self._load_metadata()

        self.pc_size = common_conf.pc_size
        self.pc_sharpedge_size = common_conf.pc_sharpedge_size
        self.return_normal = common_conf.return_normal
        self.sharpedge_label = common_conf.sharpedge_label

        self.sdf_size = common_conf.sdf_size
        self.sdf_near_size = common_conf.sdf_near_size
        self.sdf_sharpedge_size = common_conf.sdf_sharpedge_size
        self.sdf_label = common_conf.sdf_label

    def _augment_with_translation(self, rng, surface, geo_points):
        surface_min = surface[:, :3].amin(dim=0)
        surface_max = surface[:, :3].amax(dim=0)
        translation = torch.from_numpy(
            rng.uniform(surface_min.numpy(), surface_max.numpy())
        ).to(dtype=surface.dtype)
        surface[:, :3] += translation
        geo_points[:, :3] += translation

    def _augment_with_rotation(self, rng, surface, geo_points):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        rotation = torch.tensor(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=surface.dtype,
            device=surface.device,
        )

        surface[:, :3] = (rotation @ surface[:, :3].T).T
        if surface.shape[1] >= 6:
            surface[:, 3:6] = (rotation @ surface[:, 3:6].T).T
        geo_points[:, :3] = (rotation @ geo_points[:, :3].T).T

    def __len__(self):
        return len(self.samples_list)

    def _discover_samples(self):
        if not self.sdf_data_root.is_dir():
            raise ValueError(
                f"SDF data root directory does not exist: {self.sdf_data_root}"
            )

        suffix = "_full_surface.npz"
        samples = []
        for surface_path in sorted(self.sdf_data_root.rglob(f"*{suffix}")):
            relative_path = surface_path.relative_to(self.sdf_data_root).as_posix()
            samples.append(relative_path[: -len(suffix)])
        return samples

    def _load_metadata(self):
        for scene in self.samples_list:
            output_prefix = self.sdf_data_root / f"{scene}_full"
            self.data_store[scene] = {
                "surface": Path(f"{output_prefix}_surface.npz"),
                "geo_points": Path(f"{output_prefix}_{self.distance_suffix}.npz"),
            }

    def get_data(self, seq_name):
        metadata = self.data_store[seq_name]
        if self.split == "train":
            rng = np.random.default_rng()
        else:
            rng = self.deterministic_rng(type(self).__name__, self.split, seq_name)

        with np.load(metadata["surface"], allow_pickle=True) as surface_data:
            surface = self.load_surface_points(
                rng,
                random_surface=surface_data["random_surface"],
                sharpedge_surface=surface_data["sharp_surface"],
                **self.surface_rgb_kwargs(surface_data),
            )

        with np.load(metadata["geo_points"], allow_pickle=True) as sdf_data:
            geo_points = self.load_sdf_points(
                rng,
                vol_points=sdf_data["vol_points"],
                vol_label=sdf_data["vol_label"],
                near_points=sdf_data["random_near_points"],
                near_label=sdf_data["random_near_label"],
                sharpedge_points=sdf_data["sharp_near_points"],
                sharpedge_label=sdf_data["sharp_near_label"],
                **self.sdf_rgb_kwargs(sdf_data),
            )

        T = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        total_transform = torch.from_numpy(T).float()
        surface[:, :3] = (total_transform[:3, :3] @ surface[:, :3].T).T + total_transform[:3, 3]
        surface[:, 3:6] = (total_transform[:3, :3] @ surface[:, 3:6].T).T
        geo_points[:, :3] = (total_transform[:3, :3] @ geo_points[:, :3].T).T + total_transform[:3, 3]


        if self.augment_rotation:
            self._augment_with_rotation(rng, surface, geo_points)
        if self.augment_translation:
            self._augment_with_translation(rng, surface, geo_points)

        # save_batch_debug({"surface": surface}, "debug_points/front3d", batched=False)
        frame_count = 6
        images = [
            torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            for _ in range(frame_count)
        ]
        extrinsics = torch.eye(4).repeat(frame_count, 1, 1)
        intrinsics = torch.eye(3).repeat(frame_count, 1, 1)

        return {
            "seq_name": f"3DFront_{seq_name}",
            "id": seq_name,
            "subseq_ids": torch.arange(frame_count, dtype=torch.long),
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "surface": surface,
            "geo_points": geo_points,
        }

    def __getitem__(self, index):
        return self.get_data(self.samples_list[index])

    def dynamic_pad_collate_fn(self, batch):
        return default_collate(batch)
