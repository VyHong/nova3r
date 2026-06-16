"""3D-FRONT surface and SDF dataset for Mesh2SDF training."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from training.nova3r.data.base_dataset import BaseDataset


class Front3DSDFDataset(BaseDataset):
    """Load precomputed 3D-FRONT surface and signed-distance samples."""

    def __init__(
        self,
        common_conf,
        data_root,
        split="train",
        samples_list_path=None,
        sdf_data_root=None,
        augment_translation=False,
    ):
        super().__init__(common_conf)

        self.data_root = Path(data_root)
        self.sdf_data_root = Path(sdf_data_root) if sdf_data_root else self.data_root
        self.split = split
        self.augment_translation = augment_translation

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

    @staticmethod
    def save_debug_surface(
        surface, output_dir="/mnt/home/vyhong/projects/nova3r/debug_points", filename="surface.ply"
    ):
        """Save surface points, normals, and labels to an ASCII PLY file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(surface, torch.Tensor):
            surface = surface.detach().cpu().numpy()
        else:
            surface = np.asarray(surface)

        if surface.ndim != 2 or surface.shape[1] < 3:
            raise ValueError(f"Expected surface shape (N, >=3), got {surface.shape}")

        has_normals = surface.shape[1] >= 6
        has_labels = surface.shape[1] >= 7
        properties = ["property float x", "property float y", "property float z"]
        if has_normals:
            properties.extend(
                ["property float nx", "property float ny", "property float nz"]
            )
        if has_labels:
            properties.append("property uchar sharp")

        output_path = output_dir / filename
        with output_path.open("w") as file:
            file.write(
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {surface.shape[0]}\n"
                + "\n".join(properties)
                + "\nend_header\n"
            )
            for point in surface:
                values = [f"{value:.6f}" for value in point[: 6 if has_normals else 3]]
                if has_labels:
                    values.append(str(int(point[6])))
                file.write(" ".join(values) + "\n")

        return output_path

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
                "geo_points": Path(f"{output_prefix}_sdf.npz"),
            }

    def load_surface_points(self, rng, random_surface, sharpedge_surface):
        surface_normal = []
        if self.pc_size > 0:
            indices = rng.choice(
                random_surface.shape[0], self.pc_size, replace=False
            )
            sampled_surface = random_surface[indices]
            if self.sharpedge_label:
                labels = np.zeros((self.pc_size, 1))
                sampled_surface = np.concatenate(
                    (sampled_surface, labels), axis=1
                )
            surface_normal.append(sampled_surface)

        if self.pc_sharpedge_size > 0:
            indices = rng.choice(
                sharpedge_surface.shape[0],
                self.pc_sharpedge_size,
                replace=False,
            )
            sampled_sharpedge = sharpedge_surface[indices]
            if self.sharpedge_label:
                labels = np.ones((self.pc_sharpedge_size, 1))
                sampled_sharpedge = np.concatenate(
                    (sampled_sharpedge, labels), axis=1
                )
            surface_normal.append(sampled_sharpedge)

        if not surface_normal:
            raise ValueError("At least one surface sample count must be positive.")

        surface_normal = torch.from_numpy(
            np.concatenate(surface_normal, axis=0)
        ).float()
        surface = surface_normal[:, :3]
        normal = torch.nn.functional.normalize(
            surface_normal[:, 3:6], p=2, dim=1
        )

        if self.return_normal:
            surface = torch.cat((surface, normal), dim=-1)
        if self.sharpedge_label:
            surface = torch.cat((surface, surface_normal[:, -1:]), dim=-1)
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
            indices = rng.choice(vol_points.shape[0], self.sdf_size, replace=False)
            samples = np.concatenate(
                (vol_points[indices], vol_label[indices, None]), axis=1
            )
            if self.sdf_label:
                samples = np.concatenate(
                    (samples, np.zeros((self.sdf_size, 1))), axis=1
                )
            sdf_points.append(samples)

        if self.sdf_near_size > 0:
            indices = rng.choice(
                near_points.shape[0], self.sdf_near_size, replace=False
            )
            samples = np.concatenate(
                (near_points[indices], near_label[indices, None]), axis=1
            )
            if self.sdf_label:
                samples = np.concatenate(
                    (samples, np.ones((self.sdf_near_size, 1))), axis=1
                )
            sdf_points.append(samples)

        if self.sdf_sharpedge_size > 0:
            indices = rng.choice(
                sharpedge_points.shape[0],
                self.sdf_sharpedge_size,
                replace=False,
            )
            samples = np.concatenate(
                (
                    sharpedge_points[indices],
                    sharpedge_label[indices, None],
                ),
                axis=1,
            )
            if self.sdf_label:
                labels = np.full((self.sdf_sharpedge_size, 1), 2)
                samples = np.concatenate((samples, labels), axis=1)
            sdf_points.append(samples)

        if not sdf_points:
            raise ValueError("At least one SDF sample count must be positive.")

        return torch.from_numpy(np.concatenate(sdf_points, axis=0)).float()

    def get_data(self, seq_name):
        metadata = self.data_store[seq_name]
        rng = np.random.default_rng()

        with np.load(metadata["surface"], allow_pickle=True) as surface_data:
            surface = self.load_surface_points(
                rng,
                random_surface=surface_data["random_surface"],
                sharpedge_surface=surface_data["sharp_surface"],
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
            )

        T = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        total_transform = torch.from_numpy(T).float()
        surface[:, :3] = (total_transform[:3, :3] @ surface[:, :3].T).T + total_transform[:3, 3]
        surface[:, 3:6] = (total_transform[:3, :3] @ surface[:, 3:6].T).T
        geo_points[:, :3] = (total_transform[:3, :3] @ geo_points[:, :3].T).T + total_transform[:3, 3]

        # self.save_debug_surface(surface, filename=f"debug_surface.ply")

        if self.augment_translation:
            self._augment_with_translation(rng, surface, geo_points)

        frame_count = 6
        images = [
            torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            for _ in range(frame_count)
        ]
        extrinsics = torch.eye(4).repeat(frame_count, 1, 1)
        intrinsics = torch.eye(3).repeat(frame_count, 1, 1)

        return {
            "seq_name": f"3DFront_{seq_name}",
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
