"""Shared sampling helpers for Mesh2SDF datasets."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class SDFDatasetMixin(Dataset):
    """Base for Mesh2SDF datasets and their shared sampling behavior."""

    CUBEMAP_SUBSEQ_IDS = tuple(range(6))

    def __init__(self, common_conf):
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.aug_scale
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.pc_size = common_conf.pc_size
        self.pc_sharpedge_size = common_conf.pc_sharpedge_size
        self.pointcloud_sample = getattr(common_conf, "pointcloud_sample", True)
        self.return_normal = common_conf.return_normal
        self.sharpedge_label = common_conf.sharpedge_label
        self.sdf_size = common_conf.sdf_size
        self.sdf_near_size = common_conf.sdf_near_size
        self.sdf_sharpedge_size = common_conf.sdf_sharpedge_size
        self.sdf_label = common_conf.sdf_label

    def configure_conditioning_latents(self, enabled, root):
        self.load_conditioning_latent = bool(enabled)
        self.conditioning_latent_root = Path(root) if root is not None else None
        if self.load_conditioning_latent:
            if self.conditioning_latent_root is None:
                raise ValueError(
                    "conditioning_latent_root is required when load_conditioning_latent=True"
                )
            if not self.conditioning_latent_root.is_dir():
                raise ValueError(
                    f"Conditioning latent root does not exist: {self.conditioning_latent_root}"
                )

    def _pointcloud_rng(self, seq_name, sample_id, first_subseq_id):
        key = (type(self).__name__, self.split, seq_name, sample_id, int(first_subseq_id))
        if self.split == "train" and self.pointcloud_sample:
            return np.random.default_rng()
        return self.deterministic_rng(*key)

    def resize_image(self, image, interpolation=cv2.INTER_LANCZOS4):
        return cv2.resize(
            image, (self.img_size, self.img_size), interpolation=interpolation
        )

    @staticmethod
    def _image_to_tensor(image):
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)

    @staticmethod
    def _intrinsics_to_resized_pixels(intrinsics, orig_size_hw, new_size_hw):
        intrinsics = intrinsics.copy()
        if all(
            abs(intrinsics[row, col]) <= 2.0
            for row, col in ((0, 0), (1, 1), (0, 2), (1, 2))
        ):
            intrinsics[0, :] *= float(orig_size_hw[1])
            intrinsics[1, :] *= float(orig_size_hw[0])
        intrinsics[0, :] *= new_size_hw[1] / float(orig_size_hw[1])
        intrinsics[1, :] *= new_size_hw[0] / float(orig_size_hw[0])
        return intrinsics

    @staticmethod
    def _has_complete_cube_faces(path):
        return all((path / f"{face_idx:04d}.jpg").is_file() for face_idx in range(6))

    def _load_conditioning_latent(self, seq_name, ids):
        if len(ids) != 1:
            raise ValueError(
                "Precomputed conditioning latents require exactly one panorama per sample"
            )
        latent_path = self.conditioning_latent_root / seq_name / f"{ids[0]}.pt"
        checkpoint = torch.load(latent_path, map_location="cpu", weights_only=True)
        return checkpoint["latent"][0].float()

    @staticmethod
    def _normalize_extrinsics(extrinsics, translation_scale=None):
        if extrinsics is None:
            return None
        normalized = extrinsics @ torch.linalg.inv(extrinsics[:, :1])
        if translation_scale is None:
            translations = torch.linalg.inv(normalized)[..., :3, 3]
            translation_scale = translations.norm(dim=-1).median().clamp(min=1e-1)
        else:
            translation_scale = torch.as_tensor(
                translation_scale,
                dtype=normalized.dtype,
                device=normalized.device,
            ).clamp(min=1e-6)
        normalized[..., :3, 3] /= translation_scale
        return normalized

    @staticmethod
    def dynamic_pad_collate_fn(batch):
        return default_collate(batch)

    @classmethod
    def _validate_subseq_ids(cls, subseq_ids):
        """Normalize and validate a cubemap face selection."""
        if subseq_ids is None:
            return None

        normalized = tuple(int(subseq_id) for subseq_id in subseq_ids)
        if not normalized:
            raise ValueError("subseq_ids cannot be empty")

        valid_ids = set(cls.CUBEMAP_SUBSEQ_IDS)
        if any(subseq_id not in valid_ids for subseq_id in normalized):
            first_id = cls.CUBEMAP_SUBSEQ_IDS[0]
            last_id = cls.CUBEMAP_SUBSEQ_IDS[-1]
            raise ValueError(
                "subseq_ids must contain cubemap face ids in "
                f"[{first_id}, {last_id}]"
            )
        return normalized

    def configure_subseq_ids(self, subseq_ids=None):
        """Validate and store an optional fixed cubemap face selection."""
        self.subseq_ids = self._validate_subseq_ids(subseq_ids)

    def select_subseq_ids(self, subseq_ids=None):
        """Return the configured/overridden faces, or select the default faces."""
        selected = self._validate_subseq_ids(subseq_ids)
        if selected is None:
            selected = self.subseq_ids
        if selected is not None:
            return np.asarray(selected, dtype=np.int64)

        selected = np.asarray(self.CUBEMAP_SUBSEQ_IDS, dtype=np.int64)
        if self.split == "train":
            selected = np.random.choice(
                selected,
                len(selected),
                replace=self.allow_duplicate_img,
            )
        return selected

    @staticmethod
    def deterministic_rng(*key_parts):
        """Create a process-independent RNG for repeatable validation samples."""
        key = "\x1f".join(str(part) for part in key_parts).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        return np.random.default_rng(seed)

    def load_surface_points(self, rng, random_surface, sharpedge_surface):
        sample_groups = []
        for points, sample_count, label in (
            (random_surface, self.pc_size, 0),
            (sharpedge_surface, self.pc_sharpedge_size, 1),
        ):
            if sample_count <= 0:
                continue

            indices = rng.choice(points.shape[0], sample_count, replace=False)
            samples = points[indices]
            if self.sharpedge_label:
                labels = np.full((sample_count, 1), label)
                samples = np.concatenate((samples, labels), axis=1)
            sample_groups.append(samples)

        if not sample_groups:
            raise ValueError("At least one surface sample count must be positive.")

        surface_normal = torch.from_numpy(np.concatenate(sample_groups, axis=0)).float()
        surface = surface_normal[:, :3]
        if self.return_normal:
            normals = torch.nn.functional.normalize(surface_normal[:, 3:6], p=2, dim=1)
            surface = torch.cat((surface, normals), dim=-1)
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
        sample_groups = []
        for points, distances, sample_count, label in (
            (vol_points, vol_label, self.sdf_size, 0),
            (near_points, near_label, self.sdf_near_size, 1),
            (sharpedge_points, sharpedge_label, self.sdf_sharpedge_size, 2),
        ):
            if sample_count <= 0:
                continue

            indices = rng.choice(points.shape[0], sample_count, replace=False)
            samples = np.concatenate((points[indices], distances[indices, None]), axis=1)
            if self.sdf_label:
                labels = np.full((sample_count, 1), label)
                samples = np.concatenate((samples, labels), axis=1)
            sample_groups.append(samples)

        if not sample_groups:
            raise ValueError("At least one SDF sample count must be positive.")

        return torch.from_numpy(np.concatenate(sample_groups, axis=0)).float()
