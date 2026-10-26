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
    GT_LATENT_FILENAME_SUFFIX = ""

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
        self.return_color = bool(getattr(common_conf, "return_color", False))
        self.sharpedge_label = common_conf.sharpedge_label
        self.sdf_size = common_conf.sdf_size
        self.sdf_surface_size = int(getattr(common_conf, "sdf_surface_size", 0))
        self.sdf_near_size = common_conf.sdf_near_size
        self.sdf_sharpedge_size = common_conf.sdf_sharpedge_size
        self.sdf_label = common_conf.sdf_label
        self.configure_geometry_coordinate_frame(
            getattr(common_conf, "geometry_coordinate_frame", "hunyuan")
        )
        if self.sdf_surface_size < 0:
            raise ValueError("sdf_surface_size must be non-negative")

    def configure_geometry_coordinate_frame(self, coordinate_frame):
        if coordinate_frame not in ("opencv", "hunyuan"):
            raise ValueError("geometry_coordinate_frame must be 'opencv' or 'hunyuan'")
        self.geometry_coordinate_frame = coordinate_frame

    def geometry_input_transform(self, transform):
        """Convert a world-to-OpenCV transform to the selected geometry axes."""
        if self.geometry_coordinate_frame == "hunyuan":
            conversion = transform.new_tensor([1.0, -1.0, -1.0, 1.0])
            return conversion[:, None] * transform
        return transform

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

    def configure_visible_geometry_latents(self, enabled, root):
        """Configure latents encoding the geometry visible in the input views."""
        self.load_visible_geometry_latent = bool(enabled)
        self.visible_geometry_latent_root = Path(root) if root is not None else None
        if self.load_visible_geometry_latent:
            if self.visible_geometry_latent_root is None:
                raise ValueError(
                    "visible_geometry_latent_root is required when "
                    "load_visible_geometry_latent=True"
                )
            if not self.visible_geometry_latent_root.is_dir():
                raise ValueError(
                    "Visible geometry latent root does not exist: "
                    f"{self.visible_geometry_latent_root}"
                )

    def configure_gt_latents(self, root, source="latent"):
        """Configure panorama-specific precomputed VAE targets.

        Files are expected at
        ``<root>/<scene>/<frame><GT_LATENT_FILENAME_SUFFIX>.pt`` and must
        contain a ``latent`` tensor shaped ``[1, N, C]`` (or directly
        ``[N, C]``). With ``source='posterior'``, files are read from
        ``<frame>_before_sampling.pt`` and must contain ``posterior_mean`` and
        ``posterior_logvar`` so the model can sample a fresh target on-device.
        """
        self.gt_latent_root = Path(root) if root is not None else None
        self.load_gt_latent = self.gt_latent_root is not None
        self.gt_latent_source = str(source).lower()
        if self.gt_latent_source not in {"latent", "posterior"}:
            raise ValueError(
                "gt_latent_source must be 'latent' or 'posterior', got "
                f"{source!r}"
            )
        if self.load_gt_latent:
            if not self.gt_latent_root.is_dir():
                raise ValueError(
                    f"GT latent root does not exist: {self.gt_latent_root}"
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

    @staticmethod
    def _validate_single_latent(tensor, path, description):
        """Validate a saved single-sample latent and remove its batch axis."""
        if not torch.is_tensor(tensor):
            raise ValueError(f"No {description} tensor found in {path}")
        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"Expected one {description} in {path}, got {tuple(tensor.shape)}"
                )
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected {description} [N, C] or [1, N, C] in {path}, "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{description.capitalize()} contains non-finite values: {path}"
            )
        return tensor.float()

    @classmethod
    def _load_panorama_image_features(cls, checkpoint, path):
        """Validate panorama features and retain token-layout metadata."""
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Panorama feature file must include grid metadata: {path}")
        if checkpoint.get("projection") != "equirectangular":
            raise ValueError(f"Expected equirectangular panorama features in {path}")

        features = cls._validate_single_latent(
            checkpoint.get("patch_features"), path, "panorama patch features"
        )
        grid_size = checkpoint.get("grid_size")
        if not isinstance(grid_size, (list, tuple)) or len(grid_size) != 2:
            raise ValueError(f"Missing two-dimensional grid_size metadata in {path}")
        grid_size = tuple(int(value) for value in grid_size)
        if min(grid_size) < 1:
            raise ValueError(f"Invalid panorama grid_size {grid_size} in {path}")

        includes_cls = bool(checkpoint.get("includes_cls_token", False))
        expected_tokens = grid_size[0] * grid_size[1] + int(includes_cls)
        if features.shape[0] != expected_tokens:
            raise ValueError(
                f"Panorama feature/token metadata mismatch in {path}: got "
                f"{features.shape[0]} tokens, expected {expected_tokens} from "
                f"grid_size={grid_size}, includes_cls_token={includes_cls}"
            )

        model_name = str(checkpoint.get("model", "")).lower()
        if "dinov3" in model_name:
            backbone = 3
        elif "dinov2" in model_name:
            backbone = 2
        else:
            raise ValueError(f"Cannot identify DINO backbone from model metadata in {path}")

        return {
            "image_features": features,
            "image_feature_grid_size": torch.tensor(grid_size, dtype=torch.long),
            "image_features_include_cls": torch.tensor(includes_cls, dtype=torch.bool),
            "image_feature_backbone": torch.tensor(backbone, dtype=torch.uint8),
        }

    @staticmethod
    def _require_single_panorama(ids, description):
        if len(ids) != 1:
            raise ValueError(
                f"Precomputed {description} require exactly one panorama per sample"
            )
        return f"{int(ids[0]):05d}"

    def _load_conditioning_latent(self, seq_name, ids):
        frame_id = self._require_single_panorama(ids, "conditioning latents")
        latent_path = self.conditioning_latent_root / seq_name / f"{frame_id}.pt"
        checkpoint = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = checkpoint.get("latent") if isinstance(checkpoint, dict) else checkpoint
        return self._validate_single_latent(
            latent,
            latent_path,
            "conditioning latent",
        )

    def _load_visible_geometry_latent(self, seq_name, ids):
        frame_id = self._require_single_panorama(ids, "visible geometry latents")
        latent_path = self.visible_geometry_latent_root / seq_name / f"{frame_id}.pt"
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing visible geometry latent: {latent_path}")
        checkpoint = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = checkpoint.get("latent") if isinstance(checkpoint, dict) else checkpoint
        return self._validate_single_latent(
            latent,
            latent_path,
            "visible geometry latent",
        )

    def _load_gt_latent(self, seq_name, ids):
        frame_id = self._require_single_panorama(ids, "GT latents")
        filename_suffix = self.GT_LATENT_FILENAME_SUFFIX
        if self.gt_latent_source == "posterior":
            filename_suffix += "_before_sampling"
        latent_path = self.gt_latent_root / seq_name / f"{frame_id}{filename_suffix}.pt"
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing GT latent: {latent_path}")
        checkpoint = torch.load(latent_path, map_location="cpu", weights_only=True)
        if self.gt_latent_source == "posterior":
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Expected posterior dictionary in {latent_path}")
            return {
                "gt_latent_posterior_mean": self._validate_single_latent(
                    checkpoint.get("posterior_mean"),
                    latent_path,
                    "posterior_mean",
                ),
                "gt_latent_posterior_logvar": self._validate_single_latent(
                    checkpoint.get("posterior_logvar"),
                    latent_path,
                    "posterior_logvar",
                ),
            }

        latent = checkpoint.get("latent") if isinstance(checkpoint, dict) else checkpoint
        return self._validate_single_latent(latent, latent_path, "GT latent")

    def _normalize_extrinsics(self, extrinsics, translation_scale=None):
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
        if self.geometry_coordinate_frame == "hunyuan":
            normalized = normalized * normalized.new_tensor([1.0, -1.0, -1.0, 1.0])
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

    def load_surface_points(
        self,
        rng,
        random_surface,
        sharpedge_surface,
        random_surface_rgb=None,
        sharpedge_surface_rgb=None,
    ):
        sample_groups = []
        for points, colors, sample_count, label in (
            (random_surface, random_surface_rgb, self.pc_size, 0),
            (sharpedge_surface, sharpedge_surface_rgb, self.pc_sharpedge_size, 1),
        ):
            if sample_count <= 0:
                continue

            indices = rng.choice(points.shape[0], sample_count, replace=False)
            samples = points[indices]
            if self.return_color:
                if colors is None:
                    raise ValueError(
                        "return_color=true requires RGB arrays in the surface NPZ"
                    )
                if len(colors) != len(points):
                    raise ValueError(
                        f"Surface RGB/point count mismatch: {len(colors)} colors "
                        f"for {len(points)} points."
                    )
                samples = np.concatenate(
                    (samples, colors[indices].astype(np.float32) / 255.0), axis=1
                )
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
        if self.return_color:
            # Keep the legacy normal/edge-label feature prefix intact so old
            # encoder weights can be expanded with zero-initialized RGB columns.
            surface = torch.cat((surface, surface_normal[:, 6:9]), dim=-1)
        return surface

    def surface_rgb_kwargs(self, data):
        if not self.return_color:
            return {}
        color_keys = (
            ("random_surface_rgb", "sharp_surface_rgb")
            if self.pc_sharpedge_size > 0
            else ("random_surface_rgb",)
        )
        present = [key in data.files for key in color_keys]
        if any(present) and not all(present):
            missing = [key for key, exists in zip(color_keys, present) if not exists]
            raise ValueError(
                f"Incomplete surface RGB data; missing: {', '.join(missing)}"
            )
        if not all(present):
            return {}
        result = {"random_surface_rgb": data["random_surface_rgb"]}
        if self.pc_sharpedge_size > 0:
            result["sharpedge_surface_rgb"] = data["sharp_surface_rgb"]
        return result

    def load_sdf_points(
        self,
        rng,
        vol_points,
        vol_label,
        near_points,
        near_label,
        sharpedge_points,
        sharpedge_label,
        vol_rgb=None,
        vol_rgb_valid=None,
        near_rgb=None,
        sharpedge_rgb=None,
    ):
        sample_groups = []
        has_rgb = any(
            colors is not None for colors in (vol_rgb, near_rgb, sharpedge_rgb)
        )
        if has_rgb and not all(
            colors is not None for colors in (vol_rgb, near_rgb, sharpedge_rgb)
        ):
            raise ValueError("UDF RGB arrays must contain vol, random-near, and sharp-near colors.")

        for points, distances, colors, color_valid, sample_count, label in (
            (vol_points, vol_label, vol_rgb, vol_rgb_valid, self.sdf_size, 0),
            (near_points, near_label, near_rgb, None, self.sdf_near_size, 1),
            (
                sharpedge_points,
                sharpedge_label,
                sharpedge_rgb,
                None,
                self.sdf_sharpedge_size,
                2,
            ),
        ):
            if sample_count <= 0:
                continue

            indices = rng.choice(points.shape[0], sample_count, replace=False)
            samples = np.concatenate((points[indices], distances[indices, None]), axis=1)
            if has_rgb:
                if len(colors) != len(points):
                    raise ValueError(
                        f"RGB/point count mismatch: {len(colors)} colors for {len(points)} points."
                    )
                sampled_rgb = colors[indices].astype(np.float32) / 255.0
                if color_valid is None:
                    sampled_valid = np.ones((sample_count, 1), dtype=np.float32)
                else:
                    if len(color_valid) != len(points):
                        raise ValueError(
                            "RGB validity/point count mismatch: "
                            f"{len(color_valid)} flags for {len(points)} points."
                        )
                    sampled_valid = color_valid[indices, None].astype(np.float32)
                # Layout with color: XYZ, distance, RGB in [0, 1], RGB-valid.
                samples = np.concatenate(
                    (samples, sampled_rgb, sampled_valid), axis=1
                )
            if self.sdf_label:
                labels = np.full((sample_count, 1), label)
                samples = np.concatenate((samples, labels), axis=1)
            sample_groups.append(samples)

        if not sample_groups:
            raise ValueError("At least one SDF sample count must be positive.")

        return torch.from_numpy(np.concatenate(sample_groups, axis=0)).float()

    def add_exact_surface_points(self, rng, surface, geo_points):
        """Append configurable, exactly-on-surface UDF/RGB supervision."""
        if self.sdf_surface_size == 0:
            return geo_points
        if surface is None or geo_points is None:
            raise ValueError(
                "sdf_surface_size > 0 requires both surface and geo_points modalities"
            )
        if not self.return_color:
            raise ValueError("sdf_surface_size > 0 requires return_color=true")
        if self.sdf_surface_size > len(surface):
            raise ValueError(
                f"sdf_surface_size={self.sdf_surface_size} exceeds the "
                f"{len(surface)} loaded surface points"
            )

        indices = torch.from_numpy(
            rng.choice(len(surface), self.sdf_surface_size, replace=False)
        ).long()
        selected = surface[indices]
        exact = torch.cat(
            (
                selected[:, :3],
                selected.new_zeros((len(selected), 1)),
                selected[:, -3:],
                selected.new_ones((len(selected), 1)),
            ),
            dim=1,
        )
        if self.sdf_label:
            exact = torch.cat(
                (exact, exact.new_full((len(exact), 1), 3.0)),
                dim=1,
            )
        if exact.shape[1] != geo_points.shape[1]:
            raise ValueError(
                "Exact-surface and UDF point layouts do not match: "
                f"{exact.shape[1]} vs {geo_points.shape[1]} columns"
            )
        return torch.cat((geo_points, exact), dim=0)

    def sdf_rgb_kwargs(self, data):
        """Extract optional RGB arrays from a UDF NPZ without requiring them."""
        if not self.return_color:
            return {}
        color_keys = ("vol_rgb", "random_near_rgb", "sharp_near_rgb")
        present = [key in data.files for key in color_keys]
        if any(present) and not all(present):
            missing = [key for key, exists in zip(color_keys, present) if not exists]
            raise ValueError(f"Incomplete UDF RGB data; missing: {', '.join(missing)}")
        if not all(present):
            return {}
        return {
            "vol_rgb": data["vol_rgb"],
            "vol_rgb_valid": (
                data["vol_rgb_valid"] if "vol_rgb_valid" in data.files else None
            ),
            "near_rgb": data["random_near_rgb"],
            "sharpedge_rgb": data["sharp_near_rgb"],
        }
