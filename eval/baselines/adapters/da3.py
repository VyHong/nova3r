"""Official Depth Anything 3 adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch

from eval.baselines.common import (
    cubemap_z_to_range,
    InputProjection,
    SageSample,
    load_erp_input,
    load_input_cameras,
    load_scannetpp_valid_mask,
    pinhole_z_to_range,
    unproject_depth,
    unproject_erp_depth,
    unproject_erp_range,
)


def import_official_da3_api(repo: Path):
    """Import DA3's namespace package despite NOVA3R's same-named local package."""
    src = (repo / "src").resolve()
    package_dir = src / "depth_anything_3"
    if not (package_dir / "api.py").is_file():
        raise FileNotFoundError(f"DA3 API not found below {src}")

    for name in tuple(sys.modules):
        if name == "depth_anything_3" or name.startswith("depth_anything_3."):
            del sys.modules[name]
    sys.path.insert(0, str(src))
    try:
        # The official source tree intentionally has no __init__.py. A regular
        # local NOVA3R package with the same name would otherwise win over this
        # namespace package, regardless of sys.path order.
        package = importlib.import_module("depth_anything_3")
        search_path = package.__path__
        official_path = str(package_dir)
        if official_path not in search_path:
            search_path.insert(0, official_path)
        importlib.invalidate_caches()
        export_module = types.ModuleType("depth_anything_3.utils.export")

        def unsupported_export(*_args, **_kwargs):
            raise RuntimeError("DA3 exports are disabled in the evaluation adapter")

        export_module.export = unsupported_export
        sys.modules[export_module.__name__] = export_module
        return importlib.import_module("depth_anything_3.api")
    finally:
        sys.path.remove(str(src))


class DA3Adapter:
    def __init__(
        self,
        model_id: str,
        repo: Path,
        device: str,
        resolution: int,
        face_order: list[int],
        input_projection: InputProjection = "cubemap",
        erp_root: Path | None = None,
        erp_fov: float = 95.0,
    ):
        api = import_official_da3_api(repo)
        self.device = torch.device(device)
        self.model = api.DepthAnything3.from_pretrained(model_id).to(self.device).eval()
        self.resolution = resolution
        self.face_order = face_order
        self.input_projection = input_projection
        self.erp_root = erp_root
        self.erp_fov = erp_fov
        self.last_2d_prediction: dict[str, np.ndarray] = {}
        self.last_diagnostic_pointclouds: dict[str, np.ndarray] = {}

    def _load_valid_masks(
        self,
        sample: SageSample,
        image_paths: list[Path],
        height: int,
        width: int,
    ) -> np.ndarray:
        return np.stack(
            [
                load_scannetpp_valid_mask(sample, image_path, width, height)
                for image_path in image_paths
            ]
        )

    @staticmethod
    def _save_map_png(array: np.ndarray, path: Path) -> None:
        array = np.asarray(array, dtype=np.float32)
        valid = np.isfinite(array) & (array > 0)
        image = np.zeros(array.shape, dtype=np.uint8)
        if valid.any():
            low, high = np.quantile(array[valid], (0.01, 0.99))
            if high > low:
                normalized = np.clip((array - low) / (high - low), 0, 1)
                image[valid] = np.rint(normalized[valid] * 255).astype(np.uint8)
            else:
                image[valid] = 255
        Image.fromarray(image, mode="L").save(path)

    def save_2d_prediction(self, output_dir: Path, stem: str) -> None:
        """Save exact DA3 maps plus percentile-normalized PNG previews."""
        if not self.last_2d_prediction:
            raise RuntimeError("No DA3 prediction is available to save")
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / f"{stem}_da3_2d.npz", **self.last_2d_prediction)
        for name in (
            "z_depth",
            "direct_depth_range",
            "spherical_range",
            "cubemap_adjusted_range",
            "pinhole_adjusted_range",
            "valid_mask",
            "depth_conf",
            "sky",
        ):
            maps = self.last_2d_prediction.get(name)
            if maps is None:
                continue
            while maps.ndim > 3 and maps.shape[-1] == 1:
                maps = maps[..., 0]
            maps = maps[None] if maps.ndim == 2 else maps
            if maps.ndim != 3:
                continue
            for view, array in enumerate(maps):
                self._save_map_png(array, output_dir / f"{stem}_{name}_view{view:02d}.png")

    def predict(self, sample: SageSample) -> np.ndarray:
        self.last_2d_prediction = {}
        self.last_diagnostic_pointclouds = {}
        if self.input_projection == "erp":
            if self.erp_root is None:
                raise ValueError("erp_root is required when input projection is 'erp'")
            erp_path, panorama_extrinsic = load_erp_input(sample, self.erp_root)
            images, intrinsics, extrinsics = [erp_path], None, None
        else:
            images, intrinsics, extrinsics, _ = load_input_cameras(
                sample,
                self.input_projection,
                self.face_order,
                erp_root=self.erp_root,
                erp_fov=self.erp_fov,
            )
        image_paths = [Path(image) for image in images]
        images_cpu, extrinsics_cpu, intrinsics_cpu = self.model._preprocess_inputs(
            [str(image) if isinstance(image, Path) else image for image in images],
            extrinsics,
            intrinsics,
            self.resolution,
            "upper_bound_resize",
        )
        images, extrinsics_device, intrinsics_device = self.model._prepare_model_inputs(
            images_cpu, extrinsics_cpu, intrinsics_cpu
        )
        normalized_extrinsics = self.model._normalize_extrinsics(
            extrinsics_device.clone() if extrinsics_device is not None else None
        )
        with torch.inference_mode():
            raw = self.model._run_model_forward(
                images,
                normalized_extrinsics,
                intrinsics_device,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="first",
            )
        depth = raw["depth"]
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        depth_numpy = depth[0].float().cpu().numpy()
        self.last_2d_prediction["z_depth"] = depth_numpy
        valid_mask = self._load_valid_masks(
            sample,
            image_paths,
            depth_numpy.shape[-2],
            depth_numpy.shape[-1],
        )
        self.last_2d_prediction["valid_mask"] = valid_mask
        projection_depth = np.where(valid_mask, depth_numpy, np.nan)
        for source_name, saved_name in (("depth_conf", "depth_conf"), ("sky", "sky")):
            value = raw.get(source_name)
            if value is not None:
                self.last_2d_prediction[saved_name] = value[0].float().cpu().numpy()
        if self.input_projection == "erp":
            predicted_intrinsics = raw.get("intrinsics")
            cubemap_adjusted_range = cubemap_z_to_range(projection_depth)
            self.last_2d_prediction["direct_depth_range"] = projection_depth.copy()
            self.last_2d_prediction["cubemap_adjusted_range"] = cubemap_adjusted_range[None]
            direct_points = unproject_erp_range(projection_depth, panorama_extrinsic)
            cubemap_adjusted_points = unproject_erp_range(
                cubemap_adjusted_range, panorama_extrinsic
            )
            self.last_diagnostic_pointclouds = {
                "erp_direct_depth_as_range": direct_points,
                "erp_cubemap_adjusted": cubemap_adjusted_points,
            }
            if predicted_intrinsics is None:
                raise RuntimeError(
                    "DA3 did not predict intrinsics required for the default ERP pinhole adjustment"
                )
            predicted_intrinsics_numpy = predicted_intrinsics[0].float().cpu().numpy()
            adjusted_range = pinhole_z_to_range(
                projection_depth, predicted_intrinsics_numpy
            )
            adjusted_points = unproject_erp_depth(
                projection_depth,
                predicted_intrinsics_numpy,
                panorama_extrinsic,
            )
            self.last_2d_prediction["intrinsics"] = predicted_intrinsics_numpy
            self.last_2d_prediction["spherical_range"] = adjusted_range[None]
            self.last_2d_prediction["pinhole_adjusted_range"] = adjusted_range[None]
            self.last_diagnostic_pointclouds["erp_pinhole_adjusted"] = adjusted_points
            return adjusted_points
        return unproject_depth(
            projection_depth,
            intrinsics_device[0].float().cpu().numpy(),
            extrinsics_device[0].float().cpu().numpy(),
        )
