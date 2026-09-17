"""Official Depth Anything 3 adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from eval.baselines.common import SageSample, load_cameras, unproject_depth


class DA3Adapter:
    def __init__(self, model_id: str, repo: Path, device: str, resolution: int, face_order: list[int]):
        src = (repo / "src").resolve()
        if not (src / "depth_anything_3" / "api.py").is_file():
            raise FileNotFoundError(f"DA3 API not found below {src}")
        for name in tuple(sys.modules):
            if name == "depth_anything_3" or name.startswith("depth_anything_3."):
                del sys.modules[name]
        sys.path.insert(0, str(src))
        try:
            api = importlib.import_module("depth_anything_3.api")
        finally:
            sys.path.remove(str(src))
        self.device = torch.device(device)
        self.model = api.DepthAnything3.from_pretrained(model_id).to(self.device).eval()
        self.resolution = resolution
        self.face_order = face_order

    def predict(self, sample: SageSample) -> np.ndarray:
        paths, intrinsics, extrinsics = load_cameras(sample, self.face_order)
        images_cpu, extrinsics_cpu, intrinsics_cpu = self.model._preprocess_inputs(
            [str(path) for path in paths],
            extrinsics,
            intrinsics,
            self.resolution,
            "upper_bound_resize",
        )
        images, extrinsics_device, intrinsics_device = self.model._prepare_model_inputs(
            images_cpu, extrinsics_cpu, intrinsics_cpu
        )
        normalized_extrinsics = self.model._normalize_extrinsics(extrinsics_device.clone())
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
        return unproject_depth(
            depth[0].float().cpu().numpy(),
            intrinsics_device[0].float().cpu().numpy(),
            extrinsics_device[0].float().cpu().numpy(),
        )
