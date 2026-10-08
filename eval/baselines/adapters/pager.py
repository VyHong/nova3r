"""Official PaGeR panoramic-depth adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from eval.baselines.common import (
    SageSample,
    _panorama_pose,
    load_scannetpp_valid_mask,
    resolve_erp_path,
)


MODEL_TO_REPO = {
    "pager": "prs-eth/PaGeR",
    "pager-metric-depth": "prs-eth/PaGeR-metric-depth",
    "pager-normals": "prs-eth/PaGeR-normals",
}

# PaGeR ERP camera coordinates have +X forward, +Y up, +Z right. The SAGE
# panorama's OpenCV camera coordinates have +Z forward, +Y down, +X right.
PAGER_TO_OPENCV = np.asarray(
    [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
    dtype=np.float32,
)


def pager_points_to_world(points: np.ndarray, panorama_w2c: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3) @ PAGER_TO_OPENCV.T
    c2w = np.linalg.inv(np.asarray(panorama_w2c, dtype=np.float32))
    return (points @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32, copy=False)


class PaGeRAdapter:
    def __init__(
        self,
        repo: Path,
        checkpoint: str,
        device: str,
        erp_root: Path,
        scene_mode: str,
        max_depth: float,
        sky_threshold: float,
    ):
        if scene_mode not in ("indoor", "outdoor"):
            raise ValueError("PaGeR scene_mode must be 'indoor' or 'outdoor'")
        if max_depth <= 0:
            raise ValueError("PaGeR max_depth must be positive")
        repo = repo.resolve()
        if not (repo / "src" / "pager.py").is_file():
            raise FileNotFoundError(f"PaGeR repository not found: {repo}")
        checkpoint = MODEL_TO_REPO.get(checkpoint, checkpoint)

        for name in tuple(sys.modules):
            if name == "src" or name.startswith("src.") or name == "depth_anything_3" or name.startswith("depth_anything_3."):
                del sys.modules[name]
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "src"))
        try:
            from huggingface_hub import hf_hub_download
            from omegaconf import OmegaConf

            local = Path(checkpoint)
            config = local / "config.yaml" if local.is_dir() else Path(
                hf_hub_download(repo_id=checkpoint, filename="config.yaml")
            )
            pager_module = importlib.import_module("src.pager")
            geometry = importlib.import_module("src.utils.geometry_utils")
            cfg = OmegaConf.load(config)
            self.pager = pager_module.Pager(checkpoint, cfg=cfg, device=torch.device(device))
        finally:
            sys.path.remove(str(repo / "src"))
            sys.path.remove(str(repo))

        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.cfg = cfg
        self.erp_to_cubemap = geometry.erp_to_cubemap
        self.cubemap_to_erp = geometry.cubemap_to_erp
        self.erp_to_pointcloud = geometry.erp_to_pointcloud
        self.compute_edge_mask = geometry.compute_edge_mask
        self.pager.get_intrinsics_extrinsics(cfg.face_size, getattr(cfg, "cube_fov", 90.0))
        self.pager.set_depth_range(1e-2, max_depth)
        self.pager.model.to(self.device, dtype=self.pager.weight_dtype).eval()
        if "depth" not in self.pager.modalities:
            raise ValueError(f"PaGeR checkpoint has no depth head: {self.pager.modalities}")
        self.erp_root = erp_root
        self.scene_mode = scene_mode
        self.sky_threshold = sky_threshold

    def predict(self, sample: SageSample) -> np.ndarray:
        path = resolve_erp_path(sample, self.erp_root)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if image.width > 2048:
                image = image.resize(
                    (2048, round(image.height * 2048 / image.width)),
                    Image.Resampling.LANCZOS,
                )
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        cube = self.erp_to_cubemap(
            (tensor - mean) / std,
            face_w=self.cfg.face_size,
            fov=getattr(self.cfg, "cube_fov", 90.0),
        ).unsqueeze(0).to(self.device)

        scale_head = f"scale_{self.scene_mode}"
        keep = {"depth", "sky", scale_head, "scale"}
        skip = set(self.pager.modalities) - keep
        with torch.inference_mode():
            prediction = self.pager(cube, dtype=self.dtype, skip_heads=skip)
            sky = prediction.get("sky")
            depth, _ = self.pager.process_depth_output(
                prediction["depth"][0],
                (height, width),
                sky_mask=sky[0] if sky is not None else None,
                log_scale=prediction.get("scale"),
            )
            depth = depth.float().cpu().squeeze()
            points = self.erp_to_pointcloud(depth).permute(1, 2, 0).cpu().numpy()

        valid = np.isfinite(depth.numpy()) & (depth.numpy() > 0)
        valid &= self.compute_edge_mask(depth.numpy())
        valid &= load_scannetpp_valid_mask(sample, path, width, height)
        if sky is not None:
            sky_erp = geometry_sky_to_erp(
                self.cubemap_to_erp,
                self.pager.cube_fov,
                sky[0],
                height,
                width,
            )
            valid &= sky_erp < self.sky_threshold
        return pager_points_to_world(points[valid], _panorama_pose(sample))


def geometry_sky_to_erp(
    cubemap_to_erp,
    cube_fov: float,
    sky: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    """Use PaGeR's own stitcher for a diagnostic sky rejection mask."""
    return (
        cubemap_to_erp(
            torch.sigmoid(sky.float()),
            height,
            width,
            fov=cube_fov,
        )
        .cpu()
        .numpy()
        .squeeze()
    )
