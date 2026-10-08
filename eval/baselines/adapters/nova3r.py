"""Nova3R adapter following the official single/two-image demo pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from eval.baselines.common import InputProjection, SageSample, load_erp_input, load_input_cameras


def camera_to_world_points(points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    """Transform Nova3R's first-camera-frame prediction into evaluator world space."""
    w2c = np.asarray(world_to_camera, dtype=np.float32)
    if w2c.shape == (3, 4):
        w2c = np.concatenate((w2c, np.asarray([[0, 0, 0, 1]], np.float32)))
    if w2c.shape != (4, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 camera extrinsic, got {w2c.shape}")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    c2w = np.linalg.inv(w2c)
    return (points @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32, copy=False)


def prepare_official_inputs(
    paths: list[Path],
    resolution: tuple[int, int],
    num_views: int,
):
    """Reproduce official demo image normalization and N=1 duplication."""
    if not 1 <= num_views <= 6:
        raise ValueError("Nova3R supports between one and six input views")
    if not paths:
        raise ValueError("Nova3R requires at least one input image")
    selected = paths[:num_views]
    single_view = len(selected) == 1
    if single_view:
        selected = [selected[0], selected[0]]

    width, height = resolution
    normalize = transforms.Compose(
        (
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        )
    )
    images = []
    for index, path in enumerate(selected):
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        images.append(
            {
                "img": normalize(image)[None],
                "true_shape": np.int32([height, width]),
                "idx": index,
                "instance": str(index),
                "view_label": f"input_{index}",
            }
        )
    return images, single_view


class Nova3RAdapter:
    def __init__(
        self,
        checkpoint: Path,
        config: Path | None,
        device: str,
        face_order: list[int],
        input_projection: InputProjection,
        erp_root: Path | None,
        erp_fov: float,
        resolution: tuple[int, int],
        num_views: int,
        num_queries: int,
        seed: int,
        method: str | None = None,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing Nova3R checkpoint: {checkpoint}")
        config = config or checkpoint.parent / ".hydra" / "config.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"Missing Nova3R Hydra config: {config}")
        if num_queries < 1:
            raise ValueError("Nova3R num_queries must be positive")

        # Import through the repository's supported loader. FP32 parameters plus
        # inference autocast matches the official release more closely than
        # permanently converting checkpoint weights to BF16.
        from demo_nova3r import load_model

        self.device = torch.device(device)
        self.model, self.cfg = load_model(
            ckpt_path=str(checkpoint),
            config_path=str(config),
            device=self.device,
            dtype=torch.float32,
            stage="test",
        )
        self.model.eval()
        from omegaconf import OmegaConf

        OmegaConf.set_struct(self.cfg, False)
        if "fm_step_size" not in self.cfg:
            self.cfg.fm_step_size = 0.04
        if "fm_sampling" not in self.cfg:
            self.cfg.fm_sampling = "euler"
        self.method = method or self.cfg.get("fm_sampling", "euler")
        self.face_order = face_order
        self.input_projection = input_projection
        self.erp_root = erp_root
        self.erp_fov = erp_fov
        self.resolution = resolution
        self.num_views = num_views
        self.num_queries = num_queries
        self.seed = seed

    def predict(self, sample: SageSample) -> np.ndarray:
        if self.input_projection == "erp":
            if self.erp_root is None:
                raise ValueError("erp_root is required when input projection is 'erp'")
            erp_path, panorama_extrinsic = load_erp_input(sample, self.erp_root)
            paths, extrinsics = [erp_path], panorama_extrinsic[None]
        else:
            paths, _, extrinsics, _ = load_input_cameras(
                sample,
                self.input_projection,
                self.face_order,
                erp_root=self.erp_root,
                erp_fov=self.erp_fov,
            )
        paths = [Path(path) for path in paths]
        images, single_view = prepare_official_inputs(paths, self.resolution, self.num_views)

        from nova3r.inference import inference_nova3r_views
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            torch.manual_seed(self.seed + sample.frame)
            output = inference_nova3r_views(
                self.cfg,
                images,
                self.model,
                self.device,
                num_queries=self.num_queries,
                method=self.method,
            )
        points_camera = np.asarray(output["pred"]["pts3d_xyz"])[0]
        return camera_to_world_points(points_camera, extrinsics[0])
