#!/usr/bin/env python3
"""Predict a 3D scene from one equirectangular panorama using DINOv3 + Nova3R."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch
from torch import nn

from tools.panorama_to_dinov3_features import extract_features, load_model as load_dinov3
from tools.unconditional_dit_noise_sweep import mesh_level, save_latent, save_mesh


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "exp_output/dinov3_panorama_conditioned_from_unconditional_dit/checkpoints/last.ckpt"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "checkpoints/da3_large_hunyuan_mini/.hydra/config_dinov3_panorama_conditioned.yaml"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panorama", type=Path, help="Input equirectangular panorama")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("panorama_prediction"))
    parser.add_argument(
        "--dinov3-model",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Hugging Face DINOv3 ViT model ID or local directory",
    )
    parser.add_argument("--dinov3-revision", default="main")
    parser.add_argument("--dinov3-cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="DINOv3 input height; width is twice this (default: 512)",
    )
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16", "float16"),
        default="bfloat16", help="CUDA inference precision (default: bfloat16)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ode-method", default="euler")
    parser.add_argument("--ode-step-size", type=float, default=0.01)
    parser.add_argument(
        "--bounds", type=float, nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        default=(-1.2, -0.8, -1.2, 1.4, 0.8, 1.4),
    )
    parser.add_argument("--octree-resolution", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=40000)
    parser.add_argument("--mc-level", default="auto")
    parser.add_argument(
        "--skip-mesh", action="store_true",
        help="Only save the predicted ShapeVAE latent (also disables video)",
    )
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--video-frames", type=int, default=180)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--video-elevation", type=float, default=20.0)
    parser.add_argument(
        "--video-color", choices=("normal", "plasma", "viridis", "xyz"),
        default="plasma",
    )
    args = parser.parse_args(argv)
    if args.resolution <= 0 or args.resolution % 16:
        parser.error("--resolution must be a positive multiple of 16")
    if args.ode_step_size <= 0:
        parser.error("--ode-step-size must be positive")
    if args.octree_resolution < 2:
        parser.error("--octree-resolution must be at least 2")
    if args.num_chunks < 1:
        parser.error("--num-chunks must be positive")
    if args.video_frames < 1 or args.video_fps < 1 or args.video_size < 1:
        parser.error("--video-frames, --video-fps, and --video-size must be positive")
    return args


def load_panorama(path: Path, processor, resolution: int):
    with Image.open(path) as image:
        image = image.convert("RGB")
        original_size = [image.height, image.width]
        pixels = processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": resolution, "width": resolution * 2},
            do_center_crop=False,
        )["pixel_values"]
    return pixels, original_size


def render_video(mesh_path: Path, output: Path, args, device):
    """Render mesh vertices as the repository's standard 360-degree video."""
    from demo.visualization.render_points import (
        build_colored_pointcloud,
        flip_axis,
        load_ply_pointcloud,
        render_turntable_video,
    )

    point_cloud = load_ply_pointcloud(
        str(mesh_path), device=device, remove_outlier=True
    )
    point_cloud = flip_axis(point_cloud, axis="y")
    point_cloud = build_colored_pointcloud(
        point_cloud, device=device, color_type=args.video_color
    )
    render_turntable_video(
        point_cloud,
        num_frames=args.video_frames,
        fps=args.video_fps,
        image_size=args.video_size,
        elevation=args.video_elevation,
        radius=0.003,
        outfile=str(output),
    )


class ConditionedVelocity(nn.Module):
    """Bind panorama conditioning tokens to the ODE solver interface."""

    def __init__(self, head, tokens, precision):
        super().__init__()
        self.head = head
        self.tokens = tokens
        self.precision = precision

    def forward(self, x, t, **kwargs):
        del kwargs
        with torch.autocast(
            device_type=x.device.type,
            dtype=getattr(torch, self.precision),
            enabled=x.device.type == "cuda" and self.precision != "float32",
        ):
            return self.head(
                decout=[self.tokens],
                noised_latents=x,
                timestep=t.to(device=x.device, dtype=x.dtype),
            )


@torch.inference_mode()
def predict_latent(model, features, device, precision, seed, method, step_size):
    from nova3r.flow_matching.solver import ODESolver

    parameter = model.pts3d_head.x_embedder.weight
    batch = {"image_features": features.unsqueeze(0)}
    with torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, precision),
        enabled=device.type == "cuda" and precision != "float32",
    ):
        tokens = model._encode(None, batch=batch)["tokens"]
    tokens = tokens.to(device=device, dtype=parameter.dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed % (2**63 - 1))
    noise = torch.randn(
        (1, model.pts3d_head.num_latents, model.pts3d_head.in_channels),
        device=device,
        dtype=parameter.dtype,
        generator=generator,
    )
    solver = ODESolver(
        velocity_model=ConditionedVelocity(model.pts3d_head, tokens, precision)
    )
    return solver.sample(
        x_init=noise,
        time_grid=torch.tensor([0.0, 1.0], device=device, dtype=noise.dtype),
        method=method,
        step_size=step_size,
        return_intermediates=False,
    )


def main(argv=None):
    args = parse_args(argv)
    for path, description in (
        (args.panorama, "panorama"),
        (args.checkpoint, "Nova3R checkpoint"),
        (args.config, "Nova3R config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type != "cuda" and args.precision != "float32":
        print("Non-CUDA inference uses float32 precision")
        args.precision = "float32"
    if (
        device.type == "cuda"
        and args.precision == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("This GPU does not support bfloat16; use --precision float16")

    # Extract to CPU first, then release DINOv3 before loading Nova3R to keep
    # peak GPU memory low.
    dino_args = argparse.Namespace(
        model=args.dinov3_model,
        revision=args.dinov3_revision,
        cache_dir=args.dinov3_cache_dir,
        local_files_only=args.local_files_only,
    )
    dino, processor = load_dinov3(dino_args, device)
    pixels, original_size = load_panorama(args.panorama, processor, args.resolution)
    features, _ = extract_features(
        dino, pixels, device, batch_size=1, inference_dtype=args.precision
    )
    features = features[0].contiguous()
    expected_patches = (args.resolution // 16) * (args.resolution * 2 // 16)
    if features.shape != (expected_patches, dino.config.hidden_size):
        raise RuntimeError(f"Unexpected DINOv3 feature shape: {tuple(features.shape)}")
    del dino, processor, pixels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Importing Nova3R constructs the full model dependency graph, so defer it
    # until after argument validation and DINO feature extraction.
    from tools.cubemap_conditioned_dit_noise_sweep import load_model

    model, _ = load_model(args.config, args.checkpoint, device)
    if features.shape[-1] != model.img_token_proj.in_features:
        raise ValueError(
            f"DINOv3 emits {features.shape[-1]} channels but the checkpoint expects "
            f"{model.img_token_proj.in_features}"
        )
    latent = predict_latent(
        model, features, device, args.precision, args.seed,
        args.ode_method, args.ode_step_size,
    )

    bounds = np.asarray(args.bounds, dtype=np.float32)
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("Each maximum bound must be greater than its minimum")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panorama_copy = args.output_dir / f"panorama{args.panorama.suffix.lower()}"
    if args.panorama.resolve() != panorama_copy.resolve():
        shutil.copy2(args.panorama, panorama_copy)
    latent_path = args.output_dir / "prediction.pt"
    save_latent(
        latent, latent_path, source=str(args.panorama.resolve()),
        checkpoint=str(args.checkpoint), dinov3_model=args.dinov3_model,
        resolution=args.resolution, seed=args.seed, bounds=bounds.tolist(),
    )

    mesh_path = None
    if not args.skip_mesh:
        mesh_path = args.output_dir / "prediction.ply"
        save_mesh(
            model.first_stage, latent, mesh_path, bounds=bounds,
            mc_level=mesh_level(args.mc_level, bounds, args.octree_resolution),
            num_chunks=args.num_chunks,
            octree_resolution=args.octree_resolution,
        )

    video_path = None
    if mesh_path is not None and not args.skip_video:
        # The renderer also uses CUDA. Release the reconstruction networks first
        # instead of retaining several large models for the whole render.
        del model, latent
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        video_path = args.output_dir / "prediction.mp4"
        render_video(mesh_path, video_path, args, device)

    manifest = {
        "panorama": str(args.panorama.resolve()),
        "panorama_copy": str(panorama_copy),
        "original_size": original_size,
        "dinov3_input_size": [args.resolution, args.resolution * 2],
        "dinov3_feature_shape": list(features.shape),
        "dinov3_model": args.dinov3_model,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "seed": args.seed,
        "ode_method": args.ode_method,
        "ode_step_size": args.ode_step_size,
        "latent": str(latent_path),
        "mesh": str(mesh_path) if mesh_path is not None else None,
        "video": str(video_path) if video_path is not None else None,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved latent to {latent_path}")
    if mesh_path is not None:
        print(f"Saved mesh to {mesh_path}")
    if video_path is not None:
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
