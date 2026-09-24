#!/usr/bin/env python3
"""Decode a saved ShapeVAE latent into a scene mesh."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE
from nova3r.heads.hunyuan_model.pipelines import (
    colorize_trimesh_vertices,
    export_to_trimesh,
)
from training.mesh2sdf.tools.infer_shape_vae import (
    checkpoint_state,
    find_shape_vae_config,
    select_vae_state,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Decode a ShapeVAE latent .pt file into a PLY mesh.")
    parser.add_argument("latent", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "ShapeVAE config. Default: the config recorded in the latent, then "
            "CHECKPOINT_DIR/.hydra/config.yaml."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("reconstruction.ply"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Model and compute precision on CUDA (default: fp32).",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help=(
            "Mesh extraction bounds in normalized coordinates. By default, "
            "use the bounds saved with the latent."
        ),
    )
    parser.add_argument("--octree-resolution", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=20000)
    parser.add_argument(
        "--mc-level",
        default="auto",
        help="Marching-cubes level or 'auto' (default: auto).",
    )
    args = parser.parse_args()
    if args.octree_resolution < 2:
        parser.error("--octree-resolution must be at least 2")
    if args.num_chunks < 1:
        parser.error("--num-chunks must be at least 1")
    return args


def precision_dtype(precision):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def load_model(checkpoint, config, device, dtype=torch.float32):
    params, _ = find_shape_vae_config(config)
    model = ShapeVAE(**params)
    state = select_vae_state(checkpoint_state(checkpoint), model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    required = ("post_kl.", "transformer.", "geo_decoder.")
    required_missing = [key for key in missing if key.startswith(required)]
    if unexpected or required_missing:
        raise RuntimeError(f"Incompatible checkpoint: {len(required_missing)} required keys missing, " f"{len(unexpected)} unexpected keys")
    return model.to(device=device, dtype=dtype).eval()


def load_latent(path, model, device, dtype=torch.float32, saved=None):
    if saved is None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    latent = saved.get("latent") if isinstance(saved, dict) else saved
    if not torch.is_tensor(latent):
        raise ValueError(f"No latent tensor found in {path}")
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if latent.ndim != 3 or tuple(latent.shape[1:]) != tuple(model.latent_shape):
        raise ValueError(f"Expected latent [B, {model.latent_shape[0]}, {model.latent_shape[1]}], " f"got {tuple(latent.shape)}")
    if latent.shape[0] != 1:
        raise ValueError(f"Expected one latent, found a batch of {latent.shape[0]}")
    return latent.to(device=device, dtype=dtype)


def load_bounds(path, override, saved=None):
    """Use explicit bounds, otherwise recover the normalized-surface bounds."""
    if override is not None:
        return np.asarray(override, dtype=np.float32)
    if saved is None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    bounds = saved.get("bounds") if isinstance(saved, dict) else None
    if bounds is None:
        raise ValueError(
            f"No bounds found in {path}; pass --bounds explicitly"
        )
    return np.asarray(torch.as_tensor(bounds).float().cpu(), dtype=np.float32)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("medium")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    saved = torch.load(args.latent, map_location="cpu", weights_only=True)
    saved_config = saved.get("config") if isinstance(saved, dict) else None
    config = args.config or (Path(saved_config) if saved_config else None)
    config = config or args.checkpoint.parent / ".hydra" / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}; pass --config explicitly")

    dtype = precision_dtype(args.precision) if device.type == "cuda" else torch.float32
    model = load_model(args.checkpoint, config, device, dtype=dtype)
    latent = load_latent(args.latent, model, device, dtype=dtype, saved=saved)
    bounds = load_bounds(args.latent, args.bounds, saved=saved)
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("Each maximum bound must be greater than its minimum")
    if args.mc_level == "auto":
        voxel_size = float(np.max(bounds[3:] - bounds[:3])) / args.octree_resolution
        mc_level = float(np.clip(voxel_size * 128.0, 1e-4, 1.0))
    else:
        mc_level = float(args.mc_level)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        decoded = model.decode(latent)
        outputs = model.latents2mesh(
            decoded,
            output_type="trimesh",
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=args.num_chunks,
            octree_resolution=args.octree_resolution,
            mc_algo="mc",
            enable_pbar=True,
        )

    meshes = export_to_trimesh(outputs)
    meshes = meshes if isinstance(meshes, list) else [meshes]
    mesh = meshes[0]
    if mesh is None or not len(mesh.vertices) or not len(mesh.faces):
        raise RuntimeError("The VAE decoder did not produce a surface")
    mesh = colorize_trimesh_vertices(mesh, model, decoded)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(f"Saved reconstructed scene to {args.output}")


if __name__ == "__main__":
    main()
