#!/usr/bin/env python3
"""Reconstruct one ShapeVAE latent using a cubemap-conditioned Nova3R DiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torch import nn

from nova3r.models.nova3r_img_cond import Nova3rImgCond
from tools.unconditional_dit_noise_sweep import (
    load_target_latent,
    mesh_level,
    save_latent,
    save_mesh,
    timestep_label,
)


DEFAULT_MODEL_CHECKPOINT = (
    REPO_ROOT / "exp_output/da3_hunyuan_mini_scratch/checkpoints/last_1-e4.ckpt"
)
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT / "exp_output/da3_hunyuan_mini_scratch/version_1/hparams.yaml"
)
DEFAULT_FACE_ORDER = (4, 0, 5, 1, 3, 2)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Add fixed flow noise to one target latent at selected timesteps, "
            "reconstruct it while conditioning on all six posed cubemap faces, "
            "and optionally decode every result to a mesh. In this flow "
            "convention t=0 is pure noise and t=1 is clean."
        )
    )
    parser.add_argument(
        "latent",
        type=Path,
        help=(
            "Sampled latent .pt, or *_before_sampling.pt containing "
            "posterior_mean and posterior_logvar"
        ),
    )
    parser.add_argument(
        "cubemap_dir",
        type=Path,
        help="Directory containing 0000..0005 images and camera_data.json",
    )
    parser.add_argument(
        "--model-checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT,
        help=f"Conditioned Nova3R checkpoint (default: {DEFAULT_MODEL_CHECKPOINT})",
    )
    parser.add_argument(
        "--model-config", type=Path, default=DEFAULT_MODEL_CONFIG,
        help="Hydra config.yaml or Lightning hparams.yaml for the conditioned model",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("conditioned_noise_sweep"))
    parser.add_argument(
        "--face-order", type=int, nargs=6, default=list(DEFAULT_FACE_ORDER),
        metavar=("F0", "F1", "F2", "F3", "F4", "F5"),
        help="Cubemap order used during training (default: 4 0 5 1 3 2)",
    )
    parser.add_argument(
        "--image-size", type=int,
        help="Square encoder input size (default: data.img_size from model config)",
    )
    parser.add_argument(
        "--timesteps", type=float, nargs="+",
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9),
        help="Flow timesteps in [0,1]; t=0 is pure noise and t=1 is clean",
    )
    parser.add_argument(
        "--reconstruction", choices=("ode", "one-step", "both"), default="ode",
    )
    parser.add_argument(
        "--posterior-mode", choices=("sample", "mean"), default="sample",
    )
    parser.add_argument("--posterior-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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
        "--skip-meshes", action="store_true",
        help="Save reconstructed latent files without decoding meshes",
    )
    args = parser.parse_args(argv)
    if sorted(args.face_order) != list(range(6)):
        parser.error("--face-order must be a permutation of 0 1 2 3 4 5")
    if not args.timesteps or any(not 0 <= value <= 1 for value in args.timesteps):
        parser.error("--timesteps must contain values in [0,1]")
    if args.image_size is not None and args.image_size < 1:
        parser.error("--image-size must be positive")
    if args.ode_step_size <= 0:
        parser.error("--ode-step-size must be positive")
    if args.octree_resolution < 2:
        parser.error("--octree-resolution must be at least 2")
    if args.num_chunks < 1:
        parser.error("--num-chunks must be positive")
    return args


def experiment_config(path: Path):
    """Accept both Hydra configs and Lightning's saved ``cfg`` hparameter."""
    from omegaconf import OmegaConf

    root = OmegaConf.load(path)
    config = root.get("experiment") or root.get("cfg") or root
    if config.get("model") is None:
        raise ValueError(f"No experiment model configuration found in {path}")
    return root, config


def checkpoint_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    return model_state or state


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    root, config = experiment_config(config_path)
    model_config = config.model
    if model_config.get("name") != "Nova3rImgCond":
        raise ValueError(
            f"Expected experiment.model.name=Nova3rImgCond in {config_path}, "
            f"got {model_config.get('name')!r}"
        )
    # Keep the nested ``cfg`` parameter as a DictConfig. Nova3rImgCond and the
    # DA3 builder use attribute access (cfg.aggregator, cfg.pts3d_head, ...).
    model = Nova3rImgCond(**model_config.params)
    state = checkpoint_model_state(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    )
    model.load_state_dict(state, stage="test", strict=True)
    if model.pts3d_head.unconditional_training:
        raise ValueError("The supplied checkpoint/config describes an unconditional DiT")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    data_config = config.get("data") or root.get("data")
    configured_size = None if data_config is None else data_config.get("img_size")
    return model, int(configured_size or config.get("resolution") or 518)


def find_face(cubemap_dir: Path, index: int):
    matches = [cubemap_dir / f"{index:04d}{suffix}" for suffix in IMAGE_SUFFIXES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one image for face {index:04d} in {cubemap_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def resize_intrinsics(intrinsic, original_hw, image_size):
    intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
    height, width = original_hw
    if all(
        abs(intrinsic[row, column]) <= 2
        for row, column in ((0, 0), (1, 1), (0, 2), (1, 2))
    ):
        intrinsic[0] *= width
        intrinsic[1] *= height
    intrinsic[0] *= image_size / width
    intrinsic[1] *= image_size / height
    return intrinsic


def load_cubemap(cubemap_dir: Path, face_order, image_size, device):
    metadata_path = cubemap_dir / "camera_data.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing cubemap metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    images, intrinsics, extrinsics = [], [], []
    paths = []
    for index in face_order:
        key = f"{index:04d}"
        if key not in metadata:
            raise KeyError(f"Missing face {key} in {metadata_path}")
        path = find_face(cubemap_dir, index)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read cubemap face: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_hw = image.shape[:2]
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
        images.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255)
        intrinsics.append(resize_intrinsics(metadata[key]["intrinsics"], original_hw, image_size))
        extrinsic = np.asarray(metadata[key]["extrinsics"], dtype=np.float32)
        if extrinsic.shape != (4, 4):
            raise ValueError(f"Face {key} extrinsics must be 4x4, got {extrinsic.shape}")
        extrinsics.append(extrinsic)
        paths.append(str(path))

    # This matches SAGE10KSDFDataset: express every face relative to the first
    # selected face. The common panorama world pose cancels from this product.
    extrinsics = np.stack(extrinsics)
    extrinsics = extrinsics @ np.linalg.inv(extrinsics[:1])
    # The training dataloader supplies RGB images in float32 [0, 1]. Mixed
    # precision is applied around the encoder below.
    images = torch.stack(images).unsqueeze(0).to(device=device)
    batch = {
        "images": images,
        "intrinsics": torch.from_numpy(np.stack(intrinsics)).unsqueeze(0).to(device),
        "extrinsics": torch.from_numpy(extrinsics).unsqueeze(0).to(device),
    }
    return images, batch, paths


class ConditionedVelocityModel(nn.Module):
    """Bind cached cubemap tokens to the ODE solver's ``forward(x,t)`` API."""

    def __init__(self, head, tokens):
        super().__init__()
        self.head = head
        self.tokens = tokens

    def forward(self, x, t, **kwargs):
        del kwargs
        with torch.autocast(
            device_type=x.device.type,
            dtype=torch.bfloat16,
            enabled=x.device.type == "cuda",
        ):
            return self.head(
                decout=[self.tokens],
                noised_latents=x,
                timestep=t.to(device=x.device, dtype=x.dtype),
            )


def reconstruct_ode(head, tokens, noised, timestep, method, step_size):
    from nova3r.flow_matching.solver import ODESolver

    if timestep >= 1:
        return noised
    time_grid = torch.tensor(
        [timestep, 1.0], device=noised.device, dtype=noised.dtype
    )
    solver = ODESolver(velocity_model=ConditionedVelocityModel(head, tokens))
    return solver.sample(
        x_init=noised,
        time_grid=time_grid,
        method=method,
        step_size=step_size,
        return_intermediates=False,
    )


def reconstruct_one_step(head, tokens, noised, timestep):
    if timestep >= 1:
        return noised
    time = torch.full(
        (noised.shape[0],), timestep, device=noised.device, dtype=noised.dtype
    )
    velocity = ConditionedVelocityModel(head, tokens)(noised, time)
    return noised + (1.0 - timestep) * velocity


def main():
    args = parse_args()
    for path, label in (
        (args.latent, "latent"),
        (args.cubemap_dir, "cubemap directory"),
        (args.model_checkpoint, "conditioned model checkpoint"),
        (args.model_config, "conditioned model config"),
    ):
        predicate = path.is_dir if label == "cubemap directory" else path.is_file
        if not predicate():
            raise FileNotFoundError(f"Missing {label}: {path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, configured_image_size = load_model(
        args.model_config, args.model_checkpoint, device
    )
    image_size = args.image_size or configured_image_size
    parameter = model.pts3d_head.x_embedder.weight
    images, batch, image_paths = load_cubemap(
        args.cubemap_dir,
        args.face_order,
        image_size,
        device,
    )
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            encoder_data = model._encode(images, batch=batch)
    tokens = encoder_data["tokens"].to(device=device, dtype=parameter.dtype)

    expected_shape = (model.pts3d_head.num_latents, model.pts3d_head.in_channels)
    target, target_source = load_target_latent(
        args.latent,
        expected_shape,
        posterior_mode=args.posterior_mode,
        posterior_seed=args.posterior_seed,
    )
    target = target.to(device=device, dtype=parameter.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.noise_seed % (2**63 - 1))
    source_noise = torch.randn(
        target.shape, device=device, dtype=target.dtype, generator=generator
    )

    bounds = np.asarray(args.bounds, dtype=np.float32)
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("Each maximum bound must be greater than its minimum")
    level = mesh_level(args.mc_level, bounds, args.octree_resolution)
    vae = None if args.skip_meshes else model.first_stage
    if vae is None and not args.skip_meshes:
        raise ValueError("The conditioned model config does not contain a first_stage VAE")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_latent(
        target,
        args.output_dir / "target.pt",
        source=str(args.latent),
        target_source=target_source,
        posterior_seed=args.posterior_seed,
    )
    if vae is not None:
        try:
            save_mesh(
                vae, target, args.output_dir / "target.ply", bounds=bounds,
                mc_level=level, num_chunks=args.num_chunks,
                octree_resolution=args.octree_resolution,
            )
        except RuntimeError as error:
            print(f"Warning: could not decode target mesh: {error}")

    methods = (
        ("ode", "one-step")
        if args.reconstruction == "both"
        else (args.reconstruction,)
    )
    manifest = {
        "source": str(args.latent),
        "target_source": target_source,
        "cubemap": str(args.cubemap_dir),
        "cubemap_faces": image_paths,
        "face_order": args.face_order,
        "image_size": image_size,
        "model_checkpoint": str(args.model_checkpoint),
        "model_config": str(args.model_config),
        "posterior_seed": args.posterior_seed,
        "noise_seed": args.noise_seed,
        "timesteps": args.timesteps,
        "reconstruction": list(methods),
        "timestep_convention": "t=0 is pure noise; t=1 is clean",
        "results": [],
    }
    with torch.inference_mode():
        for timestep in args.timesteps:
            noised = (1.0 - timestep) * source_noise + timestep * target
            for method in methods:
                if method == "ode":
                    reconstructed = reconstruct_ode(
                        model.pts3d_head, tokens, noised, timestep,
                        args.ode_method, args.ode_step_size,
                    )
                else:
                    reconstructed = reconstruct_one_step(
                        model.pts3d_head, tokens, noised, timestep
                    )
                stem = f"t_{timestep_label(timestep)}_{method.replace('-', '_')}"
                latent_path = args.output_dir / f"{stem}.pt"
                save_latent(
                    reconstructed, latent_path, timestep=timestep, method=method,
                    noise_seed=args.noise_seed, target_source=target_source,
                    cubemap=str(args.cubemap_dir), face_order=args.face_order,
                )
                result = {
                    "timestep": timestep,
                    "method": method,
                    "latent": str(latent_path),
                    "latent_mse_to_target": float(
                        torch.mean((reconstructed.float() - target.float()) ** 2)
                        .detach().cpu()
                    ),
                }
                if vae is not None:
                    mesh_path = args.output_dir / f"{stem}.ply"
                    try:
                        save_mesh(
                            vae, reconstructed, mesh_path, bounds=bounds,
                            mc_level=level, num_chunks=args.num_chunks,
                            octree_resolution=args.octree_resolution,
                        )
                        result["mesh"] = str(mesh_path)
                    except RuntimeError as error:
                        result["mesh_error"] = str(error)
                        print(f"Warning: could not decode {method} mesh at t={timestep:g}: {error}")
                manifest["results"].append(result)
                print(f"Saved {method} reconstruction at t={timestep:g} to {latent_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved sweep manifest to {manifest_path}")


if __name__ == "__main__":
    main()
