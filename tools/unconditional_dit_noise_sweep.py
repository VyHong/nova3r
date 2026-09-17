#!/usr/bin/env python3
"""Reconstruct one ShapeVAE latent or posterior with an unconditional DiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

DEFAULT_DIT_CONFIG = REPO_ROOT / "checkpoints/unconditional_dit/.hydra/config.yaml"
DEFAULT_VAE_CHECKPOINT = REPO_ROOT / "exp_output/hunyuan_vae_v2_1_udf/checkpoints/last.ckpt"
DEFAULT_VAE_CONFIG = REPO_ROOT / "checkpoints/hunyuan/.hydra/config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add fixed flow noise to one latent at selected timesteps, reconstruct "
            "it with an unconditional DiT, and decode each result to a mesh. The "
            "input may be a sampled latent .pt or a *_before_sampling.pt containing "
            "posterior_mean and posterior_logvar. In "
            "this flow convention t=0 is pure noise and t=1 is clean."
        )
    )
    parser.add_argument(
        "latent",
        type=Path,
        help=("Sampled latent .pt, or a *_before_sampling.pt checkpoint containing " "posterior_mean and posterior_logvar."),
    )
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--dit-config", type=Path, default=DEFAULT_DIT_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("noise_sweep"))
    parser.add_argument(
        "--timesteps",
        type=float,
        nargs="+",
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9),
        help="Flow timesteps in [0, 1]; t=0 is pure noise and t=1 is clean.",
    )
    parser.add_argument(
        "--reconstruction",
        choices=("ode", "one-step", "both"),
        default="ode",
        help="ODE integration, the training-style one-step estimate, or both.",
    )
    parser.add_argument(
        "--posterior-mode",
        choices=("sample", "mean"),
        default="sample",
        help=("How to turn a *_before_sampling.pt posterior into the target latent " "(default: sample). Ignored for an already sampled latent."),
    )
    parser.add_argument(
        "--posterior-seed",
        type=int,
        default=0,
        help="Seed used with --posterior-mode sample (default: 0).",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ode-method", default="euler")
    parser.add_argument("--ode-step-size", type=float, default=0.01)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        default=(-1.2, -0.8, -1.2, 1.4, 0.8, 1.4),
    )
    parser.add_argument("--octree-resolution", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=40000)
    parser.add_argument(
        "--mc-level",
        default="auto",
        help="Marching-cubes level or 'auto'.",
    )
    parser.add_argument(
        "--skip-meshes",
        action="store_true",
        help="Save reconstructed latent files without decoding meshes.",
    )
    args = parser.parse_args()

    if not args.timesteps:
        parser.error("--timesteps requires at least one value")
    if any(timestep < 0.0 or timestep > 1.0 for timestep in args.timesteps):
        parser.error("--timesteps values must be in [0, 1]")
    if args.ode_step_size <= 0:
        parser.error("--ode-step-size must be positive")
    if args.octree_resolution < 2:
        parser.error("--octree-resolution must be at least 2")
    if args.num_chunks < 1:
        parser.error("--num-chunks must be at least 1")
    return args


def _checkpoint_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if "model" in checkpoint:
        return checkpoint["model"]
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}
    return model_state or state


def load_dit(config_path: Path, checkpoint_path: Path, device: torch.device):
    from omegaconf import OmegaConf
    from nova3r.heads.pts3d_decoder.latent_flowm_decoder import DiT

    root = OmegaConf.load(config_path)
    config = root.get("experiment", root)
    model_config = config.get("model")
    if model_config is None or model_config.get("name") != "DiT":
        raise ValueError(f"No top-level DiT model configuration in {config_path}")
    params = OmegaConf.to_container(model_config.get("params"), resolve=True)
    model = DiT(**params)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _checkpoint_model_state(checkpoint)
    model.load_state_dict(state, strict=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def _single_latent(tensor, path: Path, expected_shape):
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"Expected latent [1, {expected_shape[0]}, {expected_shape[1]}] in " f"{path}, got {tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise ValueError(f"Expected one latent in {path}, found {tensor.shape[0]}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Latent contains non-finite values: {path}")
    return tensor


def load_target_latent(
    path: Path,
    expected_shape,
    *,
    posterior_mode: str,
    posterior_seed: int,
):
    from nova3r.heads.hunyuan_model.autoencoders.model import (
        DiagonalGaussianDistribution,
    )

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if torch.is_tensor(saved):
        return _single_latent(saved, path, expected_shape), "latent"
    if not isinstance(saved, dict):
        raise ValueError(f"Unsupported latent checkpoint in {path}")

    if torch.is_tensor(saved.get("latent")):
        return _single_latent(saved["latent"], path, expected_shape), "latent"

    mean = saved.get("posterior_mean")
    logvar = saved.get("posterior_logvar")
    if not torch.is_tensor(mean) or not torch.is_tensor(logvar):
        raise ValueError(f"No latent or posterior_mean/posterior_logvar tensors found in {path}")
    mean = _single_latent(mean, path, expected_shape)
    logvar = _single_latent(logvar, path, expected_shape)
    if mean.shape != logvar.shape:
        raise ValueError(f"Posterior mean and log-variance shapes differ in {path}: " f"{tuple(mean.shape)} versus {tuple(logvar.shape)}")
    posterior = DiagonalGaussianDistribution([mean, logvar], feat_dim=-1)
    if posterior_mode == "mean":
        return posterior.mode(), "posterior_mean"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(posterior_seed % (2**63 - 1))
    return posterior.sample(generator=generator), f"posterior_sample_seed_{posterior_seed}"


class VelocityModel(nn.Module):
    """Adapt DiT to the ODE solver's ``forward(x, t)`` interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, **kwargs):
        del kwargs
        with torch.autocast(
            device_type=x.device.type,
            dtype=torch.bfloat16,
            enabled=x.device.type == "cuda",
        ):
            return self.model(
                noised_latents=x,
                timestep=t.to(device=x.device, dtype=x.dtype),
            )


def reconstruct_ode(model, noised_latent, timestep, *, method, step_size):
    from nova3r.flow_matching.solver import ODESolver

    if timestep >= 1.0:
        return noised_latent
    solver = ODESolver(velocity_model=VelocityModel(model))
    time_grid = torch.tensor(
        [timestep, 1.0],
        dtype=noised_latent.dtype,
        device=noised_latent.device,
    )
    return solver.sample(
        x_init=noised_latent,
        time_grid=time_grid,
        method=method,
        step_size=step_size,
        return_intermediates=False,
    )


def reconstruct_one_step(model, noised_latent, timestep):
    if timestep >= 1.0:
        return noised_latent
    time = torch.full(
        (noised_latent.shape[0],),
        timestep,
        dtype=noised_latent.dtype,
        device=noised_latent.device,
    )
    with torch.autocast(
        device_type=noised_latent.device.type,
        dtype=torch.bfloat16,
        enabled=noised_latent.device.type == "cuda",
    ):
        velocity = model(noised_latents=noised_latent, timestep=time)
    return noised_latent + (1.0 - timestep) * velocity


def timestep_label(timestep):
    return f"{timestep:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def save_latent(latent, path: Path, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": latent.detach().float().cpu(), **metadata}, path)


def mesh_level(value, bounds, octree_resolution):
    if str(value).lower() != "auto":
        return float(value)
    voxel_size = float(np.max(bounds[3:] - bounds[:3])) / octree_resolution
    return float(np.clip(voxel_size * 128.0, 1e-4, 1.0))


def save_mesh(
    vae,
    latent,
    path: Path,
    *,
    bounds,
    mc_level,
    num_chunks,
    octree_resolution,
):
    from nova3r.heads.hunyuan_model.pipelines import export_to_trimesh

    with torch.inference_mode():
        decoded = vae.decode(latent.to(dtype=next(vae.parameters()).dtype))
        outputs = vae.latents2mesh(
            decoded,
            output_type="trimesh",
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo="mc",
            enable_pbar=True,
        )
    meshes = export_to_trimesh(outputs)
    meshes = meshes if isinstance(meshes, list) else [meshes]
    mesh = meshes[0]
    if mesh is None or not len(mesh.vertices) or not len(mesh.faces):
        raise RuntimeError(f"The VAE decoder did not produce a surface for {path.stem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def main():
    args = parse_args()
    for path, description in (
        (args.latent, "latent"),
        (args.dit_checkpoint, "DiT checkpoint"),
        (args.dit_config, "DiT config"),
        (args.vae_checkpoint, "VAE checkpoint"),
        (args.vae_config, "VAE config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dit = load_dit(args.dit_config, args.dit_checkpoint, device)
    expected_shape = (dit.num_latents, dit.in_channels)
    target, target_source = load_target_latent(
        args.latent,
        expected_shape,
        posterior_mode=args.posterior_mode,
        posterior_seed=args.posterior_seed,
    )
    target = target.to(device=device, dtype=next(dit.parameters()).dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.noise_seed % (2**63 - 1))
    source_noise = torch.randn(
        target.shape,
        dtype=target.dtype,
        device=device,
        generator=generator,
    )

    bounds = np.asarray(args.bounds, dtype=np.float32)
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("Each maximum bound must be greater than its minimum")
    mc_level = mesh_level(args.mc_level, bounds, args.octree_resolution)
    vae = None
    if not args.skip_meshes:
        from tools.vae_latent_to_scene import load_model as load_vae_model

        vae = load_vae_model(args.vae_checkpoint, args.vae_config, device)
        if tuple(vae.latent_shape) != expected_shape:
            raise ValueError(f"DiT latent shape {expected_shape} does not match VAE latent " f"shape {tuple(vae.latent_shape)}")

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
                vae,
                target,
                args.output_dir / "target.ply",
                bounds=bounds,
                mc_level=mc_level,
                num_chunks=args.num_chunks,
                octree_resolution=args.octree_resolution,
            )
        except RuntimeError as error:
            print(f"Warning: could not decode target mesh: {error}")

    methods = ("ode", "one-step") if args.reconstruction == "both" else (args.reconstruction,)
    manifest = {
        "source": str(args.latent),
        "target_source": target_source,
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
                        dit,
                        noised,
                        timestep,
                        method=args.ode_method,
                        step_size=args.ode_step_size,
                    )
                else:
                    reconstructed = reconstruct_one_step(dit, noised, timestep)

                stem = f"t_{timestep_label(timestep)}_{method.replace('-', '_')}"
                latent_path = args.output_dir / f"{stem}.pt"
                save_latent(
                    reconstructed,
                    latent_path,
                    timestep=timestep,
                    method=method,
                    noise_seed=args.noise_seed,
                    target_source=target_source,
                )
                result = {
                    "timestep": timestep,
                    "method": method,
                    "latent": str(latent_path),
                    "latent_mse_to_target": float(torch.mean((reconstructed.float() - target.float()) ** 2).detach().cpu()),
                }
                if vae is not None:
                    mesh_path = args.output_dir / f"{stem}.ply"
                    try:
                        save_mesh(
                            vae,
                            reconstructed,
                            mesh_path,
                            bounds=bounds,
                            mc_level=mc_level,
                            num_chunks=args.num_chunks,
                            octree_resolution=args.octree_resolution,
                        )
                        result["mesh"] = str(mesh_path)
                    except RuntimeError as error:
                        result["mesh_error"] = str(error)
                        print(f"Warning: could not decode {method} mesh at " f"t={timestep:g}: {error}")
                manifest["results"].append(result)
                print(f"Saved {method} reconstruction at t={timestep:g} to {latent_path}")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved sweep manifest to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
