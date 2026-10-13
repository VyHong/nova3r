#!/usr/bin/env python3
"""Reconstruct a point cloud with ShapeVAE and render a turntable video."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_cluster import knn

from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE
from nova3r.heads.hunyuan_model.pipelines import (
    colorize_trimesh_vertices,
    export_to_trimesh,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode a point cloud with ShapeVAE, decode a mesh, and render it."
    )
    parser.add_argument("pointcloud", type=Path, help="PLY, PCD, XYZ, NPY, or NPZ point cloud")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Hydra config (default: CHECKPOINT_DIR/.hydra/config.yaml)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("vae_inference"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal-k", type=int, default=30)
    parser.add_argument("--sharp-k", type=int, default=16)
    parser.add_argument("--sharp-threshold", type=float, default=0.985)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=40000)
    parser.add_argument(
        "--mc-level", default="auto", help="Marching-cubes level or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--normalization",
        choices=("auto", "none", "median", "bbox", "cube"),
        default="auto",
        help="Point-coordinate normalization (default: infer from training config)",
    )
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def find_shape_vae_config(config_path):
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    root = cfg.get("experiment", cfg)
    root = root.get("cfg", root)
    model = root.get("model")
    if model is None:
        raise ValueError(f"No model section found in {config_path}")

    if model.get("name") == "ShapeVAE":
        params = model.get("params")
    else:
        nova_cfg = model.get("params", {}).get("cfg", {})
        first_stage = nova_cfg.get("first_stage")
        if first_stage is None or first_stage.get("name") != "ShapeVAE":
            raise ValueError(f"No ShapeVAE or ShapeVAE first_stage found in {config_path}")
        params = first_stage.get("params")

    params = OmegaConf.to_container(params, resolve=True)
    if params.get("cfg") is not None:
        raise ValueError(
            "This ShapeVAE config selects a point-output head, not the implicit "
            "geo_decoder required for mesh extraction."
        )
    return params, str(root.get("surface_geo_normalize_mode", "none"))


def checkpoint_state(path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
    if not isinstance(state, dict):
        raise ValueError(f"Could not find a state dict in {path}")
    return state


def select_vae_state(state, model):
    model_keys = set(model.state_dict())
    prefixes = ("", "model.", "first_stage.", "model.first_stage.", "vae.", "model.vae.")
    candidates = []
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix) and key[len(prefix) :] in model_keys
        }
        candidates.append((len(selected), prefix, selected))
    count, prefix, selected = max(candidates, key=lambda item: item[0])
    if count == 0:
        raise ValueError("No ShapeVAE parameters in the checkpoint matched the config")
    print(f"Loading {count}/{len(model_keys)} ShapeVAE tensors (checkpoint prefix {prefix!r})")
    return selected


def read_pointcloud(path, normal_k):
    import open3d as o3d

    suffix = path.suffix.lower()
    normals = None
    if suffix == ".npy":
        array = np.load(path)
        points = np.asarray(array[:, :3], dtype=np.float32)
        if array.shape[1] >= 6:
            normals = np.asarray(array[:, 3:6], dtype=np.float32)
    elif suffix == ".npz":
        data = np.load(path)
        key = next((key for key in ("points", "pointcloud", "pts", "xyz", "vertices", "surface") if key in data), None)
        if key is None:
            raise ValueError(f"Could not find a point array in {path}; keys: {list(data.keys())}")
        array = np.asarray(data[key])
        points = np.asarray(array[:, :3], dtype=np.float32)
        if array.shape[1] >= 6:
            normals = np.asarray(array[:, 3:6], dtype=np.float32)
        elif "normals" in data:
            normals = np.asarray(data["normals"], dtype=np.float32)
    else:
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float32)
        if cloud.has_normals():
            normals = np.asarray(cloud.normals, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Expected at least two XYZ points, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite coordinates")

    if normals is None or normals.shape != points.shape:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(
                knn=min(normal_k, len(points))
            )
        )
        if len(points) > 3:
            cloud.orient_normals_consistent_tangent_plane(
                min(normal_k, len(points) - 1)
            )
        normals = np.asarray(cloud.normals, dtype=np.float32)
    return points, normals


def normalize_points(points, mode):
    if mode in ("none", "raw", "skip"):
        return points, torch.zeros(3, device=points.device), points.new_tensor(1.0)
    if mode == "median" or mode.startswith("median_"):
        target = 1.0 if mode == "median" else float(mode.split("_")[-1])
        scale = target / torch.linalg.vector_norm(points, dim=1).median().clamp(0.01, 100.0)
        return points * scale, torch.zeros(3, device=points.device), scale
    if mode not in ("bbox", "cube"):
        raise ValueError(f"Unknown normalization mode: {mode}")
    lower, upper = points.amin(0), points.amax(0)
    center = (lower + upper) * 0.5
    extent = (upper - lower).max()
    if extent.item() <= 0:
        raise ValueError("Point cloud has zero spatial extent")
    scale = 2.0 * (1.0 if mode == "cube" else 0.98) / extent
    return (points - center) * scale, center, scale


def sample_surface(
    points, normals, random_count, sharp_count, point_feats, generator, sharp_k, sharp_threshold
):
    def sample(indices, count, weights=None):
        if weights is not None:
            return indices[torch.multinomial(
                weights, count, replacement=len(indices) < count, generator=generator
            )]
        if len(indices) < count:
            return indices[torch.randint(len(indices), (count,), device=points.device, generator=generator)]
        return indices[torch.randperm(len(indices), device=points.device, generator=generator)[:count]]

    all_indices = torch.arange(len(points), device=points.device)
    random_idx = sample(all_indices, random_count)

    if sharp_count:
        k = min(sharp_k + 1, len(points))
        if k == 1:
            score = torch.zeros(len(points), device=points.device)
        else:
            edge_index = knn(points, points, k=k).view(2, len(points), k)
            neighbors = edge_index[1, :, 1:]
            dots = (normals[:, None] * normals[neighbors]).sum(-1).abs()
            score = 1.0 - dots.amin(1)
        candidates = torch.where(score > 1.0 - sharp_threshold)[0]
        if not len(candidates):
            candidates = all_indices
        weights = score[candidates]
        weights = weights if weights.sum().item() > 0 else None
        sharp_idx = sample(candidates, sharp_count, weights)
    else:
        sharp_idx = torch.empty(0, dtype=torch.long, device=points.device)

    indices = torch.cat((random_idx, sharp_idx))
    xyz = points[indices]
    if point_feats == 0:
        return xyz
    if point_feats == 3:
        return torch.cat((xyz, normals[indices]), dim=1)
    if point_feats == 4:
        labels = torch.cat((
            torch.zeros(random_count, device=points.device),
            torch.ones(sharp_count, device=points.device),
        ))[:, None]
        return torch.cat((xyz, normals[indices], labels), dim=1)
    raise ValueError(f"Unsupported ShapeVAE point_feats={point_feats}; expected 0, 3, or 4")


def main():
    args = parse_args()
    config_path = args.config or args.checkpoint.parent / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}; pass --config explicitly")

    params, training_normalization = find_shape_vae_config(config_path)
    model = ShapeVAE(**params)
    selected = select_vae_state(checkpoint_state(args.checkpoint), model)
    missing, unexpected = model.load_state_dict(selected, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected ShapeVAE keys: {unexpected[:10]}")
    required_missing = [key for key in missing if key.startswith(("encoder.", "pre_kl.", "post_kl.", "transformer.", "geo_decoder."))]
    if required_missing:
        raise RuntimeError(f"Checkpoint is missing {len(required_missing)} required tensors; first: {required_missing[:10]}")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()

    normalization = training_normalization if args.normalization == "auto" else args.normalization
    print(f"Using coordinate normalization: {normalization}")
    points, normals = read_pointcloud(args.pointcloud, args.normal_k)
    points = torch.from_numpy(points).to(device)
    normals = torch.from_numpy(normals).to(device)
    normals = torch.nn.functional.normalize(normals, dim=1)
    points, center, scale = normalize_points(points, normalization)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    surface = sample_surface(
        points,
        normals,
        model.encoder.pc_size,
        model.encoder.pc_sharpedge_size,
        model.encoder.point_feats,
        generator,
        args.sharp_k,
        args.sharp_threshold,
    )
    surface = surface.unsqueeze(0).to(dtype=dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lower, upper = points.amin(0), points.amax(0)
    padding = ((upper - lower) * 0.1).clamp_min(1e-3)
    bounds = torch.cat((lower - padding, upper + padding)).float().cpu().numpy()
    if args.mc_level == "auto":
        voxel_size = float(np.max(bounds[3:] - bounds[:3])) / max(args.octree_resolution - 1, 1)
        mc_level = float(np.clip(voxel_size * 128.0, 1e-4, 1.0))
    else:
        mc_level = float(args.mc_level)
    print(f"Using mesh bounds {bounds.tolist()} and mc_level={mc_level:.6g}")

    with torch.inference_mode():
        encoded = model.encode(surface, sample_posterior=False)
        latent_path = args.output_dir / "latent.pt"
        torch.save(encoded.detach().cpu(), latent_path)
        decoded = model.decode(encoded)
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
    print(f"Saved latent: {latent_path}")

    meshes = export_to_trimesh(outputs)
    mesh = meshes[0] if isinstance(meshes, list) else meshes
    if mesh is None or not len(mesh.vertices) or not len(mesh.faces):
        raise RuntimeError("The VAE decoder did not produce a surface")
    mesh = colorize_trimesh_vertices(mesh, model, decoded)
    mesh.vertices = np.asarray(mesh.vertices) / scale.item() + center.cpu().numpy()
    mesh_path = args.output_dir / "reconstruction.ply"
    mesh.export(mesh_path)
    print(f"Saved mesh: {mesh_path}")

    if not args.no_video:
        from scripts.inference.demo_nova3r import render_360_video

        video_path = render_360_video(
            str(mesh_path),
            str(args.output_dir),
            num_frames=args.frames,
            fps=args.fps,
        )
        print(f"Saved video: {video_path}")


if __name__ == "__main__":
    main()
