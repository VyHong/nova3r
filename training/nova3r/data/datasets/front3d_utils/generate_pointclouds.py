#!/usr/bin/env python3
"""Convert a single 3D-Front GLB mesh into a point cloud.

This script loads one ``.glb`` file, samples points from the mesh surface,
and writes the resulting point cloud in the same directory as the input file
unless an explicit output path is provided.

Typical usage:

    python training/data/datasets/front3d_utils/generate_pointclouds.py \
        --input-path /tmp/datasets/3DFront/3D-FRONT-TEST-SCENE/.../scene_full.glb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a point cloud from a single 3D-Front GLB mesh")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to a single .glb file")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output file path; defaults to the input file stem in the same directory",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50_000,
        help="Number of points to sample per mesh (default: 50000)",
    )
    parser.add_argument(
        "--output-format",
        choices=("npy", "ply", "both"),
        default="ply",
        help="Point-cloud format to write (default: ply)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0)")
    return parser.parse_args()


def load_scene_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load a GLB file as a single mesh with all transforms applied."""
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return trimesh.Trimesh()
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        return trimesh.Trimesh()

    if mesh.is_empty:
        return trimesh.Trimesh()

    return mesh


def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample a fixed number of surface points from a mesh."""
    if mesh.is_empty:
        return np.zeros((0, 3), dtype=np.float32)

    if len(mesh.faces) == 0:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if len(vertices) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if len(vertices) >= num_points:
            indices = np.random.choice(len(vertices), num_points, replace=False)
            return vertices[indices]
        repeats = int(np.ceil(num_points / len(vertices)))
        tiled = np.tile(vertices, (repeats, 1))
        indices = np.random.choice(len(tiled), num_points, replace=False)
        return tiled[indices]

    sampled, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(sampled, dtype=np.float32)


def save_pointcloud(points: np.ndarray, output_path: Path) -> None:
    """Write a point cloud to ``.npy`` or ``.ply`` based on the suffix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npy":
        np.save(output_path, points.astype(np.float32, copy=False))
        return

    if output_path.suffix.lower() == ".ply":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        o3d.io.write_point_cloud(str(output_path), pcd)
        return

    raise ValueError(f"Unsupported output format for {output_path}")


def build_output_paths(output_root: Path, glb_path: Path, input_root: Path, output_format: str) -> list[Path]:
    rel_path = glb_path.relative_to(input_root)
    base = output_root / rel_path.with_suffix("")

    if output_format == "npy":
        return [base.with_suffix(".npy")]
    if output_format == "ply":
        return [base.with_suffix(".ply")]
    return [base.with_suffix(".npy"), base.with_suffix(".ply")]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    glb_path = args.input_path
    if not glb_path.is_file():
        raise FileNotFoundError(f"Input GLB file does not exist: {glb_path}")
    if glb_path.suffix.lower() != ".glb":
        raise ValueError(f"Input path must point to a .glb file: {glb_path}")

    mesh = load_scene_mesh(glb_path)
    points = sample_points_from_mesh(mesh, args.num_points)

    if points.shape[0] == 0:
        raise ValueError(f"Input mesh is empty: {glb_path}")

    if args.output_path is None:
        base = glb_path.with_suffix("")
        if args.output_format == "npy":
            output_paths = [base.with_suffix(".npy")]
        elif args.output_format == "ply":
            output_paths = [base.with_suffix(".ply")]
        else:
            output_paths = [base.with_suffix(".npy"), base.with_suffix(".ply")]
    else:
        if args.output_path.suffix:
            output_paths = [args.output_path]
        else:
            if args.output_format == "npy":
                output_paths = [args.output_path.with_suffix(".npy")]
            elif args.output_format == "ply":
                output_paths = [args.output_path.with_suffix(".ply")]
            else:
                output_paths = [args.output_path.with_suffix(".npy"), args.output_path.with_suffix(".ply")]

    for output_path in output_paths:
        save_pointcloud(points, output_path)


if __name__ == "__main__":
    main()
