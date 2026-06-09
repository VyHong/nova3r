#!/usr/bin/env python3
"""Generate point clouds for only the 3D-Front GLBs that contain `_full` in the name."""

from __future__ import annotations

import argparse
import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from training.nova3r.data.datasets.front3d_utils.generate_pointclouds import (
    build_output_paths,
    load_scene_mesh,
    sample_points_from_mesh,
    save_pointcloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate point clouds for GLBs with `_full` in the filename")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing .glb files")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory to save generated point clouds")
    parser.add_argument("--num-points", type=int, default=1000_000, help="Number of points to sample per mesh")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1)")
    parser.add_argument(
        "--output-format",
        choices=("npy", "ply", "both"),
        default="ply",
        help="Point-cloud format to write",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    return parser.parse_args()


def iter_full_glb_files(input_root: Path):
    for glb_path in sorted(input_root.rglob("*_full*.glb")):
        if glb_path.is_file():
            yield glb_path


def process_glb(glb_path: Path, args: argparse.Namespace) -> None:
    try:
        output_paths = build_output_paths(args.output_root, glb_path, args.input_root, args.output_format)
        if args.skip_existing and all(path.exists() for path in output_paths):
            return

        mesh = load_scene_mesh(glb_path)
        points = sample_points_from_mesh(mesh, args.num_points)

        if points.shape[0] == 0:
            print(f"Skipped empty mesh: {glb_path}")
            return

        for output_path in output_paths:
            save_pointcloud(points, output_path)
    except Exception as e:
        print(f"Failed to process {glb_path}: {e}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    glb_files = list(iter_full_glb_files(args.input_root))
    if not glb_files:
        raise FileNotFoundError(f"No *_full*.glb files found under {args.input_root}")

    if args.workers > 1:
        # Use multiprocessing Pool with maxtasksperchild=50 to prevent memory leaks and OOM while keeping overhead low
        with multiprocessing.Pool(processes=args.workers, maxtasksperchild=50) as pool:
            process_func = partial(process_glb, args=args)
            for _ in tqdm(pool.imap_unordered(process_func, glb_files, chunksize=10), total=len(glb_files), desc="Sampling full GLB meshes"):
                pass
    else:
        for glb_path in tqdm(glb_files, desc="Sampling full GLB meshes"):
            process_glb(glb_path, args)


if __name__ == "__main__":
    main()
