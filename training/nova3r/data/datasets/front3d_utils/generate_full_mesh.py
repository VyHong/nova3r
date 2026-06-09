#!/usr/bin/env python3
"""Generate or process meshes for only the 3D-Front GLBs that contain `_full` in the name."""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
from functools import partial
from pathlib import Path

# Assuming you have corresponding mesh utilities in your repository structure
# If not, you can use packages like trimesh or open3d directly here.
from training.nova3r.data.datasets.front3d_utils.generate_pointclouds import (
    build_output_paths,  # Can adapt this for mesh extensions
    load_scene_mesh,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/Process meshes for GLBs with `_full` in the filename")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing .glb files")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory to save processed meshes")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1)")
    parser.add_argument(
        "--output-format",
        choices=("glb", "obj", "ply"),
        default="glb",
        help="Mesh format to write",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    return parser.parse_args()


def iter_full_glb_files(input_root: Path):
    for glb_path in sorted(input_root.rglob("*_full*.glb")):
        if glb_path.is_file():
            yield glb_path


def process_mesh_data(mesh, output_format: str):
    """
    Placeholder for any mesh processing logic you might want to inject.
    e.g., mesh.decimate(), mesh.remove_degenerate_triangles(), etc.
    """
    # If you just want to convert or copy the file, you can return it as-is
    return mesh


def save_mesh(mesh, output_path: Path) -> None:
    """
    Saves the mesh to the target path based on its extension.
    Modify this depending on whether you use Trimesh, Open3D, or standard file copies.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Example using trimesh-like API. If doing a pure copy, use shutil.copy instead.
    if hasattr(mesh, "export"):
        mesh.export(str(output_path))
    else:
        raise NotImplementedError("Implement custom export logic for your mesh object type.")


def process_glb(glb_path: Path, args: argparse.Namespace) -> None:
    try:
        # Determine target output path
        relative_path = glb_path.relative_to(args.input_root)
        output_path = args.output_root / relative_path.with_suffix(f".{args.output_format}")

        if args.skip_existing and output_path.exists():
            return

        # If you just need a straight format conversion or file organization copy:
        if args.output_format == "glb" and not hasattr(args, "process_mesh"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(glb_path, output_path)
            return

        # If modifications to the mesh vertices/faces are required:
        mesh = load_scene_mesh(glb_path)
        processed_mesh = process_mesh_data(mesh, args.output_format)

        save_mesh(processed_mesh, output_path)

    except Exception as e:
        print(f"Failed to process mesh for {glb_path}: {e}")


def main() -> None:
    args = parse_args()

    glb_files = list(iter_full_glb_files(args.input_root))
    if not glb_files:
        raise FileNotFoundError(f"No *_full*.glb files found under {args.input_root}")

    if args.workers > 1:
        # Keeping maxtasksperchild to prevent geometry processing memory leaks
        with multiprocessing.Pool(processes=args.workers, maxtasksperchild=50) as pool:
            process_func = partial(process_glb, args=args)
            for _ in tqdm(pool.imap_unordered(process_func, glb_files, chunksize=10), total=len(glb_files), desc="Processing full GLB meshes"):
                pass
    else:
        for glb_path in tqdm(glb_files, desc="Processing full GLB meshes"):
            process_glb(glb_path, args)


if __name__ == "__main__":
    main()
