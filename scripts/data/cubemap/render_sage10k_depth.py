#!/usr/bin/env python3
"""Ray-render metric equirectangular depth panoramas from SAGE-10K GLBs."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render SAGE-10K GLBs at the saved panorama poses as depth."
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-depth", type=float, default=65.535, help="Metres")
    parser.add_argument("--ray-chunk-size", type=int, default=262144)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Process at most this many scenes")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError(f"No triangle geometry found in {path}")
    return mesh


def load_extrinsics(path):
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{path} must contain a non-empty filename-to-matrix mapping")
    result = {}
    for filename, matrix in values.items():
        w2c = np.asarray(matrix, dtype=np.float64)
        if w2c.shape != (4, 4) or not np.isfinite(w2c).all():
            raise ValueError(f"Invalid extrinsic for {filename!r} in {path}")
        result[filename] = w2c
    return result


def camera_rays(width, height):
    """Blender equirectangular rays: centre is camera -Z and +Y is up."""
    u = (np.arange(width, dtype=np.float64) + 0.5) / width
    v = (np.arange(height, dtype=np.float64) + 0.5) / height
    longitude = (u - 0.5) * (2.0 * np.pi)
    latitude = (0.5 - v) * np.pi
    longitude, latitude = np.meshgrid(longitude, latitude)
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            np.sin(longitude) * cos_latitude,
            np.sin(latitude),
            -np.cos(longitude) * cos_latitude,
        ),
        axis=-1,
    ).reshape(-1, 3)


def make_raycaster(mesh):
    raycaster = o3d.t.geometry.RaycastingScene()
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32),
    )
    raycaster.add_triangles(tensor_mesh)
    return raycaster


def render_depth(raycaster, directions_camera, w2c, shape, max_depth, chunk_size):
    c2w = np.linalg.inv(w2c)
    origin = c2w[:3, 3]
    directions = directions_camera @ c2w[:3, :3].T
    depth = np.zeros(len(directions), dtype=np.float64)
    for start in range(0, len(directions), chunk_size):
        stop = min(start + chunk_size, len(directions))
        directions_chunk = directions[start:stop]
        rays = np.concatenate(
            (np.broadcast_to(origin, directions_chunk.shape), directions_chunk),
            axis=1,
        )
        depth[start:stop] = raycaster.cast_rays(
            o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        )["t_hit"].numpy()
    valid = (depth > 0.0) & (depth <= max_depth)
    depth_mm = np.zeros(len(depth), dtype=np.uint16)
    depth_mm[valid] = np.rint(depth[valid] * 1000.0).astype(np.uint16)
    return depth_mm.reshape(shape)


def process_scene(scene_glb, input_root, output_root, args, directions):
    scene_dir = scene_glb.parent
    extrinsics_path = scene_dir / "extrinsics.json"
    if not extrinsics_path.is_file():
        raise FileNotFoundError(f"Missing sibling extrinsics: {extrinsics_path}")
    extrinsics = load_extrinsics(extrinsics_path)
    relative = scene_dir.relative_to(input_root)
    destination = (output_root / relative if output_root else scene_dir) / "depth"
    outputs = [destination / Path(name).with_suffix(".png") for name in extrinsics]
    if not args.overwrite and all(path.is_file() for path in outputs):
        print(f"Skipping complete scene: {relative}")
        return

    mesh = load_mesh(scene_glb)
    raycaster = make_raycaster(mesh)
    destination.mkdir(parents=True, exist_ok=True)
    for (filename, w2c), output_path in zip(extrinsics.items(), outputs):
        if output_path.is_file() and not args.overwrite:
            continue
        print(f"Rendering depth: {relative / filename}")
        depth = render_depth(
            raycaster,
            directions,
            w2c,
            (args.height, args.width),
            args.max_depth,
            args.ray_chunk_size,
        )
        temporary = output_path.with_name("." + output_path.name + ".tmp.png")
        Image.fromarray(depth).save(temporary)
        os.replace(temporary, output_path)


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else None
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if args.width < 1 or args.height < 1 or args.ray_chunk_size < 1:
        raise SystemExit("Dimensions and ray chunk size must be positive")
    if not 0 < args.max_depth <= 65.535:
        raise SystemExit("--max-depth must be in (0, 65.535]")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    scenes = sorted(input_root.rglob("scene.glb"))[
        args.shard_index :: args.shard_count
    ]
    if args.limit is not None:
        scenes = scenes[: args.limit]
    if not scenes:
        raise SystemExit(f"No scene.glb files found below {input_root}")

    directions = camera_rays(args.width, args.height)
    failures = []
    for scene_glb in scenes:
        try:
            process_scene(scene_glb, input_root, output_root, args, directions)
        except Exception as exc:
            failures.append((scene_glb, exc))
            print(f"Failed: {scene_glb}: {exc}")
    print(f"Completed {len(scenes) - len(failures)}/{len(scenes)} scenes")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
