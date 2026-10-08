#!/usr/bin/env python3
"""Bake SAGE-10K panorama appearance onto an existing point cloud.

Each point is projected into every equirectangular panorama using the saved
Blender-axis world-to-camera extrinsic.  A point contributes only when its
radial distance agrees with the corresponding rendered depth panorama.  The
visible RGB observations are then combined into one color per point.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pointcloud", type=Path, help="Input PLY/PCD point cloud")
    parser.add_argument(
        "scene_dir", type=Path,
        help="Directory containing panoramas, extrinsics.json, and depth/",
    )
    parser.add_argument("output", type=Path, help="Output colored point cloud (PLY recommended)")
    parser.add_argument("--num-poses", type=int, default=10)
    parser.add_argument(
        "--blend", choices=("median", "mean"), default="median",
        help="How to combine colors seen by multiple cameras (default: median)",
    )
    parser.add_argument(
        "--depth-tolerance", type=float, default=0.05,
        help="Absolute visibility tolerance in metres (default: 0.05)",
    )
    parser.add_argument(
        "--relative-depth-tolerance", type=float, default=0.01,
        help="Additional tolerance as a fraction of depth (default: 0.01)",
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--unseen-policy", choices=("nearest", "input", "black"), default="nearest",
        help=("Color points unseen by all cameras from the nearest baked point, "
              "their input color, or black (default: nearest)"),
    )
    parser.add_argument(
        "--allow-fewer-poses", action="store_true",
        help="Use all available poses instead of failing when fewer than --num-poses exist",
    )
    return parser.parse_args()


def load_cloud(path):
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_geometry()
    points = np.asarray(getattr(geometry, "vertices", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError(f"No points found in {path}")
    colors = np.asarray(getattr(geometry.visual, "vertex_colors", []))
    if colors.ndim != 2 or len(colors) != len(points) or colors.shape[1] < 3:
        colors = np.full((len(points), 4), 255, dtype=np.uint8)
    elif colors.shape[1] == 3:
        colors = np.column_stack((colors, np.full(len(points), 255, dtype=np.uint8)))
    else:
        colors = colors[:, :4].astype(np.uint8, copy=False)
    return geometry, points, colors


def load_poses(scene_dir, requested, allow_fewer):
    path = scene_dir / "extrinsics.json"
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a filename-to-matrix mapping")
    entries = sorted(data.items())
    if len(entries) < requested and not allow_fewer:
        raise ValueError(f"Requested {requested} poses, but {path} contains {len(entries)}")
    entries = entries[:requested]
    poses = []
    for filename, value in entries:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid 4x4 extrinsic for {filename!r}")
        rgb_path = scene_dir / filename
        depth_path = scene_dir / "depth" / Path(filename).with_suffix(".png")
        if not rgb_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"Missing RGB or depth for {filename}: {rgb_path}, {depth_path}")
        poses.append((filename, matrix, rgb_path, depth_path))
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses


def project_equirectangular(points, world_to_camera, width, height):
    """Return nearest pixel coordinates and radial depth for Blender PANO axes."""
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    radius = np.linalg.norm(camera, axis=1)
    safe_radius = np.maximum(radius, np.finfo(np.float64).eps)
    longitude = np.arctan2(camera[:, 0], -camera[:, 2])
    latitude = np.arcsin(np.clip(camera[:, 1] / safe_radius, -1.0, 1.0))
    x = np.floor((longitude / (2.0 * np.pi) + 0.5) * width).astype(np.int64) % width
    y = np.floor((0.5 - latitude / np.pi) * height).astype(np.int64)
    y = np.clip(y, 0, height - 1)
    return x, y, radius


def bake(points, poses, blend, absolute_tolerance, relative_tolerance, chunk_size):
    count = len(points)
    seen = np.zeros(count, dtype=np.uint16)
    if blend == "mean":
        accumulator = np.zeros((count, 3), dtype=np.float64)
    else:
        # Ten uint8 RGB observations are modest even for large point clouds.
        observations = np.zeros((len(poses), count, 3), dtype=np.uint8)
        valid_observations = np.zeros((len(poses), count), dtype=bool)

    for pose_index, (name, world_to_camera, rgb_path, depth_path) in enumerate(poses):
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth_image = np.asarray(Image.open(depth_path))
        if depth_image.ndim != 2:
            raise ValueError(f"Depth image must be single-channel: {depth_path}")
        if depth_image.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth dimensions differ for {name}: {rgb.shape[:2]} vs {depth_image.shape}"
            )
        depth = depth_image.astype(np.float64) / 1000.0
        height, width = depth.shape
        visible_total = 0
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            x, y, radius = project_equirectangular(
                points[start:stop], world_to_camera, width, height
            )
            rendered_depth = depth[y, x]
            tolerance = absolute_tolerance + relative_tolerance * rendered_depth
            valid = (rendered_depth > 0.0) & (np.abs(radius - rendered_depth) <= tolerance)
            indices = np.flatnonzero(valid) + start
            samples = rgb[y[valid], x[valid]]
            seen[indices] += 1
            visible_total += len(indices)
            if blend == "mean":
                accumulator[indices] += samples
            else:
                observations[pose_index, indices] = samples
                valid_observations[pose_index, indices] = True
        print(f"{name}: {visible_total:,}/{count:,} points visible")

    colors = np.zeros((count, 3), dtype=np.uint8)
    observed = seen > 0
    if blend == "mean":
        colors[observed] = np.rint(
            accumulator[observed] / seen[observed, None]
        ).astype(np.uint8)
    else:
        masked = np.ma.array(
            observations, mask=np.repeat(~valid_observations[:, :, None], 3, axis=2)
        )
        colors[observed] = np.ma.median(masked[:, observed], axis=0).filled(0).astype(np.uint8)
    return colors, seen


def fill_unseen_colors(points, colors, seen, input_colors, policy):
    """Retain and color all points, including surfaces absent from every view."""
    unseen = seen == 0
    if not unseen.any():
        return colors
    if policy == "input":
        colors[unseen] = input_colors[unseen, :3]
    elif policy == "nearest":
        observed = ~unseen
        if observed.any():
            tree = cKDTree(points[observed])
            observed_indices = np.flatnonzero(observed)
            unseen_indices = np.flatnonzero(unseen)
            for start in range(0, len(unseen_indices), 500_000):
                target_indices = unseen_indices[start:start + 500_000]
                _, neighbors = tree.query(points[target_indices], k=1, workers=-1)
                colors[target_indices] = colors[observed_indices[neighbors]]
        else:
            colors[unseen] = input_colors[unseen, :3]
    return colors


def main():
    args = parse_args()
    if args.num_poses < 1 or args.chunk_size < 1:
        raise SystemExit("--num-poses and --chunk-size must be positive")
    if args.depth_tolerance < 0 or args.relative_depth_tolerance < 0:
        raise SystemExit("Depth tolerances must be non-negative")
    geometry, points, input_colors = load_cloud(args.pointcloud)
    poses = load_poses(args.scene_dir, args.num_poses, args.allow_fewer_poses)
    colors, seen = bake(
        points, poses, args.blend, args.depth_tolerance,
        args.relative_depth_tolerance, args.chunk_size,
    )
    colors = fill_unseen_colors(
        points, colors, seen, input_colors, args.unseen_policy
    )
    rgba = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
        output = geometry.copy()
        output.visual = trimesh.visual.ColorVisuals(mesh=output, vertex_colors=rgba)
    else:
        output = trimesh.points.PointCloud(points, colors=rgba)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.export(args.output)
    print(
        f"Wrote all {len(points):,} points to {args.output}; "
        f"{(seen > 0).sum():,} directly observed and {(seen == 0).sum():,} "
        f"filled with policy={args.unseen_policy!r} from {len(poses)} poses"
    )


if __name__ == "__main__":
    main()
