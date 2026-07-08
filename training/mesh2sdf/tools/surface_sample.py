import argparse
import os

import numpy as np
from scipy.spatial import cKDTree


def normalize_to_centroid_sphere(V):
    centroid = V.mean(axis=0)
    V_centered = V - centroid

    distances = np.linalg.norm(V_centered, axis=1)
    max_radius = distances.max()

    scale = max_radius * 1.01
    V_normalized = V_centered / scale

    return V_normalized, scale, centroid


def load_pointcloud_npz(input_file):
    data = np.load(input_file, allow_pickle=True)
    keys = list(data.files)

    point_key = None
    for key in ["points_normals", "point_normals", "points", "pointcloud", "pts", "xyz", "vertices", "coords", "surface"]:
        if key in data:
            point_key = key
            pointcloud = np.asarray(data[key], dtype=np.float64)
            break
    else:
        if len(keys) != 1:
            raise KeyError(
                f"Could not infer pointcloud key in {input_file}. "
                f"Expected one array or one of the common keys, got: {keys}"
            )
        point_key = keys[0]
        pointcloud = np.asarray(data[point_key], dtype=np.float64)

    if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
        raise ValueError(f"Expected pointcloud shape (N, 3+) for key '{point_key}', got {pointcloud.shape}")

    points = pointcloud[:, :3]

    if pointcloud.shape[1] >= 6:
        normals = pointcloud[:, 3:6]
    else:
        normal_key = None
        for key in ["normals", "normal", "vertex_normals"]:
            if key in data:
                normal_key = key
                break

        if normal_key is None:
            raise ValueError(
                f"Pointcloud key '{point_key}' has shape {pointcloud.shape}, so normals were not in columns 3:6. "
                f"Expected a separate normals key like 'normals'. Available keys: {keys}"
            )

        normals = np.asarray(data[normal_key], dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] < 3:
            raise ValueError(f"Expected normals shape (N, 3+) for key '{normal_key}', got {normals.shape}")
        if len(normals) != len(points):
            raise ValueError(
                f"Point and normal counts do not match: key '{point_key}' has {len(points)}, "
                f"key '{normal_key}' has {len(normals)}"
            )
        normals = normals[:, :3]

    return points, normals


def normalize_normals(normals):
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)


def sample_pointcloud(points, normals, num):
    rng = np.random.default_rng()
    replace = len(points) < num
    index = rng.choice(len(points), num, replace=replace)
    return points[index], normals[index]


def sharp_sample_pointcloud(points, normals, num=16384, k=16, threshold=0.985):
    normals = normalize_normals(normals)
    valid_normal = np.linalg.norm(normals, axis=1) > 0

    if not np.any(valid_normal):
        print("Warning: no valid normals found. Falling back to random surface samples.")
        return sample_pointcloud(points, normals, num)

    query_k = min(k + 1, len(points))
    if query_k < 2:
        print("Warning: not enough points for sharp sampling. Falling back to random surface samples.")
        return sample_pointcloud(points, normals, num)

    tree = cKDTree(points)
    try:
        _, neighbor_idx = tree.query(points, k=query_k, workers=-1)
    except TypeError:
        _, neighbor_idx = tree.query(points, k=query_k)

    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]

    neighbor_idx = neighbor_idx[:, 1:]
    neighbor_normals = normals[neighbor_idx]
    normal_dot = np.sum(normals[:, None, :] * neighbor_normals, axis=-1)
    normal_dot = np.abs(normal_dot)

    VN2 = np.min(normal_dot, axis=-1)
    sharp_mask = (VN2 < threshold) & valid_normal

    sharp_idx = np.flatnonzero(sharp_mask)
    if len(sharp_idx) == 0:
        print("Warning: no sharp points found. Falling back to random surface samples.")
        return sample_pointcloud(points, normals, num)

    weights = 1.0 - VN2[sharp_idx]
    weights = np.maximum(weights, 0.0)
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = None

    replace = len(sharp_idx) < num
    sampled_idx = np.random.choice(sharp_idx, num, replace=replace, p=weights)

    return points[sampled_idx], normals[sampled_idx]


def sample_surface(points, normals, scale, centroid, sample_num, sharp_k, sharp_threshold):
    random_surface, random_normal = sample_pointcloud(points, normals, sample_num)
    sharp_surface, sharp_normal = sharp_sample_pointcloud(
        points,
        normals,
        num=sample_num,
        k=sharp_k,
        threshold=sharp_threshold,
    )

    surface_data = {
        "random_surface": np.concatenate((random_surface, normalize_normals(random_normal)), axis=1).astype(np.float16),
        "sharp_surface": np.concatenate((sharp_surface, normalize_normals(sharp_normal)), axis=1).astype(np.float16),
        "scale": np.float16(scale),
        "centroid": centroid.astype(np.float16),
    }

    return surface_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample random and sharp surface point clouds from an NPZ pointcloud."
    )

    parser.add_argument(
        "--input_obj",
        type=str,
        required=True,
        help="Path to an NPZ pointcloud with xyz+normals in one array, or xyz plus a separate normals array.",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Base path for output file, e.g. output/room_001.",
    )

    parser.add_argument(
        "--sample_num",
        type=int,
        default=(1048576 * 2) // 4,
        help="Number of random and sharp-edge surface samples to store.",
    )

    parser.add_argument(
        "--sharp_k",
        type=int,
        default=16,
        help="Number of nearest neighbors used to infer sharp points from point normals.",
    )

    parser.add_argument(
        "--sharp_threshold",
        type=float,
        default=0.985,
        help="Normal agreement threshold for inferred sharp points. Lower values are stricter.",
    )

    args = parser.parse_args()

    points, normals = load_pointcloud_npz(args.input_obj)

    # Important: no bounding box is added here.
    points, scale, centroid = normalize_to_centroid_sphere(points)

    print(f"Input points: {len(points)}")
    print(f"Scale: {scale}")
    print(f"Centroid: {centroid}")

    surface_data = sample_surface(
        points,
        normals,
        scale,
        centroid,
        args.sample_num,
        args.sharp_k,
        args.sharp_threshold,
    )

    parent_folder = os.path.dirname(args.output_prefix)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)

    export_surface = f"{args.output_prefix}_surface.npz"
    np.savez(export_surface, **surface_data)

    print(f"Saved surface samples to: {export_surface}")
