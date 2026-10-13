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


def load_pointcloud(input_file, normal_k=30, normal_orientation="camera"):
    """Load XYZ data and estimate normals when the input does not provide them."""
    suffix = os.path.splitext(input_file)[1].lower()
    if suffix == ".npz":
        try:
            return load_pointcloud_npz(input_file)
        except ValueError as exc:
            if "normals" not in str(exc).lower():
                raise
            data = np.load(input_file, allow_pickle=True)
            for key in ["points", "pointcloud", "pts", "xyz", "vertices", "coords", "surface"]:
                if key in data:
                    points = np.asarray(data[key], dtype=np.float64)[:, :3]
                    break
            else:
                raise
    elif suffix in {".glb", ".gltf"}:
        import trimesh

        loaded = trimesh.load(input_file, process=False)
        geometries = loaded.geometry.values() if isinstance(loaded, trimesh.Scene) else [loaded]
        vertices = [np.asarray(geometry.vertices, dtype=np.float64) for geometry in geometries]
        vertices = [array for array in vertices if len(array)]
        if not vertices:
            raise ValueError(f"No point geometry found in {input_file}")
        points = np.concatenate(vertices, axis=0)
    else:
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(input_file)
        points = np.asarray(cloud.points, dtype=np.float64)
        if cloud.has_normals():
            return points, np.asarray(cloud.normals, dtype=np.float64)

    if len(points) < 3:
        raise ValueError(f"Need at least three points to estimate normals, got {len(points)}")

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(normal_k, len(points)))
    )
    if normal_orientation == "camera":
        cloud.orient_normals_towards_camera_location(np.zeros(3))
    elif normal_orientation == "consistent":
        cloud.orient_normals_consistent_tangent_plane(min(normal_k, len(points) - 1))
    normals = np.asarray(cloud.normals, dtype=np.float64)
    return points, normals


def normalize_normals(normals):
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)


def sample_pointcloud(points, normals, num, rng):
    replace = len(points) < num
    index = rng.choice(len(points), num, replace=replace)
    return points[index], normals[index]


def sharp_sample_pointcloud(points, normals, rng, num=16384, k=16, threshold=0.985):
    normals = normalize_normals(normals)
    valid_normal = np.linalg.norm(normals, axis=1) > 0

    if not np.any(valid_normal):
        print("Warning: no valid normals found. Falling back to random surface samples.")
        return sample_pointcloud(points, normals, num, rng)

    query_k = min(k + 1, len(points))
    if query_k < 2:
        print("Warning: not enough points for sharp sampling. Falling back to random surface samples.")
        return sample_pointcloud(points, normals, num, rng)

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
        return sample_pointcloud(points, normals, num, rng)

    weights = 1.0 - VN2[sharp_idx]
    weights = np.maximum(weights, 0.0)
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = None

    replace = len(sharp_idx) < num
    sampled_idx = rng.choice(sharp_idx, num, replace=replace, p=weights)

    return points[sampled_idx], normals[sampled_idx]


def sample_surface(points, normals, scale, centroid, sample_num, sharp_k, sharp_threshold, rng):
    random_surface, random_normal = sample_pointcloud(points, normals, sample_num, rng)
    sharp_surface, sharp_normal = sharp_sample_pointcloud(
        points,
        normals,
        rng,
        num=sample_num,
        k=sharp_k,
        threshold=sharp_threshold,
    )

    random_surface = np.concatenate(
        (random_surface, normalize_normals(random_normal)), axis=1
    ).astype(np.float16)
    sharp_surface = np.concatenate(
        (sharp_surface, normalize_normals(sharp_normal)), axis=1
    ).astype(np.float16)
    surface = np.concatenate(
        (
            np.concatenate((random_surface, np.zeros((sample_num, 1), dtype=np.float16)), axis=1),
            np.concatenate((sharp_surface, np.ones((sample_num, 1), dtype=np.float16)), axis=1),
        ),
        axis=0,
    )
    surface_data = {
        "random_surface": random_surface,
        "sharp_surface": sharp_surface,
        "surface": surface,
        "scale": np.float16(scale),
        "centroid": centroid.astype(np.float16),
    }

    return surface_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate normals and sample VAE-ready random/sharp surface points."
    )

    parser.add_argument(
        "--input_obj",
        type=str,
        required=True,
        help="Path to an NPZ, PLY, PCD, XYZ, GLB, or glTF point cloud. Normals are estimated when absent.",
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
        help="Number of random and sharp-edge surface samples to store (default: 524288, matching udf_sample.py).",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normal_k", type=int, default=30,
        help="KNN neighborhood size for normal estimation (default: 30).",
    )
    parser.add_argument(
        "--normal_orientation", choices=["camera", "consistent", "none"], default="camera",
        help="Orient estimated normals toward origin, consistently, or not at all.",
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

    points, normals = load_pointcloud(
        args.input_obj, args.normal_k, args.normal_orientation
    )

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
        np.random.default_rng(args.seed),
    )

    parent_folder = os.path.dirname(args.output_prefix)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)

    export_surface = f"{args.output_prefix}_surface.npz"
    np.savez(export_surface, **surface_data)

    print(f"Saved surface samples to: {export_surface}")
