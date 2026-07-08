import argparse
import os
from pathlib import Path

import igl
import numpy as np
import trimesh
from scipy.stats import truncnorm


def random_sample_pointcloud(mesh, num=30000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]

    rng = np.random.default_rng()
    index = rng.choice(num, num, replace=False)

    return points[index], normals[index]


def sharp_sample_pointcloud(mesh, num=16384):
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    N = np.asarray(mesh.face_normals)
    VN = np.asarray(mesh.vertex_normals)

    VN2 = np.ones(V.shape[0])

    for i in range(3):
        dot = np.stack(
            (
                VN2[F[:, i]],
                np.sum(VN[F[:, i]] * N, axis=-1),
            ),
            axis=-1,
        )
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985

    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))

    sharp_edge = sharp_mask[edge_a] * sharp_mask[edge_b]

    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    if len(edge_a) == 0:
        print("Warning: no sharp edges found. Falling back to random surface samples.")
        return random_sample_pointcloud(mesh, num=num)

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights = weights / np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)

    index = np.searchsorted(weights.cumsum(), random_number)

    samples = w * sharp_verts_a[index] + (1.0 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1.0 - w) * sharp_verts_bn[index]

    return samples, normals


def normalize_to_centroid_sphere(V):
    centroid = V.mean(axis=0)
    V_centered = V - centroid

    distances = np.linalg.norm(V_centered, axis=1)
    max_radius = distances.max()

    scale = max_radius * 1.01
    V_normalized = V_centered / scale

    return V_normalized, scale, centroid


def normalize_to_median(V, mode="median"):
    if mode == "median":
        target_median = 1.0
    else:
        target_median = float(mode.split("_")[-1])
    if target_median == 0.0:
        raise ValueError("Median normalization target must be non-zero.")

    translation = np.array([0,0,1.6]) 
    V = V - translation

    distances = np.linalg.norm(V, axis=1)
    norm_factor = np.median(distances) if distances.size > 0 else 1.0
    norm_factor = np.clip(norm_factor, 0.01, 100.0)

    V_normalized = V / norm_factor * target_median
    V_normalized = np.clip(V_normalized, -1000.0, 1000.0)

    scale = norm_factor / target_median
    centroid = np.zeros(3, dtype=V.dtype)

    return V_normalized, scale, centroid


def normalize_vertices(V, mode="centroid_sphere"):
    if mode in ("centroid_sphere", "sphere"):
        return normalize_to_centroid_sphere(V)
    if "median" in mode:
        return normalize_to_median(V, mode=mode)

    raise ValueError(f"Unknown normalize mode: {mode}")


def load_vertices_and_faces(input_file):
    input_path = Path(input_file)

    if input_path.suffix.lower() == ".glb":
        geometry = trimesh.load(str(input_path), force=None)

        if isinstance(geometry, trimesh.Scene):
            if len(geometry.geometry) == 0:
                raise ValueError("GLB scene is empty.")
            mesh = geometry.dump(concatenate=True)
        else:
            mesh = geometry

        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)

    else:
        V, F = igl.read_triangle_mesh(str(input_path))
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)

    return V, F


def padded_mesh_bounds(mesh, padding_ratio=0.2, min_padding=0.1):
    bounds = np.array(mesh.bounds, dtype=np.float64, copy=True)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"Invalid mesh bounds: {bounds}")

    extent = bounds[1] - bounds[0]
    padding = max(float(extent.max()) * padding_ratio, min_padding)
    bounds[0] -= padding
    bounds[1] += padding
    return bounds


def sample_volume_points(bounds, num_points):
    bounds = np.asarray(bounds, dtype=np.float64)
    low, high = bounds[0], bounds[1]
    return np.random.rand(num_points, 3) * (high - low) + low


def sample_udf(mesh, random_surface, sharp_surface):
    n_volume_points = sharp_surface.shape[0] * 2

    volume_bounds = padded_mesh_bounds(mesh)
    vol_points = sample_volume_points(volume_bounds, n_volume_points)

    # Near-surface random samples
    a, b = -0.25, 0.25
    mu = 0.0

    offset1 = truncnorm.rvs(
        (a - mu) / 0.005,
        (b - mu) / 0.005,
        loc=mu,
        scale=0.005,
        size=(len(random_surface), 3),
    )

    offset2 = truncnorm.rvs(
        (a - mu) / 0.05,
        (b - mu) / 0.05,
        loc=mu,
        scale=0.05,
        size=(len(random_surface), 3),
    )

    random_near_points = np.concatenate(
        [
            random_surface + offset1,
            random_surface + offset2,
        ],
        axis=0,
    )

    # Near-sharp-edge samples
    unit_num = len(sharp_surface) // 6

    sharp_near_points = np.concatenate(
        [
            sharp_surface[:unit_num] + np.random.normal(scale=0.001, size=(unit_num, 3)),
            sharp_surface[unit_num : unit_num * 2] + np.random.normal(scale=0.003, size=(unit_num, 3)),
            sharp_surface[unit_num * 2 : unit_num * 3] + np.random.normal(scale=0.006, size=(unit_num, 3)),
            sharp_surface[unit_num * 3 : unit_num * 4] + np.random.normal(scale=0.01, size=(unit_num, 3)),
            sharp_surface[unit_num * 4 : unit_num * 5] + np.random.normal(scale=0.02, size=(unit_num, 3)),
            sharp_surface[unit_num * 5 :]
            + np.random.normal(
                scale=0.04,
                size=(len(sharp_surface) - 5 * unit_num, 3),
            ),
        ],
        axis=0,
    )

    np.random.shuffle(random_near_points)
    np.random.shuffle(sharp_near_points)

    V_np = np.asarray(mesh.vertices, dtype=np.float64, order="C")
    F_np = np.asarray(mesh.faces, dtype=np.int64, order="C")

    all_points = np.concatenate(
        [
            vol_points,
            random_near_points,
            sharp_near_points,
        ],
        axis=0,
    ).astype(np.float64)

    squared_distance, _, _ = igl.point_mesh_squared_distance(all_points, V_np, F_np)
    udf = np.sqrt(np.maximum(squared_distance, 0.0))

    idx1 = len(vol_points)
    idx2 = idx1 + len(random_near_points)

    vol_udf = udf[:idx1]
    random_near_udf = udf[idx1:idx2]
    sharp_near_udf = udf[idx2:]

    data = {
        "vol_points": vol_points.astype(np.float16),
        "vol_label": vol_udf.astype(np.float16),
        "random_near_points": random_near_points.astype(np.float16),
        "random_near_label": random_near_udf.astype(np.float16),
        "sharp_near_points": sharp_near_points.astype(np.float16),
        "sharp_near_label": sharp_near_udf.astype(np.float16),
        "bounds": np.concatenate((volume_bounds[0], volume_bounds[1])).astype(np.float32),
    }

    return data


def sample_mesh_udf(V, F, scale, centroid):
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    print(f"Watertight: {mesh.is_watertight}")
    print(f"Bounds min: {mesh.vertices.min(axis=0)}")
    print(f"Bounds max: {mesh.vertices.max(axis=0)}")
    volume_bounds = padded_mesh_bounds(mesh)
    print(f"Volume sample bounds min: {volume_bounds[0]}")
    print(f"Volume sample bounds max: {volume_bounds[1]}")

    sample_num = (1048576 * 2) // 4

    random_surface, random_normal = random_sample_pointcloud(mesh, num=sample_num)
    sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    surface_data = {
        "random_surface": np.concatenate((random_surface, random_normal), axis=1).astype(np.float16),
        "sharp_surface": np.concatenate((sharp_surface, sharp_normal), axis=1).astype(np.float16),
        "scale": np.float16(scale),
        "centroid": centroid.astype(np.float16),
        "bounds": np.concatenate((volume_bounds[0], volume_bounds[1])).astype(np.float32),
    }

    udf_data = sample_udf(mesh, random_surface, sharp_surface)

    udf_data["scale"] = np.float16(scale)
    udf_data["centroid"] = centroid.astype(np.float16)

    return surface_data, udf_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample UDF data from a mesh without adding a bounding box.")

    parser.add_argument(
        "--input_obj",
        type=str,
        required=True,
        help="Path to the input mesh file, e.g. OBJ, GLB, PLY, STL.",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Base path for output file, e.g. output/room_001.",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="median",
        help="Normalization mode: centroid_sphere, sphere, median, or median_<target>.",
    )

    args = parser.parse_args()

    V, F = load_vertices_and_faces(args.input_obj)

    # Important: no bounding box is added here.
    V, scale, centroid = normalize_vertices(V, mode=args.normalize_mode)

    print(f"Normalize mode: {args.normalize_mode}")
    print(f"Scale: {scale}")
    print(f"Centroid: {centroid}")

    surface_data, udf_data = sample_mesh_udf(V, F, scale, centroid)

    parent_folder = os.path.dirname(args.output_prefix)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)

    export_surface = f"{args.output_prefix}_surface.npz"
    np.savez(export_surface, **surface_data)

    export_udf = f"{args.output_prefix}_udf.npz"
    np.savez(export_udf, **udf_data)

    print(f"Saved surface samples to: {export_surface}")
    print(f"Saved UDF samples to: {export_udf}")

    # Optional debug export of normalized mesh
    igl.writeOBJ(f"{args.output_prefix}_normalized.obj", V, F)
