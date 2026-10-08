import argparse
import os
from pathlib import Path

import igl
import numpy as np
import trimesh
from scipy.stats import truncnorm


def random_sample_pointcloud(mesh, num=30000, return_index=False):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    if return_index:
        return points, normals, face_idx
    return points, normals


def sharp_sample_pointcloud(mesh, num=16384, vertex_rgb=None):
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
        if vertex_rgb is None:
            return random_sample_pointcloud(mesh, num=num)
        points, normals, face_idx = random_sample_pointcloud(
            mesh, num=num, return_index=True
        )
        colors = closest_surface_rgb(
            points, face_idx, V, F, vertex_rgb
        )
        return points, normals, colors

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

    if vertex_rgb is None:
        return samples, normals
    colors = w * vertex_rgb[edge_a[index]] + (1.0 - w) * vertex_rgb[edge_b[index]]
    return samples, normals, np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def mesh_radius_scale(V):
    centroid = V.mean(axis=0)
    centered = V - centroid
    distances = np.linalg.norm(centered, axis=1)
    if distances.size == 0:
        return 1.0
    return max(float(distances.max()) * 1.01, 1e-6)

def _mesh_with_vertex_colors(mesh):
    """Convert textures/material colors to vertex colors before concatenation."""
    if getattr(mesh.visual, "kind", None) not in ("vertex", "face", "texture"):
        return mesh
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors)
    except (AttributeError, TypeError, ValueError):
        colors = np.asarray(getattr(mesh.visual, "vertex_colors", []))
    if colors.ndim != 2 or len(colors) != len(mesh.vertices) or colors.shape[1] < 3:
        return mesh

    converted = mesh.copy()
    converted.visual = trimesh.visual.ColorVisuals(
        mesh=converted, vertex_colors=colors
    )
    return converted


def load_mesh(input_file):
    input_path = Path(input_file)
    geometry = trimesh.load(str(input_path), force=None, process=False)
    if isinstance(geometry, trimesh.Scene):
        if len(geometry.geometry) == 0:
            raise ValueError(f"Mesh scene is empty: {input_path}")
        scene = geometry.copy()
        for name, part in tuple(scene.geometry.items()):
            scene.geometry[name] = _mesh_with_vertex_colors(part)
        mesh = scene.dump(concatenate=True)
    else:
        mesh = _mesh_with_vertex_colors(geometry)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Input does not contain a triangle mesh: {input_path}")
    return mesh


def load_vertices_and_faces(input_file):
    mesh = load_mesh(input_file)
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def load_excluded_vertex_indices(mask_file, num_vertices):
    mask_path = Path(mask_file)
    try:
        indices = np.loadtxt(mask_path, dtype=np.int64, ndmin=1)
    except ValueError as error:
        raise ValueError(f"Could not parse vertex indices from {mask_path}: {error}") from error

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices

    invalid = indices[(indices < 0) | (indices >= num_vertices)]
    if invalid.size:
        preview = ", ".join(str(index) for index in invalid[:10])
        raise ValueError(
            f"Vertex mask {mask_path} contains {invalid.size} out-of-range indices "
            f"for a mesh with {num_vertices} vertices (first: {preview})."
        )
    return np.unique(indices)


def remove_vertices_and_incident_faces(V, F, excluded_indices, vertex_colors=None):
    excluded_indices = np.asarray(excluded_indices, dtype=np.int64).reshape(-1)
    if excluded_indices.size == 0:
        if vertex_colors is None:
            return V, F
        return V, F, vertex_colors

    excluded = np.zeros(len(V), dtype=bool)
    excluded[excluded_indices] = True
    kept_faces = F[~excluded[F].any(axis=1)]
    if kept_faces.size == 0:
        raise ValueError("Removing masked vertices would leave the mesh with no faces.")

    used_vertices = np.unique(kept_faces)
    old_to_new = np.full(len(V), -1, dtype=np.int64)
    old_to_new[used_vertices] = np.arange(len(used_vertices))
    if vertex_colors is None:
        return V[used_vertices], old_to_new[kept_faces]
    return V[used_vertices], old_to_new[kept_faces], vertex_colors[used_vertices]


def mesh_vertex_rgb(mesh):
    """Return uint8 RGB vertex colors, including colors derived from textures."""
    if getattr(mesh.visual, "kind", None) not in ("vertex", "face", "texture"):
        return None
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors)
    except (AttributeError, TypeError, ValueError):
        colors = np.asarray(getattr(mesh.visual, "vertex_colors", []))
    if colors.ndim != 2 or len(colors) != len(mesh.vertices) or colors.shape[1] < 3:
        return None
    colors = colors[:, :3]
    if np.issubdtype(colors.dtype, np.floating) and colors.size and colors.max() <= 1.0:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def closest_surface_rgb(
    closest_points,
    face_indices,
    vertices,
    faces,
    vertex_rgb,
    chunk_size=262144,
):
    rgb = np.empty((len(closest_points), 3), dtype=np.uint8)
    for start in range(0, len(closest_points), chunk_size):
        stop = min(start + chunk_size, len(closest_points))
        face_vertices = faces[face_indices[start:stop]]
        triangles = vertices[face_vertices]
        barycentric = trimesh.triangles.points_to_barycentric(
            triangles, closest_points[start:stop]
        )
        barycentric = np.nan_to_num(
            barycentric, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0
        )
        values = np.einsum("ni,nij->nj", barycentric, vertex_rgb[face_vertices])
        rgb[start:stop] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return rgb


def padded_mesh_bounds(mesh, padding_ratio=0.2, min_padding=0.1, sampling_scale=1.0):
    bounds = np.array(mesh.bounds, dtype=np.float64, copy=True)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"Invalid mesh bounds: {bounds}")

    extent = bounds[1] - bounds[0]
    padding = max(float(extent.max()) * padding_ratio, min_padding * sampling_scale)
    bounds[0] -= padding
    bounds[1] += padding
    return bounds


def sample_volume_points(bounds, num_points):
    bounds = np.asarray(bounds, dtype=np.float64)
    low, high = bounds[0], bounds[1]
    return np.random.rand(num_points, 3) * (high - low) + low


def sample_udf(
    mesh,
    random_surface,
    sharp_surface,
    sampling_scale=1.0,
    vertex_rgb=None,
    color_max_distance=0.05,
    batch_size=None,
):
    n_volume_points = sharp_surface.shape[0] * 2

    volume_bounds = padded_mesh_bounds(mesh, sampling_scale=sampling_scale)
    vol_points = sample_volume_points(volume_bounds, n_volume_points)

    # Near-surface random samples
    a, b = -0.25 * sampling_scale, 0.25 * sampling_scale
    mu = 0.0

    offset1 = truncnorm.rvs(
        (a - mu) / (0.005 * sampling_scale),
        (b - mu) / (0.005 * sampling_scale),
        loc=mu,
        scale=0.005 * sampling_scale,
        size=(len(random_surface), 3),
    )

    offset2 = truncnorm.rvs(
        (a - mu) / (0.05 * sampling_scale),
        (b - mu) / (0.05 * sampling_scale),
        loc=mu,
        scale=0.05 * sampling_scale,
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
            sharp_surface[:unit_num] + np.random.normal(scale=0.001 * sampling_scale, size=(unit_num, 3)),
            sharp_surface[unit_num : unit_num * 2] + np.random.normal(scale=0.003 * sampling_scale, size=(unit_num, 3)),
            sharp_surface[unit_num * 2 : unit_num * 3] + np.random.normal(scale=0.006 * sampling_scale, size=(unit_num, 3)),
            sharp_surface[unit_num * 3 : unit_num * 4] + np.random.normal(scale=0.01 * sampling_scale, size=(unit_num, 3)),
            sharp_surface[unit_num * 4 : unit_num * 5] + np.random.normal(scale=0.02 * sampling_scale, size=(unit_num, 3)),
            sharp_surface[unit_num * 5 :]
            + np.random.normal(
                scale=0.04 * sampling_scale,
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

    if batch_size is None:
        batch_size = len(all_points)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    udf = np.empty(len(all_points), dtype=np.float64)
    all_rgb = np.empty((len(all_points), 3), dtype=np.uint8) if vertex_rgb is not None else None
    for start in range(0, len(all_points), batch_size):
        stop = min(start + batch_size, len(all_points))
        squared_distance, face_indices, closest_points = igl.point_mesh_squared_distance(
            all_points[start:stop], V_np, F_np
        )
        udf[start:stop] = np.sqrt(np.maximum(squared_distance, 0.0))
        if all_rgb is not None:
            all_rgb[start:stop] = closest_surface_rgb(
                closest_points, face_indices, V_np, F_np, vertex_rgb
            )

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
        "sampling_scale": np.float32(sampling_scale),
    }

    if vertex_rgb is not None:
        vol_rgb_valid = vol_udf <= color_max_distance * sampling_scale
        vol_rgb = all_rgb[:idx1].copy()
        vol_rgb[~vol_rgb_valid] = 0
        data.update(
            {
                "vol_rgb": vol_rgb,
                "vol_rgb_valid": vol_rgb_valid,
                "random_near_rgb": all_rgb[idx1:idx2],
                "sharp_near_rgb": all_rgb[idx2:],
                "color_max_distance": np.float32(
                    color_max_distance * sampling_scale
                ),
            }
        )

    return data


def sample_mesh_udf(mesh, color_max_distance=0.05, sample_num=524288, include_sharp_surface=True, progress=None, batch_size=None):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    vertex_rgb = mesh_vertex_rgb(mesh)
    sampling_scale = mesh_radius_scale(V)

    print(f"Watertight: {mesh.is_watertight}")
    print(f"Bounds min: {mesh.vertices.min(axis=0)}")
    print(f"Bounds max: {mesh.vertices.max(axis=0)}")
    print(f"Sampling scale: {sampling_scale}")
    volume_bounds = padded_mesh_bounds(mesh, sampling_scale=sampling_scale)
    print(f"Volume sample bounds min: {volume_bounds[0]}")
    print(f"Volume sample bounds max: {volume_bounds[1]}")

    if vertex_rgb is None:
        random_surface, random_normal = random_sample_pointcloud(mesh, num=sample_num)
        sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)
        random_surface_rgb = sharp_surface_rgb = None
    else:
        random_surface, random_normal, random_face_idx = random_sample_pointcloud(
            mesh, num=sample_num, return_index=True
        )
        random_surface_rgb = closest_surface_rgb(
            random_surface,
            random_face_idx,
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            vertex_rgb,
        )
        sharp_surface, sharp_normal, sharp_surface_rgb = sharp_sample_pointcloud(
            mesh, num=sample_num, vertex_rgb=vertex_rgb
        )

    surface_data = {
        "random_surface": np.concatenate((random_surface, random_normal), axis=1).astype(np.float16),
        "bounds": np.concatenate((volume_bounds[0], volume_bounds[1])).astype(np.float32),
        "sampling_scale": np.float32(sampling_scale),
    }
    if include_sharp_surface:
        surface_data["sharp_surface"] = np.concatenate((sharp_surface, sharp_normal), axis=1).astype(np.float16)
    if vertex_rgb is not None:
        surface_data["random_surface_rgb"] = random_surface_rgb
        if include_sharp_surface:
            surface_data["sharp_surface_rgb"] = sharp_surface_rgb
    if progress is not None:
        progress()

    udf_data = sample_udf(
        mesh,
        random_surface,
        sharp_surface,
        sampling_scale=sampling_scale,
        vertex_rgb=vertex_rgb,
        color_max_distance=color_max_distance,
        batch_size=batch_size,
    )
    if progress is not None:
        progress()

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
        "--exclude_vertex_indices",
        type=Path,
        default=None,
        help="Text file containing mesh vertex indices to remove, one or more per line.",
    )
    parser.add_argument(
        "--color_max_distance",
        type=float,
        default=0.05,
        help=(
            "Maximum UDF for assigning RGB to uniform volume points, as a "
            "fraction of sampling_scale (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=524288,
        help="Number of random-surface and sharp-surface samples (default: 524288).",
    )
    parser.add_argument(
        "--omit_sharp_surface",
        action="store_true",
        help="Omit sharp_surface and sharp_surface_rgb from the surface NPZ; keep sharp-near UDF samples.",
    )
    parser.add_argument("--progress_events", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=None, help="Maximum UDF distance queries per batch (default: all points).")

    args = parser.parse_args()
    if args.color_max_distance < 0:
        parser.error("--color_max_distance must be non-negative")
    if args.sample_num < 1:
        parser.error("--sample_num must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch_size must be positive")

    mesh = load_mesh(args.input_obj)

    if args.exclude_vertex_indices is not None:
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)
        vertex_rgb = mesh_vertex_rgb(mesh)
        original_vertex_count = len(V)
        original_face_count = len(F)
        excluded_indices = load_excluded_vertex_indices(args.exclude_vertex_indices, original_vertex_count)
        if vertex_rgb is None:
            V, F = remove_vertices_and_incident_faces(V, F, excluded_indices)
        else:
            V, F, vertex_rgb = remove_vertices_and_incident_faces(
                V, F, excluded_indices, vertex_rgb
            )
        mesh = trimesh.Trimesh(
            vertices=V,
            faces=F,
            vertex_colors=vertex_rgb,
            process=False,
        )
        print(
            f"Excluded {len(excluded_indices)} masked vertices; mesh reduced from "
            f"{original_vertex_count} vertices/{original_face_count} faces to "
            f"{len(V)} vertices/{len(F)} faces."
        )

    progress = (lambda: print("SAGE_UDF_PROGRESS", flush=True)) if args.progress_events else None
    if progress is not None:
        progress()

    surface_data, udf_data = sample_mesh_udf(
        mesh,
        color_max_distance=args.color_max_distance,
        sample_num=args.sample_num,
        include_sharp_surface=not args.omit_sharp_surface,
        progress=progress,
        batch_size=args.batch_size,
    )

    parent_folder = os.path.dirname(args.output_prefix)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)

    export_surface = f"{args.output_prefix}_surface.npz"
    np.savez(export_surface, **surface_data)

    export_udf = f"{args.output_prefix}_udf.npz"
    np.savez(export_udf, **udf_data)
    if progress is not None:
        progress()

    print(f"Saved surface samples to: {export_surface}")
    print(f"Saved UDF samples to: {export_udf}")
