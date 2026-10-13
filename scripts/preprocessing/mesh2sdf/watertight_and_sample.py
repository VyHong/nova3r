# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
import argparse
import igl
import numpy as np
import os
from scipy.stats import truncnorm
import trimesh
from pathlib import Path


def random_sample_pointcloud(mesh, num=30000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng()
    index = rng.choice(num, num, replace=False)
    return points[index], normals[index]


def sharp_sample_pointcloud(mesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    # collect edge
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    sharp_edge = sharp_mask[edge_a] * sharp_mask[edge_b]
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals


def sample_sdf(mesh, random_surface, sharp_surface, scale):
    n_volume_points = sharp_surface.shape[0] * 2
    vol_points = (np.random.rand(n_volume_points, 3) - 0.5) * 2 * 1.05

    a, b = -0.25, 0.25
    mu = 0

    # get near points (add offset on surface points)
    offset1 = truncnorm.rvs((a - mu) / 0.005, (b - mu) / 0.005, loc=mu, scale=0.005, size=(len(random_surface), 3))
    offset2 = truncnorm.rvs((a - mu) / 0.05, (b - mu) / 0.05, loc=mu, scale=0.05, size=(len(random_surface), 3))
    random_near_points = np.concatenate([random_surface + offset1, random_surface + offset2], axis=0)

    unit_num = len(sharp_surface) // 6
    sharp_near_points = np.concatenate(
        [
            sharp_surface[:unit_num] + np.random.normal(scale=0.001, size=(unit_num, 3)),
            sharp_surface[unit_num : unit_num * 2] + np.random.normal(scale=0.003, size=(unit_num, 3)),
            sharp_surface[unit_num * 2 : unit_num * 3] + np.random.normal(scale=0.06, size=(unit_num, 3)),
            sharp_surface[unit_num * 3 : unit_num * 4] + np.random.normal(scale=0.01, size=(unit_num, 3)),
            sharp_surface[unit_num * 4 : unit_num * 5] + np.random.normal(scale=0.02, size=(unit_num, 3)),
            sharp_surface[unit_num * 5 :] + np.random.normal(scale=0.04, size=(len(sharp_surface) - 5 * unit_num, 3)),
        ],
        axis=0,
    )

    np.random.shuffle(random_near_points)
    np.random.shuffle(sharp_near_points)

    sign_type = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    # Ensure V and F are standard, contiguous NumPy arrays with correct dtypes
    V_np = np.asarray(mesh.vertices, dtype=np.float64, order="C")
    F_np = np.asarray(mesh.faces, dtype=np.int64, order="C")

    sign_type = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER

    # 1. Stack all points together
    all_points = np.concatenate([vol_points, random_near_points, sharp_near_points], axis=0)

    # 2. Query SDF once (builds the AABB tree only once!)
    try:
        all_sdf, _, _, _ = igl.signed_distance(all_points.astype(np.float64), V_np, F_np, sign_type=sign_type)
    except Exception:
        all_sdf, _, _, _ = igl.signed_distance(all_points.astype(np.float64), V_np, F_np)

    # 3. Slice the results back to their respective arrays
    idx1 = len(vol_points)
    idx2 = idx1 + len(random_near_points)

    vol_sdf = all_sdf[:idx1]
    random_near_sdf = all_sdf[idx1:idx2]
    sharp_near_sdf = all_sdf[idx2:]
    vol_label = -vol_sdf
    random_near_label = -random_near_sdf
    sharp_near_label = -sharp_near_sdf

    # vol_label = np.clip(-vol_sdf * scale, -1.0, 1.0)
    # random_near_label = np.clip(-random_near_sdf * scale, -1.0, 1.0)
    # sharp_near_label = np.clip(-sharp_near_sdf * scale, -1.0, 1.0)
    data = {
        "vol_points": vol_points.astype(np.float16),
        "vol_label": vol_label.astype(np.float16),
        "random_near_points": random_near_points.astype(np.float16),
        "random_near_label": random_near_label.astype(np.float16),
        "sharp_near_points": sharp_near_points.astype(np.float16),
        "sharp_near_label": sharp_near_label.astype(np.float16),
    }
    return data


def SampleMesh(V, F, sdf_scale, scale, centroid):
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    print(f"watertight after recon: {mesh.is_watertight}")

    area = mesh.area
    sample_num = (1048576 * 2) // 4
    # sample_num = 499712//4

    random_surface, random_normal = random_sample_pointcloud(mesh, num=sample_num)
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    # save_surface
    surface = np.concatenate((random_surface, random_normal), axis=1).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(np.float16)

    surface_data = {"random_surface": surface, "sharp_surface": sharp_surface, "scale": np.float16(scale), "centroid": centroid.astype(np.float16)}

    print(f"Max Value mesh.vertices size: {np.max(mesh.vertices, axis=0)} min: {np.min(mesh.vertices, axis=0)}")
    sdf_data = sample_sdf(mesh, random_surface, random_sharp_surface, sdf_scale)
    sdf_data["scale"] = np.float16(scale)
    sdf_data["centroid"] = centroid.astype(np.float16)

    return surface_data, sdf_data


def normalize_to_centroid_sphere(V):
    """
    Normalize the vertices V using its centroid and maximum distance.
    The centroid is moved to the origin (0,0,0) and the points are scaled
    so they fit inside a sphere of radius 1 (enclosed in a [-1, 1]^3 box).

    V: (n,3) numpy array of vertex positions.
    Returns: normalized V
    """
    # 1. Find the centroid (average position of all points)
    centroid = V.mean(axis=0)

    # 2. Shift the object so its center of mass is at the origin (0,0,0)
    V_centered = V - centroid

    # 3. Calculate the Euclidean distance of every point from the new origin
    # np.linalg.norm calculates the distance (sqrt(x^2 + y^2 + z^2)) for each row
    distances = np.linalg.norm(V_centered, axis=1)

    # 4. Find the absolute furthest point to use as our maximum radius
    max_radius = distances.max()

    # 5. Apply the 1% padding (optional, but matches your original style)
    scale = max_radius * 1.01

    # 6. Scale all vertices uniformly
    V_normalized = V_centered / scale

    return V_normalized, scale, centroid


# Given: V (n x 3 array of vertices), F (m x 3 array of faces)
# Parameters epsilon/grid_res
def Watertight(V, F, epsilon=2.0 / 256, grid_res=256):
    # Compute bounding box
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    # Create a uniform grid
    x = np.linspace(min_corner[0], max_corner[0], grid_res)
    y = np.linspace(min_corner[1], max_corner[1], grid_res)
    z = np.linspace(min_corner[2], max_corner[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Compute SDF at grid points using igl.signed_distance with pseudo normals
    sdf, _, _, _ = igl.signed_distance(grid_points, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)

    mc_verts, mc_faces, _ = igl.marching_cubes(epsilon - np.abs(sdf), grid_points, grid_res, grid_res, grid_res, 0.0)

    mesh = trimesh.Trimesh(mc_verts, mc_faces, process=False)
    mesh.export("all_components.obj")
    components = mesh.split(only_watertight=True)
    print(f"Marching cubes generated {len(components)} disconnected components.")

    components = sorted(components, key=lambda c: len(c.faces), reverse=True)
    valid_components = [components[0]]
    if len(components) > 1:
        # Calculate how big the second component is compared to the first
        size_ratio = len(components[1].faces) / len(components[0].faces)

        # If the second shell is at least 15% the size of the main shell,
        # it is almost certainly the inner wall of a closed room.
        # If it is tiny (e.g., 1%), it is just noise from an open room.
        if size_ratio > 0.30:
            valid_components.append(components[1])
            print(f"Detected Closed Room. Kept inner and outer shells. (Size Ratio: {size_ratio:.2f})")
        else:
            print(f"Detected Open Room. Discarded minor noise. (Size Ratio: {size_ratio:.2f})")

    final_mesh = trimesh.util.concatenate(valid_components)

    return final_mesh.vertices, final_mesh.faces


def load_vertices_and_faces(input_file):
    """
    Loads a 3D model and returns Vertices (V) and Faces (F).
    Uses trimesh for .glb files and igl for everything else.
    """
    input_path = Path(input_file)

    if input_path.suffix.lower() == ".glb":
        # 1. Load the GLB file using trimesh
        geometry = trimesh.load(str(input_path), force=None)

        # 2. GLB files often load as a Scene. Combine them into a single mesh.
        if isinstance(geometry, trimesh.Scene):
            if len(geometry.geometry) == 0:
                raise ValueError("GLB scene is empty.")
            # Concatenate all meshes in the scene into one
            mesh = geometry.dump(concatenate=True)
        else:
            mesh = geometry

        # 3. Extract vertices and faces as numpy arrays
        V = np.array(mesh.vertices)
        F = np.array(mesh.faces)

    else:
        # Fallback to libigl for .obj, .ply, .off, .stl, etc.
        V, F = igl.read_triangle_mesh(str(input_path))

    return V, F


def add_bounding_box(V, F, distance):
    """
    Add a closed axis-aligned bounding box around a mesh.

    The box is expanded by ``distance`` beyond the mesh bounds on every side.
    Distance uses the same coordinate units as ``V``. Returns the vertices and
    faces of the original mesh and bounding box as one disconnected mesh.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    if V.ndim != 2 or V.shape[1] != 3 or len(V) == 0:
        raise ValueError("V must be a non-empty array with shape (N, 3).")
    if F.ndim != 2 or F.size == 0:
        raise ValueError(f"F must be a non-empty 2D face array; received shape {F.shape}.")

    # Accept transposed triangle arrays and fixed-width polygon arrays.
    if F.shape[1] != 3 and F.shape[0] == 3:
        F = F.T
    if F.shape[1] > 3:
        F = np.concatenate(
            [np.column_stack((F[:, 0], F[:, i], F[:, i + 1])) for i in range(1, F.shape[1] - 1)],
            axis=0,
        )
    if F.shape[1] != 3:
        raise ValueError("F must contain triangles or fixed-width polygons; " f"received shape {F.shape}.")
    if F.min() < 0 or F.max() >= len(V):
        raise ValueError(f"F contains vertex indices outside the valid range 0..{len(V) - 1}.")
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("distance must be a finite, non-negative number.")

    min_corner = V.min(axis=0) - distance
    max_corner = V.max(axis=0) + distance
    extents = max_corner - min_corner
    if np.any(extents <= 0):
        raise ValueError("The padded bounding box must have positive extents.")

    center = (min_corner + max_corner) / 2.0
    transform = np.eye(4)
    transform[:3, 3] = center
    box = trimesh.creation.box(extents=extents, transform=transform)

    box_faces = np.asarray(box.faces, dtype=np.int64) + len(V)
    combined_vertices = np.vstack((V, np.asarray(box.vertices)))
    combined_faces = np.vstack((F, box_faces))
    return combined_vertices, combined_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an OBJ/GLB file and output surface and SDF data.")
    parser.add_argument("--input_obj", type=str, help="Path to the input file (OBJ, GLB, etc.)")
    parser.add_argument("--output_prefix", type=str, default=None, help="Base name for output files")
    args = parser.parse_args()

    input_file = args.input_obj
    name = args.output_prefix

    # Now libigl can read the .ply file flawlessly
    V, F = load_vertices_and_faces(input_file)
    V, F = add_bounding_box(V, F, distance=0.05)
    V, scale, centroid = normalize_to_centroid_sphere(V)
    print(f"Scale: {scale} Centroid: {centroid}")
    # mc_verts, mc_faces = Watertight(V, F, epsilon=2 / 512, grid_res=384)

    mc_verts, mc_faces = V, F
    sdf_scale = 1
    surface_data, sdf_data = SampleMesh(mc_verts, mc_faces, sdf_scale, scale, centroid)

    parent_folder = os.path.dirname(args.output_prefix)
    os.makedirs(parent_folder, exist_ok=True)
    export_surface = f"{name}_surface.npz"
    np.savez(export_surface, **surface_data)
    export_sdf = f"{name}_sdf.npz"
    np.savez(export_sdf, **sdf_data)
    igl.writeOBJ(f"{name}_watertight.obj", mc_verts, mc_faces)
