import numpy as np
import cv2
import argparse
import os
import json
from pathlib import Path
import utils3d
from utils import pad_to_2to1


def get_polyhedron_vertices(shape="icosahedron"):
    """
    Returns normalized vectors pointing to the centers of the faces
     of the specified polyhedron.
    """
    shape = shape.lower()
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    # Duality Logic: To look at the FACES of shape X,
    # we look at the VERTICES of its dual shape.
    if shape == "tetrahedron":
        # 4 Faces. Dual is also a tetrahedron.
        verts = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
    elif shape == "cube":
        # 6 Faces. Dual is an octahedron.
        verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    elif shape == "octahedron":
        # 8 Faces. Dual is a cube.
        verts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    elif shape == "dodecahedron":
        # 12 Faces. Dual is an icosahedron.
        verts = np.array(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ]
        )
    elif shape == "icosahedron":
        # 20 Faces. Dual is a dodecahedron.
        cube_verts = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        extra_verts = [
            [0, phi, inv_phi],
            [0, phi, -inv_phi],
            [0, -phi, inv_phi],
            [0, -phi, -inv_phi],
            [inv_phi, 0, phi],
            [-inv_phi, 0, phi],
            [inv_phi, 0, -phi],
            [-inv_phi, 0, -phi],
            [phi, inv_phi, 0],
            [phi, -inv_phi, 0],
            [-phi, inv_phi, 0],
            [-phi, -inv_phi, 0],
        ]
        verts = np.array(cube_verts + extra_verts)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    # Normalize to unit vectors
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def extrinsics_look_at(eye, look_at, up):
    """
    Creates an OpenCV-style extrinsic matrix (World-to-Camera).

    Args:
        eye: [3] Camera position (e.g., [0, 0, 0])
        look_at: [N,3] Points the camera is facing
        up: [3] World 'Up' direction (e.g., [0, 0, 1])

    Returns:
        [N, 4, 4] float32 Extrinsic matrices for each look_at point
    """
    eye = np.array(eye, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # 1. Calculate the Forward vector (Z)
    # OpenCV looks down the +Z axis
    z = look_at - eye

    # Identify which vectors are looking exactly up and down
    mask_up = np.all(np.isclose(z, [0, -1, 0]), axis=-1)
    mask_down = np.all(np.isclose(z, [0, 1, 0]), axis=-1)

    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    x = np.cross(-up, z)
    x[mask_up | mask_down] = [1, 0, 0]  # If looking straight up/down, set right to world X
    x /= np.linalg.norm(x, axis=-1, keepdims=True)

    # 3. Calculate the Down vector (Y)
    # Orthogonal to both Z and X
    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1, keepdims=True)

    # 4. Construct Rotation Matrix (R)
    # Stack as rows because extrinsics are World -> Camera
    R = np.stack([x, y, z], axis=-2)

    # 5. Construct Translation Vector (t)
    # t = -R @ eye
    t = -R @ eye

    # 6. Assemble the nx4x4 Matrix
    n, _ = look_at.shape
    extrinsic = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    extrinsic[..., :3, :3] = R
    extrinsic[:, :3, 3] = t

    return extrinsic


def get_panorama_cameras(vertices, fov=90):
    # vertices, _ = utils3d.np.create_icosahedron_mesh()
    intrinsics = utils3d.np.intrinsics_from_fov(fov_x=np.deg2rad(fov), fov_y=np.deg2rad(fov))
    extrinsics = extrinsics_look_at([0, 0, 0], vertices, [0, -1, 0]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def directions_to_spherical_uv(directions: np.ndarray, aspect_ratio=2):
    """
    Maps 3D directions to UV coordinates for a partial panorama.

    Args:
        directions: N x 3 array of vectors
        aspect_ratio: The width/height ratio of your original image (e.g., 2.5)
    """
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Horizontal (u) remains the same as it's still a 360 degree wrap
    u = (np.arctan2(directions[..., 0], directions[..., 2]) / (2 * np.pi) + 0.5) % 1.0

    # Standard spherical v (0.0 at top pole, 1.0 at bottom pole)
    v_spherical = np.arccos(-directions[..., 1]) / np.pi

    # Calculate vertical coverage factor
    # For 2:1, factor is 1.0. For 2.5:1, factor is 0.8.
    vertical_coverage = 2.0 / aspect_ratio

    # Calculate the offset (how much of the sphere is 'missing' at the top)
    offset = (1.0 - vertical_coverage) / 2.0

    # Map the spherical v to the image v
    # This stretches the middle section to fill the 0-1 range
    v = (v_spherical - offset) / vertical_coverage

    return np.stack([u, v], axis=-1)


def unproject_cv(uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """
    Unproject pixels to world coordinates using OpenCV convention.

    ## Parameters
        uv (ndarray): [..., N, 2] pixel coordinates (u, v)
        depth (ndarray): [..., N] depth values at those pixels
        intrinsics (ndarray): [..., 3, 3] camera intrinsics
        extrinsics (ndarray): [..., 4, 4] world-to-camera matrix (R|t)

    ## Returns
        (ndarray): [..., N, 3] world coordinates
    """
    # 1. Convert UV to homogeneous coordinates [u, v, 1]
    # shape: [..., N, 3]
    uv_homo = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1)

    # 2. Camera Space: P_cam = inv(K) * [u, v, 1]^T * depth
    # inv(K) shape: [..., 3, 3]
    inv_K = np.linalg.inv(intrinsics)

    # We use matmul (@) for the last two dims, so we transpose uv_homo to [..., 3, N]
    p_cam = np.matmul(inv_K, uv_homo.swapaxes(-1, -2))
    p_cam = p_cam * depth[..., None, :]  # Apply depth
    # p_cam shape: [..., 3, N]

    # 3. World Space: P_world = inv(Extrinsics) * [P_cam, 1]^T
    # Homogenize p_cam to [..., 4, N]
    p_cam_homo = np.concatenate([p_cam, np.ones_like(p_cam[..., :1, :])], axis=-2)

    inv_E = np.linalg.inv(extrinsics)
    p_world_homo = np.matmul(inv_E, p_cam_homo)  # [..., 4, N]

    # 4. Convert back from homogeneous and transpose back to [..., N, 3]
    p_world = p_world_homo[..., :3, :].swapaxes(-1, -2)

    return p_world


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, resolution: int, is_depth: bool = False):
    height, width = image.shape[:2]
    uv = utils3d.np.uv_map((resolution, resolution))
    splitted_images = []
    for i in range(len(extrinsics)):
        directions = unproject_cv(
            uv,
            np.ones_like(uv[..., 0]),
            extrinsics=extrinsics[i],
            intrinsics=intrinsics[i],
        )
        spherical_uv = directions_to_spherical_uv(directions)
        pixels = utils3d.np.uv_to_pixel(spherical_uv, (height, width)).astype(np.float32)
        ray_mag = np.linalg.norm(directions, axis=-1)

        if is_depth:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_NEAREST)
            splitted_image = splitted_image / ray_mag
        else:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)
        splitted_images.append(splitted_image)
    return splitted_images


def euler_to_rotation_matrix(rx, ry, rz):
    """
    Creates a rotation matrix given rx, ry, rz in degrees.
    Coordinate system: X is horizontal (right), Y is down, Z is forward.
    """
    rx = np.deg2rad(rx)
    ry = np.deg2rad(ry)
    rz = np.deg2rad(rz)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # Rotation about X
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    # Rotation about Y
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    # Rotation about Z
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    # Typically, R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def get_test_verts(num_views, degree_offset=0):
    start_vert = np.array([0, 0, 1])
    verts = []

    for i in range(num_views):
        angle = i * degree_offset
        # Rotation matrix around Y-axis for panning left/right
        rot_matrix = euler_to_rotation_matrix(-15, angle, 0)
        vert = rot_matrix @ start_vert
        verts.append(vert)

    return np.array(verts)


def main():
    parser = argparse.ArgumentParser(description="Split Panorama into Perspective Views")
    parser.add_argument("--image", type=str, required=True, help="Path to panorama image")
    parser.add_argument("--image_out", type=str, required=False, help="Folder to save the output images")
    parser.add_argument("--fov", type=float, default=120.0, help="FOV in degrees")
    parser.add_argument("--res", type=int, default=512, help="Output resolution")
    parser.add_argument("--num_views", type=int, default=6, help="Number of perspective views to generate")
    parser.add_argument("--degree_offset", type=float, default=0.0, help="Degree offset for camera rotation around X-axis (default: 0.0)")
    args = parser.parse_args()

    # Setup output folder structure
    img_path = Path(args.image)
    if args.image_out:
        image_out_dir = f"{args.image_out}/{img_path.stem}_{int(args.fov)}"
        base_dir = os.path.dirname(image_out_dir)
    else:
        image_out_dir = f"./{img_path.stem}_{int(args.fov)}"
        base_dir = os.path.dirname(image_out_dir)

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)

    # Prepare Camera Parameters
    verts = get_test_verts(args.num_views, degree_offset=args.degree_offset)
    image = cv2.imread(args.image)

    image = pad_to_2to1(image, mode="constant")  # Ensure 2:1 aspect ratio for proper UV mapping
    splitted_extrinsics, splitted_intriniscs = get_panorama_cameras(verts, fov=args.fov)
    splitted_resolution = args.res
    splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution)

    # Process & Save Images
    for i, im in enumerate(splitted_images):
        cv2.imwrite(os.path.join(image_out_dir, f"{i:04d}.jpg"), im)

    # Save camera_data.json
    camera_data = {}
    for i in range(len(splitted_extrinsics)):
        camera_data[f"{i:04d}"] = {
            "extrinsics": splitted_extrinsics[i].tolist(),
            "intrinsics": splitted_intriniscs[i].tolist(),
        }

    json_path = os.path.join(image_out_dir, "camera_data.json")
    with open(json_path, "w") as f:
        json.dump(camera_data, f, indent=4)

    print(f"Success! Base folder: {image_out_dir}")
    print(f"Saved: {len(splitted_images)} images")


if __name__ == "__main__":
    main()
