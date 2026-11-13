"""Shared data, point-cloud, metric, and evaluation-report helpers."""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


PANORAMA_TO_OPENCV = np.diag(np.asarray([1, -1, -1, 1], dtype=np.float32))
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_EVAL_POINTS = 100_000
DEFAULT_FSCORE_THRESHOLDS = (0.1, 0.05, 0.02)
DEFAULT_ALIGNMENT_ITERATIONS = 300
DEFAULT_ALIGNMENT_LR = 0.1
InputProjection = Literal["cubemap", "erp"]
DatasetLayout = Literal["sage10k", "scannetpp", "3dfront"]
SCANNETPP_SCANNER_TO_OPENCV = np.asarray(
    [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SageSample:
    key: str
    scene: str
    frame: int
    cubemap_dir: Path
    pose_path: Path
    surface_path: Path
    dataset_layout: DatasetLayout = "sage10k"


def read_manifest(
    path: Path,
    dataset_layout: DatasetLayout = "sage10k",
) -> list[tuple[str, int]]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"Manifest must contain a JSON list: {path}")
    parsed = []
    for entry in entries:
        try:
            if dataset_layout == "3dfront":
                from nova3r.utils.three_d_front import parse_panorama_sample

                sample = parse_panorama_sample(entry)
                parsed.append((sample.room, sample.frame))
            else:
                scene, frame = entry.split()
                parsed.append((scene, int(frame)))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid manifest entry {entry!r} in {path}") from exc
    return parsed


def resolve_scannetpp_room(root: Path, scene: str, frame: int) -> Path | None:
    matches = []
    for room in sorted((Path(root) / "rooms" / scene).glob("*")):
        poses = room / "test_camera_poses.json"
        images = room / "test_images.txt"
        if poses.is_file():
            frames = json.loads(poses.read_text())["frames"]
            contains_frame = any(int(key) == int(frame) for key in frames)
        elif images.is_file():
            contains_frame = (scene, int(frame)) in [
                (entry.split()[0], int(entry.split()[1]))
                for entry in images.read_text().splitlines() if entry.strip()
            ]
        else:
            continue
        if contains_frame:
            matches.append(room)
    if len(matches) > 1:
        raise ValueError(f"Multiple room meshes match {scene} {int(frame):05d}: {matches}")
    return matches[0] if matches else None


def resolve_scannetpp_surface(scene_root: Path, scene: str, frame: int) -> Path:
    """Select a room export by camera frame, falling back to the full scan."""
    room = resolve_scannetpp_room(scene_root, scene, frame)
    if room is not None:
        mesh = room / "mesh.ply"
        if not mesh.is_file():
            raise FileNotFoundError(f"Missing room mesh for {scene} {frame:05d}: {mesh}")
        return mesh
    scan = scene_root / scene / "scans"
    mesh = scan / "mesh_aligned_0.05.ply"
    return mesh if mesh.is_file() else scan / "mesh_aligned_0.05_surface.npz"


def resolve_sample(
    scene: str,
    frame: int,
    cubemap_root: Path,
    pose_root: Path,
    scene_root: Path,
    dataset_layout: DatasetLayout = "sage10k",
    cubemap_fov: float = 95.0,
) -> SageSample:
    fov_label = str(int(cubemap_fov)) if float(cubemap_fov).is_integer() else str(cubemap_fov)
    cube_suffix = f"cube_{fov_label}"
    if dataset_layout == "3dfront":
        room = Path(scene)
        if len(room.parts) != 2:
            raise ValueError(
                f"Expected a 3D-FRONT scene/room identifier, got {scene!r}"
            )
        room_name = room.name
        prefix = f"{room_name}_panorama_{frame:03d}"
        cubemap_dir = cubemap_root / room / f"{prefix}_{cube_suffix}"
        return SageSample(
            key=f"{room.as_posix()} {frame:05d}",
            scene=room.as_posix(),
            frame=frame,
            cubemap_dir=cubemap_dir,
            pose_path=pose_root / room / "extrinsics.json",
            surface_path=scene_root / room.parent / f"{room_name}_full_surface.npz",
            dataset_layout=dataset_layout,
        )
    if dataset_layout == "scannetpp":
        return SageSample(
            key=f"{scene} {frame:05d}",
            scene=scene,
            frame=frame,
            cubemap_dir=cubemap_root / scene / "panocam" / "images" / f"{frame}_{cube_suffix}",
            pose_path=pose_root / scene / "scans" / "scanner_poses.json",
            surface_path=resolve_scannetpp_surface(scene_root, scene, frame),
            dataset_layout=dataset_layout,
        )
    if dataset_layout != "sage10k":
        raise ValueError(f"Unsupported dataset layout: {dataset_layout!r}")
    prefix = f"{scene}_panorama_{frame:03d}"
    candidates = (
        cubemap_root / scene / f"{prefix}_{cube_suffix}",
        cubemap_root / scene / prefix / f"rgb_{cube_suffix}",
    )
    cubemap_dir = next((path for path in candidates if path.is_dir()), candidates[0])
    surface_candidates = (
        scene_root / scene / "scene_surface.npz",
        scene_root / scene / "scene.ply",
        scene_root / scene / "scene.glb",
        scene_root / f"{scene}.ply",
        scene_root / f"{scene}.glb",
    )
    surface = next((path for path in surface_candidates if path.is_file()), surface_candidates[0])
    return SageSample(
        key=f"{scene} {frame:05d}",
        scene=scene,
        frame=frame,
        cubemap_dir=cubemap_dir,
        pose_path=pose_root / f"{scene}.json",
        surface_path=surface,
    )


def load_cameras(sample: SageSample, face_order: list[int]):
    metadata_path = sample.cubemap_dir / "camera_data.json"
    required = (metadata_path, sample.pose_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SAGE sample inputs:\n  " + "\n  ".join(missing))
    metadata = json.loads(metadata_path.read_text())
    panorama_w2c = _panorama_pose(sample)

    paths, intrinsics, extrinsics = [], [], []
    for face_id in face_order:
        face_key = f"{face_id:04d}"
        face = metadata[face_key]
        image = next(
            (sample.cubemap_dir / f"{face_key}{suffix}" for suffix in IMAGE_SUFFIXES
             if (sample.cubemap_dir / f"{face_key}{suffix}").is_file()),
            None,
        )
        if image is None:
            raise FileNotFoundError(f"Missing cubemap face {face_key} in {sample.cubemap_dir}")
        paths.append(image)
        intrinsic = np.asarray(face["intrinsics"], np.float32).copy()
        with Image.open(image) as opened:
            width, height = opened.size
        if np.max(np.abs(intrinsic[:2])) <= 2:
            intrinsic[0] *= width
            intrinsic[1] *= height
        intrinsics.append(intrinsic)
        extrinsics.append(np.asarray(face["extrinsics"], np.float32) @ panorama_w2c)
    return paths, np.stack(intrinsics), np.stack(extrinsics)


def resolve_erp_path(sample: SageSample, erp_root: Path) -> Path:
    """Find the source equirectangular RGB panorama for a SAGE sample."""
    scene_path = Path(sample.scene)
    basename = scene_path.name if sample.dataset_layout == "3dfront" else sample.scene
    prefix = f"{basename}_panorama_{sample.frame:03d}"
    candidates = (
        erp_root / sample.scene / f"{prefix}.png",
        erp_root / sample.scene / f"{prefix}.jpg",
        erp_root / sample.scene / "panocam" / "images" / f"{sample.frame}.jpg",
        erp_root / sample.scene / "panocam" / "images" / f"{sample.frame}.png",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Missing ERP image; checked:\n  " + "\n  ".join(str(candidate) for candidate in candidates)
        )
    with Image.open(path) as panorama:
        width, height = panorama.size
    if width != 2 * height:
        warnings.warn(
            f"Panorama is not 2:1: {path} has size {width}x{height}. "
            "Spherical lifting assumes a full 360x180-degree ERP; cropped or "
            "unpadded inputs can produce incorrect geometry. Use a prepared 2:1 panorama.",
            UserWarning,
            stacklevel=2,
        )
    return path


def resolve_scannetpp_mask_path(image_path: Path) -> Path:
    """Resolve an ERP or cubemap anonymization mask from a ScanNet++ RGB path."""
    image_path = Path(image_path)
    parts = list(image_path.parts)
    image_dir_index = next(
        (
            index
            for index in range(len(parts) - 1, -1, -1)
            if parts[index] in ("images", "resized_images")
        ),
        None,
    )
    if image_dir_index is None:
        raise ValueError(f"Cannot derive ScanNet++ mask path from {image_path}")
    parts[image_dir_index] = (
        "resized_anon_mask"
        if parts[image_dir_index] == "resized_images"
        else "anon_mask"
    )
    return Path(*parts).with_suffix(".png")


def load_scannetpp_valid_mask(
    sample: SageSample,
    image_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    """Load a ScanNet++ mask at prediction resolution; other layouts are all valid."""
    if sample.dataset_layout != "scannetpp":
        return np.ones((height, width), dtype=bool)
    mask_path = resolve_scannetpp_mask_path(image_path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing ScanNet++ anonymization mask: {mask_path}")
    with Image.open(mask_path) as opened:
        mask = opened.convert("L").resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 0


def _panorama_pose(sample: SageSample) -> np.ndarray:
    if not sample.pose_path.is_file():
        raise FileNotFoundError(f"Missing panorama poses: {sample.pose_path}")
    poses = json.loads(sample.pose_path.read_text())
    if sample.dataset_layout == "3dfront":
        from nova3r.utils.three_d_front import world_to_opencv_camera

        panorama_name = f"{Path(sample.scene).name}_panorama_{sample.frame:03d}.png"
        if panorama_name not in poses:
            raise KeyError(f"Pose {panorama_name!r} is absent from {sample.pose_path}")
        pose = np.asarray(poses[panorama_name], dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 3D-FRONT panorama pose for {panorama_name}, "
                f"got {pose.shape}"
            )
        # The renderer stores Blender-world/Blender-camera poses, while the
        # evaluation surfaces retain their source glTF Y-up coordinates.
        return world_to_opencv_camera(pose)
    if sample.dataset_layout == "scannetpp":
        if not isinstance(poses, list):
            raise ValueError(f"Expected a pose list in {sample.pose_path}")
        if sample.frame >= len(poses):
            raise IndexError(
                f"Pose index {sample.frame} is out of range for {sample.pose_path} "
                f"with {len(poses)} poses"
            )
        scanner_c2w = np.asarray(poses[sample.frame], dtype=np.float32)
        if scanner_c2w.shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 scanner pose at index {sample.frame}, got {scanner_c2w.shape}"
            )
        return SCANNETPP_SCANNER_TO_OPENCV @ np.linalg.inv(scanner_c2w)

    panorama_name = f"{sample.scene}_panorama_{sample.frame:03d}.png"
    if panorama_name not in poses:
        raise KeyError(f"Pose {panorama_name!r} is absent from {sample.pose_path}")
    pose = np.asarray(poses[panorama_name], dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 panorama pose for {panorama_name}, got {pose.shape}")
    return PANORAMA_TO_OPENCV @ pose


def load_erp_input(sample: SageSample, erp_root: Path):
    """Load an ERP image and its pose without inventing pinhole intrinsics."""
    erp_path = resolve_erp_path(sample, erp_root)
    return erp_path, _panorama_pose(sample)


def load_erp_cameras(
    sample: SageSample,
    erp_root: Path,
    horizontal_fov: float,
):
    """Load one ERP and intentionally model it as a single pinhole camera."""
    if not 0 < horizontal_fov < 180:
        raise ValueError("ERP pinhole horizontal FOV must be between 0 and 180 degrees")
    erp_path = resolve_erp_path(sample, erp_root)
    with Image.open(erp_path) as opened:
        width, height = opened.size
    focal = 0.5 * width / np.tan(np.deg2rad(horizontal_fov) / 2)
    intrinsic = np.asarray(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    return [erp_path], intrinsic[None], _panorama_pose(sample)[None], erp_path


def pinhole_z_to_range(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Convert one pinhole camera-plane Z-depth map to Euclidean ray distance."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError(f"Expected one ERP depth map, got {depth.shape}")
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected ERP depth shaped [H, W] or [1, H, W], got {depth.shape}")

    height, width = depth.shape
    intrinsic = np.asarray(intrinsics, dtype=np.float32)
    while intrinsic.ndim > 2 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected one 3x3 intrinsic matrix, got {intrinsic.shape}")

    # DA3 predicts distance along the pinhole optical axis. Convert it to the
    # Euclidean distance along each predicted pinhole ray before assigning that
    # distance to the corresponding spherical ERP ray.
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    homogeneous_pixels = np.stack(
        (pixel_x, pixel_y, np.ones_like(pixel_x)), axis=-1
    )
    pinhole_rays = homogeneous_pixels @ np.linalg.inv(intrinsic).T
    return depth * np.linalg.norm(pinhole_rays, axis=-1)


def cubemap_z_to_range(depth: np.ndarray) -> np.ndarray:
    """Convert nearest-face cubemap Z-depth on an ERP grid to radial distance."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError(f"Expected one ERP depth map, got {depth.shape}")
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected ERP depth shaped [H, W] or [1, H, W], got {depth.shape}")

    height, width = depth.shape
    longitude = (
        (np.arange(width, dtype=np.float32) + 0.5) / width - 0.5
    ) * (2 * np.pi)
    latitude = (
        (np.arange(height, dtype=np.float32) + 0.5) / height - 0.5
    ) * np.pi
    longitude, latitude = np.meshgrid(longitude, latitude)
    cos_latitude = np.cos(latitude)
    rays = np.stack(
        (
            np.sin(longitude) * cos_latitude,
            np.sin(latitude),
            np.cos(longitude) * cos_latitude,
        ),
        axis=-1,
    )
    face_forward_component = np.max(np.abs(rays), axis=-1)
    return depth / face_forward_component


def unproject_erp_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    """Convert pinhole z-depth to range, then lift it through spherical ERP rays."""
    return unproject_erp_range(
        pinhole_z_to_range(depth, intrinsics), world_to_camera
    )


def unproject_erp_range(
    radial_depth: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    """Lift an ERP radial-distance map through spherical rays into world space."""
    radial_depth = np.asarray(radial_depth, dtype=np.float32)
    if radial_depth.ndim == 3:
        if radial_depth.shape[0] != 1:
            raise ValueError(f"Expected one ERP range map, got {radial_depth.shape}")
        radial_depth = radial_depth[0]
    if radial_depth.ndim != 2:
        raise ValueError(
            f"Expected ERP range shaped [H, W] or [1, H, W], got {radial_depth.shape}"
        )

    height, width = radial_depth.shape
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    longitude = (u - 0.5) * (2 * np.pi)
    latitude = (v - 0.5) * np.pi
    longitude, latitude = np.meshgrid(longitude, latitude)
    cos_latitude = np.cos(latitude)
    rays = np.stack(
        (
            np.sin(longitude) * cos_latitude,
            np.sin(latitude),
            np.cos(longitude) * cos_latitude,
        ),
        axis=-1,
    )
    points_camera = rays * radial_depth[..., None]

    w2c = np.asarray(world_to_camera, dtype=np.float32)
    if w2c.shape == (3, 4):
        w2c = np.vstack((w2c, np.asarray([0, 0, 0, 1], dtype=np.float32)))
    if w2c.shape != (4, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 camera extrinsic, got {w2c.shape}")
    c2w = np.linalg.inv(w2c)
    flat = points_camera.reshape(-1, 3)
    range_flat = radial_depth.reshape(-1)
    valid = np.isfinite(flat).all(axis=1) & np.isfinite(range_flat) & (range_flat > 0)
    flat = flat[valid]
    return flat @ c2w[:3, :3].T + c2w[:3, 3]


def load_input_cameras(
    sample: SageSample,
    projection: InputProjection,
    face_order: list[int],
    *,
    erp_root: Path | None = None,
    erp_fov: float = 95.0,
):
    """Load cube faces or one ERP intentionally interpreted as a pinhole image."""
    if projection == "cubemap":
        images, intrinsics, extrinsics = load_cameras(sample, face_order)
        return images, intrinsics, extrinsics, None
    if projection == "erp":
        if erp_root is None:
            raise ValueError("erp_root is required when input projection is 'erp'")
        return load_erp_cameras(
            sample,
            erp_root,
            horizontal_fov=erp_fov,
        )
    raise ValueError(f"Unsupported input projection: {projection!r}")


def load_scene_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene point cloud: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            arrays = [data[name][:, :3] for name in ("random_surface", "sharp_surface") if name in data]
        if not arrays:
            raise KeyError(f"No random_surface or sharp_surface arrays in {path}")
        points = np.concatenate(arrays)
    else:
        import open3d as o3d

        geometry = o3d.io.read_triangle_mesh(str(path)) if path.suffix.lower() == ".glb" else o3d.io.read_point_cloud(str(path))
        if path.suffix.lower() == ".glb":
            points = np.asarray(geometry.sample_points_uniformly(1_000_000).points)
        else:
            points = np.asarray(geometry.points)
    points = np.asarray(points, dtype=np.float32)
    return points[np.isfinite(points).all(axis=1)]


def sample_points(points: np.ndarray, count: int | None, seed: int) -> np.ndarray:
    if count is None or len(points) <= count:
        return points
    return points[np.random.default_rng(seed).choice(len(points), count, replace=False)]


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Area-weighted, deterministic sampling of points on a triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces)
    if count < 1:
        raise ValueError("Surface sample count must be positive")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shaped [N, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Expected a non-empty triangle array shaped [M, 3], got {faces.shape}")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"Expected integer mesh face indices, got {faces.dtype}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh faces contain out-of-range vertex indices")

    # Keep temporary expanded triangle arrays bounded for multi-million-face meshes.
    areas = np.zeros(len(faces), dtype=np.float32)
    for start in range(0, len(faces), 250_000):
        stop = min(start + 250_000, len(faces))
        triangles = vertices[faces[start:stop]]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        chunk_areas = np.linalg.norm(cross, axis=1) * 0.5
        valid = (
            np.isfinite(triangles).all(axis=(1, 2))
            & np.isfinite(chunk_areas)
            & (chunk_areas > 0)
        )
        areas[start:stop] = np.where(valid, chunk_areas, 0)
    total_area = float(areas.sum(dtype=np.float64))
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Generated mesh has no finite, non-degenerate triangles")

    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(areas, dtype=np.float64)
    selected_faces = np.searchsorted(
        cumulative,
        rng.random(count) * total_area,
        side="right",
    )
    selected = vertices[faces[selected_faces]]
    sqrt_u = np.sqrt(rng.random((count, 1), dtype=np.float32))
    v = rng.random((count, 1), dtype=np.float32)
    return (
        (1 - sqrt_u) * selected[:, 0]
        + sqrt_u * (1 - v) * selected[:, 1]
        + sqrt_u * v * selected[:, 2]
    ).astype(np.float32, copy=False)


def align_prediction_to_target(
    prediction: np.ndarray,
    target: np.ndarray,
    device: str,
    seed: int,
    max_iterations: int = 300,
    lr: float = 0.02,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Align a prediction to GT with the scale/translation fit used by evaluate_pcd."""
    import torch

    from eval.mv_recon.metric import scale_shift_alignment_chamfer

    if not len(prediction) or not len(target):
        raise ValueError("Cannot align empty point clouds")

    torch_device = torch.device(device)
    prediction_tensor = torch.as_tensor(prediction, dtype=torch.float32, device=torch_device).unsqueeze(0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=torch_device).unsqueeze(0)
    cuda_devices = []
    if torch_device.type == "cuda":
        cuda_devices = [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]

    # The Chamfer optimizer samples point subsets on every iteration. Forking the
    # RNG makes the fit repeatable without perturbing inference randomness.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        aligned, shift, scale = scale_shift_alignment_chamfer(
            prediction_tensor,
            target_tensor,
            max_iterations=max_iterations,
            lr=lr,
            return_transform=True,
        )

    return (
        aligned[0].detach().cpu().numpy(),
        float(scale.reshape(-1)[0].item()),
        shift.reshape(-1, 3)[0].detach().cpu().numpy(),
    )


def compute_ssi3d_fscores(
    prediction: np.ndarray,
    target: np.ndarray,
    thresholds: list[float],
    device: str,
) -> dict[str, float]:
    """Compute F-scores through the same SSI3DScore implementation as evaluate_pcd."""
    import torch

    from eval.mv_recon.metric import SSI3DScore

    prediction_tensor = torch.as_tensor(prediction, dtype=torch.float32, device=device).unsqueeze(0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    scorer = SSI3DScore(
        num_eval_pts=len(prediction),
        fs_thres=thresholds,
        pts_sampling_mode="uniform",
    )
    with torch.inference_mode():
        details = scorer.chamfer_and_fscore(prediction_tensor, target_tensor, eval_layers=None)
    return {name: value for name, (value, _) in details.items() if name.startswith("f_score_")}


def evaluate_point_clouds(
    prediction: np.ndarray,
    target: np.ndarray,
    thresholds: list[float],
    device: str,
    seed: int,
    alignment_iterations: int = DEFAULT_ALIGNMENT_ITERATIONS,
    alignment_lr: float = DEFAULT_ALIGNMENT_LR,
    prealigned: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run the shared baseline alignment and point-cloud metric protocol."""
    from eval.mv_recon.metric import accuracy, completion

    if prealigned:
        aligned, alignment_scale, alignment_shift = prediction, 1.0, np.zeros(3)
    else:
        aligned, alignment_scale, alignment_shift = align_prediction_to_target(
            prediction,
            target,
            device=device,
            seed=seed,
            max_iterations=alignment_iterations,
            lr=alignment_lr,
        )
    chamfer_forward, accuracy_median = accuracy(target, aligned)
    chamfer_backward, completion_median = completion(target, aligned)
    fscores = compute_ssi3d_fscores(aligned, target, thresholds, device)
    details = {
        "pred_points": len(aligned),
        "target_points": len(target),
        "alignment_scale": alignment_scale,
        "alignment_shift": alignment_shift.tolist(),
        "chamfer_forward": float(chamfer_forward),
        "chamfer_backward": float(chamfer_backward),
        "chamfer_mean": float((chamfer_forward + chamfer_backward) / 2),
        "accuracy_median": float(accuracy_median),
        "completion_median": float(completion_median),
        **fscores,
    }
    return aligned, details


def resolve_visibility_mesh(sample: SageSample, mesh_root: Path | None = None) -> Path:
    """Resolve original triangles, never silently fall back to point visibility."""
    surface = sample.surface_path
    if sample.dataset_layout == "scannetpp":
        # resolve_sample already selects the room belonging to this frame.
        if surface.name == "mesh.ply" and surface.is_file():
            return surface
        if mesh_root is not None:
            room = resolve_scannetpp_room(Path(mesh_root), sample.scene, sample.frame)
            if room is not None:
                return room / "mesh.ply"
    if mesh_root is not None:
        root = Path(mesh_root)
        if sample.dataset_layout == "scannetpp":
            candidates = [root / sample.scene / "scans" / "mesh_aligned_0.05.ply"]
        elif sample.dataset_layout == "3dfront":
            room = root / sample.scene
            candidates = [room / "room.glb", room.parent / f"{room.name}_full.glb"]
        else:
            candidates = [root / sample.scene / name for name in ("scene.glb", "scene.ply")]
    else:
        candidates = [surface] if surface.suffix.lower() in (".ply", ".glb", ".obj") else []
        if sample.dataset_layout == "scannetpp":
            candidates += [surface.with_name("mesh_aligned_0.05.ply")]
        elif sample.dataset_layout == "3dfront":
            candidates += [surface.parent / Path(sample.scene).name / "room.glb",
                           surface.with_name(surface.name.replace("_surface.npz", ".glb"))]
        else:
            candidates += [surface.parent / "scene.glb", surface.parent / "scene.ply"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No visibility mesh for {sample.key}; supply --visibility-mesh-root. Tried: {candidates}")


def load_visibility_geometry(sample, mesh_root=None, mask_root=None):
    """Return original occluders and non-anonymized triangles for GT sampling."""
    import trimesh

    path = resolve_visibility_mesh(sample, mesh_root)
    # Visibility and point sampling consume geometry only. Loading GLB textures
    # also makes scene concatenation build an unused texture atlas, which can
    # dominate evaluation time on textured SAGE/3D-FRONT scenes.
    mesh = trimesh.load(str(path), process=False, skip_materials=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError(f"Expected finite triangle geometry: {path}")
    evaluation_faces = faces
    room_mask = path.with_name("mesh_mask.txt")
    if sample.dataset_layout == "scannetpp" and path.name == "mesh.ply" and room_mask.is_file() and mask_root is None:
        indices = (np.loadtxt(room_mask, dtype=np.int64, ndmin=1).reshape(-1)
                   if room_mask.stat().st_size else np.empty(0, dtype=np.int64))
        if np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError(f"Out-of-range room vertex indices: {room_mask}")
        excluded = np.zeros(len(vertices), dtype=bool)
        excluded[indices] = True
        evaluation_faces = faces[~excluded[faces].any(axis=1)]
        if not len(evaluation_faces):
            raise ValueError(f"Anonymization excludes every triangle: {path}")
    elif sample.dataset_layout == "scannetpp":
        # Indices must refer to the original scan, not a reindexed room crop.
        mask = (Path(mask_root) / sample.scene / "scans" / "mesh_aligned_0.05_mask.txt"
                if mask_root else path.with_name("mesh_aligned_0.05_mask.txt"))
        original_vertices = vertices
        if path.name == "mesh.ply":
            # Room export removes unused vertices but preserves coordinates.
            scene_root = path.parents[3] if path.parents[2].name == "rooms" else None
            roots = [Path(root) for root in (mask_root, mesh_root, scene_root) if root is not None]
            original_path = next((root / sample.scene / "scans" / "mesh_aligned_0.05.ply"
                                  for root in roots
                                  if (root / sample.scene / "scans" / "mesh_aligned_0.05.ply").is_file()), None)
            if original_path is None:
                raise FileNotFoundError("Room anonymization requires the original scan; supply --visibility-mesh-root")
            original_vertices = np.asarray(trimesh.load(str(original_path), process=False).vertices, dtype=np.float32)
            mask = (Path(mask_root) / sample.scene / "scans" / "mesh_aligned_0.05_mask.txt"
                    if mask_root else original_path.with_name("mesh_aligned_0.05_mask.txt"))
        elif path.name != "mesh_aligned_0.05.ply":
            raise ValueError(f"Unsupported ScanNet++ mesh: {path}")
        if not mask.is_file():
            raise FileNotFoundError(f"Missing ScanNet++ vertex anonymization list: {mask}")
        indices = np.loadtxt(mask, dtype=np.int64, ndmin=1).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= len(original_vertices)):
            raise ValueError(f"Out-of-range anonymized vertex indices: {mask}")
        original_excluded = np.zeros(len(original_vertices), dtype=bool)
        original_excluded[indices] = True
        if path.name == "mesh.ply":
            from scipy.spatial import cKDTree

            # Match only masked positions: duplicate scan vertices with different
            # indices must still be treated as anonymized at that position.
            correspondence = cKDTree(original_vertices).query(vertices)[0]
            if np.any(correspondence > 1e-5):
                raise ValueError(f"Room vertices do not match original scan coordinates: {path}")
            excluded = (cKDTree(original_vertices[original_excluded]).query(vertices)[0] <= 1e-5
                        if len(indices) else np.zeros(len(vertices), dtype=bool))
        else:
            excluded = original_excluded
        evaluation_faces = faces[~excluded[faces].any(axis=1)]
        if not len(evaluation_faces):
            raise ValueError(f"Anonymization excludes every triangle: {path}")
    return vertices, faces, evaluation_faces, path


def panorama_visibility_mask(target, camera_center, vertices, faces, tolerance=0.01):
    """Visible iff the GT point agrees with the first mesh hit (scene units)."""
    import open3d as o3d

    offsets = np.asarray(target, dtype=np.float32) - camera_center
    distances = np.linalg.norm(offsets, axis=1)
    if np.any(distances <= 0) or tolerance <= 0:
        raise ValueError("Require positive camera distances and visibility tolerance")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
    visible = np.zeros(len(target), dtype=bool)
    for start in range(0, len(target), 100_000):
        stop = min(start + 100_000, len(target))
        directions = offsets[start:stop] / distances[start:stop, None]
        origins = np.broadcast_to(camera_center, directions.shape)
        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        visible[start:stop] = np.isfinite(hit) & (np.abs(hit - distances[start:stop]) <= tolerance)
    return visible


def align_lari_pointmap(pointmap, vertices, faces, world_to_camera, intrinsics, valid_pixels):
    """Fit camera-space scale and Z shift from first-layer pixel correspondences.

    Uses the least-squares transform family in LaRI's native evaluator. The
    first GT layer comes from mesh raycasting; all predicted layers share the fit.
    """
    import open3d as o3d

    height, width, layers, _ = pointmap.shape
    image_h, image_w = valid_pixels.shape
    intrinsic = np.asarray(intrinsics, dtype=np.float64).copy()
    intrinsic[0] *= width / image_w
    intrinsic[1] *= height / image_h
    xx, yy = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    directions = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(intrinsic).T
    c2w = np.linalg.inv(world_to_camera)
    world_directions = directions @ c2w[:3, :3].T
    origins = np.broadcast_to(c2w[:3, 3], world_directions.shape)
    rays = np.concatenate([origins, world_directions], axis=-1).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
    depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    px = np.minimum((xx * image_w / width).astype(int), image_w - 1)
    py = np.minimum((yy * image_h / height).astype(int), image_h - 1)
    mask = (np.isfinite(depth) & (depth > 0) & valid_pixels[py, px]
            & np.isfinite(pointmap[:, :, 0]).all(axis=-1))
    if mask.sum() < 3:
        raise ValueError("LaRI alignment requires at least 3 valid first-hit correspondences")
    pred = pointmap[:, :, 0][mask].astype(np.float64)
    target = directions[mask] * depth[mask, None]
    pred_z_mean, target_z_mean = pred[:, 2].mean(), target[:, 2].mean()
    denominator = np.mean(np.sum(pred * pred, axis=1)) - pred_z_mean ** 2
    if denominator <= 1e-12:
        raise ValueError("Degenerate LaRI pointmap scale fit")
    scale = (np.mean(np.sum(pred * target, axis=1)) - pred_z_mean * target_z_mean) / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid LaRI alignment scale: {scale}")
    z_shift = target_z_mean - scale * pred_z_mean
    aligned_camera = pointmap.reshape(-1, 3).astype(np.float64) * scale
    aligned_camera[:, 2] += z_shift
    aligned_world = aligned_camera @ c2w[:3, :3].T + c2w[:3, 3]
    aligned_world = aligned_world[np.isfinite(aligned_world).all(axis=1)]
    # Equivalent transform applied to already world-space raw predictions.
    shift_world = (1 - scale) * c2w[:3, 3] + z_shift * c2w[:3, 2]
    return aligned_world.astype(np.float32), float(scale), shift_world, {
        "camera_z_shift": float(z_shift), "correspondence_points": int(mask.sum()),
        "alignment_method": "lari_first_layer_scale_camera_z_shift",
    }


def face_target_mask(target, world_to_camera, intrinsics, valid_pixels):
    """Select all GT inside a calibrated face, including occluded surfaces."""
    camera = target @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    projected = camera @ intrinsics.T
    pixels = projected[:, :2] / np.where(camera[:, 2:3] > 0, projected[:, 2:3], 1)
    height, width = valid_pixels.shape
    inside = ((camera[:, 2] > 0)
              & np.isfinite(pixels).all(axis=1)
              & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    indices = np.flatnonzero(inside)
    uv = np.floor(pixels[indices]).astype(np.int64)
    inside[indices] &= valid_pixels[uv[:, 1], uv[:, 0]]
    return inside


def compute_visibility_metrics(prediction, target, visible, thresholds):
    """Regional CD/APD use GT-to-prediction distances; complete CD is symmetric.

    APD is thresholded GT coverage, expressed as a fraction. No regional refit
    is performed. Empty GT regions are reported as null, never as perfect scores.
    """
    from scipy.spatial import cKDTree

    visible = np.asarray(visible, dtype=bool)
    if visible.shape != (len(target),) or not len(target) or not len(prediction):
        raise ValueError("Require nonempty clouds and one visibility flag per GT point")
    backward = cKDTree(prediction).query(target)[0]
    forward = cKDTree(target).query(prediction)[0]
    metrics = {"complete_cd": float((forward.mean() + backward.mean()) / 2)}
    for name, mask in (("visible", visible), ("occluded", ~visible)):
        distances = backward[mask]
        metrics[f"{name}_points"] = int(mask.sum())
        metrics[f"{name}_cd"] = float(distances.mean()) if len(distances) else None
        # Each region is a GT subset evaluated against the entire prediction.
        regional_forward = cKDTree(target[mask]).query(prediction)[0] if mask.any() else None
        metrics[f"{name}_chamfer_forward"] = float(regional_forward.mean()) if mask.any() else None
        metrics[f"{name}_chamfer_backward"] = metrics[f"{name}_cd"]
        metrics[f"{name}_chamfer_mean"] = (
            float((regional_forward.mean() + distances.mean()) / 2) if mask.any() else None
        )
        for threshold in thresholds:
            metrics[f"{name}_apd_{threshold}"] = (
                float(np.mean(distances < threshold)) if len(distances) else None
            )
    return metrics


def build_evaluation_summary(
    records: list[dict],
    *,
    model: str,
    thresholds: list[float],
    alignment_iterations: int,
    alignment_lr: float,
    metadata: dict | None = None,
) -> dict:
    """Aggregate successful records using the baseline summary schema."""
    successful = [record for record in records if record["status"] == "ok"]
    summary = {
        "model": model,
        **(metadata or {}),
        "samples": len(records),
        "successful": len(successful),
        "fscore_thresholds": thresholds,
        "alignment": {
            "method": "chamfer_scale_translation",
            "iterations": alignment_iterations,
            "learning_rate": alignment_lr,
        },
    }
    score_names = [f"f_score_{threshold}_full" for threshold in thresholds]
    for metric in ("chamfer_forward", "chamfer_backward", "chamfer_mean", *score_names):
        summary[metric] = (
            sum(record[metric] for record in successful) / len(successful)
            if successful
            else None
        )
    regional_names = sorted({key for record in successful for key in record
                             if key.startswith(("visible_cd", "occluded_cd", "visible_apd_", "occluded_apd_", "complete_cd", "visible_chamfer_", "occluded_chamfer_"))})
    for name in regional_names:
        values = [record[name] for record in successful if record.get(name) is not None]
        summary[name] = float(np.mean(values)) if values else None
    if regional_names:
        summary["metric_samples"] = {
            region: sum(record.get(f"{region}_cd") is not None for record in successful)
            for region in ("visible", "occluded", "complete")
        }
    exceptions = [
        {"sample": record["sample"], "evaluated_faces": record.get("evaluated_faces"),
         "skipped_faces": record["skipped_faces"]}
        for record in successful if record.get("skipped_faces")
    ]
    if exceptions:
        summary["face_exceptions"] = exceptions
    return summary


def write_evaluation_results(records: list[dict], summary: dict, output_dir: Path) -> None:
    """Write the baseline result artifacts shared by standalone and Lightning evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if "complete_cd" in summary:
        columns = ["Method", "Visible CD↓", "Visible APD0.1↑", "Visible APD0.05↑", "Visible APD0.02↑",
                   "Occluded CD↓", "Occluded APD0.1↑", "Occluded APD0.05↑", "Occluded APD0.02↑",
                   "Complete CD↓", "Complete FS0.1↑", "Complete FS0.05↑", "Complete FS0.02↑"]
        keys = ["visible_cd", "visible_apd_0.1", "visible_apd_0.05", "visible_apd_0.02",
                "occluded_cd", "occluded_apd_0.1", "occluded_apd_0.05", "occluded_apd_0.02",
                "complete_cd", "f_score_0.1_full", "f_score_0.05_full", "f_score_0.02_full"]
        # Table distances use cm (assuming scene units are meters); rates use %.
        values = [summary.get(key) for key in keys]
        row = [summary["model"]] + [
            f"{value * 100:.2f}" if value is not None else "—" for value in values
        ]
        (output_dir / "table.md").write_text(
            "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns))
            + " |\n| " + " | ".join(row) + " |\n"
        )
        with (output_dir / "table.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerow(row)

    fields = sorted({key for record in records for key in record})
    with (output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Unproject z-depth through OpenCV intrinsics and world-to-camera poses."""
    all_points = []
    for z, intrinsic, w2c in zip(depth, intrinsics, extrinsics):
        if w2c.shape == (3, 4):
            w2c = np.concatenate((w2c, np.asarray([[0, 0, 0, 1]], dtype=w2c.dtype)))
        height, width = z.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        camera = np.stack(((x - intrinsic[0, 2]) * z / intrinsic[0, 0],
                           (y - intrinsic[1, 2]) * z / intrinsic[1, 1], z), axis=-1)
        valid = np.isfinite(camera).all(axis=-1) & (z > 0)
        camera_h = np.concatenate((camera[valid], np.ones((valid.sum(), 1))), axis=1)
        all_points.append((np.linalg.inv(w2c) @ camera_h.T).T[:, :3])
    return np.concatenate(all_points).astype(np.float32)
