#!/usr/bin/env python3
"""Encode scene surfaces into camera-specific VAE latent files."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import os

from nova3r.utils.three_d_front import (
    BLENDER_CAMERA_TO_OPENCV,
    GLTF_TO_BLENDER,
    parse_panorama_sample,
    world_to_opencv_camera,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_CHECKPOINT = REPO_ROOT / "exp_output/hunyuan_vae_v2_1_udf/checkpoints/last.ckpt"
DEFAULT_CONFIG = REPO_ROOT / "checkpoints/hunyuan/.hydra/config.yaml"
DEFAULT_SPLIT_LIST = REPO_ROOT / "data/sage-10k/surface_vae_latent_list.json"
PANORAMA_TO_OPENCV = BLENDER_CAMERA_TO_OPENCV
SCANNETPP_AXIS_CONVERSION = np.asarray(
    [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
GLTF_TO_BLENDER_OPENCV = PANORAMA_TO_OPENCV @ GLTF_TO_BLENDER

# Panorama split lists contain several camera views of the same room. Keep a
# small process-local cache so those views do not repeatedly read and
# decompress the same NPZ through FUSE. Multi-GPU workers each get their own
# cache. Override the size when memory is constrained.
SURFACE_CACHE_SIZE = max(1, int(os.environ.get("NOVA3R_SURFACE_CACHE_SIZE", "8")))
_SURFACE_CACHE = OrderedDict()
_SURFACE_LOADS = {}
_SURFACE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    scene: str
    frame: str
    output_path: Path


@dataclass
class PreparedSurface:
    job: Job
    surface: torch.Tensor
    source_path: Path
    cubemap_path: Path | None
    input_transform: np.ndarray
    coordinate_frame: str
    bounds: torch.Tensor
    normalization_scale: float
    room_center: torch.Tensor
    cube_bbox_center: torch.Tensor
    normalized_camera_translation: torch.Tensor
    seed: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Load scene surfaces for every listed camera, apply the "
                     "requested coordinate transform, and save a VAE latent.")
    )
    parser.add_argument(
        "--dataset-layout",
        choices=("sage10k", "scannetpp", "3dfront"),
        default="sage10k",
        help="Input path and pose convention (default: sage10k).",
    )
    parser.add_argument("--split-list", type=Path, default=DEFAULT_SPLIT_LIST)
    parser.add_argument(
        "--camera-poses",
        "--camera-pose-ids",
        dest="camera_poses",
        type=int,
        nargs="+",
        help=(
            "Panorama/camera-pose indices to encode for every scene in the "
            "split list, for example --camera-poses 0 1 2. By default, use "
            "the pose paired with each scene in the split list."
        ),
    )
    parser.add_argument(
        "--surface-root",
        type=Path,
        default=Path("/tmp/datasets/sage-10k/udf_colored"),
    )
    parser.add_argument(
        "--cubemap-root",
        type=Path,
        default=Path("/tmp/datasets/sage-10k/cubemap_data/cubemaps"),
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("/mnt/projects/theses/vyhong/sage-10k/poses"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="ShapeVAE Hydra config or Lightning hparams.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/sage10k_surface_vae_latents"),
    )
    parser.add_argument(
        "--face-id",
        type=int,
        default=4,
        choices=range(6),
        help="Cubemap face used as the camera frame (default: 4).",
    )
    parser.add_argument(
        "--no-camera-extrinsics",
        action="store_true",
        help=("Keep the latent camera-independent: only flip Y and Z, without " "loading or applying panorama poses or cubemap-face extrinsics."),
    )
    parser.add_argument(
        "--panorama-camera",
        action="store_true",
        help=(
            "For 3D-FRONT, transform geometry directly into the panorama "
            "OpenCV camera frame using extrinsics.json; no cubemap metadata "
            "is required. This matches PaGeR's output frame."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help=("Number of threads used to load and prepare surfaces (default: 8; " "use 0 for sequential loading)."),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Single device used when --devices is omitted (default: cuda).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Model and compute precision on CUDA (default: fp32).",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help=(
            "Run one encoding process per device, for example "
            "--devices cuda:0 cuda:1. Jobs are divided deterministically "
            "between the processes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-posterior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Draw a posterior sample like VAE training (default: enabled). " "Use --no-sample-posterior to save the posterior mean."),
    )
    parser.add_argument(
        "--normalization",
        default="auto",
        help=("auto, none, median, median_<target>, cube, or unit_box; unit_box fits " "XYZ inside [-1, 1]^3 with one percent padding."),
    )
    parser.add_argument(
        "--center-room",
        action="store_true",
        help=("Translate the sampled room bounding-box center to (0, 0, 0) " "before applying normalization."),
    )
    parser.add_argument(
        "--cube-center-camera",
        action="store_true",
        help=(
            "With --normalization cube, first normalize about the sampled "
            "room bounding-box midpoint, then translate the normalized camera "
            "back to (0, 0, 0). This matches cube normalization with "
            "surface_geo_use_camera_translation enabled."
        ),
    )
    parser.add_argument(
        "--save-latent-before-sampling",
        action="store_true",
        help=("Also save the posterior mean and log-variance needed to draw latent " "samples as <frame>_before_sampling.pt."),
    )
    parser.add_argument(
        "--only-save-latent-before-sampling",
        action="store_true",
        help=("Save only the posterior mean and log-variance as " "<frame>_before_sampling.pt; do not draw or save a latent sample."),
    )
    parser.add_argument("--scene", help="Only process one scene ID.")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--save-normalized-surface",
        action="store_true",
        help=("Save the normalized surface passed to the VAE as " "<frame>_normalized_surface.ply beside each latent."),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.camera_poses is not None and any(pose < 0 for pose in args.camera_poses):
        parser.error("--camera-poses values must be non-negative")
    if args.devices is not None and len(set(args.devices)) != len(args.devices):
        parser.error("--devices must not contain duplicate devices")
    if args.cube_center_camera and args.normalization not in ("cube", "auto"):
        parser.error("--cube-center-camera requires --normalization cube (or auto resolving to cube)")
    if args.cube_center_camera and args.center_room:
        parser.error("--cube-center-camera and --center-room are mutually exclusive")
    if args.cube_center_camera and args.no_camera_extrinsics:
        parser.error("--cube-center-camera requires camera extrinsics")
    if args.panorama_camera and args.no_camera_extrinsics:
        parser.error("--panorama-camera and --no-camera-extrinsics are mutually exclusive")
    if args.panorama_camera and args.dataset_layout != "3dfront":
        parser.error("--panorama-camera is currently supported only for 3dfront")
    return args


def load_vae_config(path: Path):
    """Read either a normal Hydra config or TensorBoard hparams.yaml."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    root = config.get("experiment", config)
    root = root.get("cfg", root)
    model = root.get("model")
    if model is None or model.get("name") != "ShapeVAE":
        raise ValueError(f"No top-level ShapeVAE model configuration in {path}")
    params = OmegaConf.to_container(model.get("params"), resolve=True)
    return params, str(root.get("surface_geo_normalize_mode", "none"))


def precision_dtype(precision: str):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def load_vae(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE
    from training.mesh2sdf.shape_vae import (
        checkpoint_state,
        select_vae_state,
    )

    params, training_normalization = load_vae_config(config_path)
    model = ShapeVAE(**params)
    selected = select_vae_state(checkpoint_state(checkpoint_path), model)
    missing, unexpected = model.load_state_dict(selected, strict=False)
    required_missing = [key for key in missing if key.startswith(("encoder.", "pre_kl."))]
    if unexpected or required_missing:
        raise RuntimeError(f"Incompatible VAE checkpoint: {len(required_missing)} required missing, " f"{len(unexpected)} unexpected")
    return model.to(device=device, dtype=dtype).eval(), training_normalization


def load_jobs(
    split_list: Path,
    output_root: Path,
    scene_filter=None,
    limit=None,
    camera_poses=None,
    dataset_layout="sage10k",
):
    samples = json.loads(split_list.read_text())
    jobs = []
    seen = set()
    for sample in samples:
        if dataset_layout == "3dfront":
            parsed = parse_panorama_sample(sample)
            scene, source_frame = parsed.room, parsed.frame
        else:
            scene, frame = sample.split()
            source_frame = int(frame)
        if scene_filter is not None and scene != scene_filter:
            continue
        frames = (source_frame,) if camera_poses is None else camera_poses
        for frame_index in frames:
            frame = f"{int(frame_index):05d}"
            key = (scene, frame)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(Job(scene, frame, output_root / scene / f"{frame}.pt"))
    return jobs[:limit] if limit is not None else jobs


def sample_seed(
    base_seed: int,
    scene: str,
    frame: str | None = None,
    face_id: int | None = None,
):
    parts = [str(base_seed), scene]
    if frame is not None:
        parts.append(frame)
    if face_id is not None:
        parts.append(str(face_id))
    key = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def load_world_to_camera(
    cubemap_root: Path,
    pose_root: Path,
    scene: str,
    frame: str,
    face_id: int,
    dataset_layout: str = "sage10k",
    panorama_camera: bool = False,
):
    frame_index = int(frame)
    if dataset_layout == "3dfront":
        room = Path(scene)
        room_name = room.name
        panorama_name = f"{room_name}_panorama_{frame_index:03d}.png"
        panorama_poses = json.loads(
            (pose_root / room / "extrinsics.json").read_text()
        )
        panorama_w2c = np.asarray(panorama_poses[panorama_name], dtype=np.float32)
        if panorama_w2c.shape != (4, 4):
            raise ValueError(
                f"Expected 4x4 3D-FRONT panorama pose for {panorama_name}, "
                f"got {panorama_w2c.shape}"
            )
        # Stored panorama poses use Blender camera axes and Blender's Z-up
        # world. UDF samples come directly from the Y-up glTF.
        if panorama_camera:
            return world_to_opencv_camera(panorama_w2c), None
        cube_dir = (
            cubemap_root / room / f"{room_name}_panorama_{frame_index:03d}_cube_95"
        )
        camera_data = json.loads((cube_dir / "camera_data.json").read_text())
        face_w2c = np.asarray(
            camera_data[f"{face_id:04d}"]["extrinsics"], dtype=np.float32
        )
        return world_to_opencv_camera(panorama_w2c, face_w2c), cube_dir
    if dataset_layout == "sage10k":
        cube_dir = cubemap_root / scene / f"{scene}_panorama_{frame_index:03d}_cube_95"
        camera_data = json.loads((cube_dir / "camera_data.json").read_text())
        poses = json.loads((pose_root / f"{scene}.json").read_text())
        panorama_name = f"{scene}_panorama_{frame_index:03d}.png"
        panorama_w2c = PANORAMA_TO_OPENCV @ np.asarray(
            poses[panorama_name], dtype=np.float32
        )
        face_w2c = np.asarray(
            camera_data[f"{face_id:04d}"]["extrinsics"], dtype=np.float32
        )
        return face_w2c @ panorama_w2c, cube_dir

    cube_dir = cubemap_root / scene / "panocam" / "images" / f"{frame_index}_cube_95"
    camera_data = json.loads((cube_dir / "camera_data.json").read_text())
    poses_path = pose_root / scene / "scans" / "scanner_poses.json"
    scanner_poses = json.loads(poses_path.read_text())
    if frame_index >= len(scanner_poses):
        raise IndexError(
            f"Pose index {frame_index} is out of range for {poses_path} "
            f"with {len(scanner_poses)} poses"
        )
    scanner_c2w = np.asarray(scanner_poses[frame_index], dtype=np.float32)
    if scanner_c2w.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 scanner pose at index {frame_index} in {poses_path}, "
            f"got {scanner_c2w.shape}"
        )
    scanner_w2c = (
        SCANNETPP_AXIS_CONVERSION
        @ np.linalg.inv(scanner_c2w).astype(np.float32)
        @ SCANNETPP_AXIS_CONVERSION.T
    )
    face_w2c = np.asarray(
        camera_data[f"{face_id:04d}"]["extrinsics"], dtype=np.float32
    )
    # ScanNet++ surface points need the final axis conversion as well; this is
    # identical to ScanNetPPSDFDataset._load_geometry.
    return face_w2c @ scanner_w2c @ SCANNETPP_AXIS_CONVERSION, cube_dir


def choose_rows(rng, rows, count):
    if count == 0:
        return rows[:0]
    if len(rows) < count:
        raise ValueError(f"Cannot sample {count} points without replacement from {len(rows)}")
    return rows[rng.choice(len(rows), count, replace=False)]


def load_surface_arrays(surface_path: Path, load_sharp: bool):
    """Load one surface archive once, including across concurrent workers."""
    key = (str(surface_path), bool(load_sharp))
    while True:
        with _SURFACE_CACHE_LOCK:
            cached = _SURFACE_CACHE.get(key)
            if cached is not None:
                _SURFACE_CACHE.move_to_end(key)
                return cached
            event = _SURFACE_LOADS.get(key)
            if event is None:
                event = threading.Event()
                _SURFACE_LOADS[key] = event
                break
        event.wait()

    try:
        with np.load(surface_path, allow_pickle=False) as data:
            arrays = {
                "random_surface": np.asarray(data["random_surface"]),
            }
            if load_sharp:
                arrays["sharp_surface"] = np.asarray(data["sharp_surface"])
    except BaseException:
        with _SURFACE_CACHE_LOCK:
            _SURFACE_LOADS.pop(key).set()
        raise

    with _SURFACE_CACHE_LOCK:
        _SURFACE_CACHE[key] = arrays
        _SURFACE_CACHE.move_to_end(key)
        while len(_SURFACE_CACHE) > SURFACE_CACHE_SIZE:
            _SURFACE_CACHE.popitem(last=False)
        _SURFACE_LOADS.pop(key).set()
    return arrays


def make_surface(
    surface_path: Path,
    world_to_camera,
    *,
    random_count: int,
    sharp_count: int,
    point_feats: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    arrays = load_surface_arrays(surface_path, load_sharp=sharp_count > 0)
    groups = [(choose_rows(rng, arrays["random_surface"], random_count), 0.0)]
    if sharp_count > 0:
        groups.append((choose_rows(rng, arrays["sharp_surface"], sharp_count), 1.0))

    samples = []
    for rows, label in groups:
        if not len(rows):
            continue
        xyz = np.asarray(rows[:, :3], dtype=np.float32) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        if point_feats == 0:
            sample = xyz
        else:
            if rows.shape[1] < 6:
                raise ValueError(f"Surface points in {surface_path} do not contain normals")
            normals = np.asarray(rows[:, 3:6], dtype=np.float32) @ world_to_camera[:3, :3].T
            normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(normal_lengths, 1e-12)
            if point_feats == 3:
                sample = np.concatenate((xyz, normals), axis=1)
            elif point_feats == 4:
                labels = np.full((len(rows), 1), label, dtype=np.float32)
                sample = np.concatenate((xyz, normals, labels), axis=1)
            else:
                raise ValueError(f"Unsupported ShapeVAE point_feats={point_feats}; expected 0, 3, or 4")
        samples.append(sample)
    return torch.from_numpy(np.concatenate(samples, axis=0))


def normalize_to_unit_box(surface: torch.Tensor, padding: float = 1.01):
    """Uniformly fit XYZ inside [-1, 1]^3 with configurable padding."""
    xyz = surface[:, :3]
    xyz_min = xyz.amin(dim=0)
    xyz_max = xyz.amax(dim=0)
    box_size = (xyz_max - xyz_min).amax()
    if not torch.isfinite(box_size) or box_size <= 0:
        raise ValueError("Cannot normalize a degenerate surface to a unit box")
    center = (xyz_min + xyz_max) * 0.5
    scale = 2.0 / (box_size * padding)
    result = surface.clone()
    result[:, :3] = (xyz - center) * scale
    return result, float(scale)


def normalize_surface(surface: torch.Tensor, mode: str):
    """Normalize XYZ while leaving normals and labels unchanged."""
    if mode in ("none", "raw", "skip"):
        return surface, 1.0
    if mode in ("unit_box", "cube"):
        return normalize_to_unit_box(surface, padding=1.0 if mode == "cube" else 1.01)
    if mode == "median":
        target = 1.0
    elif mode.startswith("median_"):
        target = float(mode.rsplit("_", 1)[1])
    else:
        raise ValueError(f"Unsupported normalization {mode!r}; use none, median, " "median_<target>, cube, or unit_box")
    median = torch.linalg.vector_norm(surface[:, :3], dim=1).median().clamp(0.01, 100.0)
    scale = target / median
    result = surface.clone()
    result[:, :3] = torch.clamp(result[:, :3] * scale, -1000.0, 1000.0)
    return result, float(scale)


def center_surface(surface: torch.Tensor):
    """Center surface coordinates on the midpoint of their axis-aligned bounds."""
    xyz = surface[:, :3]
    center = (xyz.amin(dim=0) + xyz.amax(dim=0)) * 0.5
    result = surface.clone()
    result[:, :3] -= center
    return result, center


def cube_normalize_and_center_camera(surface: torch.Tensor):
    """Cube-normalize about the bbox midpoint, then restore the camera origin.

    The input is already in the camera frame, so its camera is at zero. Standard
    cube normalization moves that camera to ``-bbox_center * scale``. Subtracting
    that transformed camera position puts it back at zero while retaining the
    bbox-derived uniform scale.
    """
    xyz = surface[:, :3]
    bbox_center = (xyz.amin(dim=0) + xyz.amax(dim=0)) * 0.5
    result, scale = normalize_to_unit_box(surface, padding=1.0)
    normalized_camera = -bbox_center * scale
    result[:, :3] -= normalized_camera
    return result, scale, bbox_center, -normalized_camera


def prepare_job(
    job,
    *,
    surface_root: Path,
    cubemap_root: Path,
    pose_root: Path,
    face_id: int,
    no_camera_extrinsics: bool,
    panorama_camera: bool,
    base_seed: int,
    center_room: bool,
    cube_center_camera: bool,
    normalization: str,
    random_count: int,
    sharp_count: int,
    point_feats: int,
    dataset_layout: str = "sage10k",
):
    """Load and transform one surface on a CPU worker."""
    scene, frame = job.scene, job.frame
    if dataset_layout == "scannetpp":
        surface_path = surface_root / scene / "scans" / "mesh_aligned_0.05_surface.npz"
        scene_axis_conversion = SCANNETPP_AXIS_CONVERSION
    elif dataset_layout == "3dfront":
        room = Path(scene)
        surface_path = surface_root / room.parent / f"{room.name}_full_surface.npz"
        scene_axis_conversion = GLTF_TO_BLENDER_OPENCV
    else:
        surface_path = surface_root / scene / "scene_surface.npz"
        scene_axis_conversion = PANORAMA_TO_OPENCV
    if no_camera_extrinsics:
        input_transform = scene_axis_conversion.copy()
        cube_dir = None
        seed = sample_seed(base_seed, scene, frame)
        coordinate_frame = "scene_opencv_axes"
    else:
        input_transform, cube_dir = load_world_to_camera(
            cubemap_root,
            pose_root,
            scene,
            frame,
            face_id,
            dataset_layout,
            panorama_camera,
        )
        seed = sample_seed(
            base_seed, scene, frame, None if panorama_camera else face_id
        )
        coordinate_frame = (
            "3dfront_panorama_camera"
            if panorama_camera
            else f"{dataset_layout}_camera"
        )

    surface = make_surface(
        surface_path,
        input_transform,
        random_count=random_count,
        sharp_count=sharp_count,
        point_feats=point_feats,
        seed=seed,
    )
    room_center = torch.zeros(3, dtype=surface.dtype)
    cube_bbox_center = torch.zeros(3, dtype=surface.dtype)
    normalized_camera_translation = torch.zeros(3, dtype=surface.dtype)
    if center_room:
        surface, room_center = center_surface(surface)
    if cube_center_camera:
        if normalization != "cube":
            raise ValueError(
                "--cube-center-camera requires normalization to resolve to 'cube', "
                f"got {normalization!r}"
            )
        surface, scale, cube_bbox_center, normalized_camera_translation = (
            cube_normalize_and_center_camera(surface)
        )
    else:
        surface, scale = normalize_surface(surface, normalization)

    xyz = surface[:, :3]
    lower = xyz.amin(dim=0)
    upper = xyz.amax(dim=0)
    padding = (upper - lower) * 0.1
    bounds = torch.cat((lower - padding, upper + padding))
    return PreparedSurface(
        job=job,
        surface=surface,
        source_path=surface_path,
        cubemap_path=cube_dir,
        input_transform=input_transform,
        coordinate_frame=coordinate_frame,
        bounds=bounds,
        normalization_scale=scale,
        room_center=room_center,
        cube_bbox_center=cube_bbox_center,
        normalized_camera_translation=normalized_camera_translation,
        seed=seed,
    )


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def format_duration(seconds: float) -> str:
    """Format a duration compactly for progress messages."""
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def prepared_batches(jobs, batch_size, prepare, num_workers):
    """Prepare batches in order and prefetch one batch ahead of the GPU.

    Keeping the executor alive was not sufficient to overlap CPU preparation
    with encoding: ``executor.map`` was only started after the preceding GPU
    batch had completed.  Submit the next batch before yielding the current
    one so NPZ access, transforms, and point sampling run concurrently with
    the VAE without allowing an unbounded number of large surfaces in memory.
    """
    if num_workers == 0:
        for job_batch in batched(jobs, batch_size):
            yield [prepare(job) for job in job_batch]
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        job_batches = iter(batched(jobs, batch_size))

        def submit(job_batch):
            return [executor.submit(prepare, job) for job in job_batch]

        try:
            futures = submit(next(job_batches))
        except StopIteration:
            return

        while True:
            prepared = [future.result() for future in futures]
            try:
                futures = submit(next(job_batches))
            except StopIteration:
                yield prepared
                break
            yield prepared


def normalized_surface_path(latent_path: Path):
    return latent_path.with_name(f"{latent_path.stem}_normalized_surface.ply")


def before_sampling_latent_path(latent_path: Path):
    return latent_path.with_name(f"{latent_path.stem}_before_sampling.pt")


def needs_encoding(job: Job, args):
    """Return whether any requested output for a job is missing."""
    if args.only_save_latent_before_sampling:
        requested_latent_missing = not before_sampling_latent_path(job.output_path).is_file()
    else:
        requested_latent_missing = not job.output_path.is_file() or (
            args.save_latent_before_sampling and not before_sampling_latent_path(job.output_path).is_file()
        )
    return args.overwrite or requested_latent_missing or (args.save_normalized_surface and not normalized_surface_path(job.output_path).is_file())


def encode_surfaces(
    vae,
    inputs: torch.Tensor,
    *,
    sample_posterior: bool,
    only_before_sampling: bool = False,
):
    """Encode a latent and return its posterior distribution parameters."""
    from nova3r.heads.hunyuan_model.autoencoders.model import (
        DiagonalGaussianDistribution,
    )

    pc, feats = inputs[:, :, :3], inputs[:, :, 3:]
    latents, _ = vae.encoder(pc, feats)
    moments = vae.pre_kl(latents)
    posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
    latents = None
    if not only_before_sampling:
        latents = posterior.sample() if sample_posterior else posterior.mode()
    return latents, posterior.mean, posterior.logvar


def encode_prepared_batch(
    vae,
    prepared_batch,
    device,
    sample_posterior,
    dtype,
    only_before_sampling=False,
):
    """Move one prepared CPU batch to the VAE and return CPU outputs."""
    inputs = torch.stack([item.surface for item in prepared_batch]).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        encoded = encode_surfaces(
            vae,
            inputs,
            sample_posterior=sample_posterior,
            only_before_sampling=only_before_sampling,
        )
    # Checkpoints are always stored in FP32 even when the VAE runs in BF16 or
    # FP16. This keeps generated latents compatible with the existing dataset
    # and avoids mixed checkpoint dtypes across resumed generation runs.
    latents, means, logvars = (
        None if tensor is None else tensor.detach().float().cpu()
        for tensor in encoded
    )
    return inputs, latents, means, logvars


def save_surface_ply(surface: torch.Tensor, path: Path):
    """Write one normalized VAE surface as a binary PLY point cloud."""
    if surface.ndim == 3:
        if surface.shape[0] != 1:
            raise ValueError(f"Expected one surface, got shape {tuple(surface.shape)}")
        surface = surface[0]
    if surface.ndim != 2 or surface.shape[1] < 3:
        raise ValueError(f"Expected surface [N, C>=3], got {tuple(surface.shape)}")

    values = surface.detach().float().cpu().numpy()
    properties = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if values.shape[1] >= 6:
        properties.extend((("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")))
    if values.shape[1] >= 7:
        properties.append(("label", "<f4"))

    vertices = np.empty(len(values), dtype=properties)
    for index, (name, _) in enumerate(properties):
        vertices[name] = values[:, index]

    header_properties = "\n".join(f"property float {name}" for name, _ in properties)
    header = "ply\n" "format binary_little_endian 1.0\n" f"element vertex {len(vertices)}\n" f"{header_properties}\n" "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        vertices.tofile(file)


def result_metadata(item: PreparedSurface, args, normalization: str):
    """Build metadata shared by sampled and pre-sampling checkpoints."""
    camera_transform_applied = not args.no_camera_extrinsics
    return {
        "scene": item.job.scene,
        "frame": item.job.frame,
        "face_id": (
            args.face_id
            if camera_transform_applied and not args.panorama_camera
            else None
        ),
        "source": str(item.source_path),
        "cubemap": None if item.cubemap_path is None else str(item.cubemap_path),
        "normalization": normalization,
        "normalization_scale": item.normalization_scale,
        "room_centered": args.center_room,
        "room_center": item.room_center,
        "cube_center_camera": args.cube_center_camera,
        "panorama_camera": args.panorama_camera,
        "cube_bbox_center": item.cube_bbox_center,
        "normalized_camera_translation": item.normalized_camera_translation,
        "coordinate_frame": item.coordinate_frame,
        "input_transform": torch.from_numpy(item.input_transform.copy()),
        "bounds": item.bounds,
        "camera_transform_applied": camera_transform_applied,
        "translation_applied": camera_transform_applied,
        "seed": item.seed,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
    }


def save_outputs(
    item: PreparedSurface,
    index: int,
    inputs: torch.Tensor,
    latents: torch.Tensor,
    means: torch.Tensor,
    logvars: torch.Tensor,
    *,
    args,
    normalization: str,
    completed: int,
    total: int,
):
    """Save every requested output for one prepared scene."""
    output_path = item.job.output_path
    metadata = result_metadata(item, args, normalization)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.only_save_latent_before_sampling:
        result = {
            "latent": latents[index : index + 1].clone(),
            **metadata,
            "posterior": "sample" if args.sample_posterior else "mode",
        }
        torch.save(result, output_path)
        print(f"[{completed}/{total}] Saved {tuple(result['latent'].shape)} " f"to {output_path}")

    if args.save_latent_before_sampling or args.only_save_latent_before_sampling:
        posterior_path = before_sampling_latent_path(output_path)
        posterior = {
            **metadata,
            "posterior_mean": means[index : index + 1].clone(),
            "posterior_logvar": logvars[index : index + 1].clone(),
            "posterior": "distribution",
        }
        torch.save(posterior, posterior_path)
        print(f"[{completed}/{total}] Saved sampleable posterior " f"{tuple(posterior['posterior_mean'].shape)} to {posterior_path}")

    if args.save_normalized_surface:
        surface_path = normalized_surface_path(output_path)
        save_surface_ply(inputs[index : index + 1], surface_path)
        print(f"[{completed}/{total}] Saved normalized surface " f"{tuple(inputs[index : index + 1].shape)} to {surface_path}")


def normalize_device_name(name: str):
    """Accept both CUDA indices (``0``) and normal torch device strings."""
    return f"cuda:{name}" if name.isdigit() else name


def validate_devices(names):
    devices = [torch.device(normalize_device_name(name)) for name in names]
    if len({str(device) for device in devices}) != len(devices):
        raise ValueError("--devices resolves to duplicate devices")
    if len(devices) > 1 and any(device.type == "cuda" and device.index is None for device in devices):
        raise ValueError("Multiple CUDA devices must use explicit indices, such as " "--devices cuda:0 cuda:1")
    for device in devices:
        if device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA " "devices are visible")
    return devices


def encode_jobs_on_device(args, pending, device, worker_rank=0, world_size=1):
    """Load one VAE and encode this process's non-overlapping job shard."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    prefix = f"[worker {worker_rank + 1}/{world_size} {device}] "
    print(f"{prefix}Encoding {len(pending)} samples", flush=True)

    dtype = precision_dtype(args.precision) if device.type == "cuda" else torch.float32
    vae, training_normalization = load_vae(
        args.config, args.checkpoint, device, dtype
    )
    normalization = training_normalization if args.normalization == "auto" else args.normalization
    if args.cube_center_camera and normalization != "cube":
        raise ValueError(
            "--cube-center-camera requires normalization to resolve to 'cube', "
            f"but the selected config resolved auto to {normalization!r}"
        )
    centering = " after room centering" if args.center_room else ""
    if args.cube_center_camera:
        centering = " followed by translating the normalized camera to the origin"
    if args.no_camera_extrinsics:
        axis_description = (
            "glTF Y-up -> Blender Z-up -> OpenCV axes"
            if args.dataset_layout == "3dfront"
            else "fixed (x, y, z) -> (x, -y, -z) axis conversion"
        )
        print(
            prefix + f"Using {normalization} scaling{centering} in the scene "
            f"frame with only the {axis_description}"
        )
    else:
        print(prefix + f"Using {normalization} scaling{centering} and face {args.face_id} of each " "listed panorama")

    prepare = partial(
        prepare_job,
        surface_root=args.surface_root,
        cubemap_root=args.cubemap_root,
        pose_root=args.pose_root,
        face_id=args.face_id,
        no_camera_extrinsics=args.no_camera_extrinsics,
        panorama_camera=args.panorama_camera,
        base_seed=args.seed,
        center_room=args.center_room,
        cube_center_camera=args.cube_center_camera,
        normalization=normalization,
        random_count=vae.encoder.pc_size,
        sharp_count=vae.encoder.pc_sharpedge_size,
        point_feats=vae.encoder.point_feats,
        dataset_layout=args.dataset_layout,
    )
    preparation = "sequentially" if args.num_workers == 0 else f"with {args.num_workers} worker threads"
    print(f"{prefix}Preparing surfaces {preparation}")
    batches = prepared_batches(
        pending,
        args.batch_size,
        prepare,
        args.num_workers,
    )
    started_at = time.monotonic()
    for batch_start, prepared_batch in enumerate(batches):
        inputs, latents, means, logvars = encode_prepared_batch(
            vae,
            prepared_batch,
            device,
            args.sample_posterior,
            dtype,
            args.only_save_latent_before_sampling,
        )
        for index, item in enumerate(prepared_batch):
            completed = batch_start * args.batch_size + index + 1
            save_outputs(
                item,
                index,
                inputs,
                latents,
                means,
                logvars,
                args=args,
                normalization=normalization,
                completed=completed,
                total=len(pending),
            )
        completed = min((batch_start + 1) * args.batch_size, len(pending))
        elapsed = time.monotonic() - started_at
        samples_per_second = completed / elapsed if elapsed else 0.0
        remaining = len(pending) - completed
        eta = remaining / samples_per_second if samples_per_second else 0.0
        print(
            f"{prefix}Progress {completed}/{len(pending)} "
            f"({100 * completed / len(pending):.1f}%), "
            f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}, "
            f"{samples_per_second:.2f} samples/s",
            flush=True,
        )


def multi_device_worker(rank, args, pending, device_names):
    """Entry point used by torch multiprocessing's spawn context."""
    devices = [torch.device(name) for name in device_names]
    shard = pending[rank :: len(devices)]
    encode_jobs_on_device(args, shard, devices[rank], rank, len(devices))


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("medium")
    for path, description in (
        (args.split_list, "split list"),
        (args.checkpoint, "checkpoint"),
        (args.config, "config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    jobs = load_jobs(
        args.split_list,
        args.output_root,
        args.scene,
        args.limit,
        args.camera_poses,
        args.dataset_layout,
    )
    pending = [job for job in jobs if needs_encoding(job, args)]
    print(f"Selected {len(jobs)} samples; {len(pending)} need encoding")
    if not pending:
        return

    requested_devices = args.devices or [args.device]
    devices = validate_devices(requested_devices)
    devices = devices[: len(pending)]
    if len(devices) == 1:
        encode_jobs_on_device(args, pending, devices[0])
        return

    # Spawn is required for CUDA: forking a process after CUDA initialization
    # is unsupported. Each child owns one VAE and one deterministic job shard.
    torch.multiprocessing.spawn(
        multi_device_worker,
        args=(args, pending, [str(device) for device in devices]),
        nprocs=len(devices),
        join=True,
    )


if __name__ == "__main__":
    main()
