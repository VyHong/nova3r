#!/usr/bin/env python3
"""Run batched DA3 depth on cubemaps and save separate ShapeVAE latents."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch

from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE
from tools.depth_surface import sample_depth_pointmap_for_vae, unproject_depth_to_world
from training.mesh2sdf.tools.infer_shape_vae import (
    checkpoint_state,
    find_shape_vae_config,
    select_vae_state,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_FACE_ORDER = (4, 0, 5, 1, 3, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict depth for the faces of one cubemap, unproject and sample the "
            "pointmap, then save one deterministic ShapeVAE latent."
        )
    )
    parser.add_argument(
        "cubemap_dir",
        type=Path,
        nargs="?",
        help="One rgb_cube_* directory. Omit when using --dataset-root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset root; Sage10k discovers all cubemaps, other layouts use --split-list.",
    )
    parser.add_argument(
        "--dataset-layout",
        choices=("replica_pano", "scannetpp", "sage10k"),
        default="replica_pano",
        help="Directory layout used below --dataset-root (default: replica_pano)",
    )
    parser.add_argument(
        "--split-list",
        type=Path,
        default=Path("data/replica_pano/train_list.json"),
    )
    parser.add_argument(
        "--scene",
        help="Process only this scene from --split-list.",
    )
    parser.add_argument("--cubemap-name", default="rgb_cube_95")
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N selected cubemaps (useful for a smoke test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of cubemaps to predict and encode together (default: 1)",
    )
    parser.add_argument(
        "--da3-model",
        default="depth-anything/DA3-LARGE",
        help="Official Hugging Face model ID or local pretrained directory.",
    )
    parser.add_argument(
        "--da3-repo",
        type=Path,
        default=REPO_ROOT / "third_party" / "Depth-Anything-3",
    )
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vae-config",
        type=Path,
        help="Default: VAE_CHECKPOINT_DIR/.hydra/config.yaml",
    )
    parser.add_argument("--output", type=Path, help="Output file for one cubemap")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root for Replica batch mode",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help=(
            "PyTorch intra-op CPU threads per GPU worker (default: 8). DA3 also "
            "loads images concurrently, so large values cause nested thread "
            "oversubscription."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=518,
        help="Official DA3 inference processing resolution (default: 518)",
    )
    parser.add_argument(
        "--face-order",
        type=int,
        nargs="+",
        default=list(DEFAULT_FACE_ORDER),
    )
    parser.add_argument("--confidence-percentile", type=float, default=0.0,
                        help="Discard this percentage of lowest-confidence valid points (0..100).")
    parser.add_argument("--sharp-threshold", type=float, default=0.985)
    parser.add_argument(
        "--normalization",
        choices=("auto", "none", "median", "bbox"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based job shard for independent GPU workers.")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Also save depth, confidence, sampled surface, and camera tensors.",
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require --num-shards >= 1 and 0 <= --shard-index < --num-shards")
    if not args.dataset_root and args.num_shards != 1:
        parser.error("sharding requires --dataset-root")
    if not 0 <= args.confidence_percentile < 100:
        parser.error("--confidence-percentile must be in [0, 100)")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")
    if sorted(args.face_order) != list(range(6)):
        parser.error("--face-order must be a permutation of 0 1 2 3 4 5")
    if args.dataset_root:
        if args.cubemap_dir:
            parser.error("use either cubemap_dir or --dataset-root, not both")
        if not args.output_root:
            parser.error("--output-root is required with --dataset-root")
    elif not args.cubemap_dir or not args.output:
        parser.error("single mode requires cubemap_dir and --output")
    elif args.scene:
        parser.error("--scene is only valid with --dataset-root")
    return args


def load_original_da3(repo: Path, model_id: str, device: torch.device):
    """Import DA3 from the untouched clone, isolated from the modified package."""
    src = (repo / "src").resolve()
    if not (src / "depth_anything_3" / "api.py").is_file():
        raise FileNotFoundError(f"Original DA3 API not found under {src}")

    for name in tuple(sys.modules):
        if name == "depth_anything_3" or name.startswith("depth_anything_3."):
            del sys.modules[name]
    sys.path.insert(0, str(src))
    try:
        api = importlib.import_module("depth_anything_3.api")
    finally:
        sys.path.remove(str(src))
    model = api.DepthAnything3.from_pretrained(model_id)
    print(f"Loaded official DA3 from {src}")
    return model.to(device).eval()


def load_vae(config_path: Path, checkpoint_path: Path, device: torch.device):
    params, training_normalization = find_shape_vae_config(config_path)
    model = ShapeVAE(**params)
    selected = select_vae_state(checkpoint_state(checkpoint_path), model)
    missing, unexpected = model.load_state_dict(selected, strict=False)
    required_missing = [
        key
        for key in missing
        if key.startswith(("encoder.", "pre_kl.", "post_kl.", "transformer.", "geo_decoder."))
    ]
    if unexpected or required_missing:
        raise RuntimeError(
            f"Incompatible VAE checkpoint: {len(required_missing)} required missing, "
            f"{len(unexpected)} unexpected"
        )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval(), training_normalization


def load_cubemap(cubemap_dir: Path, face_order):
    metadata_path = cubemap_dir / "camera_data.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing cubemap camera metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())

    by_index = {
        int(path.stem): path
        for path in cubemap_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = [index for index in face_order if index not in by_index]
    if missing:
        raise ValueError(f"Missing cubemap faces {missing} in {cubemap_dir}")
    paths = [by_index[index] for index in face_order]
    intrinsics = []
    for index, path in zip(face_order, paths):
        intrinsic = np.asarray(
            metadata[f"{index:04d}"]["intrinsics"], np.float32
        ).copy()
        with Image.open(path) as image:
            width, height = image.size
        if (
            abs(intrinsic[0, 0]) <= 2
            and abs(intrinsic[1, 1]) <= 2
            and abs(intrinsic[0, 2]) <= 2
            and abs(intrinsic[1, 2]) <= 2
        ):
            intrinsic[0] *= width
            intrinsic[1] *= height
        intrinsics.append(intrinsic)
    intrinsics = np.stack(intrinsics)
    extrinsics = np.stack(
        [np.asarray(metadata[f"{index:04d}"]["extrinsics"], np.float32) for index in face_order]
    )
    return paths, intrinsics, extrinsics


def dataset_jobs(
    dataset_root,
    split_list,
    dataset_layout,
    cubemap_name,
    output_root,
    scene_filter=None,
):
    samples = json.loads(split_list.read_text())
    for sample in samples:
        scene, sequence = sample.split()
        if scene_filter is not None and scene != scene_filter:
            continue
        if dataset_layout == "replica_pano":
            cubemap_dir = (
                dataset_root
                / scene
                / scene
                / "Scene_Info"
                / sequence
                / cubemap_name
            )
        elif dataset_layout == "scannetpp":
            frame_index = int(sequence)
            cubemap_dir = (
                dataset_root
                / scene
                / "panocam"
                / "images"
                / f"{frame_index}_cube_95"
            )
        else:
            raise ValueError(f"Unsupported dataset layout: {dataset_layout}")
        output = output_root / scene / f"{sequence}.pt"
        yield cubemap_dir, output


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_jobs(args):
    if not args.dataset_root:
        return [(args.cubemap_dir, args.output)]

    if args.dataset_layout == "sage10k":
        jobs = []
        for scene_dir in sorted((args.dataset_root / "cubemaps").iterdir()):
            if not scene_dir.is_dir() or args.scene and scene_dir.name != args.scene:
                continue
            prefix = f"{scene_dir.name}_panorama_"
            for cubemap_dir in sorted(scene_dir.glob(f"{prefix}*_cube_95")):
                if not cubemap_dir.is_dir():
                    continue
                frame = cubemap_dir.name.removeprefix(prefix).removesuffix("_cube_95")
                if not frame.isdigit():
                    continue
                jobs.append((cubemap_dir, args.output_root / scene_dir.name / f"{int(frame):05d}.pt"))
    else:
        jobs = list(
            dataset_jobs(
                args.dataset_root,
                args.split_list,
                args.dataset_layout,
                args.cubemap_name,
                args.output_root,
                args.scene,
            )
        )
    if args.limit is not None:
        jobs = jobs[:args.limit]
    # Shard the complete stable list before checking existing outputs, so
    # workers never race as files appear during a run or resume.
    return jobs[args.shard_index::args.num_shards]


def predict_depth_batch(da3, loaded_cubemaps, resolution):
    """Run DA3 on a batch of cubemaps using its batch-capable model forward."""
    images = []
    extrinsics = []
    intrinsics = []
    for paths, input_intrinsics, input_extrinsics in loaded_cubemaps:
        imgs_cpu, ex_cpu, in_cpu = da3._preprocess_inputs(
            [str(path) for path in paths],
            input_extrinsics,
            input_intrinsics,
            resolution,
            "upper_bound_resize",
        )
        images.append(imgs_cpu)
        extrinsics.append(ex_cpu)
        intrinsics.append(in_cpu)

    images_cpu = torch.stack(images)
    extrinsics_cpu = torch.stack(extrinsics)
    intrinsics_cpu = torch.stack(intrinsics)
    device = da3._get_model_device()
    images_device = images_cpu.to(device, non_blocking=True).float()
    extrinsics_device = extrinsics_cpu.to(device, non_blocking=True).float()
    intrinsics_device = intrinsics_cpu.to(device, non_blocking=True).float()

    # Normalize each cubemap independently. DA3's helper otherwise computes one
    # median camera distance across the entire batch.
    normalized_extrinsics = torch.cat(
        [
            da3._normalize_extrinsics(extrinsics_device[index : index + 1])
            for index in range(extrinsics_device.shape[0])
        ],
        dim=0,
    )
    raw = da3._run_model_forward(
        images_device,
        normalized_extrinsics,
        intrinsics_device,
        export_feat_layers=[],
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="first",
    )

    depth = raw["depth"]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    confidence = raw.get("depth_conf")
    if confidence is not None and confidence.ndim == 5:
        confidence = confidence.squeeze(-1)
    return depth, confidence, intrinsics_device, extrinsics_device


def encode_batch(vae, depth, confidence, intrinsics, extrinsics, args, normalization):
    with torch.inference_mode():
        pointmap = unproject_depth_to_world(depth, intrinsics, extrinsics).flatten(1, 3)
        confidence_flat = (
            confidence.reshape(confidence.shape[0], -1, 1)
            if confidence is not None
            else None
        )
        surface = sample_depth_pointmap_for_vae(
            pointmaps=pointmap,
            confidence=confidence_flat,
            num_views=depth.shape[1],
            height=depth.shape[-2],
            width=depth.shape[-1],
            random_count=vae.encoder.pc_size,
            sharp_count=vae.encoder.pc_sharpedge_size,
            point_feats=vae.encoder.point_feats,
            sharp_threshold=args.sharp_threshold,
            confidence_percentile=args.confidence_percentile,
            normalization=normalization,
            seed=args.seed,
        ).to(device=depth.device, dtype=next(vae.parameters()).dtype)
        # The encoder also randomly subsamples its point queries. A posterior
        # mode alone does not make this export deterministic.
        devices = [depth.device] if depth.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed)
            latent = vae.encode(surface, sample_posterior=True)
    return latent, pointmap, surface


def cpu_slice(tensor, index):
    return tensor[index : index + 1].detach().cpu()


def save_pointmap_ply(pointmap, path):
    """Save an XYZ tensor as a compact binary PLY point cloud."""
    points = pointmap.detach().float().cpu().reshape(-1, 3).numpy()
    points = np.ascontiguousarray(points[np.isfinite(points).all(axis=1)], dtype="<f4")
    vertices = points.view([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).reshape(-1)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        vertices.tofile(file)


def pointmap_path(output_path):
    return output_path.with_name(f"{output_path.stem}_pointmap.ply")


def make_result(
    index, cubemap_dir, paths, face_order, normalization, latent, tensors=None
):
    result = {
        "latent": cpu_slice(latent, index),
        "face_names": [path.name for path in paths],
        "face_order": list(face_order),
        "normalization": normalization,
        "source": str(cubemap_dir),
    }
    if tensors:
        result.update(
            {
                name: cpu_slice(tensor, index) if tensor is not None else None
                for name, tensor in tensors.items()
            }
        )
    return result


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    jobs = build_jobs(args)

    pending_jobs = [
        (index, *job)
        for index, job in enumerate(jobs, start=1)
        if not job[1].is_file()
    ]
    for index, (_, output_path) in enumerate(jobs, start=1):
        if not output_path.is_file():
            continue
        print(f"[{index}/{len(jobs)}] Skip existing {output_path}")
        ply_path = pointmap_path(output_path)
        if args.save_intermediates and not ply_path.is_file():
            saved = torch.load(output_path, map_location="cpu", weights_only=True)
            if "pointmap" not in saved:
                raise ValueError(f"No pointmap stored in {output_path}")
            save_pointmap_ply(saved["pointmap"], ply_path)
            print(f"Saved pointmap to {ply_path}")
    if not pending_jobs:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    vae_config = args.vae_config or args.vae_checkpoint.parent / ".hydra" / "config.yaml"
    da3 = load_original_da3(args.da3_repo, args.da3_model, device)
    print("Using ground-truth cubemap cameras without Umeyama pose alignment")
    vae, training_normalization = load_vae(
        vae_config, args.vae_checkpoint, device
    )
    normalization = (
        training_normalization
        if args.normalization == "auto"
        else args.normalization
    )
    print(
        f"Processing {len(jobs)} cubemap(s) with face order "
        f"{','.join(map(str, args.face_order))} and batch size {args.batch_size}"
    )
    for job_batch in batched(pending_jobs, args.batch_size):
        loaded = []
        for job_index, cubemap_dir, _ in job_batch:
            paths, input_intrinsics, input_extrinsics = load_cubemap(
                cubemap_dir, args.face_order
            )
            print(f"[{job_index}/{len(jobs)}] {cubemap_dir}")
            loaded.append((paths, input_intrinsics, input_extrinsics))

        depth, confidence, intrinsics, extrinsics = predict_depth_batch(
            da3, loaded, args.resolution
        )
        latent, pointmap, surface = encode_batch(
            vae, depth, confidence, intrinsics, extrinsics, args, normalization
        )
        intermediates = None
        if args.save_intermediates:
            intermediates = {
                "depth": depth,
                "confidence": confidence,
                "pointmap": pointmap,
                "surface": surface,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }

        for index, (_, cubemap_dir, output_path) in enumerate(job_batch):
            result = make_result(
                index,
                cubemap_dir,
                loaded[index][0],
                args.face_order,
                normalization,
                latent,
                intermediates,
            )
            xyz = surface[index, :, :3].float()
            lower, upper = xyz.amin(0), xyz.amax(0)
            padding = (upper - lower) * 0.1
            result.update(
                bounds=torch.cat((lower - padding, upper + padding)).cpu(),
                config=str(vae_config),
                checkpoint=str(args.vae_checkpoint),
                confidence_percentile=args.confidence_percentile,
                seed=args.seed,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, output_path)
            print(f"Saved latent {tuple(result['latent'].shape)} to {output_path}")
            if args.save_intermediates:
                ply_path = pointmap_path(output_path)
                save_pointmap_ply(result["pointmap"], ply_path)
                print(f"Saved pointmap to {ply_path}")


if __name__ == "__main__":
    main()
