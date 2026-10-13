#!/usr/bin/env python3
# Copyright (c) 2026 Weirong Chen
"""NOVA3R Demo: 3D point cloud reconstruction from images."""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import numpy as np
import open3d as o3d
import time

import PIL.Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from nova3r.models.nova3r_img_cond import Nova3rImgCond
from nova3r.models.nova3r_pts_cond import Nova3rPtsCond  # noqa: F401 — needed by load_model's eval()
from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE  # noqa: F401 — needed by load_model's eval()
from nova3r.heads.pts3d_decoder.latent_flowm_decoder import DiT  # noqa: F401 — needed by load_model's eval()
from nova3r.models.hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain  # noqa: F401 — needed by load_model's eval()
from nova3r.inference import inference_nova3r
from training.utils.checkpoint_utils import extract_model_state_dict


def _drop_mismatched_pos_embeds(model, state_dict):
    """Leave fixed positional embeddings initialized when token counts differ."""
    model_state = model.state_dict()
    filtered_state = dict(state_dict)
    for key, checkpoint_value in state_dict.items():
        model_value = model_state.get(key)
        if (
            key.endswith("pos_embed")
            and model_value is not None
            and checkpoint_value.ndim >= 2
            and model_value.ndim >= 2
            and checkpoint_value.shape[1] != model_value.shape[1]
        ):
            print(
                f"Skipping {key}: checkpoint has "
                f"{checkpoint_value.shape[1]} positions, model expects "
                f"{model_value.shape[1]}."
            )
            del filtered_state[key]
    return filtered_state


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--images", nargs="+", required=True, help="Path to 1 or 2 input images")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="demo/outputs/", help="Output directory (default: demo/outputs/)")
    parser.add_argument("--num_queries", type=int, default=50000, help="Number of query points (default: 50000)")
    parser.add_argument("--resolution", type=int, nargs="+", default=[518, 392], help="Resolution as W H (default: 518 392)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--aggregator_ckpt", default="./checkpoints/scene_n2/model.safetensors", help="Aggregator type (default: DepthAnything3Net)")
    args = parser.parse_args()

    # if len(args.images) > 2:
    #     parser.error("At most 2 images are supported")

    return args


def load_model(
    ckpt_path=None,
    device="cuda",
    config_path=None,
    dtype=torch.bfloat16,
    **kw,
):
    """Load a model from an explicit config and its configured checkpoint.

    When ``stage="test"``, ``experiment.resume_checkpoint`` is loaded as-is and
    component checkpoint substitutions are disabled by the model loader.
    ``ckpt_path`` remains an optional override. For backward compatibility, when
    ``config_path`` is omitted the config is discovered next to ``ckpt_path``.

    ``dtype`` defaults to BF16 for inference. Training entry points must request
    FP32 parameters so Lightning's mixed-precision plugin can retain FP32 master
    weights and optimizer state while autocasting supported operations.
    """
    if config_path is None:
        if ckpt_path is None:
            raise ValueError("Either config_path or ckpt_path must be provided")
        config_path = os.path.join(os.path.dirname(ckpt_path), ".hydra", "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    root_cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(root_cfg)
    cfg = root_cfg.experiment
    if "data" in root_cfg:
        cfg.data = root_cfg.data

    stage = kw.get("stage")
    # A scratch-training config may disable its initialization checkpoint, but
    # test mode must still load experiment.resume_checkpoint unless explicitly
    # overridden by the caller.
    default_load_weights = True if stage == "test" else cfg.get("load_weights", True)
    load_weights = bool(kw.pop("load_weights", default_load_weights))
    checkpoint_key = "resume_checkpoint" if stage == "test" else "checkpoint"
    ckpt_path = ckpt_path or cfg.get(checkpoint_key)
    if load_weights and not ckpt_path:
        raise ValueError(f"No experiment.{checkpoint_key} configured in {config_path}")
    if load_weights and not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    if stage == "test":
        # A trained checkpoint contains the complete model. Loading component
        # checkpoints as well would silently replace its fine-tuned weights.
        kw.pop("aggregator_ckpt", None)
        kw.pop("vae_ckpt", None)
        kw.pop("dit_ckpt", None)
        kw["strict"] = True
    else:
        for key in ("aggregator_ckpt", "vae_ckpt", "dit_ckpt"):
            if kw.get(key) is None and cfg.get(key):
                kw[key] = cfg.get(key)

    model_config = cfg.model
    model = eval(model_config["name"])(**model_config["params"])

    if not load_weights:
        model.to(device=device, dtype=dtype)
        return model, cfg

    # Avoid holding both the initialized model and checkpoint tensors on GPU.
    # Apply the state exactly once on CPU, then move the completed model.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = extract_model_state_dict(ckpt)

    state_dict = _drop_mismatched_pos_embeds(model, state_dict)
    model.load_state_dict(state_dict, **kw)

    del ckpt
    model.to(device=device, dtype=dtype)
    return model, cfg


def save_pointcloud(pts3d, output_dir):
    """Save point cloud as PLY."""
    combined_pts = pts3d.reshape(-1, 3).cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    combined_pts = np.ascontiguousarray(combined_pts, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    ply_path = os.path.join(output_dir, "pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved point cloud: {ply_path}")
    return ply_path


def render_360_video(
    ply_path,
    output_dir,
    flip_axis="y",
    color_type="source",
    num_frames=180,
    fps=30,
    elevation=20,
    radius=0.003,
    max_points=1000_000,
):
    """Render 360 turntable video from a PLY file."""
    from demo.visualization.render_points import (
        load_ply_pointcloud,
        flip_axis as flip_axis_fn,
        build_colored_pointcloud,
        render_turntable_video,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    point_cloud = load_ply_pointcloud(
        ply_path,
        device=device,
        remove_outlier=True,
        max_points=max_points,
    )
    if flip_axis:
        point_cloud = flip_axis_fn(point_cloud, axis=flip_axis)
    point_cloud = build_colored_pointcloud(point_cloud, device=device, color_type=color_type)

    video_path = os.path.join(output_dir, "pointcloud.mp4")
    render_turntable_video(
        point_cloud,
        num_frames=num_frames,
        fps=fps,
        elevation=elevation,
        radius=radius,
        outfile=video_path,
    )
    return video_path


def predict(ckpt_path, image_paths, device="cuda", resolution=(518, 392), output_path="output.ply"):
    """Simple API: image paths in, PLY file out.

    Args:
        ckpt_path: Path to model checkpoint (e.g., 'checkpoints/scene_n1/checkpoint-last.pth')
        image_paths: List of 1 or 2 image paths
        device: Device to run on
        resolution: (width, height) tuple
        output_path: Where to save the output .ply file

    Returns:
        pts3d: numpy array of shape (N, 3) with predicted 3D points
    """
    model, cfg = load_model(ckpt_path, device)
    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"

    target_W, target_H = resolution

    img_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img_paths = image_paths if len(image_paths) == 2 else [image_paths[0], image_paths[0]]
    images = []
    for i, p in enumerate(img_paths):
        img = PIL.Image.open(p).convert("RGB")
        img = img.resize((target_W, target_H), PIL.Image.LANCZOS)
        images.append(
            dict(
                img=img_norm(img)[None],
                true_shape=np.int32([target_H, target_W]),
                idx=i,
                instance=str(i),
                view_label=f"input_{i}",
            )
        )

    symmetrize = len(image_paths) == 1
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=symmetrize)

    with torch.no_grad():
        output = inference_nova3r(
            cfg,
            pairs,
            model,
            device,
            batch_size=1,
            num_queries=20000,
            method=cfg.get("fm_sampling", "euler"),
        )

    pts3d = output["pred"]["pts3d_xyz"][0].numpy()

    # Save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved {pts3d.shape[0]} points to {output_path}")

    return pts3d


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    model, cfg = load_model(args.ckpt, args.device, aggregator_ckpt=args.aggregator_ckpt)

    # Set inference defaults if not in the saved config
    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"
    OmegaConf.set_struct(cfg, True)

    for img_path in args.images:
        print(f"Processing: {img_path}")

        img_paths = [img_path, img_path] if len(args.images) == 1 else args.images
        resolution = cfg.get("resolution", 224)
        if resolution == 224 and args.resolution is None:
            images = load_images(img_paths, size=224)
        else:
            # Use explicit resolution or derive from config
            patch_size = cfg.model.params.get("patch_size", 14)
            if args.resolution is not None:
                if len(args.resolution) == 2:
                    target_W, target_H = args.resolution
                else:
                    target_W = target_H = args.resolution[0]
            else:
                target_W = target_H = resolution
            img_norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            images = []
            for i, p in enumerate(img_paths):
                img = PIL.Image.open(p).convert("RGB")
                W, H = img.size
                img = img.resize((target_W, target_H), PIL.Image.LANCZOS)
                print(f" - adding {p} with resolution {W}x{H} --> {target_W}x{target_H}")
                images.append(
                    dict(
                        img=img_norm(img)[None],
                        true_shape=np.int32([target_H, target_W]),
                        idx=i,
                        instance=str(i),
                        view_label=f"input_{i}",
                    )
                )
        # For 2-view input, don't symmetrize — run one pass with both views
        symmetrize = len(args.images) == 1
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=symmetrize)
        if len(args.images) > 2:
            pairs = [tuple(images)]

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        output = inference_nova3r(
            cfg,
            pairs,
            model,
            args.device,
            batch_size=1,
            num_queries=args.num_queries,
            n_views=len(args.images),
            method=cfg.get("fm_sampling", "euler"),
        )

        elapsed = time.time() - start_time
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Inference: {elapsed:.2f}s | Peak memory: {peak_mem:.0f} MB")
        else:
            print(f"Inference: {elapsed:.2f}s")

        pts3d = output["pred"]["pts3d_xyz"]
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        scene_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(scene_dir, exist_ok=True)

        ply_path = save_pointcloud(pts3d, scene_dir)
        render_360_video(ply_path, scene_dir)

        # if len(args.images) == 2:
        break

    print("Done!")


if __name__ == "__main__":
    main()
