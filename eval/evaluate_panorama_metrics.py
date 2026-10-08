#!/usr/bin/env python3
"""Evaluate paired predicted/ground-truth depth and RGB panoramas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEPTH_SUFFIXES = IMAGE_SUFFIXES | {".npy"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-rgb", type=Path, required=True, help="Predicted RGB image or directory")
    parser.add_argument("--gt-rgb", type=Path, required=True, help="Ground-truth RGB image or directory")
    parser.add_argument("--pred-depth", type=Path, help="Predicted depth image/.npy or directory")
    parser.add_argument("--gt-depth", type=Path, help="Ground-truth depth image/.npy or directory")
    parser.add_argument("--output", type=Path, default=Path("panorama_metrics.json"))
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="Integer depth units per metric unit; ignored for .npy (default: 1000)")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=float("inf"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--lpips-max-size", type=int, default=1024,
                        help="Downscale the longest RGB side for LPIPS; 0 keeps native size")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    args = parser.parse_args()
    if args.depth_scale <= 0 or args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("require depth-scale > 0 and max-depth > min-depth >= 0")
    if args.lpips_max_size < 0:
        parser.error("--lpips-max-size cannot be negative")
    if (args.pred_depth is None) != (args.gt_depth is None):
        parser.error("--pred-depth and --gt-depth must be supplied together")
    return args


def collect(path: Path, suffixes: set[str]) -> dict[str, Path]:
    if path.is_file():
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"Unsupported file type: {path}")
        return {"sample": path}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {
        str(item.relative_to(path).with_suffix("")): item
        for item in sorted(path.rglob("*")) if item.is_file() and item.suffix.lower() in suffixes
    }
    if not files:
        raise ValueError(f"No supported files found below {path}")
    return files


def paired_inputs(args):
    groups = {
        "pred_rgb": collect(args.pred_rgb, IMAGE_SUFFIXES),
        "gt_rgb": collect(args.gt_rgb, IMAGE_SUFFIXES),
    }
    if args.pred_depth is not None:
        groups["pred_depth"] = collect(args.pred_depth, DEPTH_SUFFIXES)
        groups["gt_depth"] = collect(args.gt_depth, DEPTH_SUFFIXES)
    keys = set(groups["pred_rgb"])
    mismatches = {name: sorted(keys ^ set(paths))[:10] for name, paths in groups.items() if set(paths) != keys}
    if mismatches:
        raise ValueError(f"Input trees do not have identical relative names: {mismatches}")
    return [
        (
            key, groups["pred_rgb"][key], groups["gt_rgb"][key],
            groups.get("pred_depth", {}).get(key), groups.get("gt_depth", {}).get(key),
        )
        for key in sorted(keys)
    ]


def load_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            depth = np.asarray(image)
        if not np.issubdtype(depth.dtype, np.integer):
            raise ValueError(f"Image depth must use an integer format: {path} ({depth.dtype})")
        depth = depth.astype(np.float32) / scale
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth at {path}, got {depth.shape}")
    return depth


def crop_depth_to_gt(pred: np.ndarray, gt: np.ndarray):
    """Width-align a 2:1 prediction, then bottom-crop it to a shorter GT."""
    resized = False
    cropped = False
    if gt.shape[1] != 2 * gt.shape[0] and pred.shape[1] == 2 * pred.shape[0]:
        if pred.shape[1] != gt.shape[1]:
            target_height = gt.shape[1] // 2
            tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
            pred = F.interpolate(
                tensor, size=(target_height, gt.shape[1]), mode="nearest"
            )[0, 0].numpy()
            resized = True
        if pred.shape[0] < gt.shape[0]:
            raise ValueError(
                f"Cannot bottom-crop predicted depth {pred.shape} to GT {gt.shape}"
            )
        pred = pred[:gt.shape[0], :]
        cropped = True
    return pred, resized, cropped


def depth_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float, max_depth: float):
    if pred.shape != gt.shape:
        raise ValueError(f"Depth shape mismatch: prediction {pred.shape}, GT {gt.shape}")
    valid = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    pred_valid = np.isfinite(pred) & (pred > min_depth) & (pred < max_depth)
    if not valid.any():
        raise ValueError("Ground-truth depth has no valid pixels")
    if not np.isfinite(pred[valid]).all():
        raise ValueError("Prediction contains NaN or infinity at valid ground-truth pixels")
    # Evaluate every GT-valid pixel. Out-of-range predictions are clipped rather
    # than silently omitted, so missing predictions cannot improve the score.
    p = np.clip(pred[valid].astype(np.float64), min_depth, max_depth)
    g = gt[valid].astype(np.float64)
    diff = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "valid_gt_pixels": int(valid.sum()),
        "evaluated_pixels": int(valid.sum()),
        "prediction_coverage": float((valid & pred_valid).sum() / valid.sum()),
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "sq_rel": float(np.mean(diff * diff / g)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def crop_rgb_to_gt(pred: Image.Image, gt: Image.Image):
    """Width-align a 2:1 prediction, then bottom-crop it to a shorter GT."""
    resized = False
    cropped = False
    gt_width, gt_height = gt.size
    pred_width, pred_height = pred.size
    if gt_width != 2 * gt_height and pred_width == 2 * pred_height:
        if pred_width != gt_width:
            pred = pred.resize((gt_width, gt_width // 2), Image.Resampling.BICUBIC)
            pred_width, pred_height = pred.size
            resized = True
        if pred_height < gt_height:
            raise ValueError(
                f"Cannot bottom-crop predicted RGB {pred.size} to GT {gt.size}"
            )
        pred = pred.crop((0, 0, gt_width, gt_height))
        cropped = True
    return pred, resized, cropped


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def resize_for_lpips(tensor: torch.Tensor, max_size: int) -> torch.Tensor:
    if not max_size or max(tensor.shape[-2:]) <= max_size:
        return tensor
    scale = max_size / max(tensor.shape[-2:])
    size = tuple(max(1, round(value * scale)) for value in tensor.shape[-2:])
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=True)


def load_rgb_models(args, device):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise RuntimeError("LPIPS requires torchmetrics and torchvision") from exc
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("CLIP image similarity requires open_clip_torch") from exc
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=args.lpips_net, normalize=True, reduction="mean"
    ).to(device).eval()
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device,
    )
    return lpips, clip.eval(), clip_preprocess


@torch.inference_mode()
def rgb_metrics(pred_image, gt_image, lpips, clip, clip_preprocess, device, lpips_max_size):
    if pred_image.size != gt_image.size:
        raise ValueError(f"RGB shape mismatch: prediction {pred_image.size}, GT {gt_image.size}")
    # Antialiased interpolation can overshoot [0, 1] by a few ULPs. TorchMetrics
    # validates this range strictly when normalize=True, so clamp after resize.
    pred_tensor = resize_for_lpips(
        image_tensor(pred_image, device), lpips_max_size
    ).clamp_(0.0, 1.0)
    gt_tensor = resize_for_lpips(
        image_tensor(gt_image, device), lpips_max_size
    ).clamp_(0.0, 1.0)
    lpips_value = float(lpips(pred_tensor, gt_tensor).item())
    lpips.reset()
    clip_pred = clip_preprocess(pred_image).unsqueeze(0).to(device)
    clip_gt = clip_preprocess(gt_image).unsqueeze(0).to(device)
    pred_features = F.normalize(clip.encode_image(clip_pred), dim=-1)
    gt_features = F.normalize(clip.encode_image(clip_gt), dim=-1)
    clip_similarity = float((pred_features * gt_features).sum(dim=-1).item())
    return {"lpips": lpips_value, "clipscore": 100.0 * clip_similarity}


def depth_summary(records):
    names = ("abs_rel", "sq_rel", "mae", "rmse", "rmse_log", "delta1", "delta2", "delta3")
    return {name: float(np.mean([item["depth"][name] for item in records])) for name in names} | {
        "valid_gt_pixels": int(sum(item["depth"]["valid_gt_pixels"] for item in records)),
        "evaluated_pixels": int(sum(item["depth"]["evaluated_pixels"] for item in records)),
        "prediction_coverage": float(np.mean([item["depth"]["prediction_coverage"] for item in records])),
    }


def main():
    args = parse_args()
    pairs = paired_inputs(args)
    device = torch.device(args.device)
    lpips, clip, clip_preprocess = load_rgb_models(args, device)
    records = []
    for index, (name, pred_rgb, gt_rgb, pred_depth, gt_depth) in enumerate(pairs, 1):
        depth = None
        depth_resized = depth_cropped = None
        if pred_depth is not None:
            pred_depth_array = load_depth(pred_depth, args.depth_scale)
            gt_depth_array = load_depth(gt_depth, args.depth_scale)
            pred_depth_array, depth_resized, depth_cropped = crop_depth_to_gt(
                pred_depth_array, gt_depth_array
            )
            depth = depth_metrics(
                pred_depth_array, gt_depth_array,
                args.min_depth, args.max_depth,
            )
        pred_rgb_image = load_rgb(pred_rgb)
        gt_rgb_image = load_rgb(gt_rgb)
        pred_rgb_image, rgb_resized, rgb_cropped = crop_rgb_to_gt(
            pred_rgb_image, gt_rgb_image
        )
        rgb = rgb_metrics(
            pred_rgb_image, gt_rgb_image, lpips, clip, clip_preprocess,
            device, args.lpips_max_size,
        )
        record = {
            "name": name,
            "alignment": {
                "rgb_prediction_resized_to_gt_width": rgb_resized,
                "rgb_prediction_bottom_cropped": rgb_cropped,
            },
            "rgb": rgb,
        }
        if depth is not None:
            record["alignment"]["depth_prediction_resized_to_gt_width"] = depth_resized
            record["alignment"]["depth_prediction_bottom_cropped"] = depth_cropped
            record["depth"] = depth
        records.append(record)
        depth_text = f"AbsRel={depth['abs_rel']:.6f} " if depth is not None else ""
        print(f"[{index}/{len(pairs)}] {name}: {depth_text}"
              f"LPIPS={rgb['lpips']:.6f} CLIPScore={rgb['clipscore']:.4f}", flush=True)

    summary = {
        "count": len(records),
        "rgb": {
            name: float(np.mean([item["rgb"][name] for item in records]))
            for name in ("lpips", "clipscore")
        },
    }
    if args.pred_depth is not None:
        summary["depth"] = depth_summary(records)
    result = {
        "definitions": {
            "depth": "Metrics are macro-averaged over samples on all GT-valid pixels; predictions are range-clipped and pre-clip coverage is reported.",
            "clipscore": "100 * cosine similarity between normalized CLIP image embeddings.",
            "panorama_alignment": "If GT is not 2:1 and prediction is 2:1, prediction is resized to GT width while preserving 2:1, then rows are removed from the bottom to match GT height. RGB uses bicubic and depth uses nearest-neighbor resizing.",
            "depth_scale": args.depth_scale,
            "depth_range": [args.min_depth, args.max_depth if np.isfinite(args.max_depth) else None],
            "lpips_net": args.lpips_net,
            "lpips_max_size": args.lpips_max_size,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
        },
        "summary": summary,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
