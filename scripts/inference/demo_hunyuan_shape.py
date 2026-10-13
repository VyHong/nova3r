#!/usr/bin/env python3
"""Generate an untextured Hunyuan3D-2.1 mesh from one reference image."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nova3r.models.hy3dshape.pipelines import (  # noqa: E402
    Hunyuan3DDiTFlowMatchingPipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an untextured 3D mesh from a single image."
    )
    parser.add_argument("image", type=Path, help="Input PNG/JPEG image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/hunyuan_dit/model.fp16.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "scripts/inference/hunyuan_shape_v2_1.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path("demo/outputs/hunyuan.glb"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument(
        "--remove-background",
        action="store_true",
        help="Run rembg first (may download its segmentation weights on first use)",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Reduce VRAM use with accelerate at the cost of speed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Inference config not found: {args.config}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    image = Image.open(args.image).convert("RGBA")
    if args.remove_background:
        from nova3r.models.hy3dshape.rembg import BackgroundRemover

        image = BackgroundRemover()(image)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    print(f"Loading Hunyuan3D checkpoint: {args.checkpoint}")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(args.checkpoint),
        str(args.config),
        device=args.device,
        dtype=dtype,
    )
    if args.cpu_offload:
        if not args.device.startswith("cuda"):
            raise ValueError("--cpu-offload requires a CUDA device")
        pipeline.enable_model_cpu_offload(device=args.device)

    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    print(f"Generating mesh from: {args.image}")
    meshes = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        output_type="trimesh",
    )
    if not meshes or meshes[0] is None:
        raise RuntimeError("Mesh extraction returned no mesh")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meshes[0].export(args.output)
    print(f"Saved mesh: {args.output}")


if __name__ == "__main__":
    main()
