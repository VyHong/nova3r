#!/usr/bin/env python3
"""Generate an untextured Hunyuan3D-2.1 mesh from one reference image."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nova3r.models.hy3dshape.pipelines import (  # noqa: E402
    Hunyuan3DDiTFlowMatchingPipeline,
)


class PrecomputedImageConditioner(nn.Module):
    """Expose precomputed image tokens through the pipeline conditioner API."""

    def __init__(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        includes_cls_token: bool,
        backbone: int = 3,
    ):
        super().__init__()
        self.register_buffer("tokens", tokens)
        self.grid_size = tuple(int(value) for value in grid_size)
        self.includes_cls_token = bool(includes_cls_token)
        self.backbone = int(backbone)

    def forward(self, image, **kwargs):
        batch_size = image.shape[0]
        if batch_size != 1:
            raise ValueError("Precomputed DINOv3 inference currently supports one image")
        return {
            "main": self.tokens.expand(batch_size, -1, -1),
            "backbone": torch.full(
                (batch_size,), self.backbone, dtype=torch.uint8, device=self.tokens.device
            ),
            "grid_size": torch.tensor(
                self.grid_size,
                dtype=torch.long,
                device=self.tokens.device,
            ).expand(batch_size, -1),
            "includes_cls_token": torch.zeros(
                batch_size, dtype=torch.bool, device=self.tokens.device
            ).fill_(self.includes_cls_token),
        }

    def unconditional_embedding(self, batch_size, **kwargs):
        result = self.forward(torch.empty(batch_size, 0, device=self.tokens.device))
        result["main"] = torch.zeros_like(self.tokens).expand(batch_size, -1, -1)
        return result


def load_image_features(path: Path) -> tuple[torch.Tensor, tuple[int, int], bool, int]:
    """Load panorama DINOv2/DINOv3 features or a raw token tensor."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    grid_size = None
    includes_cls_token = False
    backbone = 3
    if isinstance(value, dict):
        if "patch_features" not in value:
            raise ValueError(
                f"DINOv3 latent dictionary has no 'patch_features' key: {path}"
            )
        grid_size = value.get("grid_size")
        includes_cls_token = bool(value.get("includes_cls_token", False))
        model_name = str(value.get("model", "")).lower()
        if "dinov2" in model_name:
            backbone = 2
        elif "dinov3" in model_name:
            backbone = 3
        else:
            raise ValueError(f"Cannot identify DINO backbone from metadata: {path}")
        value = value["patch_features"]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"DINOv3 latents must be a tensor, got {type(value).__name__}")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 1024:
        raise ValueError(
            "Expected DINOv3 latents shaped [N, 1024] or [1, N, 1024], "
            f"got {tuple(value.shape)}"
        )
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("DINOv3 latents must contain finite floating-point values")
    if grid_size is None:
        patch_count = value.shape[1] - int(includes_cls_token)
        grid_height = round((patch_count / 2) ** 0.5)
        grid_size = (grid_height, grid_height * 2)
    grid_size = tuple(int(item) for item in grid_size)
    expected_tokens = grid_size[0] * grid_size[1] + int(includes_cls_token)
    if value.shape[1] != expected_tokens:
        raise ValueError(
            f"DINOv3 token count {value.shape[1]} does not match grid_size="
            f"{grid_size}, includes_cls_token={includes_cls_token}"
        )
    return value.contiguous(), grid_size, includes_cls_token, backbone


def load_modified_vae(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    dtype: torch.dtype,
) -> nn.Module:
    """Load a Nova3R ShapeVAE in place of the original signed-field VAE."""
    from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE
    from training.mesh2sdf.shape_vae import (
        checkpoint_state,
        find_shape_vae_config,
        select_vae_state,
    )

    params, _ = find_shape_vae_config(config_path)
    vae = ShapeVAE(**params)
    selected = select_vae_state(checkpoint_state(checkpoint_path), vae)
    missing, unexpected = vae.load_state_dict(selected, strict=False)
    decoder_missing = [
        key
        for key in missing
        if key.startswith(("post_kl.", "transformer.", "geo_decoder."))
    ]
    if decoder_missing or unexpected:
        raise RuntimeError(
            "Incompatible modified VAE checkpoint: "
            f"{len(decoder_missing)} decoder keys missing, "
            f"{len(unexpected)} unexpected keys"
        )
    return vae.requires_grad_(False).to(device=device, dtype=dtype).eval()


def mesh_level_from_config(config_path: Path, octree_resolution: int, box_v: float):
    """Read mesh_mc_level and resolve the training config's ``auto`` mode."""
    import numpy as np
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    root = config.get("experiment", config)
    root = root.get("cfg", root)
    value = root.get("mesh_mc_level", 0.0)
    if not (isinstance(value, str) and value.lower() == "auto"):
        return float(value)
    extent = 2.0 * float(box_v)
    voxel_world_size = extent / max(int(octree_resolution) - 1, 1)
    return float(np.clip(voxel_world_size * 128.0, 1e-4, 1.0))


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
    parser.add_argument(
        "--image-features",
        "--dinov3-latents",
        dest="image_features",
        type=Path,
        help=(
            "Use precomputed DINOv2 or DINOv3 patch features instead of encoding "
            "the input image. Accepts panorama feature-generator .pt output or "
            "a raw [N, 1024] tensor."
        ),
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        help="Replace the original Hunyuan VAE with a modified Nova3R ShapeVAE.",
    )
    parser.add_argument(
        "--vae-config",
        type=Path,
        default=REPO_ROOT / "checkpoints/hunyuan/.hydra/config.yaml",
        help="ShapeVAE training config used with --vae-checkpoint.",
    )
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
    if args.image_features is not None and not args.image_features.is_file():
        raise FileNotFoundError(f"Image features not found: {args.image_features}")
    if args.vae_checkpoint is not None:
        if not args.vae_checkpoint.is_file():
            raise FileNotFoundError(f"Modified VAE checkpoint not found: {args.vae_checkpoint}")
        if not args.vae_config.is_file():
            raise FileNotFoundError(f"Modified VAE config not found: {args.vae_config}")
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
    if args.image_features is not None:
        tokens, grid_size, includes_cls_token, backbone = load_image_features(
            args.image_features
        )
        tokens = tokens.to(dtype=dtype)
        pipeline.conditioner = PrecomputedImageConditioner(
            tokens, grid_size, includes_cls_token, backbone
        ).to(args.device)
        print(
            f"Using DINOv{backbone} patch features: {args.image_features} "
            f"{tuple(tokens.shape)}"
        )
    if args.vae_checkpoint is not None:
        pipeline.vae = load_modified_vae(
            args.vae_config,
            args.vae_checkpoint,
            args.device,
            dtype,
        )
        print(f"Using modified ShapeVAE: {args.vae_checkpoint}")
    if args.cpu_offload:
        if not args.device.startswith("cuda"):
            raise ValueError("--cpu-offload requires a CUDA device")
        pipeline.enable_model_cpu_offload(device=args.device)

    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    box_v = 1.01
    mc_level = -1 / 512
    if args.vae_checkpoint is not None:
        mc_level = mesh_level_from_config(
            args.vae_config,
            octree_resolution=args.octree_resolution,
            box_v=box_v,
        )
        print(f"Using mesh surface level from VAE config: {mc_level:.6g}")
    print(f"Generating mesh from: {args.image}")
    meshes = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        box_v=box_v,
        octree_resolution=args.octree_resolution,
        mc_level=mc_level,
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
