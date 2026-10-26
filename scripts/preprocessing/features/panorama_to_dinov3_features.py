#!/usr/bin/env python3
"""Extract DINOv2 or DINOv3 ViT features from equirectangular panoramas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from nova3r.utils.three_d_front import parse_panorama_sample

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
RESOLUTION = 512
INFERENCE_DTYPE = 'bfloat16'
SAVE_DTYPE = 'float16'


def pad_panorama_bottom_to_2to1(image):
    """Restore ScanNet++'s missing nadir region without distorting valid pixels."""
    target_height = image.width // 2
    bottom_padding = max(0, target_height - image.height)
    if not bottom_padding:
        return image, 0
    return ImageOps.expand(image, border=(0, 0, 0, bottom_padding), fill=0), bottom_padding


class PanoramaPreprocessDataset(Dataset):
    """Decode and preprocess panoramas inside DataLoader worker processes."""

    def __init__(self, jobs, resolution, processor, scannetpp=False):
        self.jobs = jobs
        self.resolution = resolution
        self.processor = processor
        self.scannetpp = scannetpp

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, index):
        path, _ = self.jobs[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
            original_size = torch.tensor([image.height, image.width], dtype=torch.int64)
            masked_pixels = 0
            if self.scannetpp:
                mask_path = path.parents[1] / 'anon_mask' / f'{path.stem}.png'
                if not mask_path.is_file():
                    raise FileNotFoundError(f'Missing ScanNet++ anonymization mask: {mask_path}')
                with Image.open(mask_path) as mask_image:
                    mask = np.asarray(mask_image.convert('L'))
                if mask.shape != (image.height, image.width):
                    raise ValueError(
                        f'Anonymization mask {mask_path} has shape {mask.shape}, '
                        f'expected {(image.height, image.width)}'
                    )
                invalid = mask == 0
                masked_pixels = int(invalid.sum())
                rgb = np.asarray(image).copy()
                rgb[invalid] = 0
                image = Image.fromarray(rgb)
                image, bottom_padding = pad_panorama_bottom_to_2to1(image)
            else:
                bottom_padding = 0
            pixels = self.processor(
                images=image,
                return_tensors='pt',
                do_resize=True,
                size={'height': self.resolution, 'width': self.resolution * 2},
                do_center_crop=False,
            )['pixel_values'][0]
        return index, pixels, original_size, bottom_padding, masked_pixels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panorama_root', type=Path)
    parser.add_argument('output_root', type=Path)
    parser.add_argument(
        '--dataset-layout', choices=('generic', 'sage10k', 'scannetpp', '3dfront'),
        default='generic', help='Input discovery layout (default: generic).',
    )
    parser.add_argument('--scannetpp', action='store_true',
                        help='Legacy alias for --dataset-layout scannetpp.')
    parser.add_argument('--split-list', type=Path,
                        help=("Optional '<scene> <frame>' JSON list. For 3D-FRONT, "
                              "scene is '<scene-id>/<room-id>'."))
    parser.add_argument('--backbone', choices=('dinov2', 'dinov3'), default='dinov3',
                        help='Backbone family (default: dinov3).')
    parser.add_argument('--model',
                        help='Hugging Face model ID or local pretrained directory.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Panoramas per forward (default: 4).')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Panorama decode/preprocess workers (default: 8).')
    parser.add_argument(
        '--include-cls-token',
        action=argparse.BooleanOptionalAction, default=None,
        help='Prepend CLS to patch_features (default: enabled for DINOv2).',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.num_workers < 0:
        parser.error('--num-workers cannot be negative')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('require 0 <= --shard-index < --num-shards')
    if args.scannetpp:
        if args.dataset_layout not in ('generic', 'scannetpp'):
            parser.error('--scannetpp conflicts with --dataset-layout')
        args.dataset_layout = 'scannetpp'
    args.scannetpp = args.dataset_layout == 'scannetpp'
    if args.split_list is not None and args.dataset_layout == 'generic':
        parser.error('--split-list requires an explicit --dataset-layout')
    args.model = args.model or {
        'dinov2': 'facebook/dinov2-large',
        'dinov3': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    }[args.backbone]
    if args.include_cls_token is None:
        args.include_cls_token = args.backbone == 'dinov2'
    args.resolution = 518 if args.backbone == 'dinov2' else RESOLUTION
    args.device = 'cuda'
    args.inference_dtype = INFERENCE_DTYPE
    args.save_dtype = SAVE_DTYPE
    args.prefetch_factor = 2
    args.pin_memory = True
    args.revision = 'main'
    args.cache_dir = None
    args.local_files_only = False
    return args


def load_model(args, device):
    # Transformers 5 imports scikit-learn's generation helpers even for vision
    # models. Some environments contain an optional pyarrow wheel built for a
    # newer NumPy, which then prevents AutoModel from importing. Feature
    # extraction does not use pyarrow, so treat that broken optional install as
    # unavailable without changing the environment's NumPy stack.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        for module_name in tuple(sys.modules):
            if module_name == 'pyarrow' or module_name.startswith('pyarrow.'):
                sys.modules.pop(module_name, None)
        sys.modules['pyarrow'] = None
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(f'Could not import Transformers vision support: {exc}') from exc
    options = dict(
        revision=getattr(args, 'revision', 'main'),
        local_files_only=getattr(args, 'local_files_only', False),
    )
    cache_dir = getattr(args, 'cache_dir', None)
    if cache_dir is not None:
        options['cache_dir'] = str(cache_dir)
    # Hugging Face automatically uses HF_TOKEN or the token saved by hf auth login.
    processor = AutoImageProcessor.from_pretrained(args.model, **options)
    model = AutoModel.from_pretrained(args.model, **options)
    expected_type = {'dinov2': 'dinov2', 'dinov3': 'dinov3_vit'}[args.backbone]
    expected_patch_size = 14 if args.backbone == 'dinov2' else 16
    if model.config.model_type != expected_type:
        raise ValueError(f'--model must match --backbone {args.backbone}')
    if model.config.patch_size != expected_patch_size:
        raise ValueError(f'Expected patch size {expected_patch_size} for {args.backbone}')
    if args.backbone == 'dinov2' and model.config.hidden_size != 1024:
        raise ValueError('Expected a DINOv2-L backbone with width 1024')
    return model.to(device=device, dtype=torch.float32).eval(), processor


@torch.inference_mode()
def extract_features(model, images, device, batch_size, inference_dtype='float32'):
    patches, cls = [], []
    patch_start = 1 + getattr(model.config, 'num_register_tokens', 0)
    for chunk in images.split(batch_size):
        with torch.autocast(
            device_type=device.type,
            dtype=getattr(torch, inference_dtype),
            enabled=inference_dtype != 'float32',
        ):
            features = model(
                pixel_values=chunk.to(device, non_blocking=True), return_dict=True
            ).last_hidden_state
        patches.append(features[:, patch_start:].detach().cpu())
        cls.append(features[:, 0].detach().cpu())
    return torch.cat(patches), torch.cat(cls)


def build_jobs(args):
    if not args.panorama_root.is_dir():
        raise FileNotFoundError(args.panorama_root)
    if args.dataset_layout == 'scannetpp':
        selected = None
        if args.split_list is not None:
            selected = {
                (scene, f'{int(frame):05d}')
                for scene, frame in (
                    sample.split() for sample in json.loads(args.split_list.read_text())
                )
            }
        images = sorted(
            path for path in args.panorama_root.glob('*/panocam/images/*.*')
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            and path.stem.isdigit()
        )
        jobs = []
        for path in images:
            scene = path.relative_to(args.panorama_root).parts[0]
            frame = f'{int(path.stem):05d}'
            if selected is None or (scene, frame) in selected:
                jobs.append((path, args.output_root / scene / f'{frame}.pt'))
    elif args.dataset_layout == 'sage10k':
        selected = None
        if args.split_list is not None:
            selected = {
                (scene, int(frame))
                for scene, frame in (
                    sample.split() for sample in json.loads(args.split_list.read_text())
                )
            }
        jobs = []
        for path in sorted(args.panorama_root.glob('*/*_panorama_*.*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            scene = path.relative_to(args.panorama_root).parts[0]
            marker = '_panorama_'
            frame_text = path.stem.rsplit(marker, 1)[-1]
            if marker not in path.stem or not frame_text.isdigit():
                continue
            frame = int(frame_text)
            if selected is not None and (scene, frame) not in selected:
                continue
            jobs.append(
                (path, args.output_root / scene / f'{path.stem}.pt')
            )
    elif args.dataset_layout == '3dfront':
        selected = None
        if args.split_list is not None:
            selected = {
                (sample.room, sample.frame)
                for sample in map(
                    parse_panorama_sample,
                    json.loads(args.split_list.read_text()),
                )
            }
        jobs = []
        for path in sorted(args.panorama_root.glob('*/*/*_panorama_*.png')):
            # The current room renderer writes metadata.json. Retain support for
            # archives produced by the older renderer, which used .complete.json.
            room_complete = any(
                (path.parent / marker).is_file()
                for marker in ('metadata.json', '.complete.json')
            )
            if not path.is_file() or not room_complete:
                continue
            relative_room = path.parent.relative_to(args.panorama_root).as_posix()
            marker = '_panorama_'
            frame_text = path.stem.rsplit(marker, 1)[-1]
            if marker not in path.stem or not frame_text.isdigit():
                continue
            frame = int(frame_text)
            if selected is not None and (relative_room, frame) not in selected:
                continue
            jobs.append(
                (path, args.output_root / path.relative_to(args.panorama_root).parent
                 / f'{frame:05d}.pt')
            )
    else:
        images = sorted(
            path for path in args.panorama_root.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        jobs = [
            (path, args.output_root / path.relative_to(args.panorama_root).parent
             / f'{path.stem}.pt')
            for path in images
        ]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    return jobs[args.shard_index::args.num_shards]


def main():
    args = parse_args()
    jobs = build_jobs(args)
    pending = [(p, out) for p, out in jobs if args.overwrite or not out.is_file()]
    print(f'{len(jobs)} panoramas selected; {len(pending)} to extract', flush=True)
    if not pending:
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    if args.inference_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA device does not support bfloat16')
    model, processor = load_model(args, device)
    process_panoramas(args, pending, model, processor, device)


def process_panoramas(args, pending, model, processor, device):
    """Batch panoramas and save one independent CPU feature file per image."""
    dtype = getattr(torch, args.save_dtype)
    patch_size = model.config.patch_size
    grid = [args.resolution // patch_size, args.resolution * 2 // patch_size]
    loader_options = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory and device.type == 'cuda',
    }
    if args.num_workers:
        loader_options.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            multiprocessing_context='spawn',
        )
    loader = DataLoader(
        PanoramaPreprocessDataset(
            pending, args.resolution, processor,
            scannetpp=args.scannetpp,
        ),
        **loader_options,
    )
    print(
        f'Panorama loader: {args.num_workers} workers, prefetch factor '
        f'{args.prefetch_factor if args.num_workers else 0}, '
        f'pin_memory={loader_options["pin_memory"]}',
        flush=True,
    )
    progress = tqdm(total=len(pending), desc=f'Extracting {args.backbone}', unit='panorama')
    try:
        for indices, images, original_sizes, bottom_paddings, masked_pixel_counts in loader:
            patches, cls = extract_features(
                model, images, device, batch_size=len(images),
                inference_dtype=args.inference_dtype,
            )
            if patches.ndim != 3 or patches.shape[1] != grid[0] * grid[1]:
                raise RuntimeError(f'Unexpected patch feature shape: {tuple(patches.shape)}')
            for batch_index, (index, original_size, bottom_padding, masked_pixels) in enumerate(
                zip(
                    indices.tolist(), original_sizes.tolist(), bottom_paddings.tolist(),
                    masked_pixel_counts.tolist(),
                )
            ):
                path, output = pending[index]
                mask_path = path.parents[1] / 'anon_mask' / f'{path.stem}.png'
                patch_features = patches[batch_index]
                if args.include_cls_token:
                    patch_features = torch.cat(
                        (cls[batch_index].unsqueeze(0), patch_features), dim=0
                    )
                result = {
                    'patch_features': patch_features.to(dtype).clone(),
                    'cls_features': cls[batch_index].to(dtype).clone(),
                    'grid_size': grid,
                    'patch_size': patch_size,
                    'includes_cls_token': args.include_cls_token,
                    'projection': 'equirectangular',
                    'dataset_layout': args.dataset_layout,
                    'original_size': original_size,
                    'padded_size': [original_size[0] + bottom_padding, original_size[1]],
                    'bottom_padding': bottom_padding,
                    'anonymization_mask': str(mask_path.resolve()) if args.scannetpp else None,
                    'masked_pixels': masked_pixels,
                    'masked_color_rgb': [0, 0, 0] if args.scannetpp else None,
                    'preprocessing': (
                        'apply_anon_mask_then_pad_bottom_to_2to1_then_resize'
                        if args.scannetpp
                        else 'resize'
                    ),
                    'input_size': [args.resolution, args.resolution * 2],
                    'normalization_mean': list(processor.image_mean),
                    'normalization_std': list(processor.image_std),
                    'source': str(path.resolve()),
                    'model': args.model,
                    'revision': args.revision,
                    'model_commit': getattr(model.config, '_commit_hash', None),
                    'inference_dtype': args.inference_dtype,
                }
                if args.include_cls_token:
                    result['cls_token_index'] = 0
                output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.tmp', delete=False) as f:
                    temporary = Path(f.name)
                try:
                    torch.save(result, temporary)
                    temporary.replace(output)
                finally:
                    temporary.unlink(missing_ok=True)
                progress.update(1)
                progress.set_postfix_str(str(output), refresh=False)
            del images, patches, cls
    finally:
        progress.close()


if __name__ == '__main__':
    main()
