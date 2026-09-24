#!/usr/bin/env python3
"""Extract DINOv3 ViT features from cubemaps or equirectangular panoramas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

FACE_ORDER = (4, 0, 5, 1, 3, 2)
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}


class PanoramaPreprocessDataset(Dataset):
    """Decode and preprocess panoramas inside DataLoader worker processes."""

    def __init__(self, jobs, resolution, processor):
        self.jobs = jobs
        self.resolution = resolution
        self.processor = processor

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, index):
        path, _ = self.jobs[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
            original_size = torch.tensor([image.height, image.width], dtype=torch.int64)
            pixels = self.processor(
                images=image,
                return_tensors='pt',
                do_resize=True,
                size={'height': self.resolution, 'width': self.resolution * 2},
                do_center_crop=False,
            )['pixel_values'][0]
        return index, pixels, original_size


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cubemap-dir', type=Path)
    source.add_argument('--dataset-root', type=Path,
                        help='Recursively discover cubemap directories under this root.')
    source.add_argument('--panorama-root', type=Path,
                        help='Recursively process panorama images under this root.')
    parser.add_argument('--output', type=Path, help='Single-cubemap .pt output.')
    parser.add_argument('--output-root', type=Path,
                        help='Dataset output root; preserves relative cubemap paths.')
    parser.add_argument('--dataset-layout', choices=('generic', 'scannetpp'),
                        default='generic',
                        help='Dataset discovery and output naming convention.')
    parser.add_argument('--split-list', type=Path,
                        help="Optional '<scene> <frame>' JSON list (ScanNet++ only).")
    parser.add_argument('--cubemap-glob', default='*cube*',
                        help='Directory glob for dataset discovery (default: *cube*).')
    parser.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m',
                        help='Hugging Face DINOv3 ViT model ID or local pretrained directory.')
    parser.add_argument('--revision', default='main', help='Hugging Face branch, tag, or commit.')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--local-files-only', action='store_true',
                        help='Load only already-cached model and processor files.')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Cubemap face size, or panorama height; panorama width is twice this.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Items per forward; each cubemap item contains six face images.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Panorama decode/preprocess workers (default: 8).')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Batches prefetched by each panorama worker (default: 2).')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Pin prefetched image batches for asynchronous CUDA copies.')
    parser.add_argument('--face-order', nargs=6, type=int, default=list(FACE_ORDER))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inference-dtype', choices=('float32', 'bfloat16', 'float16'),
                        default='float32',
                        help='CUDA autocast dtype for model inference (default: float32).')
    parser.add_argument('--save-dtype', choices=('float32', 'float16'), default='float32')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.resolution <= 0 or args.resolution % 16:
        parser.error('--resolution must be a positive multiple of 16')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.num_workers < 0:
        parser.error('--num-workers cannot be negative')
    if args.prefetch_factor < 1:
        parser.error('--prefetch-factor must be positive')
    if sorted(args.face_order) != list(range(6)):
        parser.error('--face-order must be a permutation of 0 1 2 3 4 5')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('require 0 <= --shard-index < --num-shards')
    if args.cubemap_dir and (not args.output or args.output_root):
        parser.error('--cubemap-dir requires --output and excludes --output-root')
    if (args.dataset_root or args.panorama_root) and (not args.output_root or args.output):
        parser.error('dataset modes require --output-root and exclude --output')
    return args


def face_paths(directory, order):
    by_index = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.isdigit():
            index = int(path.stem)
            if index in by_index:
                raise ValueError(f'Duplicate face {index} in {directory}')
            by_index[index] = path
    missing = set(order) - by_index.keys()
    if missing:
        raise ValueError(f'Missing faces {sorted(missing)} in {directory}')
    return [by_index[index] for index in order]


def load_faces(paths, resolution, processor):
    images, sizes = [], []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
            sizes.append([image.height, image.width])
    # Override the default size to preserve full-face coverage at our resolution.
    inputs = processor(images=images, return_tensors='pt', do_resize=True,
                       size={'height': resolution, 'width': resolution},
                       do_center_crop=False)
    return inputs['pixel_values'], sizes


def load_panoramas(paths, resolution, processor):
    images, sizes = [], []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
            sizes.append([image.height, image.width])
    inputs = processor(images=images, return_tensors='pt', do_resize=True,
                       size={'height': resolution, 'width': resolution * 2},
                       do_center_crop=False)
    return inputs['pixel_values'], sizes


def load_model(args, device):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError('Install DINOv3 support: pip install "transformers>=4.56.0" torchvision') from exc
    options = dict(revision=args.revision, local_files_only=args.local_files_only)
    if args.cache_dir is not None:
        options['cache_dir'] = str(args.cache_dir)
    # Hugging Face automatically uses HF_TOKEN or the token saved by hf auth login.
    processor = AutoImageProcessor.from_pretrained(args.model, **options)
    model = AutoModel.from_pretrained(args.model, **options)
    if model.config.model_type != 'dinov3_vit':
        raise ValueError('--model must be a DINOv3 ViT backbone')
    if model.config.patch_size != 16:
        raise ValueError('Expected a DINOv3 ViT backbone with patch size 16')
    return model.to(device=device, dtype=torch.float32).eval(), processor


@torch.inference_mode()
def extract_features(model, images, device, batch_size, inference_dtype='float32'):
    patches, cls = [], []
    patch_start = 1 + model.config.num_register_tokens
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


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_jobs(args):
    if args.cubemap_dir:
        jobs = [(args.cubemap_dir, args.output)]
    elif args.dataset_root:
        if not args.dataset_root.is_dir():
            raise FileNotFoundError(args.dataset_root)
        if args.dataset_layout == 'scannetpp':
            selected = None
            if args.split_list is not None:
                selected = {
                    (scene, f'{int(frame):05d}')
                    for scene, frame in (
                        sample.split()
                        for sample in json.loads(args.split_list.read_text())
                    )
                }
            directories = []
            for directory in sorted(args.dataset_root.glob('*/panocam/images/*_cube_95')):
                scene = directory.parts[-4]
                frame = f'{int(directory.name.removesuffix("_cube_95")):05d}'
                if selected is None or (scene, frame) in selected:
                    directories.append((directory, scene, frame))
            jobs = [
                (directory, args.output_root / scene / f'{frame}.pt')
                for directory, scene, frame in directories
            ]
        else:
            directories = sorted(p for p in args.dataset_root.rglob(args.cubemap_glob)
                                 if p.is_dir() and any(
                                     f.is_file() and f.stem.isdigit() and
                                     f.suffix.lower() in IMAGE_SUFFIXES for f in p.iterdir()))
            jobs = [(p, args.output_root / p.relative_to(args.dataset_root) / 'dinov3_features.pt')
                    for p in directories]
    else:
        if not args.panorama_root.is_dir():
            raise FileNotFoundError(args.panorama_root)
        if args.dataset_layout == 'scannetpp':
            selected = None
            if args.split_list is not None:
                selected = {
                    (scene, f'{int(frame):05d}')
                    for scene, frame in (
                        sample.split()
                        for sample in json.loads(args.split_list.read_text())
                    )
                }
            images = sorted(
                p for p in args.panorama_root.glob('*/panocam/images/*.*')
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p.stem.isdigit()
            )
            jobs = []
            for path in images:
                scene = path.relative_to(args.panorama_root).parts[0]
                frame = f'{int(path.stem):05d}'
                if selected is None or (scene, frame) in selected:
                    jobs.append((path, args.output_root / scene / f'{frame}.pt'))
        else:
            images = sorted(p for p in args.panorama_root.rglob('*')
                            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            jobs = [
                (p, args.output_root / p.relative_to(args.panorama_root).parent / f'{p.stem}.pt')
                for p in images
            ]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    return jobs[args.shard_index::args.num_shards]


def main():
    args = parse_args()
    jobs = build_jobs(args)
    pending = [(p, out) for p, out in jobs if args.overwrite or not out.is_file()]
    kind = 'panoramas' if args.panorama_root else 'cubemaps'
    print(f'{len(jobs)} {kind} selected; {len(pending)} to extract', flush=True)
    if not pending:
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    if args.inference_dtype != 'float32' and device.type != 'cuda':
        raise ValueError('--inference-dtype bfloat16/float16 currently requires CUDA')
    if args.inference_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA device does not support bfloat16; use float16 or float32')
    model, processor = load_model(args, device)
    if args.panorama_root:
        process_panoramas(args, pending, model, processor, device)
        return

    completed = 0
    for job_batch in batched(pending, args.batch_size):
        loaded = []
        for directory, output in job_batch:
            paths = face_paths(directory, args.face_order)
            images, original_sizes = load_faces(paths, args.resolution, processor)
            loaded.append((directory, output, paths, original_sizes, images))

        # One forward processes all six faces from every cubemap in this batch.
        image_batch = torch.cat([item[4] for item in loaded], dim=0)
        patches, cls = extract_features(
            model, image_batch, device, batch_size=len(image_batch),
            inference_dtype=args.inference_dtype,
        )
        patches = patches.reshape(len(loaded), 6, *patches.shape[1:])
        cls = cls.reshape(len(loaded), 6, *cls.shape[1:])
        grid = args.resolution // 16
        if patches.ndim != 4 or patches.shape[1:3] != (6, grid * grid):
            raise RuntimeError(f'Unexpected patch feature shape: {tuple(patches.shape)}')
        dtype = getattr(torch, args.save_dtype)

        for batch_index, (directory, output, paths, original_sizes, _) in enumerate(loaded):
            # Indexing keeps a view, so clone each CPU tensor to give every saved
            # cubemap independent storage and release the full batch afterward.
            cubemap_patches = patches[batch_index].to(dtype).clone()
            cubemap_cls = cls[batch_index].to(dtype).clone()
            result = {
                'patch_features': cubemap_patches,
                'cls_features': cubemap_cls,
                'grid_size': [grid, grid],
                'patch_size': 16,
                'face_order': list(args.face_order),
                'face_names': [p.name for p in paths],
                'original_sizes': original_sizes,
                'input_size': [args.resolution, args.resolution],
                'normalization_mean': list(processor.image_mean),
                'normalization_std': list(processor.image_std),
                'source': str(directory.resolve()),
                'model': args.model,
                'revision': args.revision,
                'model_commit': getattr(model.config, '_commit_hash', None),
                'inference_dtype': args.inference_dtype,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            # Publish only complete files, so interrupted exports can be resumed.
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.tmp', delete=False) as f:
                temporary = Path(f.name)
            try:
                torch.save(result, temporary)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            completed += 1
            print(f'[{completed}/{len(pending)}] Saved {tuple(cubemap_patches.shape)} to {output}', flush=True)

        del image_batch, patches, cls, loaded


def process_panoramas(args, pending, model, processor, device):
    """Batch panoramas and save one independent CPU feature file per image."""
    completed = 0
    dtype = getattr(torch, args.save_dtype)
    grid = [args.resolution // 16, args.resolution * 2 // 16]
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
        PanoramaPreprocessDataset(pending, args.resolution, processor),
        **loader_options,
    )
    print(
        f'Panorama loader: {args.num_workers} workers, prefetch factor '
        f'{args.prefetch_factor if args.num_workers else 0}, '
        f'pin_memory={loader_options["pin_memory"]}',
        flush=True,
    )
    for indices, images, original_sizes in loader:
        patches, cls = extract_features(
            model, images, device, batch_size=len(images),
            inference_dtype=args.inference_dtype,
        )
        if patches.ndim != 3 or patches.shape[1] != grid[0] * grid[1]:
            raise RuntimeError(f'Unexpected patch feature shape: {tuple(patches.shape)}')
        for batch_index, (index, original_size) in enumerate(
            zip(indices.tolist(), original_sizes.tolist())
        ):
            path, output = pending[index]
            result = {
                'patch_features': patches[batch_index].to(dtype).clone(),
                'cls_features': cls[batch_index].to(dtype).clone(),
                'grid_size': grid,
                'patch_size': 16,
                'projection': 'equirectangular',
                'original_size': original_size,
                'input_size': [args.resolution, args.resolution * 2],
                'normalization_mean': list(processor.image_mean),
                'normalization_std': list(processor.image_std),
                'source': str(path.resolve()),
                'model': args.model,
                'revision': args.revision,
                'model_commit': getattr(model.config, '_commit_hash', None),
                'inference_dtype': args.inference_dtype,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.tmp', delete=False) as f:
                temporary = Path(f.name)
            try:
                torch.save(result, temporary)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            completed += 1
            print(f'[{completed}/{len(pending)}] Saved {tuple(result["patch_features"].shape)} to {output}', flush=True)
        del images, patches, cls


if __name__ == '__main__':
    main()
