#!/usr/bin/env python3
"""Compare dense DINOv3 features for a panorama and its cubemap using shared PCA."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from tools.cubemap_to_dinov3_features import FACE_ORDER, face_paths, load_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panorama', type=Path, required=True)
    parser.add_argument('--cubemap-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--face-resolution', type=int, default=512)
    parser.add_argument('--panorama-height', type=int, default=512)
    parser.add_argument('--face-order', nargs=6, type=int, default=list(FACE_ORDER))
    args = parser.parse_args()
    if args.face_resolution % 16 or args.panorama_height % 16:
        parser.error('resolutions must be divisible by 16')
    if sorted(args.face_order) != list(range(6)):
        parser.error('--face-order must be a permutation of 0 1 2 3 4 5')
    return args


def process_images(processor, images, height, width):
    return processor(
        images=images,
        return_tensors='pt',
        do_resize=True,
        size={'height': height, 'width': width},
        do_center_crop=False,
    )['pixel_values']


@torch.inference_mode()
def dense_features(model, pixels, device):
    hidden = model(pixel_values=pixels.to(device), return_dict=True).last_hidden_state
    start = 1 + model.config.num_register_tokens
    return hidden[:, start:].detach().float().cpu()


def shared_pca(*features):
    counts = [len(x) for x in features]
    joined = torch.cat(features)
    centered = joined - joined.mean(0, keepdim=True)
    # Feature dimension is modest; SVD is deterministic and avoids sklearn.
    _, _, axes = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:3].T
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    return projected.split(counts)


def color_image(values, grid, output_size):
    tensor = values.T.reshape(1, 3, *grid)
    tensor = F.interpolate(tensor, size=output_size, mode='bilinear', align_corners=False)[0]
    return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def labeled_panel(image, label, width):
    image = image.copy()
    image.thumbnail((width, width), Image.Resampling.LANCZOS)
    panel = Image.new('RGB', (width, image.height + 28), 'white')
    panel.paste(image, ((width - image.width) // 2, 28))
    ImageDraw.Draw(panel).text((6, 7), label, fill='black')
    return panel


def row(images, gap=8):
    canvas = Image.new('RGB', (sum(x.width for x in images) + gap * (len(images) - 1),
                               max(x.height for x in images)), 'white')
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, processor = load_model(args, device)
    with Image.open(args.panorama) as image:
        panorama = image.convert('RGB')
    paths = face_paths(args.cubemap_dir, args.face_order)
    faces = []
    for path in paths:
        with Image.open(path) as image:
            faces.append(image.convert('RGB'))

    pano_width = args.panorama_height * 2
    pano_pixels = process_images(processor, [panorama], args.panorama_height, pano_width)
    cube_pixels = process_images(processor, faces, args.face_resolution, args.face_resolution)
    pano_features = dense_features(model, pano_pixels, device)[0]
    cube_features = dense_features(model, cube_pixels, device)
    pano_rgb, cube_rgb = shared_pca(pano_features, cube_features.flatten(0, 1))

    pano_grid = (args.panorama_height // 16, pano_width // 16)
    face_grid = (args.face_resolution // 16, args.face_resolution // 16)
    pano_map = color_image(pano_rgb, pano_grid, (args.panorama_height, pano_width))
    cube_rgb = cube_rgb.reshape(6, -1, 3)
    cube_maps = [color_image(x, face_grid, (args.face_resolution, args.face_resolution))
                 for x in cube_rgb]

    width = 256
    output_rows = [
        row([labeled_panel(panorama, 'Equirectangular RGB', width * 2),
             labeled_panel(pano_map, 'Equirectangular DINOv3 PCA', width * 2)]),
        row([labeled_panel(image, f'Face {index} RGB', width)
             for index, image in zip(args.face_order, faces)]),
        row([labeled_panel(image, f'Face {index} DINOv3 PCA', width)
             for index, image in zip(args.face_order, cube_maps)]),
    ]
    canvas = Image.new('RGB', (max(x.width for x in output_rows),
                               sum(x.height for x in output_rows) + 16), 'white')
    y = 0
    for output_row in output_rows:
        canvas.paste(output_row, (0, y))
        y += output_row.height + 8
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f'Saved shared-PCA comparison to {args.output}')


if __name__ == '__main__':
    main()
