#!/usr/bin/env python3
"""Predict a scene mesh from an ERP panorama using the epoch 9 panorama model."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOCAL_DEPS = ROOT / '.deps/panorama-inference'
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path, nargs='?', default=ROOT / 'archive/overview.png')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'exp_output/hunyuandit_panorama/checkpoints/epoch=09-val_loss=1.4962.ckpt')
    p.add_argument('--config', type=Path, default=ROOT / 'checkpoints/hunyuan_dit/config_hunyuandit_panorama_visible_latent.yaml')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'demo/outputs/panorama_epoch09_meshudf768_padding15')
    p.add_argument('--dino-model', default='facebook/dinov2-large')
    p.add_argument('--pager-checkpoint', default='pager')
    p.add_argument('--pager-repo', type=Path, default=ROOT / 'third_party/PaGeR')
    p.add_argument('--pager-scene-mode', choices=['indoor', 'outdoor'], default='indoor')
    p.add_argument('--pager-max-depth', type=float, default=75.0)
    p.add_argument('--pager-sky-threshold', type=float, default=0.5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--guidance-scale', type=float, default=5.0)
    p.add_argument('--mesh-resolution', type=int, default=768)
    p.add_argument('--bounds-padding', type=float, default=0.15, help='Padding on each side as a fraction of each axis extent')
    p.add_argument('--bounds-outlier-neighbors', type=int, default=20, help='Nearest neighbours for statistical bounds filtering; 0 disables filtering')
    p.add_argument('--bounds-outlier-std-ratio', type=float, default=2.0)
    p.add_argument('--mesh-method', choices=['latent2mesh', 'mesh_udf'], default='mesh_udf')
    args = p.parse_args()
    for key in ('image', 'checkpoint', 'config'):
        path = getattr(args, key).resolve()
        if not path.is_file():
            p.error(f'{key} does not exist: {path}')
        setattr(args, key, path)
    if args.steps < 1 or args.mesh_resolution < 2:
        p.error('steps must be positive and mesh resolution >= 2')
    if args.bounds_padding < 0:
        p.error('bounds padding must be nonnegative')
    import math
    if args.bounds_outlier_neighbors < 0 or not math.isfinite(args.bounds_outlier_std_ratio) or args.bounds_outlier_std_ratio <= 0:
        p.error('outlier neighbours must be nonnegative and std ratio finite and positive')
    return args


def filter_bounds_points(points, neighbors=20, std_ratio=2.0):
    """Reject isolated points before fitting bounds, without changing conditioning."""
    import numpy as np
    from scipy.spatial import cKDTree

    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    before = len(points)
    if neighbors and before > 2:
        k = min(neighbors, before - 1)
        distances = cKDTree(points).query(points, k=k + 1)[0][:, 1:].mean(axis=1)
        limit = distances.mean() + std_ratio * distances.std()
        points = points[distances <= limit]
    if len(points) < 3 or np.any(np.ptp(points, axis=0) <= 0):
        raise RuntimeError('Insufficient nondegenerate points remain for meshing bounds')
    return points, {'finite_points': before, 'retained_points': len(points), 'removed_points': before - len(points)}


def load_rgb_image(path):
    """Use OpenCV when Pillow recognizes an image but lacks its decoder."""
    from PIL import Image
    try:
        with Image.open(path) as opened:
            return opened.convert('RGB')
    except (OSError, RuntimeError) as exc:
        import cv2
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is None:
            raise RuntimeError(f'Neither Pillow nor OpenCV could decode {path}') from exc
        print(f'Using OpenCV image decoder for {path}: {exc}', flush=True)
        return Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))


def main():
    args = parse_args()
    import os
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    # Config component paths are relative to the repository, even when invoked elsewhere.
    os.chdir(ROOT)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run this script in the nova3r environment on a GPU node.')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    torch.manual_seed(args.seed)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def release():
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    from scripts.preprocessing.features.panorama_to_dinov3_features import load_model as load_dino, extract_features
    print('Extracting panorama DINOv2 features...', flush=True)
    dino, processor = load_dino(SimpleNamespace(model=args.dino_model, backbone='dinov2'), device)
    with load_rgb_image(args.image) as image:
        # Preserve all source pixels. For odd widths, add one column on the right
        # so an integer-height canvas can have an exact 2:1 aspect ratio.
        source_size = image.size
        canvas_width = image.width + image.width % 2
        canvas_height = canvas_width // 2
        if image.height > canvas_height:
            raise ValueError('Input is taller than a 2:1 panorama; bottom padding cannot restore it')
        right_padding = canvas_width - image.width
        bottom_padding = canvas_height - image.height
        image = ImageOps.expand(image, border=(0, 0, right_padding, bottom_padding), fill=0)
        print(f'Padded {source_size} to {image.size}: bottom={bottom_padding}, right={right_padding}; no source pixels resized.', flush=True)
        image.save(out / 'panorama.png')
        pixels = processor(images=image, return_tensors='pt', do_resize=True,
                           size={'height': 518, 'width': 1036}, do_center_crop=False)['pixel_values']
    patches, cls = extract_features(dino, pixels, device, 1, 'bfloat16' if device.type == 'cuda' else 'float32')
    tokens = torch.cat((cls[:, None], patches), dim=1).to(device=device, dtype=dtype)
    del dino, processor, pixels
    release()

    from eval.baselines.adapters.pager import PaGeRAdapter, PAGER_TO_OPENCV
    # PaGeR loads its own DA3 fork under the same top-level package name.
    # Restore our imports after its inference so the scene model uses this repo.
    original_depth_modules = {name: module for name, module in sys.modules.items()
                              if name == 'depth_anything_3' or name.startswith('depth_anything_3.')}
    print('Predicting panorama geometry with PaGeR...', flush=True)
    pager = PaGeRAdapter(args.pager_repo, args.pager_checkpoint, str(device), out,
                        args.pager_scene_mode, args.pager_max_depth, args.pager_sky_threshold)
    pointmap, valid = pager.predict_pointmap(out / 'panorama.png')
    # Match the panorama OpenCV frame used by the existing cubemap preprocessing.
    pointmap = pointmap @ PAGER_TO_OPENCV.T
    height, width = valid.shape
    # Exclude black canvas padding from geometry, even after PaGeR resizing.
    valid[int(np.floor(source_size[1] * height / image.height)):, :] = False
    valid[:, int(np.floor(source_size[0] * width / image.width)):] = False
    pointmap[~valid] = np.nan
    np.savez_compressed(out / 'pager_geometry.npz', points=pointmap, valid=valid)
    pointmap = torch.from_numpy(pointmap).reshape(1, -1, 3).to(device)
    del pager
    for name in tuple(sys.modules):
        if name == 'depth_anything_3' or name.startswith('depth_anything_3.'):
            del sys.modules[name]
    sys.modules.update(original_depth_modules)
    release()
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    vae_path = Path(config.experiment.vae_ckpt)
    print(f'Encoding visible geometry with {vae_path}...', flush=True)
    from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE
    from training.mesh2sdf.shape_vae import checkpoint_state, select_vae_state
    vae = ShapeVAE(**OmegaConf.to_container(config.experiment.vae.params, resolve=True))
    vae.load_state_dict(select_vae_state(checkpoint_state(vae_path), vae), strict=True)
    vae = vae.to(device=device, dtype=dtype).eval()
    normalization = str(config.experiment.get('surface_geo_normalize_mode', 'none'))
    from nova3r.utils.depth_surface import sample_depth_pointmap_for_vae
    with torch.no_grad():
        surface = sample_depth_pointmap_for_vae(
            pointmaps=pointmap, confidence=None, num_views=1, height=height, width=width,
            random_count=vae.encoder.pc_size, sharp_count=vae.encoder.pc_sharpedge_size,
            point_feats=vae.encoder.point_feats, sharp_threshold=0.985,
            normalization=normalization, seed=args.seed).to(device=device, dtype=dtype)
        devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed)
            visible = vae.encode(surface, sample_posterior=True)
    del vae, pointmap
    release()

    from scripts.inference.demo_nova3r import load_model
    from nova3r.models.hy3dshape.models.denoisers.hunyuandit import HunYuanDiTVelocityWrapper
    from nova3r.flow_matching.solver import ODESolver
    print(f'Loading panorama checkpoint: {args.checkpoint}', flush=True)
    model, cfg = load_model(ckpt_path=str(args.checkpoint), config_path=str(args.config),
                            device=device, dtype=dtype, stage='test', low_cpu_mem_usage=True)
    model.eval()
    contexts = {'main': tokens, 'backbone': 2, 'grid_size': (37, 74), 'includes_cls_token': True}
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn((1, model.input_size, model.in_channels), device=device, dtype=dtype, generator=generator)
    print(f'Generating scene latents ({args.steps} steps, CFG {args.guidance_scale})...', flush=True)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
        latents = ODESolver(velocity_model=HunYuanDiTVelocityWrapper(model)).sample(
            x_init=noise, time_grid=torch.tensor([0., 1.], device=device), method='euler',
            step_size=1.0 / args.steps, return_intermediates=False, contexts=contexts,
            visible_geometry_latents=visible, guidance_scale=args.guidance_scale)
        latents = latents / model.latent_scale_factor
    torch.save({'latents': latents.cpu(), 'visible_geometry_latents': visible.cpu(),
                'surface': surface.cpu()}, out / 'latents.pt')
    del tokens, visible, contexts, noise
    release()
    from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
    cfg.mesh_method = args.mesh_method
    cfg.mesh_octree_resolution = args.mesh_resolution
    cfg.mesh_mc = args.mesh_resolution
    cfg.mesh_udf_streamed_grid = True
    cfg.mesh_udf_fast_cleanup = True
    module = Mesh2SDFLightningModule(cfg, model).to(device=device, dtype=dtype).eval()
    decoder = module._latent_decoder().float()
    print('Extracting scene mesh...', flush=True)
    with torch.no_grad():
        decoded = module._decode_latents_for_geometry(latents.float(), decoder)
        # PaGeR points share the normalized coordinates used for visible conditioning.
        filtered, bounds_filter = filter_bounds_points(
            surface[0, :, :3].detach().float().cpu().numpy(),
            args.bounds_outlier_neighbors, args.bounds_outlier_std_ratio)
        print(f'Bounds outlier filtering: {bounds_filter}', flush=True)
        points = torch.as_tensor(filtered, device=device, dtype=torch.float32)
        lower, upper = points.amin(dim=0), points.amax(dim=0)
        extent = upper - lower
        if torch.any(extent <= 0):
            raise RuntimeError('Predicted geometry has degenerate meshing bounds')
        padding = extent * args.bounds_padding
        bounds = torch.cat((lower - padding, upper + padding)).cpu().numpy()
        print(f'Predicted geometry bounds with {args.bounds_padding:.0%} padding per side: {bounds.tolist()}', flush=True)
        mesh = module.mesh_from_latents(decoded, 0, bounds, decoder, artifact_dir=out)
    if mesh is None or not len(mesh.vertices) or not len(mesh.faces):
        raise RuntimeError('Mesh extraction produced no triangles; generated latents are saved for inspection.')
    mesh.export(out / 'prediction.ply')
    mesh.export(out / 'prediction.glb')
    (out / 'run.json').write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_size': list(source_size), 'padded_size': list(image.size),
        'bottom_padding': bottom_padding, 'right_padding': right_padding,
        'normalization': normalization, 'coordinate_space': 'normalized decoder coordinates; scale is not metric',
        'meshing_bounds': bounds.tolist(), 'bounds_source': 'statistically filtered normalized PaGeR surface points',
        'bounds_outlier_filter': bounds_filter,
        'vertices': len(mesh.vertices), 'faces': len(mesh.faces)}, indent=2) + '\n')
    print(f'Saved {out / "prediction.glb"}', flush=True)


if __name__ == '__main__':
    main()
