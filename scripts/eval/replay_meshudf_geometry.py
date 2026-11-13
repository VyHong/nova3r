"""Evaluate geometry-only variants from a trusted local diagnostic cache."""
import argparse
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT)]
from training.lightning_data import Nova3RDataModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.mesh2sdf.module_options import MeshOptions, ValidationOptions, EvaluationOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache_directory', type=Path)
    parser.add_argument('--variant', required=True, help='Output variant name')
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--reference-resolution', type=int)
    parser.add_argument('--surface-threshold', type=float, default=.01)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--stop-keepalive-pid', type=int)
    args = parser.parse_args()
    folder = args.cache_directory.resolve()
    b = torch.load(folder / 'geometry_cache.pt', map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(b['cfg'])
    cfg.data.test_datasets = {'scannetpp': cfg.data.test_datasets.scannetpp}
    resolution = args.resolution or (512 if args.variant == 'reference512' else 1024)
    reference_resolution = args.reference_resolution
    if args.resolution is None and args.variant == 'matched_cutoff1024':
        reference_resolution = 512
    cfg.update(mesh_mc=resolution,
               mesh_bounds_padding=.05, diagnostic_fp32=False,
               mesh_udf_normal_band_voxels=2.,
               mesh_udf_reference_resolution=reference_resolution,
               mesh_udf_surface_threshold=args.surface_threshold,
               mesh_udf_query_batch_size=262144,
               mesh_udf_vertex_batch_size=131072,
               mesh_udf_fast_cleanup=True,
               mesh_udf_streamed_grid=True, smooth_borders=False)
    out = args.output_dir.resolve() if args.output_dir else folder / args.variant
    out.mkdir(parents=True, exist_ok=True)
    scene, frame = folder.name.removeprefix('scannetpp_').rsplit('_', 2)[:2]
    test_list = out / 'test_list.json'
    test_list.write_text(json.dumps([f'{scene} {frame}']) + '\n')
    cfg.data.test_datasets.scannetpp.params.list_path = str(test_list)
    (out / 'config.json').write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)+'\n')
    dm = Nova3RDataModule(cfg.data)
    dm.setup('test')
    batch = next(iter(dm.test_dataloader()))
    def device(value):
        if isinstance(value, torch.Tensor): return value.cuda()
        if isinstance(value, dict): return {k:device(v) for k,v in value.items()}
        if isinstance(value, list): return [device(v) for v in value]
        if isinstance(value, tuple): return tuple(device(v) for v in value)
        return value
    batch = device(batch)
    module = Mesh2SDFLightningModule.__new__(Mesh2SDFLightningModule)
    pl.LightningModule.__init__(module)
    module.cfg = cfg
    module.model = torch.nn.Module()
    module.model.geo_decoder = b['geo_decoder'].cuda().float()
    latents = b['latents'].cuda()
    module.model.lightning_forward = lambda *a, **kw: (None, None, latents)
    module.first_stage = None
    module.val_criterion = None
    module.validation_options = ValidationOptions.from_config(cfg)
    module.mesh_options = MeshOptions.from_config(cfg)
    module.evaluation_options = EvaluationOptions.from_config(cfg)
    module._validation_lpips_evaluator = None
    module.test_results = []
    module._trainer = SimpleNamespace(world_size=1, global_rank=0, is_global_zero=True,
                                     logger=SimpleNamespace(log_dir=str(out), save_dir=str(out)))
    module.cuda().eval()
    print('Geometry replay model loaded', flush=True)
    if args.stop_keepalive_pid:
        cmdline = Path(f'/proc/{args.stop_keepalive_pid}/cmdline').read_bytes()
        if b'gpu_keepalive.py' not in cmdline:
            raise RuntimeError('Refusing to stop a process that is not GPU keepalive')
        os.kill(args.stop_keepalive_pid, signal.SIGTERM)
        for _ in range(100):
            stat = Path(f'/proc/{args.stop_keepalive_pid}/stat')
            if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Keepalive did not stop after SIGTERM')
        print(f'Stopped GPU keepalive PID {args.stop_keepalive_pid}', flush=True)
    factor = min(1., (reference_resolution-1)/(resolution-1)) if reference_resolution else 1.
    torch.set_float32_matmul_precision('medium')
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = module.test_step(batch, 2)
    record = dict(variant=args.variant, resolution=resolution,
                  reference_resolution=reference_resolution,
                  surface_threshold=args.surface_threshold,
                  query_batch_size=262144, vertex_batch_size=131072,
                  fast_cleanup=True, precision='bfloat16 autocast',
                  seconds=time.perf_counter()-start, native_distance_scale=factor, results=result)
    (out / 'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    if any(r.get('status') != 'ok' for r in result): raise RuntimeError('Geometry replay failed')


if __name__ == '__main__': main()
