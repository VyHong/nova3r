"""Compare saved MeshUDF meshes with 30 Taubin iterations, without re-extraction."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT)]
from training.lightning_data import Nova3RDataModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.mesh2sdf.module_options import EvaluationOptions, ValidationOptions
from eval.baselines.common import sample_mesh_surface, evaluate_point_clouds, compute_visibility_metrics
from report_meshudf_resolution import report


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--watch-pid', type=int)
    parser.add_argument('--expected-scenes', type=int, default=3)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--skip-baseline-smoothing', action='store_true')
    args = parser.parse_args()
    source_root, out = args.source_root.resolve(), args.output_root.resolve()
    if source_root == out or source_root in out.parents:
        parser.error('Output must be outside the source tree')
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config).cfg
    cfg.data.test_datasets = {'scannetpp': cfg.data.test_datasets.scannetpp}
    cfg.data.num_workers = cfg.data.test_num_workers = 0
    module = Mesh2SDFLightningModule.__new__(Mesh2SDFLightningModule)
    pl.LightningModule.__init__(module)
    module.cfg = cfg
    module.validation_options = ValidationOptions.from_config(cfg)
    opts = module.evaluation_options = EvaluationOptions.from_config(cfg)
    targets = {}

    def target_for(name):
        if name not in targets:
            scene, frame, _ = name.removeprefix('scannetpp_').rsplit('_', 2)
            split = out / f'{name}_test_list.json'
            split.write_text(json.dumps([f'{scene} {frame}'])+'\n')
            data = copy.deepcopy(cfg.data)
            data.test_datasets.scannetpp.params.list_path = str(split)
            dm = Nova3RDataModule(data)
            dm.setup('test')
            batch = next(iter(dm.test_dataloader()))
            batch = module.normalize_surface_and_geo_points(batch, module.validation_options.normalization)
            seed = module._test_sample_seed(name)
            target, visible, mesh = module._test_visibility_target(batch, 0, seed, opts.points or 100000)
            targets[name] = target, visible, mesh, seed
        return targets[name]

    def evaluate(source, variant):
        name = source['sample']
        modes = ('raw',) if args.skip_baseline_smoothing and variant == 'reference512' else ('raw', 'taubin30')
        paths = {mode: out / mode / variant / 'test/meshes' / name / 'metrics.json'
                 for mode in modes}
        if all(p.is_file() for p in paths.values()):
            return
        target, visible, gt_mesh, seed = target_for(name)
        mesh = o3d.io.read_triangle_mesh(source['mesh'])
        if mesh.is_empty() or not mesh.has_triangles():
            raise RuntimeError(f'Invalid source mesh: {source["mesh"]}')
        vertices = np.asarray(mesh.vertices).copy()
        if source.get('mesh_alignment_applied'):
            vertices = (vertices - np.asarray(source['alignment_shift'])) / source['alignment_scale']
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        for mode, path in paths.items():
            if path.is_file():
                continue
            started = time.perf_counter()
            current = mesh if mode == 'raw' else mesh.filter_smooth_taubin(
                number_of_iterations=30, lambda_filter=.5, mu=-.53)
            xyz, faces = np.asarray(current.vertices), np.asarray(current.triangles)
            pred = sample_mesh_surface(xyz, faces, opts.points or 100000, seed)
            aligned, metrics = evaluate_point_clouds(pred, target, opts.fscore_thresholds,
                'cuda:0', seed, opts.alignment_iterations, opts.alignment_lr)
            metrics.update(compute_visibility_metrics(aligned, target, visible, opts.fscore_thresholds))
            path.parent.mkdir(parents=True, exist_ok=True)
            if mode == 'taubin30':
                exported = copy.deepcopy(current)
                exported.vertices = o3d.utility.Vector3dVector(
                    xyz * metrics['alignment_scale'] + np.asarray(metrics['alignment_shift']))
                mesh_path = path.parent / 'test_mesh.ply'
                if not o3d.io.write_triangle_mesh(str(mesh_path), exported):
                    raise RuntimeError(f'Could not write {mesh_path}')
            else:
                mesh_path = Path(source['mesh'])
            record = dict(sample=name, status='ok', mesh=str(mesh_path),
                source_mesh=source['mesh'], preprocessing=mode,
                taubin_iterations=0 if mode == 'raw' else 30,
                lambda_filter=.5, mu=-.53, topology_unchanged=True,
                vertices=len(xyz), faces=len(faces), mesh_alignment_applied=True,
                visibility_mesh=str(gt_mesh), visibility_protocol='mesh_raycast_v3_directional',
                visibility_tolerance=float(cfg.get('test_visibility_tolerance', .01)),
                seconds=time.perf_counter()-started, **metrics)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2)+'\n')
            temporary.replace(path)
            print(f'{variant} {name} {mode}: '+json.dumps({k:metrics[k] for k in (
                'occluded_cd', 'occluded_apd_0.1', 'occluded_apd_0.05', 'complete_cd', 'f_score_0.05_full')}), flush=True)

    while True:
        variants = set()
        for path in sorted(source_root.rglob('metrics.json')):
            try:
                source = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if source.get('status') != 'ok':
                continue
            variant = path.relative_to(source_root).parts[0]
            if args.variants and variant not in args.variants:
                continue
            variants.add(variant)
            evaluate(source, variant)
        for variant in variants:
            if not (args.skip_baseline_smoothing and variant == 'reference512'):
                report(out / 'raw' / variant, out / 'taubin30' / variant,
                    out / f'comparison_{variant}', args.expected_scenes, variant, variant+' + Taubin30')
            if variant != 'reference512':
                for mode in ('raw', 'taubin30'):
                    report(out / 'raw/reference512', out / mode / variant,
                        out / f'vs512_{mode}_{variant}', args.expected_scenes,
                        '512 raw', variant + (' + Taubin30' if mode == 'taubin30' else ' raw'))
        if args.watch_pid is None or not alive(args.watch_pid):
            break
        time.sleep(10)
    (out / 'status.json').write_text(json.dumps(dict(state='complete', variants=sorted(variants),
        note='Compare matched completed scenes; raw controls are re-evaluated from the same serialized meshes.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
