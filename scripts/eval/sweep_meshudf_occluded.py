"""Controlled MeshUDF APD sweep. Run with the nova3r environment Python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import statistics

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / 'exp_output/hunyuandit_panorama/eval_epoch09_mesh_udf_768/nova3r_training/version_0'
ROOT = Path(os.environ.get('MESH_UDF_SEARCH_ROOT', str(PROJECT / 'exp_output/hunyuandit_panorama/meshudf_apd_search')))
# Keep checkpoint, CFG and metric protocol fixed; change extraction only.
VARIANTS = {
    'reference': (768, .05, .01),
    'padding075': (768, .075, .01),
    'padding10': (768, .10, .01),
    'r512_padding075': (512, .075, .01),
    'r1024_padding075': (1024, .075, .01),
    'threshold02_padding075': (768, .075, .02),
    'threshold005_padding075': (768, .075, .005),
}

def prepare():
    import yaml
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((BASE / 'hparams.yaml').read_text())
    full_list = json.loads((PROJECT / cfg['cfg']['data']['test_datasets']['scannetpp']['params']['list_path']).read_text())
    indices = {f"scannetpp_{item.replace(' ', '_')}_4": i for i, item in enumerate(full_list)}
    (ROOT / 'indices.json').write_text(json.dumps(indices, indent=2))
    # Uniform coverage plus difficult examples; this is a tuning split, not final validation.
    saved = json.loads((BASE / 'test/results.json').read_text())
    worst = {r['sample'] for r in sorted(saved, key=lambda r: r['occluded_apd_0.05'])[:5]}
    subset = [item for i, item in enumerate(full_list) if i % 8 == 0 or f"scannetpp_{item.replace(' ', '_')}_4" in worst]
    (ROOT / 'tuning_list.json').write_text(json.dumps(subset, indent=2))
    for name, (resolution, padding, threshold) in VARIANTS.items():
        variant = yaml.safe_load(yaml.safe_dump(cfg))
        variant['cfg'].update(mesh_method='mesh_udf', mesh_mc=resolution,
                              mesh_bounds_padding=padding, mesh_udf_surface_threshold=threshold)
        (ROOT / f'{name}.yaml').write_text(yaml.safe_dump(variant, sort_keys=False))
    print(f'Prepared {len(VARIANTS)} variants on {len(subset)} tuning panoramas', flush=True)


def worker():
    sys.path[:0] = [str(PROJECT / 'training'), str(PROJECT)]
    from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
    indices = json.loads((ROOT / 'indices.json').read_text())
    original = Mesh2SDFLightningModule.test_step
    completed = {}
    resume_dir = os.environ.get('MESH_UDF_RESUME_COMPLETED')
    if resume_dir:
        for path in sorted(Path(resume_dir).rglob('metrics.json')):
            record = json.loads(path.read_text())
            if record.get('status') == 'ok' and Path(record.get('mesh', '')).is_file():
                completed[record['sample']] = record
        print(f'Resuming with {len(completed)} completed samples; preserving original inference indices', flush=True)
    def controlled(self, batch, batch_idx):
        name = self._test_sample_name(batch, 0)
        if name in completed:
            self.test_results.append(completed[name])
            print(f'Reusing completed sample: {name}', flush=True)
            return None
        return original(self, batch, indices[name])
    Mesh2SDFLightningModule.test_step = controlled
    from training.test import main
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    main()


def report():
    lines = ['# MeshUDF occluded APD tuning', '',
             'GenRecon repaired 100-sample targets: APD10 > 61.3064%, APD5 > 35.6611%, CD < 13.3973 cm.',
             'Tuning results require full-set validation. Only complete successful runs are comparable.', '',
             '| Variant | Success/expected | Occ APD10 % | Occ APD5 % | Occ CD cm |',
             '|---|---:|---:|---:|---:|']
    rows = []
    for path in sorted(ROOT.glob('*/nova3r_training/version_*/test/results.json')):
        records = json.loads(path.read_text())
        good = [r for r in records if r.get('status') == 'ok']
        if not good:
            continue
        name = path.relative_to(ROOT).parts[0]
        expected = len(json.loads((ROOT / ('full_list.json' if name.endswith('_full') else 'tuning_list.json')).read_text()))
        row = dict(variant=name, success=len(good), expected=expected)
        for key in ('occluded_apd_0.1', 'occluded_apd_0.05', 'occluded_cd'):
            row[key] = statistics.mean(r[key] for r in good)
        rows.append(row)
        lines.append(f"| {name} | {len(good)}/{expected} | {100*row['occluded_apd_0.1']:.3f} | {100*row['occluded_apd_0.05']:.3f} | {100*row['occluded_cd']:.3f} |")
    (ROOT / 'comparison.json').write_text(json.dumps(rows, indent=2))
    (ROOT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT)
    prepare()
    if args.run:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('CUDA unavailable; run outside the sandbox.')
        import yaml
        cfg = yaml.safe_load((BASE / 'hparams.yaml').read_text())['cfg']
        full = json.loads((PROJECT / cfg['data']['test_datasets']['scannetpp']['params']['list_path']).read_text())
        (ROOT / 'full_list.json').write_text(json.dumps(full, indent=2))
        for name in args.variants:
            out = ROOT / (name + ('_full' if args.full else ''))
            if list(out.glob('nova3r_training/version_*/test/results.json')):
                print(f'Skipping existing results: {out}', flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                       '--config', str(ROOT / f'{name}.yaml'), '--devices', '1',
                       '--low-cpu-mem-usage', '--test-dataset', 'scannetpp',
                       '--test-list', str(ROOT / ('full_list.json' if args.full else 'tuning_list.json')),
                       '--mesh-method', 'mesh_udf', '--output-dir', str(out)]
            print(f'Running {name}: {out}', flush=True)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            with (out / 'evaluation.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            report()
            if result.returncode:
                raise SystemExit(f'{name} failed; inspect {out / "evaluation.log"}')
    report()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker()
    else:
        main()
