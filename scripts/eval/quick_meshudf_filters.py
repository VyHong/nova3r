"""Four-panorama MeshUDF filter or resolution screening, two concurrent GPU jobs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import yaml

PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT / 'exp_output/hunyuandit_panorama/meshudf_apd_search'
ROOT = PROJECT / 'exp_output/hunyuandit_panorama/meshudf_apd_filters4'
SAMPLES = ['09c1414f1b 00001', '27dd4da69e 00004', '578511c8a9 00000', 'ac48a9b736 00002']
VARIANTS = {'looser02': .02, 'tighter005': .005}

def main():
    global ROOT, VARIANTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['filters', 'resolution', 'band'], default='filters')
    args = parser.parse_args()
    if args.mode == 'band':
        ROOT = PROJECT / 'exp_output/hunyuandit_panorama/meshudf_apd_band4'
        VARIANTS = {'band3': (.01, 768), 'band4': (.01, 768)}
    elif args.mode == 'resolution':
        ROOT = PROJECT / 'exp_output/hunyuandit_panorama/meshudf_apd_resolution4'
        VARIANTS = {'resolution512': (.01, 512), 'resolution1024': (.01, 1024)}
    else:
        VARIANTS = {name: (value, 768) for name, value in VARIANTS.items()}
    os.chdir(PROJECT)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'tuning_list.json').write_text(json.dumps(SAMPLES, indent=2))
    (ROOT / 'indices.json').write_text((OLD / 'indices.json').read_text())
    (ROOT / 'full_list.json').write_text((OLD / 'full_list.json').read_text())
    names = {f"scannetpp_{s.replace(' ', '_')}_4" for s in SAMPLES}
    reference = [json.loads(p.read_text()) for p in (OLD / 'reference').rglob('metrics.json')]
    reference = [r for r in reference if r['sample'] in names and r.get('status') == 'ok']
    if {r['sample'] for r in reference} != names:
        raise RuntimeError('Selected reference panoramas must all be complete')
    out = ROOT / 'reference/nova3r_training/version_0/test'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(reference, indent=2))
    env = dict(os.environ, MESH_UDF_SEARCH_ROOT=str(ROOT), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    os.environ['MESH_UDF_SEARCH_ROOT'] = str(ROOT)
    from sweep_meshudf_occluded import report
    cfg = yaml.safe_load((OLD / 'reference.yaml').read_text())
    # Limit input prefetching when sharing a GPU; extraction and metrics stay unchanged.
    cfg['cfg']['data']['num_workers'] = 0
    cfg['cfg']['data']['test_num_workers'] = 0
    def run(name, settings):
        threshold, resolution = settings
        variant = yaml.safe_load(yaml.safe_dump(cfg))
        variant['cfg'].update(mesh_bounds_padding=.05, mesh_udf_surface_threshold=threshold, mesh_mc=resolution)
        if args.mode == 'band':
            variant['cfg']['mesh_udf_normal_band_voxels'] = float(name.removeprefix('band'))
        config = ROOT / f'{name}.yaml'
        config.write_text(yaml.safe_dump(variant, sort_keys=False))
        output = ROOT / name
        output.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(PROJECT / 'scripts/eval/sweep_meshudf_occluded.py'), 'worker',
               '--config', str(config), '--devices', '1', '--low-cpu-mem-usage',
               '--test-dataset', 'scannetpp', '--test-list', str(ROOT / 'tuning_list.json'),
               '--mesh-method', 'mesh_udf', '--output-dir', str(output)]
        print(f'Starting {name}, threshold={threshold}, padding=.05, resolution={resolution}', flush=True)
        with (output / 'evaluation.log').open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return name, result.returncode
    report()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, name, value) for name, value in VARIANTS.items()]
        for future in as_completed(futures):
            name, code = future.result()
            print(f'{name} exit={code}', flush=True)
            report()
            if code:
                print(f'Inspect {ROOT / name / "evaluation.log"}', flush=True)
    report()

if __name__ == '__main__':
    main()
