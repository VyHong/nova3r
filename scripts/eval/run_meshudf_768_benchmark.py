"""Paired 512/768 extraction pilot with shared latents and fixed 100 flow steps."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'exp_output/hunyuandit_panorama/meshudf_768_tuning/pilot'
SOURCE = ROOT / 'exp_output/hunyuandit_panorama/meshudf_apd_resolution4'
VARIANTS = {
    'reference512': dict(mesh_mc=512, mesh_udf_reference_resolution=None),
    'matched768_reference512': dict(mesh_mc=768, mesh_udf_reference_resolution=512),
}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    full = json.loads((SOURCE / 'full_list.json').read_text())
    positions = [0, len(full)//3, 2*len(full)//3]
    samples = [full[i] for i in positions]
    (OUT / 'test_list.json').write_text(json.dumps(samples, indent=2)+'\n')
    (OUT / 'indices.json').write_text(json.dumps({
        f"scannetpp_{item.replace(' ', '_')}_4": i for i, item in enumerate(full)
    }, indent=2)+'\n')
    (OUT / 'selection.json').write_text(json.dumps(dict(
        positions=positions, samples=samples, selection='Uniform list positions; no metric-based selection',
        flow_steps=100, diffusion_seed=42, scope='Three-scene tuning pilot, not full-set validation'
    ), indent=2)+'\n')
    cfg = yaml.safe_load((SOURCE / 'resolution512.yaml').read_text())
    cfg['cfg']['data']['test_datasets'] = {'scannetpp': cfg['cfg']['data']['test_datasets']['scannetpp']}
    cfg['cfg']['data'].update(num_workers=0, test_num_workers=0)
    cfg['cfg'].update(mesh_mc=512, mesh_bounds_padding=.05,
        mesh_udf_surface_threshold=.01, mesh_udf_normal_band_voxels=2.,
        mesh_udf_reference_resolution=None, mesh_udf_streamed_grid=True,
        mesh_udf_fast_cleanup=True, mesh_udf_query_batch_size=262144,
        mesh_udf_vertex_batch_size=131072, mesh_udf_mc_tile_cells=0,
        smooth_borders=False, test_flow_matching_steps=100)
    (OUT / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT / 'variants.json').write_text(json.dumps(VARIANTS, indent=2)+'\n')


def summarize():
    from report_meshudf_resolution import report
    reports = {}
    for name in VARIANTS:
        if name == 'reference512':
            continue
        reports[name] = report(OUT / 'reference512', OUT / name,
            OUT / f'comparison_{name}', 3, '512', name)
    (OUT / 'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare()
    if not args.run:
        print(OUT)
        return
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        PYTHONUNBUFFERED='1', MESH_UDF_SAVE_GEOMETRY_CACHE='1', MESH_UDF_DIAGNOSTIC_RESUME='1',
        MESH_UDF_DIAGNOSTIC_DIR=str(OUT), MESH_UDF_DIAGNOSTIC_INDICES=str(OUT / 'indices.json'),
        MESH_UDF_DIAGNOSTIC_VARIANTS=json.dumps(VARIANTS))
    command = [sys.executable, '-u', str(ROOT / 'scripts/eval/diagnose_meshudf_resolution.py'),
        '--config', str(OUT / 'config.yaml'), '--devices', '1', '--low-cpu-mem-usage',
        '--test-dataset', 'scannetpp', '--test-list', str(OUT / 'test_list.json'),
        '--mesh-method', 'mesh_udf', '--output-dir', str(OUT / 'lightning')]
    with (OUT / 'evaluation.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
        status = dict(state='running', pid=child.pid, command=command,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (OUT / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)
        while child.poll() is None:
            summarize()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    reports = summarize()
    complete = all(r['matched'] == 3 and not r['candidate_failed'] for r in reports.values())
    status.update(state='complete' if child.returncode == 0 and complete else 'failed',
        returncode=child.returncode)
    (OUT / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    if status['state'] != 'complete':
        raise SystemExit('Pilot incomplete; inspect evaluation.log')


if __name__ == '__main__':
    main()
