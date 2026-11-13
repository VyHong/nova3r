"""Run the three full tuned-768 evaluations concurrently, without smoothing."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from run_flow_steps_benchmark import sample_name
from report_meshudf_resolution import report

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'exp_output/hunyuandit_panorama'
OUT = BASE / 'meshudf_768_tuning'
SOURCES = {
    'scannetpp': (OUT / 'tuned768.yaml', BASE / 'meshudf_apd_resolution4/full_list.json',
                  BASE / 'meshudf_apd_resolution4/resolution512_full'),
    **{ds: (BASE / f'eval_epoch09_mesh_udf_512_sage3dfront/{ds}/config.yaml',
            BASE / f'eval_epoch09_mesh_udf_512_sage3dfront/{ds}/test_list.json',
            BASE / f'eval_epoch09_mesh_udf_512_sage3dfront/{ds}') for ds in ('3dfront', 'sage10k')},
}
MOUNTS = {
    'scannetpp': [f'/tmp/datasets/scannetpp/{s}' for s in (
        'cubemap_data', 'udf_color', 'dinov2_latents', 'fix_gt_latents', 'fix_vis_latents')],
    '3dfront': [f'/tmp/datasets/3dfront/{s}' for s in (
        'panorama_scenes', 'better_udf', 'better_gt_latents', 'better_dinov2_latents', 'better_vis_latents')],
    'sage10k': [f'/tmp/datasets/sage-10k/{s}' for s in (
        'cubemap_data', 'udf', 'dinov2_latents', 'fix_gt_latents', 'fix_vis_latents', 'panorama-scenes')],
}


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def run(dataset):
    source, split, reference = SOURCES[dataset]
    out = OUT / f'{dataset}_tuned768_full'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = yaml.safe_load(source.read_text())
    samples = json.loads(split.read_text())
    assert len(samples) == len(set(samples)) == 100
    write(out / 'full_list.json', samples)
    write(out / 'indices.json', {sample_name(dataset, item): i for i, item in enumerate(samples)})
    spec = cfg['cfg']['data']['test_datasets'][dataset]
    spec['params']['list_path'] = str(out / 'full_list.json')
    cfg['cfg']['data']['test_datasets'] = {dataset: spec}
    cfg['cfg']['data'].update(num_workers=0, test_num_workers=2, prefetch_factor=2,
                              persistent_workers=True, test_batch_size=1)
    cfg['cfg'].update(mesh_method='mesh_udf', mesh_mc=768, mesh_octree_resolution=768,
        mesh_udf_reference_resolution=512, mesh_bounds_padding=.05,
        mesh_udf_normal_band_voxels=2., mesh_udf_surface_threshold=.01,
        mesh_udf_query_batch_size=262144, mesh_udf_vertex_batch_size=131072,
        mesh_udf_streamed_grid=True, mesh_udf_fast_cleanup=True, mesh_udf_mc_tile_cells=0,
        test_flow_matching_steps=100, smooth_borders=False, output_dir=str(out))
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    state = dict(dataset=dataset, state='waiting_for_mounts', expected_samples=100,
        resolution=768, reference_resolution=512, taubin=False,
        speed=dict(cpu_threads=16, openblas_threads=8, data_workers=2, prefetch_factor=2,
                   field_batch=262144, gradient_batch=65536, vertex_batch=131072,
                   fast_cleanup=True, concurrent_dataset_workers=3))
    deadline = time.monotonic() + 3600
    while True:
        missing = [p for p in MOUNTS[dataset] if not os.path.ismount(p)]
        state['missing_mounts'] = missing
        write(out / 'status.json', state)
        if not missing:
            break
        if time.monotonic() > deadline:
            state.update(state='failed', error='Dataset mount readiness timed out')
            write(out / 'status.json', state)
            return state
        time.sleep(10)
    command = [sys.executable, '-u', str(ROOT / 'scripts/eval/sweep_meshudf_occluded.py'),
        'worker', '--config', str(out / 'config.yaml'), '--devices', '1', '--low-cpu-mem-usage',
        '--test-dataset', dataset, '--test-list', str(out / 'full_list.json'),
        '--mesh-method', 'mesh_udf', '--output-dir', str(out)]
    env = dict(os.environ, MESH_UDF_SEARCH_ROOT=str(out), MESH_UDF_RESUME_COMPLETED=str(out),
        OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1')
    with (out / 'evaluation.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(state='running', pid=child.pid, command=command,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out / 'launch.json', state)
        write(out / 'status.json', state)
        print(f'Started {dataset}: PID {child.pid}', flush=True)
        while True:
            comparison = report(reference, out, out / 'comparison_to_512', 100, '512', 'Tuned768 raw')
            state.update(successful=comparison['candidate_successful'], failed=comparison['candidate_failed'])
            write(out / 'status.json', state)
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
    comparison = report(reference, out, out / 'comparison_to_512', 100, '512', 'Tuned768 raw')
    state.update(state='complete' if code == 0 and comparison['candidate_successful'] == 100
                 and not comparison['candidate_failed'] else 'failed', returncode=code,
                 successful=comparison['candidate_successful'], failed=comparison['candidate_failed'])
    write(out / 'status.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(run, ds): ds for ds in args.datasets}
        result = {}
        for future in as_completed(jobs):
            ds = jobs[future]
            try:
                result[ds] = future.result()
            except Exception as exc:
                result[ds] = dict(state='failed', error=str(exc))
            write(OUT / 'full_suite_result.json', result)
            print(json.dumps(result[ds]), flush=True)


if __name__ == '__main__':
    main()
