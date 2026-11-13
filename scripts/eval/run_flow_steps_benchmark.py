"""Compare 50/100/200 flow steps on three fixed scenes with MeshUDF1024."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'exp_output/hunyuandit_panorama'
OUT = BASE / 'flow_steps_1024_generalization'
SOURCES = {
    'scannetpp': (BASE / 'meshudf_apd_resolution4/resolution512.yaml', BASE / 'meshudf_apd_resolution4/full_list.json'),
    '3dfront': (BASE / 'eval_epoch09_mesh_udf_512_sage3dfront/3dfront/config.yaml', BASE / 'eval_epoch09_mesh_udf_512_sage3dfront/3dfront/test_list.json'),
    'sage10k': (BASE / 'eval_epoch09_mesh_udf_512_sage3dfront/sage10k/config.yaml', BASE / 'eval_epoch09_mesh_udf_512_sage3dfront/sage10k/test_list.json'),
}


def sample_name(dataset, item):
    if dataset == '3dfront':
        path = Path(item)
        frame = int(re.search(r'_panorama_(\d+)\.', path.name).group(1))
        return f'3dfront_{str(path.parent).replace("/", "_")}_{frame:05d}_4'
    scene, frame = item.split()
    return f'{dataset}_{scene}_{int(frame):05d}_4'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    variants = {f'steps_{steps}': dict(test_flow_matching_steps=steps, mesh_mc=1024,
                                    mesh_octree_resolution=1024, mesh_bounds_padding=.05,
                                    mesh_udf_reference_resolution=512) for steps in (100, 200, 50)}
    (OUT / 'variants.json').write_text(json.dumps(variants, indent=2)+'\n')
    for dataset, (config, split) in SOURCES.items():
        out = OUT / dataset
        out.mkdir(exist_ok=True)
        full = json.loads(split.read_text())
        selected = sorted({0, len(full)//3, 2*len(full)//3, len(full)-1})
        if dataset == 'scannetpp': selected = selected[:3]
        subset = [full[i] for i in selected]
        (out / 'test_list.json').write_text(json.dumps(subset, indent=2)+'\n')
        (out / 'indices.json').write_text(json.dumps({sample_name(dataset, v):i for i,v in enumerate(full)}, indent=2)+'\n')
        cfg = yaml.safe_load(config.read_text())
        spec = cfg['cfg']['data']['test_datasets'][dataset]
        cfg['cfg']['data']['test_datasets'] = {dataset:spec}
        spec['params']['list_path'] = str(out/'test_list.json')
        cfg['cfg'].update(mesh_mc=1024, mesh_bounds_padding=.05, mesh_udf_reference_resolution=512,
                          mesh_udf_normal_band_voxels=2., mesh_udf_streamed_grid=True,
                          smooth_borders=False, test_flow_matching_steps=100)
        (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / 'selection.json').write_text(json.dumps(dict(dataset=dataset, indices=selected,
             samples=subset, selection='Uniform list positions, chosen independently of metrics',
             expected_per_step_count=len(subset)), indent=2)+'\n')
    return variants


def run(variants):
    for dataset in ('scannetpp',):
        out = OUT / dataset
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', MESH_UDF_SAVE_GEOMETRY_CACHE='0',
                   MESH_UDF_DIAGNOSTIC_RESUME='1',
                   MESH_UDF_DIAGNOSTIC_DIR=str(out), MESH_UDF_DIAGNOSTIC_INDICES=str(out/'indices.json'),
                   MESH_UDF_DIAGNOSTIC_VARIANTS=json.dumps(variants))
        command = [sys.executable, '-u', str(ROOT/'scripts/eval/diagnose_meshudf_resolution.py'),
                   '--config',str(out/'config.yaml'),'--devices','1','--low-cpu-mem-usage',
                   '--test-dataset',dataset,'--test-list',str(out/'test_list.json'),
                   '--mesh-method','mesh_udf','--output-dir',str(out/'lightning')]
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with (out/'evaluation.log').open('a') as log:
            log.write(json.dumps(dict(event='benchmark_reconfigured',steps=[50,100,200],started_utc=stamp))+'\n')
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            state = dict(state='running',dataset=dataset,pid=child.pid,command=command,
                         started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(state),flush=True)
            code = child.wait()
        state.update(state='finished' if code == 0 else 'failed',returncode=code)
        (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        if code:
            (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            raise SystemExit(code)
        for steps in (50,200):
            subprocess.run([sys.executable,str(ROOT/'scripts/eval/report_meshudf_resolution.py'),
                            str(out/'steps_100'),str(out/f'steps_{steps}'),
                            '--expected','3','--output',str(out/f'comparison_steps_{steps}'),
                            '--reference-label','100 steps','--candidate-label',f'{steps} steps'],check=True)
    (OUT/'status.json').write_text(json.dumps(dict(state='finished',datasets=['scannetpp']),indent=2)+'\n')


if __name__ == '__main__':
    variants = prepare()
    if '--run' in sys.argv: run(variants)
    else: print(OUT)
