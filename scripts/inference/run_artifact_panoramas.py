"""Infer the three requested artifact panoramas, preserving per-image logs."""
from pathlib import Path
import os
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ['logo_make_11_06_2023_383.jpg',
          '360_F_649911283_osXl0lH6BQezO9P6otjnvQzjzDo4jR0c.jpg',
          '360_F_645250947_QleB30Sc6ozLbpC2wsOZRM8NtaLKyyE4.jpg']
os.chdir(ROOT)
output = ROOT / 'demo/outputs/artifact_panoramas_epoch09'
output.mkdir(parents=True, exist_ok=True)
results = []
for name in IMAGES:
    destination = output / Path(name).stem
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'scripts/inference/predict_panorama.py'),
               str(ROOT / 'artifacts' / name), '--output-dir', str(destination)]
    print(f'Starting {name}', flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with (destination / 'inference.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(image=name, exit_code=result.returncode, output=str(destination)))
    (output / 'status.json').write_text(json.dumps(results, indent=2))
    print(f'Finished {name}: exit={result.returncode}', flush=True)
if any(r['exit_code'] for r in results):
    raise SystemExit(1)
