#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/vyhong/projects/nova3r
export PATH=/opt/conda/envs/nova3r/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log_dir=exp_output/hunyuandit_panorama/queued_training
mkdir -p "$log_dir"
exec 9>"$log_dir/queue.lock"
flock -n 9 || { echo 'Training queue is already running.'; exit 1; }
echo "Waiting for evaluations: $(date -Is)"
while python - <<'PY'
from pathlib import Path
import sys
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        args = (p / 'cmdline').read_bytes().split(b'\0')
        if b'training/test.py' in args or b'eval.baselines.run' in args:
            sys.exit(0)
    except OSError:
        pass
sys.exit(1)
PY
do
    sleep 30
done
python - <<'PY'
from pathlib import Path
import json
root = Path('exp_output/hunyuandit_panorama')
for dataset in ('scannetpp', 'sage10k', '3dfront'):
    samples = set()
    for run in root.glob(f'eval_{dataset}_final100*'):
        for path in run.glob('**/meshes/*/metrics.json'):
            record = json.loads(path.read_text())
            if record.get('status') == 'ok':
                samples.add(record['sample'])
    if len(samples) != 100:
        raise SystemExit(f'Refusing to start: {dataset} has {len(samples)}/100 successful evaluations')
print('All 300 evaluation results verified.')
PY
echo "Starting resumed training: $(date -Is)"
python -u training/resume_train.py \
    --config checkpoints/hunyuan_dit/config_hunyuandit_panorama_visible_latent.yaml \
    --devices 1 > "$log_dir/training.log" 2>&1
echo "Training complete: $(date -Is)"
