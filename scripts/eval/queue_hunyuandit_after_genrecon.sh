#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/vyhong/projects/nova3r
export PATH=/opt/conda/envs/nova3r/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
log_dir=exp_output/hunyuandit_panorama/eval_final100_visibility_100k
mkdir -p "$log_dir"
echo "Waiting for GenRecon jobs: $(date -Is)"
while python - <<'PY'
from pathlib import Path
import sys
active=False
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        args=(p/'cmdline').read_bytes().split(b'\0')
        if b'eval.baselines.run' in args and b'--model' in args:
            if args[args.index(b'--model')+1] == b'genrecon':
                active=True
    except (OSError, IndexError):
        pass
sys.exit(0 if active else 1)
PY
do
    sleep 30
done
echo "Starting Hunyuan DiT evaluation: $(date -Is)"
python -u training/test.py \
    --config checkpoints/hunyuan_dit/config_hunyuandit_panorama_visible_latent_eval100k.yaml \
    --ckpt exp_output/hunyuandit_panorama/checkpoints/last-v1.ckpt \
    --mesh-method latent2mesh \
    --devices 1 \
    --device cuda \
    > "$log_dir/evaluation.log" 2>&1
 echo "Evaluation complete: $(date -Is)"
