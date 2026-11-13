#!/usr/bin/env bash
# Resume the interrupted final-100 GenRecon runs after all dataset ZIP mounts.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec 9>eval_results/baselines/genrecon/resume_after_mounts.lock
flock -n 9 || { echo 'A GenRecon restart supervisor is already running.'; exit 1; }
export PATH=/opt/conda/envs/nova3r/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers
export HF_HOME="$PWD/checkpoints/hf_baselines"
export HF_TOKEN_PATH=/mnt/home/vyhong/.cache/huggingface/token

required_mounts=()
for name in cubemap_data udf_color dinov2_latents fix_vis_latents fix_gt_latents scannetpp_panorama; do
    required_mounts+=("/tmp/datasets/scannetpp/$name")
done
for name in cubemap_data udf dinov2_latents fix_vis_latents fix_gt_latents panorama-scenes; do
    required_mounts+=("/tmp/datasets/sage-10k/$name")
done
for name in panorama_scenes better_udf better_gt_latents better_dinov2_latents better_vis_latents; do
    required_mounts+=("/tmp/datasets/3dfront/.original/$name" "/tmp/datasets/3dfront/$name")
done
for name in add_panorama_scenes add_udf add_gt_latents add_dinov2_latents add_vis_latents; do
    required_mounts+=("/tmp/datasets/3dfront/$name")
done
echo "Waiting for all ${#required_mounts[@]} dataset mounts: $(date -Is)"
while true; do
    missing=()
    for path in "${required_mounts[@]}"; do
        mountpoint -q "$path" || missing+=("$path")
    done
    if ((${#missing[@]} == 0)); then break; fi
    echo "$(date -Is) Still waiting: ${missing[*]}"
    sleep 30
done
echo "All ZIPs and merged datasets are mounted: $(date -Is)"
# Avoid duplicating an evaluation that was started manually while mounts loaded.
python - <<'PY'
from pathlib import Path
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        args = (p / 'cmdline').read_bytes().split(b'\0')
        if b'eval.baselines.run' in args and b'--model' in args and args[args.index(b'--model') + 1] == b'genrecon':
            raise SystemExit('GenRecon is already running; refusing to duplicate it.')
    except (OSError, IndexError):
        pass
PY
nvidia-smi
failed=0
for layout in scannetpp sage10k; do
    case "$layout" in
        scannetpp)
            split=data/scannetpp/final_test_list.json
            cube=datasets/scannetpp/cubemaps/updated_final_test_100_prepared_cube_90
            mesh=datasets/scannetpp/test_scenes ;;
        sage10k)
            split=data/sage-10k/final_test_list.json
            cube=datasets/sage-10k/cubemaps/final_test_100_prepared_cube_90
            mesh=/tmp/datasets/sage-10k/panorama-scenes ;;
    esac
    output="eval_results/baselines/genrecon/${layout}_4workers_cube90_visibility_100k"
    cp "$output/results.json" "$output/results.before_resume_$(date -u +%Y%m%dT%H%M%SZ).json"
    echo "Resuming $layout with four workers: $(date -Is)"
    if python -m eval.baselines.run \
        --model genrecon --dataset-layout "$layout" \
        --input-projection cubemap --cubemap-fov 90 \
        --split-list "$split" --cubemap-root "$cube" \
        --genrecon-scannet-chunk-source evaluation_surface \
        --workers 4 --device cuda:0 --eval-points 100000 \
        --visibility-mesh-root "$mesh" --visibility-metrics \
        --fscore-thresholds 0.1 0.05 0.02 --save-predictions --resume \
        --output-dir "$output" >> "$output.log" 2>&1; then
        echo "Finished $layout: $(date -Is)"
    else
        echo "Evaluation process failed for $layout; see $output.log"
        failed=1
    fi
done
echo "GenRecon restart sequence finished: $(date -Is)"
exit "$failed"
