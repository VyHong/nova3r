#!/usr/bin/env bash
# Direct single-ERP ablation across the final test splits.
set -euo pipefail
benchmark_model=${1:-all}
benchmark_dataset=${2:-all}
benchmark_extra_args=("${@:3}")
case "$benchmark_model" in lari|nova3r|volfill|all) ;; *) echo 'Unknown model' >&2; exit 2;; esac
case "$benchmark_dataset" in scannetpp|sage10k|3dfront|all) ;; *) echo 'Unknown dataset' >&2; exit 2;; esac
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$PWD/checkpoints/hf_baselines}"
benchmark_python=${BENCHMARK_PYTHON:-python}
if [[ "$benchmark_model" == all || "$benchmark_dataset" == all ]]; then
  benchmark_log_root="eval_results/baselines/erp_suite${BENCHMARK_SUFFIX:+_$BENCHMARK_SUFFIX}"
  mkdir -p "$benchmark_log_root"
  benchmark_models=("$benchmark_model")
  benchmark_datasets=("$benchmark_dataset")
  if [[ "$benchmark_model" == all ]]; then benchmark_models=(lari nova3r volfill); fi
  if [[ "$benchmark_dataset" == all ]]; then benchmark_datasets=(scannetpp sage10k 3dfront); fi
  benchmark_pids=()
  for benchmark_method in "${benchmark_models[@]}"; do
    for benchmark_layout in "${benchmark_datasets[@]}"; do
      benchmark_log="$benchmark_log_root/${benchmark_method}_${benchmark_layout}.log"
      bash scripts/eval/benchmark_erp.sh "$benchmark_method" "$benchmark_layout" "${benchmark_extra_args[@]}" \
        >> "$benchmark_log" 2>&1 &
      benchmark_pids+=("$!")
      echo "Started $benchmark_method / $benchmark_layout: PID $!, log $benchmark_log"
    done
  done
  benchmark_failed=0
  for benchmark_pid in "${benchmark_pids[@]}"; do
    if ! wait "$benchmark_pid"; then benchmark_failed=1; fi
  done
  exit "$benchmark_failed"
fi
mkdir -p "eval_results/baselines/$benchmark_model"
exec 9>"eval_results/baselines/$benchmark_model/erp_${benchmark_dataset}${BENCHMARK_SUFFIX:+_$BENCHMARK_SUFFIX}.lock"
flock -n 9 || { echo "An ERP job for $benchmark_model / $benchmark_dataset is already running" >&2; exit 1; }
"$benchmark_python" -c 'import torch; assert torch.cuda.is_available(), "ERP benchmark requires an available CUDA GPU"'
benchmark_models=("$benchmark_model")
benchmark_datasets=("$benchmark_dataset")
if [[ "$benchmark_dataset" == all ]]; then benchmark_datasets=(scannetpp sage10k 3dfront); fi
for benchmark_method in "${benchmark_models[@]}"; do
  for benchmark_layout in "${benchmark_datasets[@]}"; do
    benchmark_options=()
    case "$benchmark_layout" in
      scannetpp)
        benchmark_split=data/scannetpp/final_test_list.json
        benchmark_erp=datasets/scannetpp/panos/updated_final_test_100_prepared_erp
        benchmark_pose=datasets/scannetpp/test_scenes
        benchmark_scene=datasets/scannetpp/test_scenes
        benchmark_mesh=datasets/scannetpp/test_scenes ;;
      sage10k)
        benchmark_split=data/sage-10k/final_test_list.json
        benchmark_erp=/tmp/datasets/sage-10k/panorama-scenes
        benchmark_pose=/mnt/projects/theses/vyhong/sage-10k/poses
        benchmark_scene=/tmp/datasets/sage-10k/udf
        benchmark_mesh=/tmp/datasets/sage-10k/panorama-scenes ;;
      3dfront)
        benchmark_split=data/3dfront/final_test_list.json
        benchmark_erp=/tmp/datasets/3dfront/panorama_scenes
        benchmark_pose=$benchmark_erp
        benchmark_scene=/tmp/datasets/3dfront/better_udf
        benchmark_mesh=$benchmark_erp ;;
    esac
    if [[ "$benchmark_method" == nova3r ]]; then benchmark_options+=(--nova3r-num-views 1); fi
    "$benchmark_python" -u -m eval.baselines.run \
      --model "$benchmark_method" --dataset-layout "$benchmark_layout" \
      --input-projection erp --erp-root "$benchmark_erp" \
      --split-list "$benchmark_split" --pose-root "$benchmark_pose" \
      --scene-root "$benchmark_scene" --visibility-mesh-root "$benchmark_mesh" \
      --visibility-metrics --fscore-thresholds 0.1 0.05 0.02 \
      --save-predictions --resume \
      --output-dir "eval_results/baselines/$benchmark_method/${benchmark_layout}_single_erp${BENCHMARK_SUFFIX:+_$BENCHMARK_SUFFIX}" \
      "${benchmark_options[@]}" "${benchmark_extra_args[@]}"
  done
done
