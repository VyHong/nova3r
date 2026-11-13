#!/usr/bin/env bash
# Run LaRI/VolFill with six independent 90-degree cubemap faces.
set -euo pipefail
benchmark_model=${1:?Usage: benchmark_single_image.sh lari|volfill [scannetpp|sage10k|3dfront|all]}
benchmark_dataset=${2:-all}
benchmark_extra_args=("${@:3}")
case "$benchmark_model" in lari|volfill) ;; *) echo 'Model must be lari or volfill' >&2; exit 2;; esac
case "$benchmark_dataset" in scannetpp|sage10k|3dfront|all) ;; *) echo 'Unknown dataset' >&2; exit 2;; esac
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$PWD/checkpoints/hf_baselines}"
benchmark_datasets=("$benchmark_dataset")
if [[ "$benchmark_dataset" == all ]]; then benchmark_datasets=(scannetpp sage10k 3dfront); fi
for benchmark_layout in "${benchmark_datasets[@]}"; do
    case "$benchmark_layout" in
        scannetpp)
            benchmark_split=data/scannetpp/final_test_list.json
            benchmark_cube=datasets/scannetpp/cubemaps/updated_final_test_100_prepared_cube_90
            benchmark_pose=datasets/scannetpp/test_scenes
            benchmark_scene=datasets/scannetpp/test_scenes
            benchmark_mesh=datasets/scannetpp/test_scenes ;;
        sage10k)
            benchmark_split=data/sage-10k/final_test_list.json
            benchmark_cube=datasets/sage-10k/cubemaps/final_test_100_prepared_cube_90
            benchmark_pose=/mnt/projects/theses/vyhong/sage-10k/poses
            benchmark_scene=/tmp/datasets/sage-10k/udf
            benchmark_mesh=/tmp/datasets/sage-10k/panorama-scenes ;;
        3dfront)
            benchmark_split=data/3dfront/final_test_list.json
            benchmark_cube=datasets/3dfront/cubemaps/final_test_100_prepared_cube_90
            benchmark_pose=/tmp/datasets/3dfront/panorama_scenes
            benchmark_scene=/tmp/datasets/3dfront/better_udf
            benchmark_mesh=/tmp/datasets/3dfront/panorama_scenes ;;
    esac
    "${BENCHMARK_PYTHON:-python}" -m eval.baselines.run \
        --model "$benchmark_model" --dataset-layout "$benchmark_layout" \
        --input-projection cubemap --cubemap-fov 90 \
        --split-list "$benchmark_split" --cubemap-root "$benchmark_cube" \
        --pose-root "$benchmark_pose" --scene-root "$benchmark_scene" \
        --visibility-mesh-root "$benchmark_mesh" --visibility-metrics \
        --fscore-thresholds 0.1 0.05 0.02 --save-predictions --resume \
        --output-dir "eval_results/baselines/$benchmark_model/${benchmark_layout}_independent_faces_cube90${BENCHMARK_SUFFIX:+_$BENCHMARK_SUFFIX}" "${benchmark_extra_args[@]}"
done
