#!/usr/bin/env bash
set -euo pipefail

scene_root=${1:?Usage: render_anon_masks_sharded.sh SCENE_ROOT PANORAMA_ROOT [SHARDS] [OUTPUT_ROOT]}
panorama_root=${2:?Usage: render_anon_masks_sharded.sh SCENE_ROOT PANORAMA_ROOT [SHARDS] [OUTPUT_ROOT]}
shards=${3:-12}
output_root=${4:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
log_root=${output_root:-$panorama_root}
log_dir="$log_root/logs/anon_masks"
mkdir -p "$log_dir"

seq 0 $((shards - 1)) | xargs -P "$shards" -I '{}' bash -c '
  shard=$1
  shard_count=$2
  scene_root=$3
  panorama_root=$4
  script=$5
  log_dir=$6
  output_root=$7
  output_args=()
  if [[ -n "$output_root" ]]; then
    output_args=(--output-root "$output_root")
  fi
  XDG_CACHE_HOME=/tmp/blender-mask-cache-"$shard" \
    conda run --no-capture-output -n blender python -u "$script" \
      "$scene_root" "$panorama_root" \
      "${output_args[@]}" \
      --shard-count "$shard_count" --shard-index "$shard" \
      >"$log_dir/worker_${shard}.log" 2>&1
' _ '{}' "$shards" "$scene_root" "$panorama_root" \
  "$repo_root/scripts/data/3dfront/render_anon_masks.py" "$log_dir" "$output_root"

