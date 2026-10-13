#!/usr/bin/env bash

# Generate sampleable ShapeVAE posteriors for every 3D-FRONT panorama.
# Existing *_before_sampling.pt files are skipped, so interrupted runs resume.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
cd "$repo_root"

split_list=${SPLIT_LIST:-data/3dfront/panorama_list_2000.json}
surface_root=${SURFACE_ROOT:-/tmp/datasets/3dfront/udf}
pose_root=${POSE_ROOT:-/tmp/datasets/3dfront/panoramas}
output_root=${OUTPUT_ROOT:-$repo_root/datasets/3dfront/gt_latent_posteriors}
checkpoint=${VAE_CHECKPOINT:-exp_output/hunyuan_vae/checkpoints/last.ckpt}
config=${VAE_CONFIG:-checkpoints/hunyuan/.hydra/config.yaml}
devices=${DEVICES:-cuda:0}
batch_size=${BATCH_SIZE:-24}
num_workers=${NUM_WORKERS:-8}
# ShapeVAE training already uses BF16. Encoding its large 81,920-point input
# in FP32 is substantially slower and consumes roughly twice the GPU memory.
# Posterior checkpoints are converted back to FP32 before they are saved. Set
# PRECISION=fp32 only when the VAE computation itself must also be full precision.
precision=${PRECISION:-bf16}

for required in "$split_list" "$checkpoint" "$config"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file does not exist: $required" >&2
    exit 1
  fi
done
for required in "$surface_root" "$pose_root"; do
  if [[ ! -d "$required" ]]; then
    echo "Required directory does not exist: $required" >&2
    exit 1
  fi
done

if mountpoint -q "$output_root" 2>/dev/null; then
  echo "Output root is a mount point and may be read-only: $output_root" >&2
  echo "Choose a writable directory with OUTPUT_ROOT=/path/to/output." >&2
  exit 1
fi
mkdir -p "$output_root"
if [[ ! -w "$output_root" ]]; then
  echo "Output root is not writable: $output_root" >&2
  exit 1
fi

read -r -a device_args <<< "$devices"
if ((${#device_args[@]} == 0)); then
  echo "DEVICES must contain at least one device, for example 'cuda:0 cuda:1'." >&2
  exit 1
fi

echo "Generating 3D-FRONT posteriors"
echo "  split:   $split_list"
echo "  output:  $output_root"
echo "  devices: ${device_args[*]}"
echo "  compute precision: $precision"
echo "  saved precision:   fp32"

python -u scripts/preprocessing/latents/surface_to_vae_latent.py \
  --dataset-layout 3dfront \
  --split-list "$split_list" \
  --surface-root "$surface_root" \
  --pose-root "$pose_root" \
  --panorama-camera \
  --checkpoint "$checkpoint" \
  --config "$config" \
  --output-root "$output_root" \
  --devices "${device_args[@]}" \
  --precision "$precision" \
  --batch-size "$batch_size" \
  --num-workers "$num_workers" \
  --normalization cube \
  --cube-center-camera \
  --only-save-latent-before-sampling \
  --seed 0 \
  "$@"

python - "$split_list" "$output_root" <<'PY'
import json
from pathlib import Path, PurePosixPath
import sys

split_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
samples = json.loads(split_path.read_text())
expected = set()
for sample in samples:
    path = PurePosixPath(sample)
    frame = int(path.stem.rsplit("_", 1)[1])
    expected.add(output_root / path.parent / f"{frame:05d}_before_sampling.pt")

missing = [path for path in sorted(expected) if not path.is_file()]
if missing:
    print(f"Missing {len(missing)}/{len(expected)} posterior files.", file=sys.stderr)
    for path in missing[:20]:
        print(f"  {path}", file=sys.stderr)
    raise SystemExit(1)

print(f"Verified {len(expected)} posterior files under {output_root}")
PY

echo
echo "Generation complete. Set the 3D-FRONT gt_latent_root to:"
echo "  $output_root"
