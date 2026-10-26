#!/usr/bin/env bash

# Mount every dataset used by the DiT-L and Hunyuan DiT configs in parallel.

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MOUNT_HELPER="$SCRIPT_DIR/bulk_mount_mount_zip.sh"
SHARED_DATA_ROOT=${SHARED_DATA_ROOT:-/mnt/projects/theses/vyhong}
MOUNT_ROOT=${MOUNT_ROOT:-/tmp/datasets}

if [[ ! -f "$MOUNT_HELPER" ]]; then
  echo "Mount helper not found: $MOUNT_HELPER" >&2
  exit 1
fi

if ! command -v mount-zip >/dev/null 2>&1; then
  echo "mount-zip is not installed or is not available in PATH." >&2
  exit 1
fi

mount_specs=(
  "$SHARED_DATA_ROOT/scannetpp/cubemap_data.zip|$MOUNT_ROOT/scannetpp|"
  "$SHARED_DATA_ROOT/scannetpp/udf_color.zip|$MOUNT_ROOT/scannetpp|"
  "$SHARED_DATA_ROOT/scannetpp/dinov2_latents.zip|$MOUNT_ROOT/scannetpp|"
  "$SHARED_DATA_ROOT/scannetpp/fix_vis_latents.zip|$MOUNT_ROOT/scannetpp|"
  "$SHARED_DATA_ROOT/scannetpp/fix_gt_latents.zip|$MOUNT_ROOT/scannetpp|"
  "$SHARED_DATA_ROOT/sage-10k/udf.zip|$MOUNT_ROOT/sage-10k|"
  "$SHARED_DATA_ROOT/sage-10k/cubemap_data.zip|$MOUNT_ROOT/sage-10k|-o notrim"
  "$SHARED_DATA_ROOT/sage-10k/dinov2_latents.zip|$MOUNT_ROOT/sage-10k|"
  "$SHARED_DATA_ROOT/sage-10k/fix_vis_latents.zip|$MOUNT_ROOT/sage-10k|"
  "$SHARED_DATA_ROOT/sage-10k/fix_gt_latents.zip|$MOUNT_ROOT/sage-10k|"
  "$SHARED_DATA_ROOT/3dfront/panorama_scenes.zip|$MOUNT_ROOT/3dfront|"
  "$SHARED_DATA_ROOT/3dfront/better_udf.zip|$MOUNT_ROOT/3dfront|"
  "$SHARED_DATA_ROOT/3dfront/better_gt_latents.zip|$MOUNT_ROOT/3dfront|"
  "$SHARED_DATA_ROOT/3dfront/better_dinov2_latents.zip|$MOUNT_ROOT/3dfront|"
  "$SHARED_DATA_ROOT/3dfront/better_vis_latents.zip|$MOUNT_ROOT/3dfront|"
)

# Validate every source before starting, avoiding a partially mounted dataset
# when an archive has moved or is unavailable on the current machine.
missing=0
for spec in "${mount_specs[@]}"; do
  IFS='|' read -r source _ _ <<< "$spec"
  if [[ ! -f "$source" ]]; then
    echo "Missing dataset archive: $source" >&2
    missing=1
  fi
done
if ((missing)); then
  exit 1
fi

pids=()
labels=()

stop_pending_mounts() {
  echo "Stopping pending dataset mounts..." >&2
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}
trap stop_pending_mounts INT TERM

echo "Starting ${#mount_specs[@]} dataset mounts in parallel..."
for spec in "${mount_specs[@]}"; do
  IFS='|' read -r source destination options <<< "$spec"
  args=("$source" "$destination")
  if [[ -n "$options" ]]; then
    read -r -a option_args <<< "$options"
    args+=("${option_args[@]}")
  fi

  bash "$MOUNT_HELPER" "${args[@]}" &
  pids+=("$!")
  labels+=("$source")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Mount failed: ${labels[$index]}" >&2
    failed=1
  fi
done
trap - INT TERM

if ((failed)); then
  echo "One or more dataset mounts failed." >&2
  exit 1
fi

echo "All datasets are mounted under $MOUNT_ROOT."
