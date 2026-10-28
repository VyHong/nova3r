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
)

FRONT_ROOT="$MOUNT_ROOT/3dfront"
UNIONFS_BIN=${UNIONFS_BIN:-$(command -v unionfs || true)}
UNIONFS_BIN=${UNIONFS_BIN:-$SCRIPT_DIR/.deps/unionfs/usr/bin/unionfs}
BASES=(panorama_scenes better_udf better_gt_latents better_dinov2_latents better_vis_latents)
ADDITIONS=(add_panorama_scenes add_udf add_gt_latents add_dinov2_latents add_vis_latents)

for tool in mount-zip mountpoint findmnt fusermount; do
  command -v "$tool" >/dev/null || { echo "Required command not found: $tool" >&2; exit 1; }
done
if [[ ! -x "$UNIONFS_BIN" ]]; then
  echo "unionfs-fuse is required; install it or set UNIONFS_BIN to its executable." >&2
  exit 1
fi
for name in "${BASES[@]}" "${ADDITIONS[@]}"; do
  [[ -f "$SHARED_DATA_ROOT/3dfront/$name.zip" ]] || {
    echo "Missing dataset archive: $SHARED_DATA_ROOT/3dfront/$name.zip" >&2
    exit 1
  }
done

mount_3dfront_merged() (
  set -euo pipefail
  for index in "${!BASES[@]}"; do
    base=${BASES[$index]}
    addition=${ADDITIONS[$index]}
    original="$FRONT_ROOT/.original/$base"
    extra="$FRONT_ROOT/$addition"
    merged="$FRONT_ROOT/$base"
    if mountpoint -q "$merged"; then
      fstype=$(findmnt -rn -M "$merged" -o FSTYPE)
      if [[ "$fstype" == fuse.unionfs* ]]; then
        echo "Skipping already-merged path: $merged"
        continue
      fi
      if [[ "$fstype" != fuse.mount-zip ]]; then
        echo "Unexpected filesystem at $merged: $fstype" >&2
        exit 1
      fi
    fi

    bash "$SCRIPT_DIR/bulk_mount_mount_zip.sh" "$SHARED_DATA_ROOT/3dfront/$base.zip" "$FRONT_ROOT/.original"
    bash "$SCRIPT_DIR/bulk_mount_mount_zip.sh" "$SHARED_DATA_ROOT/3dfront/$addition.zip" "$FRONT_ROOT"
    # Upgrade an old single-archive mount only after both branches are ready.
    while mountpoint -q "$merged"; do
      fusermount -u "$merged"
    done
    mkdir -p "$merged"
    if [[ -n $(find "$merged" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
      echo "Merge mount point is not empty: $merged" >&2
      exit 1
    fi
    if ! "$UNIONFS_BIN" -o ro "$original=RO:$extra=RO" "$merged"; then
      echo "Failed to merge $base; restoring the original ZIP mount." >&2
      bash "$SCRIPT_DIR/bulk_mount_mount_zip.sh" "$SHARED_DATA_ROOT/3dfront/$base.zip" "$FRONT_ROOT" || true
      exit 1
    fi
    echo "Merged $base.zip + $addition.zip at $merged"
  done
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
  if [[ ${SCANNETPP_PRECACHE:-0} == 1 && "$source" == "$SHARED_DATA_ROOT/scannetpp/"* ]]; then
    options="${options:+$options }-o precache"
  fi
  args=("$source" "$destination")
  if [[ -n "$options" ]]; then
    read -r -a option_args <<< "$options"
    args+=("${option_args[@]}")
  fi

  bash "$MOUNT_HELPER" "${args[@]}" &
  pids+=("$!")
  labels+=("$source")
done

# Run the five merged 3D-FRONT pairs alongside the other dataset mounts.
mount_3dfront_merged &
pids+=("$!")
labels+=("merged 3D-FRONT datasets")

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
