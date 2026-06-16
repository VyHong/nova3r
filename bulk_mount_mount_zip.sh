#!/usr/bin/env bash

# Usage: ./bulk_mount_mount_zip.sh <zip_or_directory> <mount_root>

set -u

SRC_TARGET=${1:-}
DEST_DIR=${2:-}

if [[ -z "$SRC_TARGET" || -z "$DEST_DIR" ]]; then
    echo "Usage: $0 <source_folder_or_single_zip> <target_mount_folder>" >&2
    exit 1
fi

if ! command -v mount-zip >/dev/null 2>&1; then
    echo "Error: mount-zip is not installed or not available in PATH." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

mounted_count=0
failed_count=0

mount_single_zip() {
    local zip_path=$1
    local zip_name
    local mount_point

    zip_name=$(basename "$zip_path")
    zip_name=${zip_name%.[zZ][iI][pP]}
    mount_point="$DEST_DIR/$zip_name"

    if mountpoint -q "$mount_point" 2>/dev/null; then
        echo "Skipping already-mounted path: $mount_point"
        return 0
    fi

    mkdir -p "$mount_point"
    if [[ -n "$(find "$mount_point" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Failed: mount point is not empty: $mount_point" >&2
        ((failed_count += 1))
        return 1
    fi

    echo "Mounting: $zip_path"
    echo "Mount point: $mount_point"

    if mount-zip "$zip_path" "$mount_point"; then
        echo "Mounted: $zip_name"
        ((mounted_count += 1))
        return 0
    fi

    echo "Failed to mount: $zip_name" >&2
    ((failed_count += 1))
    return 1
}

if [[ -f "$SRC_TARGET" ]]; then
    if [[ "$SRC_TARGET" =~ \.[zZ][iI][pP]$ ]]; then
        mount_single_zip "$SRC_TARGET" || true
    else
        echo "Error: file is not a ZIP archive: $SRC_TARGET" >&2
        exit 1
    fi
elif [[ -d "$SRC_TARGET" ]]; then
    found_zip=false
    while IFS= read -r -d '' zip_path; do
        found_zip=true
        mount_single_zip "$zip_path" || true
    done < <(
        find "$SRC_TARGET" -maxdepth 1 -type f -iname '*.zip' -print0 | sort -z
    )

    if [[ "$found_zip" == false ]]; then
        echo "Error: no ZIP archives found in $SRC_TARGET" >&2
        exit 1
    fi
else
    echo "Error: source is neither a file nor a directory: $SRC_TARGET" >&2
    exit 1
fi

echo "Mounted: $mounted_count; failed: $failed_count"
echo "Mounts are available in $DEST_DIR"

if ((failed_count > 0)); then
    exit 1
fi
