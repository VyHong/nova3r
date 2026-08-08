#!/usr/bin/env bash
set -euo pipefail

SEVEN_Z_DIR="${1:-/mnt/projects/theses/vyhong/sage-10k/panoramas-2}"
ZIP_DIR="${2:-/mnt/projects/theses/vyhong/sage-10k/panoramas-2-zips}"
OUTPUT_DIR="${3:-/mnt/projects/theses/vyhong/sage-10k/panoramas-2-extracted}"
THREADS="${THREADS:-$(nproc)}"
DELETE_ARCHIVES="${DELETE_ARCHIVES:-0}"

command -v 7z >/dev/null || {
    echo "Error: 7z is required." >&2
    exit 1
}

if [[ ! "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: THREADS must be a positive integer." >&2
    exit 1
fi

if [[ "$DELETE_ARCHIVES" != 0 && "$DELETE_ARCHIVES" != 1 ]]; then
    echo "Error: DELETE_ARCHIVES must be 0 or 1." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SEVEN_Z_DIR="$(realpath "$SEVEN_Z_DIR")"
ZIP_DIR="$(realpath "$ZIP_DIR")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

declare -A archives=()
shopt -s nullglob

# Prefer converted ZIPs when both formats exist for a batch.
for marker in "$SEVEN_Z_DIR"/batch_*.7z.done; do
    archive="${marker%.done}"
    [[ -f "$archive" ]] || continue
    batch="${archive##*/}"
    batch="${batch%.7z}"
    archives["$batch"]="$archive"
done

for marker in "$ZIP_DIR"/batch_*.zip.done; do
    archive="${marker%.done}"
    [[ -f "$archive" ]] || continue
    batch="${archive##*/}"
    batch="${batch%.zip}"
    archives["$batch"]="$archive"
done

if ((${#archives[@]} == 0)); then
    echo "No completed .7z or .zip archives found." >&2
    exit 1
fi

mapfile -t batches < <(printf '%s\n' "${!archives[@]}" | sort)

echo "Extracting ${#batches[@]} batches into: $OUTPUT_DIR"
echo "Using $THREADS 7-Zip threads"

for batch in "${batches[@]}"; do
    archive="${archives[$batch]}"
    archive_marker="${archive}.done"
    extracted_marker="$OUTPUT_DIR/.${batch}.extracted.done"

    if [[ -f "$extracted_marker" ]]; then
        echo "Skipping completed batch: $batch"
        continue
    fi

    echo
    echo "Extracting: $archive"

    # A successful extraction verifies every extracted member's CRC.
    7z x -y -mmt="$THREADS" -o"$OUTPUT_DIR" "$archive"
    touch -- "$extracted_marker"

    if [[ "$DELETE_ARCHIVES" == 1 ]]; then
        echo "Deleting successfully extracted archive: $archive"
        rm -- "$archive" "$archive_marker"
    fi

    echo "Completed: $batch"
done

echo
echo "All completed archives have been extracted into: $OUTPUT_DIR"
