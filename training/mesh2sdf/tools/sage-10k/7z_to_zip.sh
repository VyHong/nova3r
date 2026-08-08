#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-/mnt/projects/theses/vyhong/sage-10k/panoramas-2}"
OUTPUT_DIR="${2:-/mnt/projects/theses/vyhong/sage-10k/panoramas-2-zips}"
THREADS="${THREADS:-$(nproc)}"

if [[ ! "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: THREADS must be a positive integer." >&2
    exit 1
fi

command -v 7z >/dev/null || {
    echo "Error: 7z is required." >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"

SOURCE_DIR="$(realpath "$SOURCE_DIR")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

echo "Using $THREADS 7-Zip threads"

shopt -s nullglob
markers=("$SOURCE_DIR"/*.7z.done)

if ((${#markers[@]} == 0)); then
    echo "No completed .7z archives found in: $SOURCE_DIR" >&2
    exit 1
fi

for marker in "${markers[@]}"; do
    archive="${marker%.done}"
    filename="${archive##*/}"
    stem="${filename%.7z}"
    output="$OUTPUT_DIR/$stem.zip"
    partial="$OUTPUT_DIR/.${stem}.zip.partial"

    if [[ ! -f "$archive" ]]; then
        echo "Skipping missing archive: $archive"
        continue
    fi

    if [[ -e "$output" ]]; then
        echo "Skipping because output already exists: $output"
        continue
    fi

    temp_dir="$(mktemp -d "$OUTPUT_DIR/.convert-${stem}.XXXXXX")"

    cleanup() {
        rm -rf -- "$temp_dir"
        rm -f -- "$partial"
    }

    trap cleanup EXIT INT TERM

    echo
    echo "Extracting: $archive"
    7z x -y -mmt="$THREADS" -o"$temp_dir" "$archive"

    echo "Creating (multithreaded store mode): $output"
    (
        cd "$temp_dir"
        7z a -tzip -mx=0 -mmt="$THREADS" "$partial" .
    )

    echo "Verifying ZIP..."
    7z t -tzip -mmt="$THREADS" "$partial"

    mv -- "$partial" "$output"

    echo "ZIP verified successfully."
    echo "Deleting source archive: $archive"
    rm -- "$archive"

    # Replace the old archive marker with a ZIP completion marker.
    rm -- "$marker"
    touch -- "$output.done"

    rm -rf -- "$temp_dir"
    trap - EXIT INT TERM

    echo "Completed: $output"
done

echo
echo "All completed archives have been processed."
