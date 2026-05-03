#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scannet_cubemap.sh "<scannet_root>" "<path/to/image-generation.py>" "<output_root>"
#
# Example (WSL):
#   bash scannet_cubemap.sh "D:\scannet\data" "/home/vy/project/panorama/code/image-generation.py" "/home/vy/project/panorama/output"
#
# Expected ScanNet layout:
#   <scannet_root>/<scan_id>/panocam/images/*.jpg




INPUT_TARGET="${1:-/mnt/d/scannet/data}"
PY_SCRIPT="image_generation.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "image_generation.py not found: $PY_SCRIPT"
    exit 1
fi

if [[ -f "$INPUT_TARGET" && "$INPUT_TARGET" == *.json ]]; then
    echo "Reading scene paths from JSON: $INPUT_TARGET"
    mapfile -t img_dirs < <(jq -r '.scenes[] | . + "/panocam/images"' "$INPUT_TARGET")
elif [[ -d "$INPUT_TARGET" ]]; then
    echo "Finding scenes in root directory: $INPUT_TARGET"
    mapfile -t img_dirs < <(find "$INPUT_TARGET" -type d -path "*/panocam/images")
else
    echo "Input target not found or invalid: $INPUT_TARGET"
    exit 1
fi


# Process all panocam/images directories
for img_dir in "${img_dirs[@]}"; do
    if [[ ! -d "$img_dir" ]]; then
        continue
    fi
    scan_dir="$(dirname "$(dirname "$img_dir")")"
    scan_id="$(basename "$scan_dir")"

    echo "Processing scan: $scan_id"

    while IFS= read -r img_path; do
        img_name="$(basename "$img_path")"
        img_stem="${img_name%.*}"
        img_out_dir="$img_dir/$img_stem"

        depth_dir="$(dirname "$img_dir")/depth"
        depth_path="$depth_dir/$img_stem.png"
        depth_out_dir="$depth_dir/$img_stem"

        echo "  -> $img_name"

        # Adjust flags below if your image-generation.py uses different arguments.
        python "$PY_SCRIPT" \
            --image "$img_path" \
            --image_out "$img_out_dir" \
            --depth "$depth_path" \
            --depth_out "$depth_out_dir"
        
        echo "    Cubemap generated at: $img_out_dir"

    done < <(find "$img_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)

done

echo "Done."