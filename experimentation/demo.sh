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
PY_SCRIPT="demo_nova3r.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo " not found: $PY_SCRIPT"
    exit 1
fi

if [[ -f "$INPUT_TARGET" && "$INPUT_TARGET" == *.json ]]; then
    echo "Reading scene paths from JSON: $INPUT_TARGET"
    mapfile -t img_dirs < <(jq -r '.scenes[] | . + "/panocam/images"' "$INPUT_TARGET")
fi


# Process all panocam/images directories
for img_dir in "${img_dirs[@]}"; do
    if [[ ! -d "$img_dir" ]]; then
        continue
    fi
    scan_dir="$(dirname "$(dirname "$img_dir")")"
    scan_id="$(basename "$scan_dir")"

    echo "Processing scan: $scan_id"

    while IFS= read -r sub_dir; do
        folder_name="$(basename "$sub_dir")"
        echo "  -> Found folder: $folder_name"

        shopt -s nullglob
        img_array=( "$sub_dir"/*.jpg "$sub_dir"/*.png "$sub_dir"/*.jpeg )
        shopt -u nullglob

        if [[ ${#img_array[@]} -gt 0 ]]; then
            # Run demo_nova3r.py with all images in the folder
            /home/vy/miniconda3/envs/nova3r/bin/python "$PY_SCRIPT" \
                --images "${img_array[0]}" "${img_array[1]}" "${img_array[4]}" "${img_array[5]}" \
                --ckpt checkpoints/scene_n2/checkpoint-last.pth \
                --output_dir "../nova3r_output_4/$scan_id/$folder_name"\
                --num_queries 200000
            
            echo "    demo_nova3r.py executed for folder: $sub_dir"
        else
            echo "    No images found in $sub_dir"
        fi

    done < <(find "$img_dir" -mindepth 1 -maxdepth 1 -type d | sort)

done

echo "Done."