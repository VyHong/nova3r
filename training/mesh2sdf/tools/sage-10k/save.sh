#!/usr/bin/env bash
set -u

input_dir="/tmp/datasets/sage-10k/panoramas-2"
output_dir="/mnt/projects/theses/vyhong/sage-10k/panoramas-2"
batch_size=100

mkdir -p "$output_dir"

mapfile -d '' folders < <(
    find "$input_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)

total=${#folders[@]}

for ((start=0; start<total; start+=batch_size)); do
    batch=$((start / batch_size))
    final="$output_dir/batch_$(printf '%04d' "$batch").zip"
    marker="${final}.done"

    if [[ -f "$marker" ]]; then
        echo "Skipping completed batch $batch"
        continue
    fi

    list_file=$(mktemp)

    end=$((start + batch_size))
    (( end > total )) && end=$total

    for ((i=start; i<end; i++)); do
        printf '%s\n' "${folders[$i]}"
    done > "$list_file"

    echo "Archiving folders $start to $((end - 1))"

    rm -f "$final"

    if 7z a -tzip -mx=1 "$final" @"$list_file" && 7z t "$final"; then
        touch "$marker"
    else
        echo "Batch $batch failed"
        rm -f "$final"
        rm -f "$list_file"
        exit 1
    fi

    rm -f "$list_file"
done
