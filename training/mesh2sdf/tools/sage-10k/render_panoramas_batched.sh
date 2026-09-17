#!/usr/bin/env bash
# Render SAGE-10k panoramas in short-lived Blender Python processes.
#
# A new process is used for every small batch so Blender/Cycles allocations are
# returned to the OS. render_panoramas.py's completion markers make reruns safe.

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
renderer="$script_dir/render_panoramas.py"

input_dir=""
output_dir=""
batch_size=5
jobs=4
conda_env=blender
gpu_ids=0
log_dir=""
reverse=false
declare -a completion_dirs=()
declare -a renderer_args=()

usage() {
    sed -n '2,5p' "$0"
    cat <<'EOF'

Usage:
  render_panoramas_batched.sh INPUT_DIR OUTPUT_DIR [options] [-- RENDERER_ARGS...]

Options:
  --batch-size N    Scenes per fresh process (default: 5)
  --jobs N          Concurrent processes (default: 4)
  --conda-env NAME  Conda environment containing Blender Python (default: blender)
  --gpus LIST       Comma-separated physical GPU IDs, e.g. 0,1 (default: 0)
  --log-dir DIR     Batch log directory (default: OUTPUT_DIR/logs-batched)
  --completion-dir DIR
                    Skip scenes with DIR/SCENE/.complete.json. May be repeated
                    to check multiple directories (default: let the renderer
                    check OUTPUT_DIR and its expected files)
  --reverse         Process scenes in reverse order, starting with scenes that
                    would normally be scheduled last
  -h, --help        Show this help

Arguments after -- are passed directly to render_panoramas.py. For example:
  -- --num-panoramas 10 --device OPTIX --width 2048 --height 1024
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

[[ $# -gt 0 ]] || { usage; exit 2; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size)
            [[ $# -ge 2 ]] || die "--batch-size requires a value"
            batch_size=$2
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || die "--jobs requires a value"
            jobs=$2
            shift 2
            ;;
        --conda-env)
            [[ $# -ge 2 ]] || die "--conda-env requires a value"
            conda_env=$2
            shift 2
            ;;
        --gpus)
            [[ $# -ge 2 ]] || die "--gpus requires a value"
            gpu_ids=$2
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "--log-dir requires a value"
            log_dir=$2
            shift 2
            ;;
        --completion-dir)
            [[ $# -ge 2 ]] || die "--completion-dir requires a value"
            completion_dirs+=("$2")
            shift 2
            ;;
        --reverse)
            reverse=true
            shift
            ;;
        --)
            shift
            renderer_args=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown launcher option: $1 (put renderer options after --)"
            ;;
        *)
            if [[ -z "$input_dir" ]]; then
                input_dir=$1
            elif [[ -z "$output_dir" ]]; then
                output_dir=$1
            else
                die "unexpected argument: $1 (put renderer options after --)"
            fi
            shift
            ;;
    esac
done

[[ -n "$input_dir" && -n "$output_dir" ]] || die "INPUT_DIR and OUTPUT_DIR are required"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ -d "$input_dir" ]] || die "input directory does not exist: $input_dir"
[[ -f "$renderer" ]] || die "renderer not found: $renderer"
command -v conda >/dev/null 2>&1 || die "conda is not available on PATH"
if ! conda run -n "$conda_env" python -c 'import sys' >/dev/null 2>&1; then
    die "conda environment is unavailable or cannot run Python: $conda_env"
fi

IFS=',' read -r -a gpus <<< "$gpu_ids"
[[ ${#gpus[@]} -gt 0 ]] || die "--gpus must contain at least one GPU ID"
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"
done

input_dir=$(cd -- "$input_dir" && pwd)
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
for index in "${!completion_dirs[@]}"; do
    completion_dir=${completion_dirs[index]}
    [[ -d "$completion_dir" ]] || die "completion directory does not exist: $completion_dir"
    completion_dirs[index]=$(cd -- "$completion_dir" && pwd)
done
log_dir=${log_dir:-"$output_dir/logs-batched"}
mkdir -p -- "$log_dir"
log_dir=$(cd -- "$log_dir" && pwd)

declare -a scenes=()
input_count=0
completed_count=0
while IFS= read -r -d '' scene; do
    ((input_count += 1))
    if ((${#completion_dirs[@]} > 0)); then
        scene_name=$(basename -- "$scene" .zip)
        completed=false
        for completion_dir in "${completion_dirs[@]}"; do
            if [[ -f "$completion_dir/$scene_name/.complete.json" ]]; then
                completed=true
                break
            fi
        done
        if [[ "$completed" == true ]]; then
            ((completed_count += 1))
            continue
        fi
    fi
    scenes+=("$scene")
# The centered-cubemap driver stages selected scenes as symlinks. Follow them
# for the file-type check, while still ignoring broken links and directories.
done < <(find -L "$input_dir" -maxdepth 1 -type f -name '*.zip' -print0 | sort -z)

if [[ "$reverse" == true ]]; then
    declare -a reversed_scenes=()
    for ((index=${#scenes[@]} - 1; index>=0; index--)); do
        reversed_scenes+=("${scenes[index]}")
    done
    scenes=("${reversed_scenes[@]}")
fi

if ((input_count == 0)); then
    die "no .zip scenes found directly in $input_dir"
fi

if [[ ${#scenes[@]} -eq 0 ]]; then
    if ((${#completion_dirs[@]} > 0)); then
        printf 'No scenes to render; all %d input scenes have completion markers in one of:\n' \
            "$input_count"
        printf '  %s\n' "${completion_dirs[@]}"
        exit 0
    fi
    die "no .zip scenes found in $input_dir"
fi

if ((completed_count > 0)); then
    printf 'Skipped %d of %d input scenes using completion markers\n' \
        "$completed_count" "$input_count"
fi

batch_root=$(mktemp -d "${TMPDIR:-/tmp}/sage-panorama-batches.XXXXXXXX")
declare -a child_pids=()

cleanup() {
    trap - INT TERM EXIT
    for pid in "${child_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    rm -rf -- "$batch_root"
}
trap cleanup INT TERM EXIT

run_batch() {
    local batch_number=$1
    local batch_dir=$2
    local gpu=$3
    local log_file
    log_file=$(printf '%s/batch-%05d.log' "$log_dir" "$batch_number")
    printf 'Starting batch %d on GPU %s; log: %s\n' "$batch_number" "$gpu" "$log_file"
    local status=0
    CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n "$conda_env" \
        python -u "$renderer" "$batch_dir" --output-dir "$output_dir" \
        --allow-scene-failures "${renderer_args[@]}" >"$log_file" 2>&1
    status=$?
    if ((status != 0)); then
        printf 'Batch %d exited with status %d; last log lines (%s):\n' \
            "$batch_number" "$status" "$log_file" >&2
        tail -n 25 -- "$log_file" >&2
    fi
    return "$status"
}

total_batches=$(( (${#scenes[@]} + batch_size - 1) / batch_size ))
printf 'Rendering %d scenes in %d batches (%d scenes/batch, %d concurrent processes)\n' \
    "${#scenes[@]}" "$total_batches" "$batch_size" "$jobs"

active=0
failures=0
stop_dispatch=false
for ((batch=0; batch<total_batches; batch++)); do
    batch_dir=$(printf '%s/batch-%05d' "$batch_root" "$batch")
    mkdir -- "$batch_dir"
    first=$((batch * batch_size))
    last=$((first + batch_size))
    ((last > ${#scenes[@]})) && last=${#scenes[@]}
    for ((index=first; index<last; index++)); do
        ln -s -- "${scenes[index]}" "$batch_dir/$(basename -- "${scenes[index]}")"
    done

    gpu=${gpus[batch % ${#gpus[@]}]}
    run_batch "$batch" "$batch_dir" "$gpu" &
    child_pids+=("$!")
    ((active += 1))

    if ((active >= jobs)); then
        if ! wait -n; then
            ((failures += 1))
            stop_dispatch=true
        fi
        ((active -= 1))
        if [[ "$stop_dispatch" == true ]]; then
            printf 'A batch failed; stopping new batch dispatch. Waiting for active batches...\n' >&2
            break
        fi
    fi
done

while ((active > 0)); do
    if ! wait -n; then
        ((failures += 1))
    fi
    ((active -= 1))
done

if ((failures > 0)); then
    printf '%d batch process(es) failed; inspect %s\n' "$failures" "$log_dir" >&2
    exit 1
fi
printf 'All batches completed successfully. Logs: %s\n' "$log_dir"
