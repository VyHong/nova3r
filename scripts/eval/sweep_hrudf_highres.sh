#!/usr/bin/env bash
# Sweep HRUDF at higher resolutions after the 129-point baseline.
# Usage: scripts/eval/sweep_hrudf_highres.sh [RESOLUTION ...]

set -uo pipefail

NOVA3R_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$NOVA3R_ROOT"

HRUDF_CONFIG="${HRUDF_CONFIG:-exp_output/hunyuandit_panorama/nova3r_training/version_18/hparams.yaml}"
HRUDF_CHECKPOINT="${HRUDF_CHECKPOINT:-exp_output/hunyuandit_panorama/checkpoints/last.ckpt}"
HRUDF_TEST_LIST="${HRUDF_TEST_LIST:-artifacts/dualmeshudf_0d2ee665be_00001/split.json}"
HRUDF_CONDA_ENV="${HRUDF_CONDA_ENV:-nova3r}"
HRUDF_TIMEOUT_SECONDS="${HRUDF_TIMEOUT_SECONDS:-3600}"
HRUDF_QUERY_BATCH_SIZE="${HRUDF_QUERY_BATCH_SIZE:-10000}"
HRUDF_RECURSIONS="${HRUDF_RECURSIONS:-5}"
HRUDF_DEVICES="${HRUDF_DEVICES:-1}"
HRUDF_FORCE="${HRUDF_FORCE:-0}"
HRUDF_LOG_DIR="${HRUDF_LOG_DIR:-logs/hrudf_highres}"

if (( $# > 0 )); then
    HRUDF_RESOLUTIONS=("$@")
else
    HRUDF_RESOLUTIONS=(193 257)
fi

for HRUDF_REQUIRED_PATH in "$HRUDF_CONFIG" "$HRUDF_CHECKPOINT" "$HRUDF_TEST_LIST"; do
    if [[ ! -f "$HRUDF_REQUIRED_PATH" ]]; then
        echo "Missing required file: $HRUDF_REQUIRED_PATH" >&2
        exit 2
    fi
done

mkdir -p "$HRUDF_LOG_DIR"
HRUDF_SUMMARY="$HRUDF_LOG_DIR/summary.csv"
if [[ ! -f "$HRUDF_SUMMARY" ]]; then
    echo "resolution,cells_per_axis,status,elapsed_seconds,recursions,query_batch_size" > "$HRUDF_SUMMARY"
fi

echo "HRUDF high-resolution sweep"
echo "  resolutions: ${HRUDF_RESOLUTIONS[*]}"
echo "  recursions: $HRUDF_RECURSIONS"
echo "  timeout per run: ${HRUDF_TIMEOUT_SECONDS}s"
echo "  logs: $HRUDF_LOG_DIR"

HRUDF_ANY_FAILURE=0
for HRUDF_RESOLUTION in "${HRUDF_RESOLUTIONS[@]}"; do
    if ! [[ "$HRUDF_RESOLUTION" =~ ^[0-9]+$ ]] || (( HRUDF_RESOLUTION < 17 )); then
        echo "Invalid resolution: $HRUDF_RESOLUTION" >&2
        HRUDF_ANY_FAILURE=1
        continue
    fi

    HRUDF_CASE="res${HRUDF_RESOLUTION}_r${HRUDF_RECURSIONS}"
    HRUDF_LOG="$HRUDF_LOG_DIR/${HRUDF_CASE}.log"
    HRUDF_DONE="$HRUDF_LOG_DIR/${HRUDF_CASE}.done"
    if [[ -f "$HRUDF_DONE" && "$HRUDF_FORCE" != "1" ]]; then
        echo "Skipping completed case: $HRUDF_CASE"
        continue
    fi

    echo "Starting $HRUDF_CASE"
    HRUDF_START=$(date +%s)
    timeout --signal=INT --kill-after=30s "${HRUDF_TIMEOUT_SECONDS}s" \
        conda run --no-capture-output -n "$HRUDF_CONDA_ENV" \
        python training/test.py \
        --config "$HRUDF_CONFIG" \
        --ckpt "$HRUDF_CHECKPOINT" \
        --test-list "$HRUDF_TEST_LIST" \
        --devices "$HRUDF_DEVICES" \
        --mesh-method hrudf \
        --mesh-hrudf-resolution "$HRUDF_RESOLUTION" \
        --mesh-hrudf-query-batch-size "$HRUDF_QUERY_BATCH_SIZE" \
        --mesh-hrudf-recursions "$HRUDF_RECURSIONS" \
        2>&1 | tee "$HRUDF_LOG"
    HRUDF_STATUS=${PIPESTATUS[0]}
    HRUDF_END=$(date +%s)
    HRUDF_ELAPSED=$((HRUDF_END - HRUDF_START))
    HRUDF_CELLS=$((HRUDF_RESOLUTION - 1))

    echo "$HRUDF_RESOLUTION,$HRUDF_CELLS,$HRUDF_STATUS,$HRUDF_ELAPSED,$HRUDF_RECURSIONS,$HRUDF_QUERY_BATCH_SIZE" >> "$HRUDF_SUMMARY"
    echo "Finished $HRUDF_CASE: status=$HRUDF_STATUS elapsed=${HRUDF_ELAPSED}s"

    if (( HRUDF_STATUS == 0 )); then
        printf 'status=0\nelapsed_seconds=%s\n' "$HRUDF_ELAPSED" > "$HRUDF_DONE"
    else
        HRUDF_ANY_FAILURE=1
    fi
done

echo "Sweep summary: $HRUDF_SUMMARY"
exit "$HRUDF_ANY_FAILURE"
