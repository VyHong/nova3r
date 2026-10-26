#!/usr/bin/env bash
# Sweep dense NSDUDF grid resolutions on one fixed evaluation split.
#
# Usage:
#   scripts/eval/sweep_nsdudf_highres.sh
#   scripts/eval/sweep_nsdudf_highres.sh 641 769 897 1025
#
# Optional environment overrides:
#   NSDUDF_TIMEOUT_SECONDS=2400
#   NSDUDF_QUERY_BATCH_SIZE=20000
#   NSDUDF_MAX_AVG_FACTOR=1.2
#   NSDUDF_MAX_MAX_FACTOR=2.0
#   NSDUDF_DEVICES=1
#   NSDUDF_FORCE=1                  # rerun cases with a successful marker
#   NSDUDF_LOG_DIR=logs/nsdudf_highres

set -uo pipefail

NOVA3R_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$NOVA3R_ROOT"

NSDUDF_CONFIG="${NSDUDF_CONFIG:-exp_output/hunyuandit_panorama/nova3r_training/version_18/hparams.yaml}"
NSDUDF_CHECKPOINT="${NSDUDF_CHECKPOINT:-exp_output/hunyuandit_panorama/checkpoints/last.ckpt}"
NSDUDF_TEST_LIST="${NSDUDF_TEST_LIST:-artifacts/dualmeshudf_0d2ee665be_00001/split.json}"
NSDUDF_CONDA_ENV="${NSDUDF_CONDA_ENV:-nova3r}"
NSDUDF_TIMEOUT_SECONDS="${NSDUDF_TIMEOUT_SECONDS:-2400}"
NSDUDF_QUERY_BATCH_SIZE="${NSDUDF_QUERY_BATCH_SIZE:-20000}"
NSDUDF_MAX_AVG_FACTOR="${NSDUDF_MAX_AVG_FACTOR:-1.2}"
NSDUDF_MAX_MAX_FACTOR="${NSDUDF_MAX_MAX_FACTOR:-2.0}"
NSDUDF_DEVICES="${NSDUDF_DEVICES:-1}"
NSDUDF_FORCE="${NSDUDF_FORCE:-0}"
NSDUDF_LOG_DIR="${NSDUDF_LOG_DIR:-logs/nsdudf_highres}"

if (( $# > 0 )); then
    NSDUDF_RESOLUTIONS=("$@")
else
    # The 513-point (512-cell) run is already available as version_47.
    NSDUDF_RESOLUTIONS=(641 769 897 1025)
fi

for NSDUDF_REQUIRED_PATH in "$NSDUDF_CONFIG" "$NSDUDF_CHECKPOINT" "$NSDUDF_TEST_LIST"; do
    if [[ ! -f "$NSDUDF_REQUIRED_PATH" ]]; then
        echo "Missing required file: $NSDUDF_REQUIRED_PATH" >&2
        exit 2
    fi
done

mkdir -p "$NSDUDF_LOG_DIR"
NSDUDF_SUMMARY="$NSDUDF_LOG_DIR/summary.csv"
if [[ ! -f "$NSDUDF_SUMMARY" ]]; then
    echo "resolution,cells_per_axis,status,elapsed_seconds,max_avg_factor,max_max_factor,query_batch_size" > "$NSDUDF_SUMMARY"
fi

echo "NSDUDF high-resolution sweep"
echo "  resolutions: ${NSDUDF_RESOLUTIONS[*]}"
echo "  thresholds: avg=$NSDUDF_MAX_AVG_FACTOR max=$NSDUDF_MAX_MAX_FACTOR"
echo "  timeout per run: ${NSDUDF_TIMEOUT_SECONDS}s"
echo "  logs: $NSDUDF_LOG_DIR"

NSDUDF_ANY_FAILURE=0
for NSDUDF_RESOLUTION in "${NSDUDF_RESOLUTIONS[@]}"; do
    if ! [[ "$NSDUDF_RESOLUTION" =~ ^[0-9]+$ ]] || (( NSDUDF_RESOLUTION < 17 )); then
        echo "Invalid resolution: $NSDUDF_RESOLUTION" >&2
        NSDUDF_ANY_FAILURE=1
        continue
    fi

    NSDUDF_CASE="res${NSDUDF_RESOLUTION}_avg${NSDUDF_MAX_AVG_FACTOR}_max${NSDUDF_MAX_MAX_FACTOR}"
    NSDUDF_LOG="$NSDUDF_LOG_DIR/${NSDUDF_CASE}.log"
    NSDUDF_DONE="$NSDUDF_LOG_DIR/${NSDUDF_CASE}.done"
    if [[ -f "$NSDUDF_DONE" && "$NSDUDF_FORCE" != "1" ]]; then
        echo "Skipping completed case: $NSDUDF_CASE"
        continue
    fi

    echo "Starting $NSDUDF_CASE"
    NSDUDF_START=$(date +%s)
    timeout --signal=INT --kill-after=30s "${NSDUDF_TIMEOUT_SECONDS}s" \
        conda run --no-capture-output -n "$NSDUDF_CONDA_ENV" \
        python training/test.py \
        --config "$NSDUDF_CONFIG" \
        --ckpt "$NSDUDF_CHECKPOINT" \
        --test-list "$NSDUDF_TEST_LIST" \
        --devices "$NSDUDF_DEVICES" \
        --mesh-method nsdudf \
        --mesh-nsdudf-resolution "$NSDUDF_RESOLUTION" \
        --mesh-nsdudf-query-batch-size "$NSDUDF_QUERY_BATCH_SIZE" \
        --mesh-nsdudf-max-avg-factor "$NSDUDF_MAX_AVG_FACTOR" \
        --mesh-nsdudf-max-max-factor "$NSDUDF_MAX_MAX_FACTOR" \
        2>&1 | tee "$NSDUDF_LOG"
    NSDUDF_STATUS=${PIPESTATUS[0]}
    NSDUDF_END=$(date +%s)
    NSDUDF_ELAPSED=$((NSDUDF_END - NSDUDF_START))
    NSDUDF_CELLS=$((NSDUDF_RESOLUTION - 1))

    echo "$NSDUDF_RESOLUTION,$NSDUDF_CELLS,$NSDUDF_STATUS,$NSDUDF_ELAPSED,$NSDUDF_MAX_AVG_FACTOR,$NSDUDF_MAX_MAX_FACTOR,$NSDUDF_QUERY_BATCH_SIZE" >> "$NSDUDF_SUMMARY"
    echo "Finished $NSDUDF_CASE: status=$NSDUDF_STATUS elapsed=${NSDUDF_ELAPSED}s"

    if (( NSDUDF_STATUS == 0 )); then
        printf 'status=0\nelapsed_seconds=%s\n' "$NSDUDF_ELAPSED" > "$NSDUDF_DONE"
    else
        NSDUDF_ANY_FAILURE=1
    fi
done

echo "Sweep summary: $NSDUDF_SUMMARY"
exit "$NSDUDF_ANY_FAILURE"
