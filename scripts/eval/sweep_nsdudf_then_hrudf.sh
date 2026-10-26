#!/usr/bin/env bash
# Run the default NSDUDF high-resolution sweep, then the HRUDF sweep.

set -uo pipefail

NOVA3R_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$NOVA3R_ROOT"

scripts/eval/sweep_nsdudf_highres.sh
NSDUDF_SWEEP_STATUS=$?

# Continue to HRUDF after NSDUDF finishes even if an individual NSDUDF case
# timed out. Both sweep scripts record per-case status and resume markers.
scripts/eval/sweep_hrudf_highres.sh
HRUDF_SWEEP_STATUS=$?

if (( NSDUDF_SWEEP_STATUS != 0 || HRUDF_SWEEP_STATUS != 0 )); then
    exit 1
fi
