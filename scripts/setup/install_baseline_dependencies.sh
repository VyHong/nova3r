#!/usr/bin/env bash
# Extract and install the offline extra-dependency bundle; never installs Torch.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage:
  conda activate nova3r
  bash scripts/setup/install_baseline_dependencies.sh [PATH/TO/deps.zip] [--dry-run]
  bash scripts/setup/install_baseline_dependencies.sh [PATH/TO/deps.zip] --install-system-deps

Defaults to deps.zip in the home directory containing this repository.
Requires the same Linux/Python/PyTorch/CUDA platform as the bundle.
Extracts into a temporary directory and installs extra packages offline.
Checks GenReCon's C compiler, Python headers, and shared-library linker first.
--install-system-deps repairs compiler libraries via apt (network and root/sudo required).
VolFill-specific packages install under the active environment's share folder.
Does not install/replace PyTorch, GPU drivers, model weights or baseline repos.
HELP
    exit 0
fi
if [[ ${CONDA_DEFAULT_ENV:-} != nova3r ]]; then
    echo 'Activate the nova3r conda environment first.' >&2
    exit 1
fi
exec python "$root/scripts/setup/install_baseline_bundle.py" "$@"
