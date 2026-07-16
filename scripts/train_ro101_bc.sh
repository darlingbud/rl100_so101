#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}/RL-100"
export HYDRA_FULL_ERROR=1

python train_bc.py --config-name=rl100_2d_epsilon_ro101 "$@"
