#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/donquixote/miniconda3/envs/lerobot/bin/python}"

export PYTHONPATH="${REPO_ROOT}/RL-100${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/RL-100/benchmark_policy_server.py" "$@"
