#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    LEROBOT_PYTHON="${HOME}/miniconda3/envs/lerobot/bin/python"
    if [[ -x "${LEROBOT_PYTHON}" ]]; then
        PYTHON_BIN="${LEROBOT_PYTHON}"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

export PYTHONPATH="${REPO_ROOT}/RL-100${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/RL-100/lerobot_policy_client.py" "$@"
