#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RL100_ROOT="${WORKSPACE_ROOT}/RL-100"

DEFAULT_RUN_DIR="${RL100_ROOT}/data/outputs/ro101_clip/42/2026.07.15/22.20.16_train_ro101_2d_bc_ro101_clip"
DEFAULT_CHECKPOINT="${DEFAULT_RUN_DIR}/checkpoints/best.ckpt"
DEFAULT_CONFIG="${DEFAULT_RUN_DIR}/config.yaml"
CHECKPOINT="${RL100_CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
CONFIG="${RL100_CONFIG:-${DEFAULT_CONFIG}}"
DEVICE="${RL100_DEVICE:-cuda:0}"
HOST="${RL100_HOST:-0.0.0.0}"
PORT="${RL100_PORT:-8000}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    echo "Set RL100_CHECKPOINT to an existing workspace .ckpt." >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Training config not found: ${CONFIG}" >&2
    echo "Set RL100_CONFIG to the resolved config.yaml from the checkpoint run." >&2
    exit 1
fi

export PYTHONPATH="${RL100_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${RL100_ROOT}"

exec python serve_policy.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --weights auto \
    --device "${DEVICE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
