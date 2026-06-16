#!/usr/bin/env bash
set -euo pipefail

# Plan 2: no action encoder, raw chunk action concatenated, wider Q + dynamics.

ALG_NAME=${1:-dp3}
TASK_NAME=${2:-adroit_door_medium}
ADDITION_INFO=${3:-0422noae_wide}
SEED=${4:-100}
NUM_GPUS=${5:-4}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

USE_ACTION_EMBED=False \
USE_CONV_ACTION_EMBED=False \
Q_HIDDEN_DIM=1024 \
DYNAMICS_HIDDEN_DIMS="[1024,1024,512,512]" \
bash "${SCRIPT_DIR}/train_policy_chunk_two_stage1.sh" \
    "${ALG_NAME}" "${TASK_NAME}" "${ADDITION_INFO}" "${SEED}" "${NUM_GPUS}"
