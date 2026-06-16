#!/usr/bin/env bash
set -euo pipefail

# Plan 1: Conv action encoder with latent_cz=64 and recon_beta=0.

ALG_NAME=${1:-dp3}
TASK_NAME=${2:-adroit_door_medium}
ADDITION_INFO=${3:-0422convablation_cz64_rb0}
SEED=${4:-100}
NUM_GPUS=${5:-4}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

USE_CONV_ACTION_EMBED=True \
USE_ACTION_EMBED=True \
CONV_LATENT_CZ=64 \
ACTION_RECON_BETA=0 \
Q_HIDDEN_DIM=1024 \
DYNAMICS_HIDDEN_DIMS="[1024,1024,512,512]" \
bash "${SCRIPT_DIR}/train_policy_chunk_two_stage_flow.sh" \
    "${ALG_NAME}" "${TASK_NAME}" "${ADDITION_INFO}" "${SEED}" "${NUM_GPUS}"
