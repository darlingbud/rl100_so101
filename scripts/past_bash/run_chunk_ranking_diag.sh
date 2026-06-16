#!/usr/bin/env bash
set -euo pipefail

# Run chunk ranking diagnostic for both stage1 experiments.
# Usage: bash scripts/run_chunk_ranking_diag.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="${SCRIPT_DIR}/.."

cd "${ROOT_DIR}/RL-100"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

RUN_0112161="data/outputs_two_stage_chunk/adroit_door_medium-dp3-0112161_seed100/mish/dp3vib/dp3"
RUN_0112183="data/outputs_two_stage_chunk/adroit_door_medium-dp3-0112183_seed100/mish/dp3vib/dp3"

NUM_STATES=${DIAG_NUM_STATES:-64}
NUM_CANDIDATES=${DIAG_NUM_CANDIDATES:-32}

echo "=== Chunk Ranking Diagnostic: 0112161 (n_obs_steps=1, horizon=16, beta_kl=1e-5) ==="
DIAG_RUN_DIR="${RUN_0112161}" \
DIAG_CRITIC_ARTIFACT_DIR="${RUN_0112161}/critic_c16_f16" \
DIAG_NUM_STATES="${NUM_STATES}" \
DIAG_NUM_CANDIDATES="${NUM_CANDIDATES}" \
DIAG_OUTPUT_DIR="${RUN_0112161}/chunk_ranking_diagnostic" \
MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 \
python chunk_ranking_diagnostic.py \
    --config-name=rl100_3d_epsilon.yaml \
    hydra.run.dir="${RUN_0112161}" \
    2>&1 | tee "${RUN_0112161}/chunk_ranking_diagnostic.log"

echo ""
echo "=== Chunk Ranking Diagnostic: 0112183 (n_obs_steps=3, horizon=18, beta_kl=1e-3) ==="
DIAG_RUN_DIR="${RUN_0112183}" \
DIAG_CRITIC_ARTIFACT_DIR="${RUN_0112183}/critic_c16_f16" \
DIAG_NUM_STATES="${NUM_STATES}" \
DIAG_NUM_CANDIDATES="${NUM_CANDIDATES}" \
DIAG_OUTPUT_DIR="${RUN_0112183}/chunk_ranking_diagnostic" \
MUJOCO_EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 \
python chunk_ranking_diagnostic.py \
    --config-name=rl100_3d_epsilon.yaml \
    hydra.run.dir="${RUN_0112183}" \
    2>&1 | tee "${RUN_0112183}/chunk_ranking_diagnostic.log"

echo ""
echo "=== Done ==="
