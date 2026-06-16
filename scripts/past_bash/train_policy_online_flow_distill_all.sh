#!/usr/bin/env bash

# Plan B pipeline controller:
# 1) Run after_dp distillation + online RL
# 2) Then run online interleaved distillation
#
# Usage examples:
#   bash scripts/train_policy_online_flow_distill_all.sh rl100 adroit_door_medium 0112 100
#   bash scripts/train_policy_online_flow_distill_all.sh rl100 adroit_door_medium 0112 100
#
# Arguments:
#   $1 = alg_name      (e.g., dp3)
#   $2 = task_name     (e.g., adroit_hammer)
#   $3 = addition_info (e.g., 0322)
#   $4 = seed          (e.g., 0)
#   $5 = gpu_id (optional, will be passed through to sub-scripts if provided)

set -e  # exit immediately if a command exits with a non-zero status

alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}

if [ -z "${alg_name}" ] || [ -z "${task_name}" ] || [ -z "${addition_info}" ] || [ -z "${seed}" ]; then
    echo "Usage: bash scripts/train_policy_online_flow_distill_all.sh <alg_name> <task_name> <addition_info> <seed> [gpu_id]"
    exit 1
fi

echo "=============================="
echo "[Step 1] after_dp distillation + online RL"
echo "=============================="

if [ -z "${gpu_id}" ]; then
    bash scripts/train_policy_online_flow_distill.sh "${alg_name}" "${task_name}" "${addition_info}" "${seed}"
else
    bash scripts/train_policy_online_flow_distill.sh "${alg_name}" "${task_name}" "${addition_info}" "${seed}" "${gpu_id}"
fi

echo "============================================="
echo "[Step 2] online interleaved distillation"
echo "============================================="

if [ -z "${gpu_id}" ]; then
    bash scripts/train_policy_online_flow_distill_online.sh "${alg_name}" "${task_name}" "${addition_info}" "${seed}"
else
    bash scripts/train_policy_online_flow_distill_online.sh "${alg_name}" "${task_name}" "${addition_info}" "${seed}" "${gpu_id}"
fi

echo "All stages finished."

