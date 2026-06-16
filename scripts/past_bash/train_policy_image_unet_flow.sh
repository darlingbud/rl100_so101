#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   bash scripts/train_policy_image_unet_flow.sh rl100 adroit_door_medium 0112 100
#   bash scripts/train_policy_image_unet_flow.sh rl100 adroit_door_medium 0112 100

DEBUG=False
save_ckpt=True

alg_name=${1:?alg_name is required}
task_name=${2:?task_name is required}
addition_info=${3:?addition_info is required}
seed=${4:?seed is required}

config_name='rl100_2d_flow'
exp_name=${task_name}-${alg_name}-flow-${addition_info}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CLIP_STD_MAX_VALUES=${CLIP_STD_MAX_VALUES:-"0.1 0.8"}

gpu_id=$(bash scripts/find_gpu.sh)
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ "${DEBUG}" = True ]; then
    wandb_mode=offline
else
    wandb_mode=offline
fi

cd RL-100

export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=${gpu_id}
export MUJOCO_EGL_DEVICE_ID=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1

act='mish'
encoder_type='resnet'
model='skipnet'
encoder_tag="${encoder_type}"
run_dir="data/outputs_2d_flow/${exp_name}_seed${seed}/${act}/${encoder_tag}/${model}"

for lr in 1e-6 2e-6 1e-5
do
    for rollout_length in 10 5 15 20
    do
        for clip_std_max in 0.1 0.8
        do
            python train.py --config-name=${config_name}.yaml \
                task=${task_name} \
                hydra.run.dir=${run_dir} \
                training.debug=${DEBUG} \
                training.seed=${seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                checkpoint.save_ckpt=${save_ckpt} \
                unio4.bppo_lr=${lr} \
                unio4.rollout_length=${rollout_length} \
                training.resume=True \
                use_action_embed=True \
                horizon=3 \
                n_action_steps=1 \
                n_obs_steps=3 \
                ft_all_actions=False \
                num_inference_steps=10 \
                flow_inference_steps=10 \
                flow_sde_type='cps' \
                flow_noise_level=0.7 \
                flow_sde_window_size=0 \
                flow_logit_normal_sampling=False \
                flow_noise_on_final_step=False \
                flow_cps_logprob_mode='gaussian' \
                only_bc=True \
                offline=True \
                policy.model=${model} \
                policy.act=${act} \
                task.env_runner.eval_episodes=30 \
                task.env_runner.env_num=1 \
                task.dataset.pre_image_norm=True \
                policy.img_shape=[3,224,224] \
                policy.use_aug=True \
                use_wandb=True \
                unio4.idql_eval=False \
                critic.omega=0.7 \
                critic.gamma=0.99 \
                use_vib=True \
                use_recon=True \
                dynamics_type='diffusion' \
                training.num_epochs=600 \
                training.num_critic_epochs=600 \
                dynamics.dynamics_max_epochs=350 \
                dataloader.batch_size=128 \
                val_dataloader.batch_size=128 \
                clip_std_max=${clip_std_max} \
                distill_phase=null
        done
    done
done
