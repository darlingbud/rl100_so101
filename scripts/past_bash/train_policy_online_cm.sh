# Examples:
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_3d_epsilon'
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


gpu_id=$(bash scripts/find_gpu.sh)
# gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

cd RL-100


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${gpu_id}
export EGL_DEVICE_ID=${gpu_id}
act='relu'
encoder_type='dp3vib'
model='skipnet'
run_dir="data/outputs_vib_ablation_compress_fix_seed/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"


for lr_a in 3e-6; do
    for K_epochs in 10; do
            python train.py --config-name=${config_name}.yaml \
                task=${task_name} \
                hydra.run.dir=${run_dir} \
                training.debug=$DEBUG \
                training.seed=${seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                checkpoint.save_ckpt=${save_ckpt} \
                unio4.bppo_lr=3e-6 \
                unio4.rollout_length=30 \
                training.resume=True \
                policy._target_=rl_100.policy.rl100_3d.RL1003D \
                policy.ddim_noise_scheduler.num_train_timesteps=100 \
                policy.cm_noise_scheduler.num_train_timesteps=100 \
                use_action_embed=True \
                horizon=3 \
                n_action_steps=1 \
                n_obs_steps=3 \
                ft_all_actions=False \
                cm_inference_steps=1 \
                num_inference_steps=10 \
                policy.model=$model \
                policy.encoder_type=$encoder_type \
                policy.act=$act \
                unio4.bppo_steps=5000 \
                online=True \
                policy.scheduler_type='ddim' \
                critic.load_pretrain=True \
                ppo.lr_a=${lr_a} \
                ppo.K_epochs=${K_epochs} \
                ppo.lr_c=3e-4 \
                ppo.mini_batch_size=128 \
                ppo.batch_size=2048 \
                ++ppo.use_vec_env_online=False \
                task.env_runner.env_num=4 \
                task.env_runner.eval_episodes=30 \
                ppo.share_encoder=False \
                unio4.eval_times=1 \
                ppo.fix_encoder=False \
                ppo.max_train_steps=1000000 \
                policy.img_shape=[3,84,84] \
                distill_phase='onlin' \
                update_phase='step' \
                distill_loss_type='action_same_noise' \
                ppo.save_online_cp=False \
                ppo.online_cp_save_freq=50 \
                distill2mean=True \
                load_bc=False \
                clip_std_max=0.8 \
                ppo.load_online_cp=False \
                ppo.iql_ft=False \
                ppo.idql_eval=False \
                policy.use_vib=True \
                policy.use_recon=True \
                dynamics_type='diffusion' \
                ppo.recon=True \
                ppo.value_recon=True \
                ppo.online_iql_recon=True \
                ppo.fix_iql_encoder=False \
                ppo.is_share_iql_encoder=False \
                ppo.iql_encoder_update_with='q' \
                policy.joint_opt_encoder=False \
                ppo.per_step_recon=False \
                ppo.force_stochastic_online=True \
                policy.beta_kl=1e-3
    done
done
