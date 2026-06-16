# Examples:
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_2d_epsilon'
addition_info=${3}
seed=${4}
train_env_num=${5:-4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/vrl3_outputs_0628_vit_woloss/${exp_name}_seed${seed}"


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


export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${gpu_id}
export EGL_DEVICE_ID=${gpu_id}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
act='mish'
model='skipnet'
encoder_type='resnet'
run_dir="data/outputs_3d/${exp_name}_seed${seed}/${act}/${model}"
for lr_a in 4.6e-6; do
    for K_epochs in 5; do
        for clip in 0.8; do
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
                policy._target_=rl_100.policy.rl100_2d.RL1002D \
                policy.ddim_noise_scheduler.num_train_timesteps=100 \
                use_action_embed=True \
                horizon=3 \
                n_action_steps=1 \
                n_obs_steps=3 \
                ft_all_actions=False \
                num_inference_steps=10 \
                unio4.bppo_steps=5000 \
                online=True \
                policy.encoder_output_dim=64 \
                policy.down_dims="[256,512,1024]" \
                policy.scheduler_type='ddim' \
                critic.load_pretrain=True \
                ppo.lr_a=${lr_a} \
                ppo.K_epochs=${K_epochs} \
                ppo.lr_c=3e-4 \
                ppo.mini_batch_size=128 \
                ppo.batch_size=2048 \
                task.env_runner.env_num=4 \
                task.env_runner.eval_episodes=5 \
                ++task.env_runner.with_pointcloud=False \
                use_agent_pos=True \
                policy.use_visual=True \
                feature_type='2D' \
                ppo.fix_encoder=False \
                ppo.share_encoder=True \
                policy.model=$model \
                policy.act=$act \
                policy.mlp_policy_depth=3 \
                ++ppo.use_vec_env_online=True \
                ++ppo.train_env_num=${train_env_num} \
                ++ppo.eval_env_num=${train_env_num} \
                policy.use_aug=False \
                ppo.max_train_steps=1000000 \
                encoder_type='resnet' \
                encoders.resnet.share_rgb_model=False \
                encoders.resnet.rgb_model.weights='r3m' \
                task.dataset.pre_image_norm=True \
                task.critic_dataset.pre_image_norm=True \
                task.scale_dataset.pre_image_norm=True \
                only_bc=False \
                clip_std_max=${clip} \
                ppo.scale_strategy='dynamic' \
                ppo.per_step_recon=False \
                ppo.encoder_lr_scale=0.1 \
                use_vib=False \
                use_recon=False \
                dynamics_type='diffusion' \
                ppo.recon=False \
                ppo.value_recon=False \

        done
    done
done
