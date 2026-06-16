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
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/vrl3_outputs_mlp/${exp_name}_seed${seed}"


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
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=${gpu_id}
export MUJOCO_EGL_DEVICE_ID=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
# CROP_SIZE="[74,74]" 
act='mish'
model='skipnet'
run_dir="data/outputs_3d/${exp_name}_seed${seed}/${act}/${model}"
for lr in 1e-6 2e-6 1e-5
do
    for rollout_length in 10 5 15 20
    do
        for clip_std_max in 0.1 0.8
    do
    python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            unio4.bppo_lr=${lr} \
                            unio4.rollout_length=${rollout_length} \
                            policy._target_=rl_100.policy.rl100_2d.RL1002D \
                            policy.ddim_noise_scheduler.num_train_timesteps=100 \
                            training.resume=True \
                            use_action_embed=True \
                            horizon=3 \
                            n_action_steps=1 \
                            n_obs_steps=3 \
                            ft_all_actions=False \
                            num_inference_steps=10 \
                            unio4.bppo_steps=6000 \
                            offline=True \
                            policy.use_visual=True \
                            policy.model=$model \
                            policy.act=$act \
                            feature_type='2D' \
                            policy.scheduler_type='ddim' \
                            task.env_runner.eval_episodes=30 \
                            task.env_runner.env_num=1 \
                            ++task.env_runner.with_pointcloud=False \
                            policy.use_aug=True \
                            policy.img_shape=[3,224,224] \
                            task.dataset.pre_image_norm=True \
                            only_bc=True \
                            use_recon=False \
                            use_vib=False \
                            training.num_epochs=600 \
                            training.num_critic_epochs=500 \
                            dynamics.dynamics_max_epochs=200 \
                            encoder_type='resnet' \
                            encoders.resnet.share_rgb_model=False \
                            encoders.resnet.rgb_model.weights='r3m' \
                            encoders.resnet.kl_beta=1e-3 \
                            dataloader.batch_size=128 \
                            val_dataloader.batch_size=128 \
                            distill_phase='after_offlin' \
                            distill2mean=True \
                            distill_loss_type='action_same_noise' \
                            dynamics_type='diffusion' \
                            kl_annealing=True \
                            unio4.idql_eval=False \
                            clip_std_max=${clip_std_max}
        done

    done               
done
