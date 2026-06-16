# Examples:
# bash scripts/train_policy_online_cm_flip.sh rl100 adroit_door_medium 0112 100



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_3d_epsilon'
addition_info=${3}
seed=${4}
ft_seed=${seed}
exp_name=${task_name}-${alg_name}-${addition_info}


gpu_id=$(bash scripts/find_gpu.sh)
# gpu_id=${0}
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
act='relu'
encoder_type='dp3vib'
model='dp3'
run_dir="data/outputs_flipping_0113_800_1024/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"

# 2025-08-17-21-45-47 1999
for lr_a in 1e-6; do
    for K_epochs in 10; do
            python train_real.py --config-name=${config_name}.yaml \
                task=${task_name} \
                hydra.run.dir=${run_dir} \
                training.debug=$DEBUG \
                training.seed=${ft_seed} \
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
                horizon=20 \
                n_action_steps=20 \
                n_obs_steps=1 \
                ft_all_actions=False \
                cm_inference_steps=1 \
                num_inference_steps=10 \
                policy.model=$model \
                policy.encoder_type=$encoder_type \
                policy.act=$act \
                unio4.bppo_steps=5000 \
                online=True \
                policy.encoder_output_dim=64 \
                policy.diffusion_step_embed_dim=128 \
                policy.down_dims="[256,512,1024]" \
                policy.scheduler_type='ddim' \
                critic.load_pretrain=True \
                ppo.lr_a=${lr_a} \
                ppo.K_epochs=${K_epochs} \
                ppo.lr_c=3e-4 \
                ppo.mini_batch_size=128 \
                ppo.batch_size=1024 \
                task.env_runner.env_num=4 \
                task.env_runner.eval_episodes=10 \
                ppo.share_encoder=False \
                unio4.eval_times=1 \
                ppo.fix_encoder=False \
                ppo.max_train_steps=1000000 \
                policy.img_shape=[3,84,84] \
                policy.use_agent_pos=True \
                update_phase='step' \
                distill_loss_type='action_same_noise' \
                policy.mlp_policy_depth=3 \
                ppo.save_online_cp=True \
                ppo.online_cp_save_freq=1 \
                distill2mean=True \
                load_bc=False \
                ppo.load_online_cp=True \
                ppo.iql_ft=False \
                ppo.idql_eval=False \
                policy.use_vib=False \
                policy.use_recon=True \
                dynamics_type='mlp' \
                ppo.recon=True \
                ppo.value_recon=True \
                ppo.online_iql_recon=True \
                ppo.fix_iql_encoder=False \
                ppo.is_share_iql_encoder=False \
                ppo.iql_encoder_update_with=q \
                predict_r=False \
                chunk_as_single_action=True \
                clip_std_max=0.001 \
                offline_cp_timestamp='2026-01-15-19-34-01' \
                offline_cp_timestep='score_4999'\
                task.dataset.zarr_path='data/data_flipping_0103_122_0.01_0.05_True_True_0109_700.zarr' \
                data_collect=True \
                use_wandb=False \
                +cm_eval=False \
                distill_phase='online'  \
                gamma=0.997 \
                dynamics.prediction_mode='full'
                # ppo.scale_strategy='dynamic' \
                # ppo.save_online_cp=True \
                # ppo.online_cp_save_freq=1 \
                # policy.mlp_policy_depth=3 \
                # distill2mean=True \
                # ppo.load_online_cp=False \
                # task.dataset.zarr_path='data/data_rotate_1024.zarr' \
                # clip_std_max=0.8 \
                # ppo.idql_eval=True \
                # task.shape_meta.obs.point_cloud.shape=[2048,3] \
                # task.dataset._target_=diffusion_policy_3d.dataset.rotate_dataset_pc.RotateDataset \
                # policy.use_pc_color=True \
                # task.env_runner.use_rgb=True \
                # task.shape_meta.obs.point_cloud.shape=[2048,6] \
                # task.dataset._target_=diffusion_policy_3d.dataset.rotate_dataset_pc.RotateDataset \
                # policy.use_pc_color=True \
                # task.env_runner.use_rgb=True \
                # task.dataset.zarr_path='data/rotate_17.zarr' \
                # task.critic_dataset.zarr_path='data/push_t_static_20_600.zarr' \
                # task.scale_dataset.zarr_path='data/push_t_static_20_600.zarr' \
                # policy.pointcloud_encoder_cfg.use_layernorm=False \
                # policy.pointcloud_encoder_cfg.final_norm='none' \
                # ppo.scale_strategy='dynamic' \
                # task.env_runner.fake_env=True \
                # clip_std_max=${clip} \
                # task.env_runner._target_=diffusion_policy_3d.env_runner_vec.adroit_runner.AdroitRunner \
                # clip_std_max=${clip_std_max} \
                # ppo.scale_strategy='dynamic' \
        # done
    done
done
