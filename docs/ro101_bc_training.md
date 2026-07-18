# RO101 新数据集 BC 训练

本文记录如何使用新采集并转换的 RO101 双摄像头数据集执行 Behavior Cloning（BC）训练。所有命令均从仓库根目录 `/home/tianma/work/RL-100` 执行。

## 数据集

当前训练数据：

```text
/home/tianma/.cache/huggingface/lerobot/ro101.zarr
```

已验证的数据结构：

```text
30 episodes
13,926 frames at 30 Hz
12,442 training windows
1,484 validation windows

data/state          (13926, 6)              float32
data/action         (13926, 6)              float32
data/image_front    (13926, 480, 640, 3)    uint8 NHWC
data/image_side     (13926, 480, 640, 3)    uint8 NHWC
meta/episode_ends   (30,)                    int64
```

## 训练环境

训练使用 `rl100` Conda 环境：

```bash
conda activate rl100
cd /home/tianma/work/RL-100
```

`lerobot` 环境用于采集和数据转换，但当前没有安装训练入口需要的 Hydra 等依赖，不能直接运行本项目的 BC 训练。

## 推荐训练命令

激活环境后运行：

```bash
cd /home/tianma/work/RL-100

CUDA_VISIBLE_DEVICES=0 bash scripts/train_ro101_bc.sh \
  task.dataset.zarr_path=/home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  training.device=cuda:0
```

也可以不激活环境，直接通过 Conda 运行：

```bash
cd /home/tianma/work/RL-100

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n rl100 \
  bash scripts/train_ro101_bc.sh \
  task.dataset.zarr_path=/home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  training.device=cuda:0
```

命令行指定的 `task.dataset.zarr_path` 会覆盖 `ro101_clip.yaml` 中原有的旧数据路径。

## 后台运行与日志

```bash
cd /home/tianma/work/RL-100
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n rl100 \
  bash scripts/train_ro101_bc.sh \
  task.dataset.zarr_path=/home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  training.device=cuda:0 \
  > logs/ro101_bc.log 2>&1 &
```

查看进程和日志：

```bash
jobs -l
tail -f logs/ro101_bc.log
```

## 不使用 Weights & Biases

默认配置会在线连接 W&B。如果当前没有登录、网络不可用或不需要记录到 W&B，增加覆盖参数：

```bash
cd /home/tianma/work/RL-100

CUDA_VISIBLE_DEVICES=0 bash scripts/train_ro101_bc.sh \
  task.dataset.zarr_path=/home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  training.device=cuda:0 \
  use_wandb=false
```

## 当前训练参数

```text
horizon                       8
n_obs_steps                   5
n_action_steps                4
batch_size                    32
gradient_accumulate_every     2
effective batch size          64
num_epochs                    500
validation interval           5 epochs
checkpoint interval           25 epochs
random seed                   42
kl_annealing_epoch            null (默认使用 num_epochs)
```

每个长度为 8 的窗口使用前 5 帧观测，预测从当前时刻开始的 4 步动作。一个 mini-batch 有 32 个窗口，累积两个 mini-batch 后执行一次参数更新。

## 显存不足时

把 mini-batch 降为 16、梯度累积提高到 4，可以继续保持有效 batch size 为 64：

```bash
cd /home/tianma/work/RL-100

CUDA_VISIBLE_DEVICES=0 bash scripts/train_ro101_bc.sh \
  task.dataset.zarr_path=/home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  training.device=cuda:0 \
  dataloader.batch_size=16 \
  training.gradient_accumulate_every=4
```

如果验证阶段也显存不足，可以同时设置：

```text
val_dataloader.batch_size=8
```

## 只检查最终配置

以下命令只解析和打印配置，不开始训练：

```bash
cd /home/tianma/work/RL-100

conda run --no-capture-output -n rl100 \
  bash scripts/train_ro101_bc.sh \
  task.dataset.zarr_path=/home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  training.device=cuda:0 \
  --cfg job
```

重点确认输出中包含：

```text
zarr_path: /home/tianma/.cache/huggingface/lerobot/ro101.zarr
device: cuda:0
batch_size: 32
num_epochs: 500
gradient_accumulate_every: 2
stop_after_bc: true
```

## 输出目录和 checkpoint

默认输出目录格式：

```text
/home/tianma/work/RL-100/RL-100/data/outputs/ro101_clip/42/YYYY.MM.DD/HH.MM.SS_train_ro101_2d_bc_ro101_clip/
```

其中包含：

```text
config.yaml
checkpoints/latest.ckpt
checkpoints/best.ckpt
```

`best.ckpt` 按验证集动作 RMSE 选择；`latest.ckpt` 每 25 个 epoch 保存，并在训练正常结束时再次保存。

## 从已有运行恢复

恢复时必须指定原运行目录，因为默认 Hydra 输出目录带有新的时间戳。假设原运行目录为 `/absolute/path/to/run`：

```bash
cd /home/tianma/work/RL-100

CUDA_VISIBLE_DEVICES=0 bash scripts/train_ro101_bc.sh \
  task.dataset.zarr_path=/home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  training.device=cuda:0 \
  training.resume=true \
  hydra.run.dir=/absolute/path/to/run
```

该目录下必须存在 `checkpoints/latest.ckpt`。恢复训练时不要更换数据集、模型结构、`horizon` 或观测/动作维度。

## 常用覆盖参数

```bash
# 修改训练轮数
training.num_epochs=300

# 修改随机种子
training.seed=100

# 关闭 W&B
use_wandb=false

# 修改 checkpoint 间隔
training.checkpoint_every=10

# 修改验证间隔
training.val_every=2

# 在前 100 个 epoch 内把 beta_kl 线性增加到 kl_beta，之后保持不变
kl_annealing_epoch=100
```

将覆盖参数追加到推荐训练命令末尾即可。

`kl_annealing_epoch=null` 保持原行为，在完整的 `training.num_epochs` 内完成 KL 退火。指定正整数时，`beta_kl` 在对应 epoch 数内升到目标值，后续 epoch 保持目标值；设置为 `1` 表示从第一个 epoch 起直接使用目标值。
