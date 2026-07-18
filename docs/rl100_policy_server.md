# RL-100 轻量策略服务端

本文说明当前已实现的 C/S 上位机部分。LeRobot 下位机不在本仓库，本阶段只冻结其数据契约，不实现任何 LeRobot API。

## 组件

```text
RL-100/serve_policy.py
  → WebSocketPolicyServer
    → MessagePack + NumPy protocol
      → RL100PolicyAdapter
        → RL1002D/RL1003D.predict_action()
```

- `rl_100.serving.protocol`：协议版本、MessagePack/NumPy 编解码和结构化错误。
- `rl_100.serving.policy_adapter`：恢复 checkpoint、normalizer 和 Hydra 配置，完成 NumPy/Torch 转换。
- `rl_100.serving.websocket_server`：metadata 握手、推理、reset、健康检查和耗时统计。
- `serve_policy.py`：独立服务端入口。

## 安装通信依赖

```bash
python -m pip install -r RL-100/requirements-serving.txt
```

## 启动

### 启动当前 RO101 latest checkpoint

当前已验证的 checkpoint 是：

```text
/home/tianma/work/RL-100/RL-100/data/outputs/ro101_clip/42/2026.07.18/18.21.28_train_ro101_2d_bc_ro101_clip/checkpoints/latest.ckpt
```

启动时同时使用该次训练保存的完整解析配置：

```text
/home/tianma/work/RL-100/RL-100/data/outputs/ro101_clip/42/2026.07.18/18.21.28_train_ro101_2d_bc_ro101_clip/config.yaml
```

它由 `train_bc.py` 从 `rl100_2d_epsilon_ro101.yaml` 组合并解析后保存，和该
checkpoint 一一对应。服务端会校验 `policy`、`shape_meta`、观测/动作步数与 horizon；
不匹配时拒绝启动。

从工作区根目录启动：

```bash
cd /home/tianma/work/RL-100

RUN_DIR=/home/tianma/work/RL-100/RL-100/data/outputs/ro101_clip/42/2026.07.18/18.21.28_train_ro101_2d_bc_ro101_clip
RL100_CHECKPOINT="$RUN_DIR/checkpoints/latest.ckpt" \
RL100_CONFIG="$RUN_DIR/config.yaml" \
RL100_DEVICE=cuda:0 \
RL100_PORT=8000 \
conda run --no-capture-output -n rl100 ./scripts/serve_ro101_best.sh
```

脚本默认使用 `cuda:0`、监听 `0.0.0.0:8000`，并以 `--weights auto`
优先恢复 checkpoint 中的 EMA 权重。可用环境变量覆盖：

```bash
RL100_PORT=8001 \
RL100_DEVICE=cuda:0 \
RL100_CHECKPOINT=/path/to/another/best.ckpt \
RL100_CONFIG=/path/to/the/same/run/config.yaml \
./scripts/serve_ro101_best.sh
```

当前 checkpoint 的加载结果应为：

```text
policy_name: train_ro101_2d_bc
task_name: ro101_clip
weights_source: model
n_obs_steps: 5
action_horizon: 4
action_dim: 6
```

该 latest checkpoint 对应 epoch 150、global step 29,250，且没有 EMA state，
因此 `--weights auto` 会加载 `model`。

### 通用启动方式

从仓库根目录执行：

```bash
export PYTHONPATH=$(pwd)/RL-100:${PYTHONPATH}

python RL-100/serve_policy.py \
  --checkpoint /path/to/run/checkpoints/best.ckpt \
  --config /path/to/run/config.yaml \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8000
```

默认 `--weights auto`：checkpoint 中存在 `ema_model` 时优先使用，否则使用 `model`。其他参数：

```text
--weights auto|model|ema_model
--stochastic
--use-cm | --no-use-cm
--distill2mean
--max-message-mib 64
--non-strict-checkpoint
```

生产部署应保持 strict checkpoint 加载；`--non-strict-checkpoint` 只用于明确了解缺失字段影响的迁移场景。

## Checkpoint 约束

服务端要求训练产生的 workspace `.ckpt`。`predict_action()` 内部使用 `policy.normalizer` 对 observation 归一化并对 action 反归一化，而 workspace checkpoint 的 `model`/`ema_model` state dict 包含 normalizer。

单独策略目录通常只有 `model.pt` 和 `encoder.pt`，不足以证明 normalizer 已完整恢复，所以不是默认加载格式。

## Metadata 与下位机预留接口

连接成功后，服务端首先发送从 checkpoint 配置生成的 metadata：

- `task_name`、`weights_source`；
- `n_obs_steps`；
- `action_horizon`、`action_dim`；
- `observation_spec`、`action_spec`；
- `deterministic`、`use_cm`、`distill2mean`。

当前 RO101 双相机策略的 observation 是：

```python
{
    "image_front": np.ndarray((5, 3, 480, 640), dtype=np.uint8),
    "image_side": np.ndarray((5, 3, 480, 640), dtype=np.uint8),
    "agent_pos": np.ndarray((5, 6), dtype=np.float32),
}
```

图像在线路上采用 `TCHW`、`uint8`、`[0,255]`，避免把消息体放大为
float32 的四倍。服务端接收后统一转换为 float32；策略内部执行与训练时一致的
`/255`、resize 和 ImageNet mean/std 标准化。相机原始输出通常是 `HWC`，下位机
必须先转成 `CHW`，不能交换 front/side。

未来下位机应维护长度为 `n_obs_steps` 的 observation history。服务端不隐式缓存历史，避免断线、重连或多客户端时混用旧观测。

## 推理协议

请求：

```python
{
    "message_type": "infer_request",
    "protocol_version": 1,
    "request_id": 0,
    "episode_id": "episode-001",
    "step_id": 0,
    "observation": {
        # 必须匹配 metadata 中的 policy_input_shape
    },
}
```

响应：

```python
{
    "message_type": "infer_response",
    "protocol_version": 1,
    "request_id": 0,
    "episode_id": "episode-001",
    "actions": np.ndarray((action_horizon, action_dim), dtype=np.float32),
    "server_time_ns": ...,
    "timing": {
        "preprocess_ms": ...,
        "policy_ms": ...,
        "postprocess_ms": ...,
        "total_ms": ...,
    },
}
```

服务端只返回 `predict_action()` 的 `action`，不传输完整的调试输出 `action_pred`。
当前响应形状为 `(4, 6)`。服务端不裁剪动作、不限制速度，也不完成急停；这些真机
安全约束属于 LeRobot 下位机执行层。

## 两端边界

未来 LeRobot 下位机负责：

- 获取 observation 并映射成 metadata 定义的 key；
- 维护 `n_obs_steps` 历史；
- 相机、点云采集和几何预处理；
- 请求策略并逐步执行 action chunk。

当前上位机负责：

- 校验 key、shape、数值 dtype 和有限值；
- 转 float32、增加 batch 维并搬到计算设备；
- RL-100 normalizer、encoder 和策略推理；
- action 反归一化；
- 返回 NumPy float32 action chunk。

## LeRobot SO101 下位机客户端

仓库提供 `RL-100/lerobot_policy_client.py`，直接使用 LeRobot 连接 SO101 follower
和 front/side 两路 OpenCV 相机。默认是 dry-run，只采集真实观测并请求推理，不驱动电机：

### 使用当前 0718 epoch 150 checkpoint

client 的 LeRobot 环境也需要 MessagePack 和 WebSocket 依赖。首次运行前安装一次：

```bash
cd /home/tianma/work/RL-100
/home/tianma/miniconda3/envs/lerobot/bin/python -m pip install \
  -r RL-100/requirements-serving.txt
```

先在 RL-100 机器的终端 A 启动模型服务：

```bash
cd /home/tianma/work/RL-100

RUN_DIR=/home/tianma/work/RL-100/RL-100/data/outputs/ro101_clip/42/2026.07.18/18.21.28_train_ro101_2d_bc_ro101_clip
RL100_CHECKPOINT="$RUN_DIR/checkpoints/latest.ckpt" \
RL100_CONFIG="$RUN_DIR/config.yaml" \
RL100_DEVICE=cuda:0 \
RL100_HOST=0.0.0.0 \
RL100_PORT=8000 \
conda run --no-capture-output -n rl100 ./scripts/serve_ro101_best.sh
```

在 client 机器的终端 B 先运行 10 秒 dry-run。当前本机 follower 固定端口为
`/dev/follower`，两路有效视频节点为 `/dev/video0` 和 `/dev/video2`：

```bash
cd /home/tianma/work/RL-100

PYTHON_BIN=/home/tianma/miniconda3/envs/lerobot/bin/python \
./scripts/run_lerobot_policy_client.sh \
  --url ws://127.0.0.1:8000 \
  --port /dev/follower \
  --front-camera 0 \
  --side-camera 2 \
  --control-fps 10 \
  --inference-fps 3 \
  --duration 10 \
  --action-log outputs/latest_0718_epoch150_dry_run.jsonl
```

如果 server 和 client 不在同一台机器，将 `127.0.0.1` 替换为 server 的局域网
IP。确认图像、关节顺序、动作幅度和 dry-run 日志后，再启用实机执行：

```bash
cd /home/tianma/work/RL-100

PYTHON_BIN=/home/tianma/miniconda3/envs/lerobot/bin/python \
./scripts/run_lerobot_policy_client.sh \
  --url ws://127.0.0.1:8000 \
  --port /dev/follower \
  --front-camera 0 \
  --side-camera 2 \
  --control-fps 10 \
  --inference-fps 3 \
  --max-relative-target 5 \
  --execute \
  --action-log outputs/latest_0718_epoch150_execute.jsonl
```

执行命令不会跳过确认，仍需在终端输入 `MOVE`。需要跳过重新标定时增加
`--no-calibrate`；只有已经确认当前标定文件与机器人匹配时才能使用。
当前 checkpoint 的本机 smoke 推理耗时约 259 ms，因此先使用 3 Hz 推理；4 步
action chunk 在 10 Hz 控制频率下覆盖约 400 ms。若日志持续出现 underrun，应继续
降低 `--inference-fps` 或控制频率，而不是让请求并发堆积。

### 通用 client 命令

下面是可替换 URL、端口和相机 ID 的通用形式：

```bash
./scripts/run_lerobot_policy_client.sh \
  --url ws://192.168.0.135:8000 \
  --port /dev/follower \
  --front-camera 0 \
  --side-camera 2 \
  --once
```

确认图像映射、关节顺序、返回动作和延迟均正确，并清空机械臂工作空间后，才启用执行：

```bash
./scripts/run_lerobot_policy_client.sh \
  --url ws://192.168.0.135:8000 \
  --port /dev/follower \
  --front-camera 0 \
  --side-camera 2 \
  --control-fps 10 \
  --inference-fps 3 \
  --execute
```

执行模式需要输入 `MOVE` 二次确认。推理和控制使用独立异步循环：控制循环按
`--control-fps` 消费当前 action chunk，通信循环按 `--inference-fps` 请求新 chunk，
推理期间不会暂停动作执行。新 chunk 会替换尚未执行的旧计划，避免积压过时动作。
action chunk 长度完全由服务端返回值决定，客户端连接后打印，不提供手工设置参数。
LeRobot 将每关节单次目标变化限制为 5 个归一化单位，可用
`--max-relative-target` 调整。程序每 5 秒打印实际频率、
平均 RTT 和动作缓冲欠载次数。`Ctrl-C` 会停止循环并断开设备。

需要诊断跨推理 action 抖动时，可增加：

```bash
--action-log outputs/action_chunks.jsonl
```

JSONL 使用 `record_type` 区分记录：`inference` 包含原始 action chunk、对应实时关节
state、RTT、`first_action_minus_state` 和 `first_action_minus_previous_first`；
`plan_update` 记录新 chunk 替换了多少未执行动作；`control` 则包含每个控制周期选中的
chunk/action 索引、原始目标、`send_action()` 返回的限幅后真实发送目标、是否发生裁剪、
发送耗时与调度延迟。由此可以完整区分策略输出、chunk 调度和电机指令问题。

服务端不代替下位机完成相机、点云或机器人坐标系转换；这些规则必须与训练数据一致，待具体 LeRobot 机器人确定后实现。

## 健康检查

```bash
curl --noproxy '*' http://127.0.0.1:8000/healthz
```

模型在监听 WebSocket 前同步加载，因此连接成功时 checkpoint 已恢复。

## 当前 checkpoint 端到端 smoke

先在终端 A 启动服务：

```bash
cd /home/tianma/work/RL-100

RUN_DIR=/home/tianma/work/RL-100/RL-100/data/outputs/ro101_clip/42/2026.07.18/18.21.28_train_ro101_2d_bc_ro101_clip
RL100_CHECKPOINT="$RUN_DIR/checkpoints/latest.ckpt" \
RL100_CONFIG="$RUN_DIR/config.yaml" \
conda run --no-capture-output -n rl100 ./scripts/serve_ro101_best.sh
```

终端 B 检查健康状态，并用训练 Zarr 的前 5 帧发送一次真实请求：

```bash
curl --noproxy '*' http://127.0.0.1:8000/healthz

cd /home/tianma/work/RL-100/RL-100
conda activate rl100
python smoke_policy_server.py \
  --url ws://127.0.0.1:8000 \
  --dataset /home/tianma/.cache/huggingface/lerobot/ro101.zarr
```

脚本使用数据集前 5 帧构造 observation history。成功时会打印 metadata、`(4,6)`
actions 和服务端各阶段耗时。这个脚本只用于验证
checkpoint、协议和 GPU 推理链路，不是 LeRobot 真机控制客户端。

## 数据集动作回放服务

`serve_dataset_actions.py` 使用同一套 WebSocket 协议，但不加载神经网络，适合把
网络、序列化、action queue 和 LeRobot 执行链路从策略质量中隔离出来诊断。

第一步使用 hold 模式。服务端把请求中最新的 `agent_pos` 重复为 4 个动作，机器人应
保持当前位置：

```bash
cd /home/tianma/work/RL-100/RL-100
conda activate rl100
python serve_dataset_actions.py \
  --dataset /home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  --mode hold \
  --port 8000
```

下位机先保持 dry-run，不发送电机命令：

```bash
python lerobot_policy_client.py \
  --url ws://<server-ip>:8000 \
  --control-fps 30 \
  --inference-fps 10 \
  --duration 10
```

确认日志中的 action、RTT 和 underrun 正常后，hold 模式才可增加 `--execute`。客户端
会再次要求输入 `MOVE` 确认。

确认 dry-run 和 hold 正常后，选择一条 episode 回放专家动作：

```bash
python serve_dataset_actions.py \
  --dataset /home/tianma/.cache/huggingface/lerobot/ro101.zarr \
  --mode replay \
  --episode-index 0 \
  --start-step 0 \
  --action-horizon 4 \
  --port 8000
```

回放服务使用请求中的 `step_id`，按
`episode_start + start_step + step_id` 读取动作。到达 episode 末尾后用最后一个动作
填满 chunk；传入 `--loop` 才会循环。数据集是 30 FPS，回放客户端建议使用
`--control-fps 30 --inference-fps 10`。

`replay` 是绝对关节目标轨迹。加 `--execute` 前必须让机械臂处于所选 episode 的初始
姿态附近，并先完成默认 dry-run 和 hold 测试。建议先为 replay 运行同一条不带
`--execute` 的客户端命令，确认动作顺序和范围后再操作真机。

## 测试

```bash
export PYTHONPATH=$(pwd)/RL-100:${PYTHONPATH}
python -m unittest discover -s RL-100/tests -p 'test_*.py'
```

单元测试使用假策略，不依赖 LeRobot、数据集、真机或 GPU。上面的 smoke 脚本则使用
当前真实 0718 epoch 150 `latest.ckpt` 和 RO101 数据完成完整请求。
