# Vec Env 并行 Rollout 实施指南

## 目标
将 `online_ft` (train_cm_mid.py:989-1438) 的单环境 rollout 改为 vec_env 并行环境，加速数据收集。

## PPO 正确性说明
vec_env **不会**损害 PPO 的 ratio/loss 计算：
- **Ratio** `exp(new_logprob - old_logprob)` 是 per-transition 的，每个 transition 独立存储自己的 old_logprob，与数据来自几个 env 无关
- **GAE** 必须 per-env 计算（现有 `dp_align_update_no_share_vec` 有 bug：flatten 后再算 GAE 会跨 env 传播）。本方案用 `compute_gae_per_env()` 修复
- **Advantage normalization**：全局归一化，标准做法
- **Mini-batch sampling**：flatten 后随机采样，标准做法

## batch_size 语义
保持 `batch_size` 为总 transition 数不变（如 256）。`steps_per_update = batch_size // train_env_num`（如 256//8=32 步/env）。每次 PPO update 的数据总量不变。

## 设计决策

### Feature Flag 守卫
用 `ppo.use_vec_env_online`（默认 False）控制是否启用 vec rollout。当 False 时走**完全不变的原始单 env 路径**，保证零风险回退。

### 独立的 env 数量参数
用 `ppo.train_env_num`（默认 1）控制训练用的并行 env 数量，与 `env_num`（eval 用）解耦。

### v1 Scope 限制
v1 版本**仅支持**主路径：
- `all_step_action_logprob()` rollout
- 标准 reward 和 `scale_strategy == 'number'`
- 标准 PPO loss / ratio / clipping

v1 **显式阻断**以下分支（assert 报错）：
- `ppo.iql_ft`
- `ppo.iql_adv`
- `ppo.idql_rollout`
- `update_phase == 'outloop'`
- `ppo.scale_strategy == 'dynamic'`

这些分支后续版本再逐个适配。

### 手动 env 管理（非 SubprocVecEnv）
手动管理 env 列表，逐个 step。原因：
- `MultiStepWrapper.step()` 有自定义 kwargs（`reward_agg_method`, `gamma`），SubprocVecEnv 不支持
- 对 diffusion policy，**policy 推理（多步 UNet forward）是主要瓶颈**，env stepping（MuJoCo）相对快，真并行 env 的额外加速有限
- 手动管理避免了 SubprocVecEnv 的 IPC、pickling、dict obs 序列化等复杂性
- 手动管理天然避免了 terminal_observation 问题（reset 前就保存了 next_obs）

### 复用现有 PPO update
通过 `precomputed` 参数复用 `dp_align_update_no_share`，不创建新的 vec update 方法。好处：未来单 env 路径的 PPO 改动自动同步到 vec 路径。

## 当前实现现状与已知陷阱

### 现有 `dp_align_update_no_share_vec()` 不能直接接入
`RL-100/rl_100/unidpg/uni_ppo.py` 里虽然已经有 `dp_align_update_no_share_vec()`，但它的实现是：
- 先对 vec buffer 调 `flatten()`
- 再把 `done` / `reward` 当作单条时间序列倒序计算 GAE

这样会把不同 env 的时间序列串起来，导致 **GAE 跨 env 传播**。因此本方案**不复用**这个函数作为主路径，而是：
- 保留 vec buffer 的 `(steps, env_num, *)` 结构
- 先做 per-env GAE
- 再 flatten 成旧 PPO update 可接受的 flat buffer
- 通过 `precomputed` 复用 `dp_align_update_no_share()`

### 现有 `online_buffer_vec.py` 是半成品
`RL-100/rl_100/dppo/online_buffer_vec.py` 当前存在，但仍有以下问题：
- 第一维仍然使用总 `batch_size`，没有切换到 `steps_per_update`
- 没有 `numpy_to_tensor_vec()`，无法保留 `(steps, env_num, *)` 维度直接做 per-env GAE
- `flatten()` 会直接把 `(steps, env_num)` 展平，不能在 GAE 前使用
- 没有同步 `online_buffer.py` 中的 image HWC -> CHW shape 处理

因此该文件需要按本方案显式修复，不能直接拿现状接到训练主循环里。

### `online_ft` 仍然是单 env 主循环
`RL-100/train_cm_mid.py` 当前 `online_ft()` 仍然直接使用：
- `env_runner.env`
- 单个 `obs/reset/step`
- `total_steps += 1`
- `replay_buffer.count == batch_size` 触发 PPO update

这意味着当前不是“已有 vec rollout 只差收尾”，而是**主 rollout 组织方式仍需重写**。实现时应避免在现有双层 while 上做零碎修补，而要按 `train_env_num` 重新组织采样、存储、auto-reset、评估触发和 PPO update。

### Outloop distillation 频率锁定
vec env 版本里，如果未来重新放开 `update_phase == 'outloop'`，频率约定为：
- **每个收集步做一次 distill**

不是：
- 每个 env 各做一次
- 也不是每次 PPO update 前只做一次

这样可以保持与单 env 版本最接近的更新语义，同时避免更新次数随 `train_env_num` 成倍放大。

### Evaluation 触发不能继续使用取模
vec env 路径下 `total_steps` 会按 `train_env_num` 累加，因此：
- 不能再依赖 `total_steps % evaluate_freq == 0`

否则可能直接跳过评估点。应改为：
- `next_eval_at = evaluate_freq`
- 当 `total_steps >= next_eval_at` 时触发评估
- 之后 `next_eval_at += evaluate_freq`

---

## 修改文件清单

| # | 文件路径 | 改动摘要 |
|---|---------|---------|
| 1 | `RL-100/rl_100/env_runner/base_runner.py` | 添加 `make_env()` 方法 |
| 2 | `RL-100/rl_100/env_runner/adroit_runner.py` | 实现 `make_env()` |
| 3 | `RL-100/rl_100/env_runner/metaworld_runner.py` | 实现 `make_env()` |
| 4 | `RL-100/rl_100/env_runner/dmc_runner.py` | 实现 `make_env()` |
| 5 | `RL-100/rl_100/env_runner/dexart_runner.py` | 实现 `make_env()` |
| 6 | `RL-100/rl_100/dppo/online_buffer_vec.py` | 修复 vec buffer，添加 `numpy_to_tensor_vec()` |
| 7 | `RL-100/rl_100/unidpg/uni_ppo.py` | 添加 `compute_gae_per_env()` + `dp_align_update_no_share` 支持 precomputed adv |
| 8 | `RL-100/train_cm_mid.py` | 重写 `online_ft` rollout 循环（feature flag 守卫） |

---

## Task 1: Runner 添加 `make_env()` 方法

### 1.1 修改 `RL-100/rl_100/env_runner/base_runner.py`

在 `BaseRunner` 类中添加：
```python
def make_env(self):
    """Create and return a new independent environment instance."""
    raise NotImplementedError()
```

### 1.2 修改 `RL-100/rl_100/env_runner/adroit_runner.py`

当前代码 (line 44-57):
```python
def env_fn():
    return MultiStepWrapper(
        SimpleVideoRecordingWrapper(
            MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                          env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        reward_agg_method='sum',
        gamma=gamma,
    )
self.eval_episodes = eval_episodes
self.env = env_fn()
```

改为：
1. 先存储必要参数到 self（`use_point_crop`, `gamma` 等 — 检查哪些 `__init__` 参数还没存到 self 上）
2. 将 `env_fn()` 的内容移入 `make_env()` 方法，使用 `self.*` 引用参数
3. 将 `self.env = env_fn()` 改为 `self.env = self.make_env()`

```python
def make_env(self):
    return MultiStepWrapper(
        SimpleVideoRecordingWrapper(
            MujocoPointcloudWrapperAdroit(
                env=AdroitEnv(env_name=self.task_name, use_point_cloud=True),
                env_name='adroit_'+self.task_name, use_point_crop=self.use_point_crop)),
        n_obs_steps=self.n_obs_steps,
        n_action_steps=self.n_action_steps,
        max_episode_steps=self.max_steps,
        reward_agg_method='sum',
        gamma=self.gamma,
    )
```

**注意**：检查 `__init__` 中 `use_point_crop` 和 `gamma` 是否已存为 `self.use_point_crop`, `self.gamma`，如未存需添加。当前 `self.task_name`, `self.n_obs_steps`, `self.n_action_steps`, `self.max_steps` 已存在。

### 1.3 修改 `RL-100/rl_100/env_runner/metaworld_runner.py`

当前 (line 41-52):
```python
def env_fn(task_name):
    return MultiStepWrapper(
        SimpleVideoRecordingWrapper(
            MetaWorldEnv(task_name=task_name, device=device,
                         use_point_crop=use_point_crop, num_points=num_points, rgb_size=84)),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        reward_agg_method='sum',
    )
self.env = env_fn(self.task_name)
```

同理：确保 `device`, `use_point_crop`, `num_points` 存到 self，然后移入 `make_env()`。

### 1.4 修改 `RL-100/rl_100/env_runner/dmc_runner.py`

找到 env 创建代码，同理提取为 `make_env()`。

### 1.5 修改 `RL-100/rl_100/env_runner/dexart_runner.py`

找到 env 创建代码，同理提取为 `make_env()`。DexArt 的 `env_fn` 可能有 `is_test` 参数，`make_env()` 中默认用 `is_test=False`。

---

## Task 2: 修复 vec buffer (`RL-100/rl_100/dppo/online_buffer_vec.py`)

### 2.1 当前 buffer 结构

```python
class ReplayBuffer:
    def __init__(self, args, shape_info, device, env_num, wo_visual=False):
        # 在 reset() 中分配
    def reset(self):
        # 分配 shape (args.batch_size, env_num, *)
        self.action = np.zeros((args.batch_size, env_num, args.num_inference_steps + 1, *shape_info['action']))
        ...
    def store(self, obs, action, a_logprob, reward, next_obs, done, dw):
        # 存一个 timestep 的所有 env 数据
        self.action[self.count] = action  # action shape: (env_num, ...)
        ...
    def flatten(self):
        # reshape(total_size, ...) — 有 bug，混淆 env 时序
    def numpy_to_tensor(self):
        # 转 tensor — 当前只在 flatten 后调用
```

### 2.2 需要的修改

**2.2.1** 添加 `numpy_to_tensor_vec()` 方法 — 返回保留 `(steps, env_num, *)` 维度的 tensor，不做 flatten：

```python
def numpy_to_tensor_vec(self):
    """返回 shape 为 (steps_per_update, env_num, *) 的 tensor，不做 flatten。"""
    if not self.wo_visual:
        point_cloud = torch.tensor(self.point_cloud[:self.count], dtype=torch.float).to(self.device)
        image = torch.tensor(self.image[:self.count], dtype=torch.float).to(self.device)
        if self.use_imagin_robot:
            imagin_robot = torch.tensor(self.imagin_robot[:self.count], dtype=torch.float).to(self.device)
    agent_pos = torch.tensor(self.agent_pos[:self.count], dtype=torch.float).to(self.device)
    action = torch.tensor(self.action[:self.count], dtype=torch.float).to(self.device)
    a_logprob = torch.tensor(self.a_logprob[:self.count], dtype=torch.float).to(self.device)
    reward = torch.tensor(self.reward[:self.count], dtype=torch.float).to(self.device)
    if not self.wo_visual:
        next_point_cloud = torch.tensor(self.next_point_cloud[:self.count], dtype=torch.float).to(self.device)
        next_image = torch.tensor(self.next_image[:self.count], dtype=torch.float).to(self.device)
        if self.use_imagin_robot:
            next_imagin_robot = torch.tensor(self.next_imagin_robot[:self.count], dtype=torch.float).to(self.device)
    next_agent_pos = torch.tensor(self.next_agent_pos[:self.count], dtype=torch.float).to(self.device)
    done = torch.tensor(self.done[:self.count], dtype=torch.float).to(self.device)
    dw = torch.tensor(self.dw[:self.count], dtype=torch.float).to(self.device)

    # 构建 obs/next_obs dict — shape 保持 (steps, env_num, *)
    if not self.wo_visual:
        if self.use_imagin_robot:
            obs = {'point_cloud': point_cloud, 'agent_pos': agent_pos, 'image': image, 'imagin_robot': imagin_robot}
            next_obs = {'point_cloud': next_point_cloud, 'agent_pos': next_agent_pos, 'image': next_image, 'imagin_robot': next_imagin_robot}
        else:
            obs = {'point_cloud': point_cloud, 'agent_pos': agent_pos, 'image': image}
            next_obs = {'point_cloud': next_point_cloud, 'agent_pos': next_agent_pos, 'image': next_image}
    else:
        obs = {'agent_pos': agent_pos}
        next_obs = {'agent_pos': next_agent_pos}

    return obs, action, a_logprob, reward, next_obs, dw, done
```

**2.2.2** 处理 batch_size 参数：当前 `args.batch_size` 在 `reset()` 中用于第一维。对 vec env，这个第一维应该是 `steps_per_update = batch_size // env_num`。

方案：在 `__init__` 中接受一个 `steps_per_update` 参数（或在调用端计算后覆盖 `args.batch_size`）：

```python
def __init__(self, args, shape_info, device, env_num, wo_visual=False, steps_per_update=None):
    ...
    self.steps_per_update = steps_per_update if steps_per_update is not None else args.batch_size
    # reset() 中用 self.steps_per_update 替代 args.batch_size
```

在 `reset()` 中：
```python
self.point_cloud = np.zeros((self.steps_per_update, env_num, *shape_info['obs']['point_cloud']))
# ... 所有数组同理
```

**2.2.3** 图片 shape 处理：检查 `online_buffer.py` line 13-18 的 HWC→CHW 转换。如果 image shape 是 `(n_obs, H, W, 3)` 需要转为 `(n_obs, 3, H, W)`。在 `__init__` 中添加相同逻辑：

```python
if shape_info['obs']['image'][-1] == 3:
    shape_info['obs']['image'] = (
        shape_info['obs']['image'][0],
        shape_info['obs']['image'][-1],
        shape_info['obs']['image'][1],
        shape_info['obs']['image'][2],
    )
```

---

## Task 3: 添加 per-env GAE 计算 (`RL-100/rl_100/unidpg/uni_ppo.py`)

### 3.1 在文件顶部或 ABPPO 类外添加工具函数

```python
def compute_gae_per_env(rewards, dones, dws, vs, vs_, gamma, lamda, n_action_steps=1):
    """
    Per-env GAE computation，避免跨 env 传播。

    Args:
        rewards: (steps, env_num, 1)
        dones:   (steps, env_num, 1)
        dws:     (steps, env_num, 1) — true termination (非 truncation)
        vs:      (steps, env_num, 1) — V(s)
        vs_:     (steps, env_num, 1) — V(s')
        gamma:   discount factor
        lamda:   GAE lambda
        n_action_steps: action chunk size

    Returns:
        adv:      (steps * env_num, 1)
        v_target: (steps * env_num, 1)
    """
    steps, env_num = rewards.shape[0], rewards.shape[1]
    gamma_eff = gamma ** n_action_steps
    deltas = rewards + gamma_eff * (1.0 - dws) * vs_ - vs  # (steps, env_num, 1)

    adv = torch.zeros_like(rewards)  # (steps, env_num, 1)
    gae = torch.zeros(env_num, 1, device=rewards.device)  # per-env accumulator

    for t in reversed(range(steps)):
        gae = deltas[t] + gamma_eff * lamda * gae * (1.0 - dones[t])
        adv[t] = gae

    v_target = adv + vs
    return adv.reshape(-1, 1), v_target.reshape(-1, 1)
```

### 3.2 修改 `dp_align_update_no_share` (line 777)

添加可选参数 `precomputed=None`：

```python
def dp_align_update_no_share(self, replay_buffer, total_steps, precomputed=None):
```

在 GAE 计算块 (lines 789-808) 加条件分支：

```python
s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()
a, a_logprob = a.transpose(0,1), a_logprob.transpose(0, 1)

if precomputed is not None:
    # Vec env 路径：使用预计算的 advantage
    adv = precomputed['adv']
    v_target = precomputed['v_target']
    vs = precomputed['vs']
    if self.args.use_adv_norm:
        adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
else:
    # 原有单 env 路径：完全不变
    adv = []
    gae = 0
    with torch.no_grad():
        if self.args.share_encoder:
            vs, vs_ = self._compute_critic_values_in_chunks(s, s_, use_obs2latent=True)
        else:
            vs, vs_ = self._compute_critic_values_in_chunks(s, s_, use_obs2latent=False)
        deltas = r + (self.args.gamma ** self.cfg.n_action_steps) * (1.0 - dw) * vs_ - vs
        for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
            gae = delta + (self.args.gamma ** self.cfg.n_action_steps) * self.args.lamda * gae * (1.0 - d)
            adv.insert(0, gae)
        adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self._device)
        v_target = adv + vs
        if self.args.use_adv_norm:
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
```

**后续 PPO 优化循环 (lines 813-947) 完全不变**。需要注意 `self.args.batch_size` 在 `BatchSampler` 中的使用 (line 815)：
```python
for index in BatchSampler(SubsetRandomSampler(range(self.args.batch_size)), self.args.mini_batch_size, False):
```
对 vec env，实际数据量是 `steps_per_update * env_num = batch_size`，所以 `self.args.batch_size` 值不变，此处无需修改。但需确保传入的 flat replay_buffer 的 count 等于 `batch_size`。

---

## Task 4: 重写 `online_ft` rollout 循环 (`RL-100/train_cm_mid.py`)

这是最大的改动。以下逐段说明。

### 4.1 添加辅助函数 `stack_obs_dicts`

在 `online_ft` 方法之前或文件顶部添加：

```python
def stack_obs_dicts(obs_list, device, task_name):
    """将 N 个 single-env obs dict 堆叠为 batched obs dict (N, ...)"""
    keys = ['point_cloud', 'agent_pos', 'image']
    if 'dexart' in task_name:
        keys.append('imagin_robot')
    batched = {}
    for key in keys:
        if key in obs_list[0]:
            batched[key] = torch.from_numpy(
                np.stack([obs[key] for obs in obs_list], axis=0)
            ).to(device=device)
    if 'image' in batched:
        batched['image'] = batched['image'].to(torch.float)
    return batched
```

### 4.2 Feature flag + 不支持路径阻断（在 `online_ft` 方法入口处）

```python
use_vec = getattr(self.cfg.ppo, 'use_vec_env_online', False)
train_env_num = getattr(self.cfg.ppo, 'train_env_num', 1) if use_vec else 1

if use_vec and train_env_num > 1:
    # v1 scope 限制：阻断不支持的路径
    assert not getattr(self.cfg.ppo, 'iql_ft', False), \
        "vec_env v1 does not support ppo.iql_ft=True"
    assert not getattr(self.cfg.ppo, 'iql_adv', False), \
        "vec_env v1 does not support ppo.iql_adv=True"
    assert not getattr(self.cfg.ppo, 'idql_rollout', False), \
        "vec_env v1 does not support ppo.idql_rollout=True"
    assert getattr(self.cfg, 'update_phase', 'inloop') != 'outloop', \
        "vec_env v1 does not support update_phase='outloop'"
    assert getattr(self.cfg.ppo, 'scale_strategy', 'none') != 'dynamic', \
        "vec_env v1 does not support ppo.scale_strategy='dynamic'"
```

**当 `use_vec=False` 或 `train_env_num=1` 时，走完全不变的原始单 env 路径。**

### 4.3 环境创建

```python
if use_vec and train_env_num > 1:
    env_runner = self.env_runner
    envs = [env_runner.make_env() for _ in range(train_env_num)]
    steps_per_update = self.cfg.ppo.batch_size // train_env_num
    assert self.cfg.ppo.batch_size % train_env_num == 0, \
        f"batch_size ({self.cfg.ppo.batch_size}) must be divisible by train_env_num ({train_env_num})"
else:
    # 原始路径
    env_runner = self.env_runner
    env = env_runner.env
```

### 4.4 Buffer 初始化（vec 分支）

```python
if use_vec and train_env_num > 1:
    from rl_100.dppo.online_buffer_vec import ReplayBuffer as VecReplayBuffer
    replay_buffer = VecReplayBuffer(
        args=self.cfg.ppo, shape_info=self.shape_info,
        device=self.device, env_num=train_env_num,
        steps_per_update=steps_per_update)
    replay_buffer.reset()
else:
    # 原始路径
    replay_buffer = ReplayBuffer(args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)
```

### 4.5 初始化 per-env 状态（vec 分支，在 rollout 循环前）

```python
if use_vec and train_env_num > 1:
    obs_list = [envs[i].reset() for i in range(train_env_num)]
    episode_rewards = [0.0] * train_env_num
    episode_steps_per_env = [0] * train_env_num
    next_eval_at = self.cfg.ppo.evaluate_freq
```

### 4.6 重写主循环（vec 分支，替换 lines ~1207-1285 的双层 while 循环）

**关键变化**：原来是外层 `while total_steps < max` 套内层 `while not done`（单 env episode）。新版改为单层 `while total_steps < max`，每次迭代 step 所有 env 一次。

**当 `use_vec=False` 时，保留完全不变的原始双层循环。** 以下仅描述 vec 分支：

```python
while total_steps < self.cfg.ppo.max_train_steps:
    # -- clip_std_decay (原 1220-1222) --
    if self.cfg.ppo.clip_std_decay:
        decay_value = self.value_decay(
            initial_value=self.cfg.clip_std_max,
            total_steps=total_steps,
            max_train_steps=self.cfg.ppo.max_train_steps)
        self.unio4._policy.noise_scheduler.clip_std_max = decay_value

    # -- 保存 step 前的 obs (用于 buffer store) --
    # 注意：必须在 step 前保存，auto-reset 后 obs_list[i] 会变
    obs_before_step = [dict(obs) for obs in obs_list]  # shallow copy per env

    # -- 批量 policy 推理 (原 1225-1244) --
    obs_dict_input = stack_obs_dicts(obs_list, self.device, self.cfg.task_name)

    with torch.no_grad():
        # v1 只支持 all_step_action_logprob 路径（idql_rollout 已 assert 阻断）
        action, all_x, a_logprob = self.unio4._policy.all_step_action_logprob(
            obs_dict_input, fix_encoder=self.cfg.ppo.fix_encoder)
        # action:    (train_env_num, n_action_steps, action_dim)
        # all_x:     (T+1, train_env_num, horizon, action_dim)
        # a_logprob: (T, train_env_num, horizon, action_dim) 或 (T, train_env_num, action_dim)

    # 转 numpy
    all_x_np = all_x.detach().cpu().numpy()
    a_logprob_np = a_logprob.detach().cpu().numpy()
    action_np = action.detach().cpu().numpy()

    # -- 逐 env step (原 1247-1282) --
    next_obs_list = [None] * train_env_num
    step_rewards = np.zeros(train_env_num)
    step_dones = np.zeros(train_env_num)
    step_dws = np.zeros(train_env_num)

    for i in range(train_env_num):
        next_obs, reward, done, info = envs[i].step(
            action_np[i], reward_agg_method='discounted_sum', gamma=self.cfg.gamma)

        episode_rewards[i] += reward
        episode_steps_per_env[i] += 1

        # dw: true termination (非 max_steps truncation)
        if done and episode_steps_per_env[i] != self.cfg.task.env_runner.max_steps:
            dw = True
        else:
            dw = False

        # reward scaling (v1 仅支持 'number' 和 'none')
        if self.cfg.ppo.scale_strategy == 'number':
            step_rewards[i] = reward * 0.1
        else:
            step_rewards[i] = reward

        step_dones[i] = float(done)
        step_dws[i] = float(dw)
        next_obs_list[i] = next_obs  # 保存 terminal obs（reset 前）

        # auto-reset
        if done:
            total_episode_r.append(episode_rewards[i])
            print(f'env {i} episode reward: {episode_rewards[i]:.2f}, steps: {episode_steps_per_env[i]}')
            episode_rewards[i] = 0.0
            episode_steps_per_env[i] = 0
            obs_list[i] = envs[i].reset()
        else:
            obs_list[i] = next_obs

    # -- 构建 batched data 存入 vec buffer --
    obs_keys = list(obs_before_step[0].keys())
    obs_batch_np = {k: np.stack([obs_before_step[i][k] for i in range(train_env_num)], axis=0)
                    for k in obs_keys}
    next_obs_batch_np = {k: np.stack([next_obs_list[i][k] for i in range(train_env_num)], axis=0)
                         for k in obs_keys}

    # all_x: (T+1, train_env_num, ...) → (train_env_num, T+1, ...)
    all_x_for_buffer = np.moveaxis(all_x_np, 1, 0)
    a_logprob_for_buffer = np.moveaxis(a_logprob_np, 1, 0)

    replay_buffer.store(obs_batch_np, all_x_for_buffer, a_logprob_for_buffer,
                       step_rewards, next_obs_batch_np, step_dones, step_dws)

    total_steps += train_env_num

    # -- Buffer 满时触发 PPO update (原 1286-1338) --
    if replay_buffer.count == steps_per_update:
        update_num += 1

        # -- 预计算 per-env GAE --
        s_vec, a_vec, a_logprob_vec, r_vec, s_vec_, dw_vec, done_vec = \
            replay_buffer.numpy_to_tensor_vec()
        # 所有 shape: (steps_per_update, train_env_num, *)

        with torch.no_grad():
            # 计算 V(s), V(s') — flatten 送入 critic，再 reshape 回来
            flat_s = dict_apply(s_vec, lambda x: x.reshape(-1, *x.shape[2:]))
            flat_s_ = dict_apply(s_vec_, lambda x: x.reshape(-1, *x.shape[2:]))
            if self.unio4.args.share_encoder:
                flat_vs, flat_vs_ = self.unio4._compute_critic_values_in_chunks(
                    flat_s, flat_s_, use_obs2latent=True)
            else:
                flat_vs, flat_vs_ = self.unio4._compute_critic_values_in_chunks(
                    flat_s, flat_s_, use_obs2latent=False)
            vs = flat_vs.reshape(steps_per_update, train_env_num, 1)
            vs_ = flat_vs_.reshape(steps_per_update, train_env_num, 1)

            from rl_100.unidpg.uni_ppo import compute_gae_per_env
            adv, v_target = compute_gae_per_env(
                r_vec, done_vec, dw_vec, vs, vs_,
                self.cfg.gamma, self.cfg.ppo.lamda, self.cfg.n_action_steps)
            # adv, v_target: (steps_per_update * train_env_num, 1)

            if self.unio4.args.use_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # -- 创建临时 flat buffer 供 dp_align_update_no_share 使用 --
        import copy
        from rl_100.dppo.online_buffer import ReplayBuffer as FlatReplayBuffer
        flat_args = copy.copy(self.cfg.ppo)
        flat_args.batch_size = steps_per_update * train_env_num  # 总 transition 数
        flat_replay = FlatReplayBuffer(args=flat_args, shape_info=self.shape_info,
                                       device=self.device, wo_visual=replay_buffer.wo_visual)

        # 将 vec buffer 数据 flatten 写入 flat buffer 的 numpy 数组
        if not replay_buffer.wo_visual:
            flat_replay.point_cloud = replay_buffer.point_cloud[:steps_per_update].reshape(
                -1, *replay_buffer.point_cloud.shape[2:])
            flat_replay.image = replay_buffer.image[:steps_per_update].reshape(
                -1, *replay_buffer.image.shape[2:])
            if replay_buffer.use_imagin_robot:
                flat_replay.imagin_robot = replay_buffer.imagin_robot[:steps_per_update].reshape(
                    -1, *replay_buffer.imagin_robot.shape[2:])
        flat_replay.agent_pos = replay_buffer.agent_pos[:steps_per_update].reshape(
            -1, *replay_buffer.agent_pos.shape[2:])
        flat_replay.action = replay_buffer.action[:steps_per_update].reshape(
            -1, *replay_buffer.action.shape[2:])
        flat_replay.a_logprob = replay_buffer.a_logprob[:steps_per_update].reshape(
            -1, *replay_buffer.a_logprob.shape[2:])
        flat_replay.reward = replay_buffer.reward[:steps_per_update].reshape(-1, 1)
        if not replay_buffer.wo_visual:
            flat_replay.next_point_cloud = replay_buffer.next_point_cloud[:steps_per_update].reshape(
                -1, *replay_buffer.next_point_cloud.shape[2:])
            flat_replay.next_image = replay_buffer.next_image[:steps_per_update].reshape(
                -1, *replay_buffer.next_image.shape[2:])
            if replay_buffer.use_imagin_robot:
                flat_replay.next_imagin_robot = replay_buffer.next_imagin_robot[:steps_per_update].reshape(
                    -1, *replay_buffer.next_imagin_robot.shape[2:])
        flat_replay.next_agent_pos = replay_buffer.next_agent_pos[:steps_per_update].reshape(
            -1, *replay_buffer.next_agent_pos.shape[2:])
        flat_replay.done = replay_buffer.done[:steps_per_update].reshape(-1, 1)
        flat_replay.dw = replay_buffer.dw[:steps_per_update].reshape(-1, 1)
        flat_replay.count = steps_per_update * train_env_num

        precomputed = {
            'adv': adv,
            'v_target': v_target,
            'vs': flat_vs.reshape(-1, 1),
        }

        # -- PPO update --
        time2 = time.time()
        pre_training_time = time.time()
        actor_loss, critic_loss, bc_loss, distill_loss = self.unio4.dp_align_update_no_share(
            flat_replay, total_steps, precomputed=precomputed)
        if distill_loss != 0:
            distill_losses.append(distill_loss)
        post_training_time = time.time()
        print(f'pure policy updated time: {post_training_time - pre_training_time}')

        time3 = time.time()
        ppo_elapsed = getattr(self.unio4, 'last_ppo_elapsed', None)
        ppo_time_str = f'; ppo loop: {ppo_elapsed:.2f}s' if ppo_elapsed is not None else ''
        print(f'step {total_steps}; collecting data time: {time2 - time1}; '
              f'update time: {time3 - time2}{ppo_time_str}')

        replay_buffer.reset()  # 重置 vec buffer
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        bc_losses.append(bc_loss)
        time1 = time.time()

        if self.cfg.ppo.save_online_cp and update_num % self.cfg.ppo.online_cp_save_freq == 0:
            self.save_online_checkpoints(online_ft_path, update_num, iql)

    # -- Evaluation (原 1343-1432) --
    # 改 == 为 >= 阈值检查，避免 total_steps 增量为 env_num 时跳过
    if total_steps >= next_eval_at:
        next_eval_at += self.cfg.ppo.evaluate_freq
        evaluate_num += 1
        # ... 原有 eval/log/save 逻辑完全不变 ...
        # (self.eval(), self.unio4_eval(), wandb.log, np.savetxt 等)
        # 注意：eval 使用 self.env_runner 自带的 eval env，与 vec train envs 无关
```

### 4.7 整体结构：if/else 分流

```python
def online_ft(self, ...):
    use_vec = getattr(self.cfg.ppo, 'use_vec_env_online', False)
    train_env_num = getattr(self.cfg.ppo, 'train_env_num', 1) if use_vec else 1

    if use_vec and train_env_num > 1:
        # assert 阻断不支持路径（见 4.2）
        # vec 分支（见 4.3-4.6）
        ...
    else:
        # ===== 原始单 env 路径：完全不修改 =====
        # 保留现有 lines 1126-1438 的所有代码不变
        ...
```

这样原始路径**零改动**，最大程度降低回归风险。

---

## Task 5: Config 变更

在使用 3D 在线 finetune 的 config 文件中添加默认值：

```yaml
ppo:
  use_vec_env_online: false    # 是否启用 vec env rollout
  train_env_num: 1             # 训练用并行 env 数量（仅 use_vec_env_online=true 时生效）
```

优先方案：
- 在 `online_ft` 中用 `getattr(..., default)` 提供默认值，这样 **v1 可以不改任何现有 config 文件**

如果后续为了显式配置而落盘到 yaml，则只改**当前实际使用的 DP3CM / vec online-ft 入口 config**，不要顺手扩散到其他 policy variant（如 `dp_image_unet*`、`dp_state*` 等）。

---

## v2 后续扩展（不在本次 scope 内）

当 v1 验证通过后，可逐步添加：
- [ ] `ppo.idql_rollout` 支持（需处理 `sample_action_with_logprob` 的 batch>1）
- [ ] `ppo.scale_strategy == 'dynamic'` 支持（per-env `RewardScaling` 实例）
- [ ] `ppo.iql_ft` 支持（逐 env 存入 iql_buffer）
- [ ] `update_phase == 'outloop'` 支持（outloop distillation 频率调整）
- [ ] SubprocVecEnv 真并行（当 env stepping 成为瓶颈时）

---

## 验证清单

- [ ] **Feature flag 关闭**：`use_vec_env_online=false`（默认），确认走原始路径，行为完全不变
- [ ] **train_env_num=1 回归**：`use_vec_env_online=true, train_env_num=1`，运行几个 PPO update，对比 actor_loss / critic_loss / ratio 与原始路径一致
- [ ] **train_env_num=4 功能**：`use_vec_env_online=true, train_env_num=4`，adroit_door 任务，验证：
  - 多个 env 各自打印 episode reward
  - PPO update 按预期频率触发（每 64 步 = 256 transitions）
  - ratio 在合理范围（~1.0）
  - collecting data time / update 节奏无明显异常；样本吞吐不应明显回退
- [ ] **train_env_num=8 功能**：同上，train_env_num=8
- [ ] **GAE 正确性**：构造简单数据，验证 `compute_gae_per_env` 在 `train_env_num=1` 时与原始标量 GAE 结果完全一致
- [ ] **不支持路径阻断**：开启 vec + iql_ft=true，确认 assert 报错
- [ ] **Eval 不受影响**：确认 eval 仍使用 env_runner 自带的单 eval env，不受 vec train envs 影响

---

## 第一轮 Review 结果（需先修复）

以下问题来自第一轮 work agent 实现后的代码 review。这些问题未修复前，vec env 路径**不能认为是正确接入**。

### 1. 不能继续使用现有 `dp_align_update_vec()`
当前 work agent 在 vec 分支里调用了 `self.unio4.dp_align_update_vec(...)`。

这条路径不符合本方案要求，因为 `RL-100/rl_100/unidpg/uni_ppo.py` 中的现有实现会：
- 先 `replay_buffer.flatten()`
- 再把 flatten 后的数据当作单条时间序列计算 GAE

这会导致 **不同 env 的 GAE 串联传播**。必须改回本方案原定实现：
- 保留 `(steps, train_env_num, *)` 结构
- 先计算 per-env GAE
- 再 flatten 成 flat replay buffer
- 通过 `precomputed` 复用 `dp_align_update_no_share()`

换句话说：
- `dp_align_update_vec()` 不能作为本轮主路径
- `uni_ppo.py` 和 `online_buffer_vec.py` 仍然必须按前文 Task 2 / Task 3 修改

### 2. PPO update 触发条件按错了维度
当前实现使用：

```python
if replay_buffer.count == self.cfg.ppo.batch_size:
```

但现有 `online_buffer_vec.ReplayBuffer.count` 的语义是“存了多少个 vec step”，不是“存了多少条 transition”。

在 vec env 下，每次 `store()` 实际写入的是：
- `train_env_num` 条 transition

因此如果仍然等到 `count == batch_size` 才 update，就会变成：
- 实际收集了 `batch_size * train_env_num` 条 transition 才做一次 PPO update

这会直接改变 PPO 数据量与更新频率。

必须改为：
- 引入 `steps_per_update = batch_size // train_env_num`
- vec buffer 第一维使用 `steps_per_update`
- 当 `replay_buffer.count == steps_per_update` 时触发 PPO update

### 3. 当前 `SubprocVecEnv` 自动 reset 语义与训练数据不兼容
当前实现改成了 `SubprocVecEnv`，但本仓库里的 `SubprocVecEnv` worker 会在 `done=True` 时：
- 把真正的 terminal observation 放进 `info["terminal_observation"]`
- 然后立刻 `reset()`
- 返回给上层的是 reset 后的新 observation

这与当前 vec rollout 写 buffer 的方式冲突，因为代码现在直接把返回的 `next_obs` 存进 replay buffer。

结果是：
- done transition 的 `next_obs` 不是 terminal observation
- `dw` / `done` / `s'` 三者可能不一致
- critic target 和 GAE 都会被污染

如果继续使用 `SubprocVecEnv`，必须显式处理 `info["terminal_observation"]`。

但按本方案，**更推荐不要走 `SubprocVecEnv`**，而是：
- 手动管理 env 列表
- 逐 env `step(...)`
- 在 reset 前保留真实 `next_obs`

这样也能继续传 `reward_agg_method` 和 `gamma`。

### 4. vec rollout 改坏了 reward 聚合语义
原单 env 主路径调用的是：

```python
env.step(action, reward_agg_method='discounted_sum', gamma=self.cfg.gamma)
```

当前 vec 实现改成了：
- `SubprocVecEnv.step(action_batch)`

这样无法向底层 `MultiStepWrapper.step()` 传 `reward_agg_method='discounted_sum'` 和 `gamma`，实际会退回 wrapper 初始化时的默认配置（当前 runner 里多处是 `'sum'`）。

这会直接改变：
- rollout reward
- value target
- GAE
- PPO 学到的目标

因此本轮实现不能继续依赖 `SubprocVecEnv.step()` 作为训练主路径。

### 5. `dw` 不能用全局 `episode_steps` 计算
当前 vec 分支用单个 `episode_steps` 和 `max_steps` 来判断：
- 一个 env 的 `done` 是否属于 true termination

但 vec env 下每个 env 的 episode 长度都可能不同，`dw` 必须是 **per-env 状态**。

必须改为：
- 维护 `episode_steps_per_env[i]`
- 每个 env 独立判断其 `dw`

### 6. 当前配置与 runner 改动不匹配
这次改动只给：
- `metaworld_runner.py`
- `dexart_runner.py`

加了 `vec_env`，但 `dp3_vec.yaml` 默认 task 仍然是：
- `adroit_hammer`

这意味着只要直接启用：

```yaml
ppo.use_vec_env_online: true
```

默认配置就会在 adroit 路径上因为 `env_runner.vec_env` 不存在而失败。

因此二选一：
- 要么补齐 adroit runner 的 vec 训练路径支持
- 要么把 config / 文档明确收窄到当前真正支持的 task，不能保留 adroit 默认入口

## 对下一轮 work agent 的明确要求

- 不要再沿着 `dp_align_update_vec()` 修补，回到 `todo_vec.md` 前文定义的 per-env GAE + `precomputed` 方案。
- 不要把 `SubprocVecEnv` 当作本轮训练主路径；优先改为手动 env 列表管理。
- 修复 vec buffer 的第一维语义，引入 `steps_per_update`。
- 在修改完成前，重新对照本节 6 个问题逐项自检。

---

## 第二轮 Review 结果（剩余修复项）

第二轮实现已经修掉了第一轮 review 中最核心的数学和数据流问题：
- 不再用 `SubprocVecEnv` 做训练主路径
- 已切到手动 env 列表管理
- 已引入 `steps_per_update`
- 已加入 `compute_gae_per_env()` 和 `precomputed` 路径

但仍有以下剩余问题需要修复。

### 1. `ppo.idql_eval` 在 vec 分支里被静默丢失
当前单 env `online_ft` 在 `ppo.idql_eval=True` 时会：
- 调用 `self.unio4_eval(..., idql_eval=True)`
- 记录 `all_idql_success_rates` / `all_idql_returns`
- 写出 `idql_success_rates.csv` / `idql_returns.csv`

但当前 vec 分支没有对应逻辑，也没有显式 assert 把它列为“不支持路径”。

这会造成：
- 配置打开时行为悄悄变化
- 用户以为还在做 idql_eval，实际没有执行

必须二选一：
- 要么在 vec 分支里补齐与单 env 一致的 `idql_eval` 逻辑
- 要么在 vec v1 的入口 guard 里显式 `assert not self.cfg.ppo.idql_eval`

不能保持当前这种“配置存在但被静默忽略”的状态。

### 2. `distill_phase == 'online'` 的 cm 指标日志/落盘回归了
当前 vec 分支仍会做：
- `self.eval(..., use_cm=True)`

但没有再把以下内容完整写回：
- wandb 中的 `cm_success rates`
- wandb 中的 `cm_returns`
- `cm_success_rates.csv`
- `cm_returns.csv`

这和单 env 路径相比是明显的可观察行为回归。

必须二选一：
- 要么恢复与单 env 路径一致的 cm logging / csv 落盘
- 要么在 vec v1 中显式阻断 `distill_phase == 'online'`

当前更推荐：
- 直接恢复日志与 CSV 行为，保持输出口径一致

### 3. DMC runner 的并行 env 仍然共用同一个 seed
`DMCRunner.make_env()` 现在每次都用同一个：

```python
self._seed
```

这会让多个并行 DMC env 高度相关，甚至可能退化为重复采样。

必须改成：
- 每个新 env 使用不同 seed

例如：
- `self._seed + env_idx`
- 或 runner 内维护一个递增计数器，在每次 `make_env()` 时分配新 seed

要求是：
- 单 env 默认行为不变
- vec env 下不同副本不能共享同一个 seed

## 对下一轮 work agent 的明确要求

- 先决定 `ppo.idql_eval` 是“支持”还是“显式阻断”，但不能继续静默忽略。
- 恢复 vec 分支下 `cm` 指标的 wandb / CSV 输出，除非明确改成阻断路径。
- 修复 DMC vec env 的 seed 分配，避免多 env 重复采样。
- 修改完成后，补一次最小自检：
  - vec + `idql_eval=true` 的行为是否明确
  - vec + `distill_phase=online` 时是否仍写出 cm 指标
  - DMC 多 env 的 seed 是否不同

---

## 第二轮执行（加入 vec `iql_ft` + `idql_eval`）

这一轮不再停留在“阻断不支持路径”，而是在当前已通过第一轮 vec `online_ft` 主路径上，继续接通：
- `ppo.iql_ft`
- `ppo.idql_eval`

同时明确保持以下分支**继续阻断**：
- `ppo.idql_rollout`
- `ppo.iql_adv`
- `update_phase == 'outloop'`
- `ppo.scale_strategy == 'dynamic'`

### 执行目标

本轮目标只有两件事：
- 在 vec `online_ft` 中加入并行 online IQL fine-tune 过程
- 恢复 vec 分支下与单 env 一致的 `idql_eval` 评估/日志/CSV 输出

本轮**不做**：
- batch>1 的 `sample_action_with_logprob()` rollout
- `idql_rollout` 训练路径
- `iql_adv` 路径
- outloop distillation 与 vec iql_ft 的联动
- 动态 reward scaling

### 1. 调整 vec 分支入口 guard

在 `train_cm_mid.py::_online_ft_vec()` 入口处：
- 删除 `assert not self.cfg.ppo.iql_ft`
- 删除 `assert not self.cfg.ppo.idql_eval`

保留以下 guard 不变：
- `assert not self.cfg.ppo.idql_rollout`
- `assert not self.cfg.ppo.iql_adv`
- `assert self.cfg.update_phase != 'outloop'`
- `assert self.cfg.ppo.scale_strategy != 'dynamic'`

说明：
- `idql_eval` 本轮只恢复评估逻辑，不代表放开 `idql_rollout`
- 训练 rollout 仍然只允许 `all_step_action_logprob()`

### 2. 在 vec rollout 中加入 IQL buffer

在 vec 分支中显式创建 `IqlBuffer`，不要继续依赖单 env 路径中的局部变量：

```python
from rl_100.dppo.online_buffer import IqlBuffer
if self.cfg.ppo.iql_ft:
    iql_buffer = IqlBuffer(None, args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)
    iql = iql_online
```

要求：
- 复用现有 `online_buffer.IqlBuffer`
- 不新建 vec 专用的 IQL 训练数据流

### 3. 在逐 env step 循环中逐条写入 IQL buffer

在 vec rollout 的 `for i in range(train_env_num)` 循环内部，每个 env 的 transition 产生后立刻写入 `iql_buffer`。

存储语义必须与单 env 路径保持一致：
- `obs` 使用 `obs_before_step[i]`
- `action` 使用 `all_x_np[-1, i]`
- `reward` 使用该 env 的**原始 reward**
- `next_obs` 使用 reset 前保留的 `next_obs_list[i]`
- `done` 使用该 env 的 done

这里要明确区分两条 reward 路径：
- `replay_buffer`：服务 PPO，可继续使用 `scale_strategy` 之后的 reward
- `iql_buffer`：服务 IQL，**必须始终存原始 environment reward**

不要这样做：
- 把 `step_rewards[i]`（例如 `reward * 0.1`）写进 `iql_buffer`

不要这样做：
- 等到 PPO buffer 满了以后才统一存一次
- 用 `action_np[i]` 替代 `all_x_np[-1, i]`
- 用 reset 之后的新 obs 写入 `next_obs`

原因：
- `iql_buffer` 本质上存的是 online 单条 transition
- vec env 下如果不按 env 逐条写，会丢失大部分 online 样本
- offline IQL 使用的是原始 reward，online IQL 也必须保持同一 reward 口径
- PPO 的 reward scaling / value scaling 是独立路径，不能反向污染 IQL 的 Q/V 训练目标

### 4. 在 vec PPO update 后接入 online IQL training

在 vec PPO update 触发点（即 `replay_buffer.count == steps_per_update`）之后，沿用单 env 的 online IQL 更新逻辑：

前置条件：
- `if self.cfg.ppo.iql_ft`
- `if total_steps > self.cfg.ppo.online_start_training`

更新过程保持如下：
- 循环 `self.cfg.ppo.iql_steps`
- 计算
  - `alpha = self.cfg.ppo.data_ratio + (1 - self.cfg.ppo.data_ratio) * (total_steps / self.cfg.ppo.max_train_steps)`
  - `online_sample_size = int(alpha * 256)`
  - `offline_sample_size = 256 - online_sample_size`
- `online_batch = iql_buffer.sample(batch_size=online_sample_size)`
- `offline_batch = next(iter(self.train_dataloader))`
- `offline_batch` 继续做与单 env 相同的裁切：
  - `action[:, self.cfg.n_obs_steps - 1:]`
  - `reward[:, self.cfg.n_obs_steps - 1:]`
  - `not_done[:, self.cfg.n_obs_steps - 1:]`
- 若 `self.cfg.action_norm` 为真，继续复用现有 `normalizer['action']`
- `merged_batch = iql_buffer.merge(online_batch, offline_batch)`
- `merged_batch = dict_apply(merged_batch, lambda x: x[:256])`
- `Q_bc_loss, value_loss = iql.update(batch=merged_batch, online=True, pre_cut=True, online_recon=self.cfg.ppo.online_iql_recon)`

后处理也必须保持一致：
- 若 `self.cfg.ppo.fix_encoder` 且 `iql_q_encoder=True`，回灌 `iql._Q._obs_encoder`
- 若 `self.cfg.ppo.fix_encoder` 且 `iql_v_encoder=True`，回灌 `iql._value._obs_encoder`

### 5. 恢复 vec 分支下的 `idql_eval`

在 vec 分支中恢复与单 env 一致的 `ppo.idql_eval` 逻辑。

初始评估阶段：
- 若 `self.cfg.ppo.idql_eval=True`
  - 调用 `self.unio4_eval(idql_eval=True, dynamics=dynamics, first_action=self.cfg.unio4.first_action, get_np=True, use_gae=self.cfg.unio4.use_gae, iql=iql, Q=Q, repeat_num=128, eval_times=self.cfg.unio4.eval_times)`
  - 写入 `all_idql_success_rates`
  - 写入 `all_idql_returns`

周期评估阶段：
- 同样在 eval 分支中调用上述 `self.unio4_eval(..., idql_eval=True, ...)`
- 继续维护：
  - `all_idql_success_rates`
  - `all_idql_returns`

日志与 CSV：
- wandb 恢复：
  - `idql_success rates`
  - `idql_returns`
- 输出目录恢复：
  - `idql_success_rates.csv`
  - `idql_returns.csv`

注意：
- 这里恢复的是评估分支
- 不是开放 `idql_rollout` 训练采样

### 6. 范围与配置约束

本轮只保证**当前 CM 在线路径**：
- 当前 vec `online_ft` 主路径
- 当前在线 CM 训练脚本 / override 组合

本轮不要求：
- 顺手补齐所有 vec config
- 泛化到所有 policy variant
- 放开 `dp3_vec.yaml` 之外的任意历史遗留分支

换句话说：
- 如果当前在线实验主用 `dp3_cm_epsilon` + override，就只保证这条路径能用
- `dp3_vec.yaml` 可只补最小必要项，不要求一次性补齐所有 IQL 在线参数面

### 7. 第二轮执行后的最小验证集

- [ ] `ppo.iql_ft=true`, `ppo.idql_eval=false`, `train_env_num=4`
  - 至少完成 1 次 vec PPO update
  - 至少完成 1 次 online IQL update
  - `iql_buffer.sample()` / `merge()` / `iql.update()` 无 shape 错误
- [ ] `ppo.iql_ft=true`, `ppo.idql_eval=true`, `train_env_num=4`
  - 初始评估和周期评估都能产出 `idql` 指标
  - wandb 包含 `idql_success rates` / `idql_returns`
  - 输出目录写出 `idql_success_rates.csv` / `idql_returns.csv`
- [ ] vec + `ppo.idql_rollout=true`
  - 仍然 assert 阻断
- [ ] vec + `ppo.iql_adv=true`
  - 仍然 assert 阻断
- [ ] vec + `update_phase='outloop'`
  - 仍然 assert 阻断
- [ ] vec + `ppo.scale_strategy='dynamic'`
  - 仍然 assert 阻断

### 8. 对本轮 work agent 的明确要求

- 不要改回 `dp_align_update_vec()`，继续沿用当前 per-env GAE + `precomputed` 主路径。
- 只在现有 vec `online_ft` 分支上追加 `iql_ft` 和 `idql_eval`，不要扩散范围。
- `iql_buffer` 的写入必须 per-env、per-transition，即时发生。
- `idql_eval` 只恢复评估，不要顺手打开 `idql_rollout`。
- 修改完成后，先按“第二轮执行后的最小验证集”自检，再进入下一轮 review。

---

## 第三轮执行（加入 vec `scale_strategy='dynamic'`）

这一轮的目标是在当前已经可用的 vec `online_ft` 主路径上，补齐：
- `ppo.scale_strategy == 'dynamic'`

同时明确保持以下分支**继续阻断**：
- `ppo.idql_rollout`
- `ppo.iql_adv`
- `update_phase == 'outloop'`

这一轮不是重写 reward normalization 体系，而是把单 env 已有的 dynamic reward scaling 语义，按“每个 env 一个独立 scaler”的方式接到 vec rollout 上。

### 执行目标

本轮目标只有一件事：
- 在 vec `online_ft` 中加入并行环境下的 dynamic reward scaling

本轮**不做**：
- batch 版 `RewardScaling`
- 修改 `RewardScaling` 类接口
- 改写单 env `online_ft`
- 打开 `idql_rollout`
- 打开 `iql_adv`
- 打开 `update_phase == 'outloop'`

### 1. 调整 vec 分支入口 guard

在 `train_cm_mid.py::_online_ft_vec()` 入口处：
- 删除 `assert self.cfg.ppo.scale_strategy != 'dynamic'`

保留以下 guard 不变：
- `assert not self.cfg.ppo.idql_rollout`
- `assert not self.cfg.ppo.iql_adv`
- `assert self.cfg.update_phase != 'outloop'`

要求：
- `scale_strategy in {'none', 'number', 'dynamic'}` 都走同一套 vec rollout 主循环
- 不新增新的 vec 专用配置项

### 2. 初始化 per-env `RewardScaling`

先明确数据来源与函数签名：
- `reward_scaler = scale_dataset.reward_norm` 目前是 `online_ft` 内的局部变量
- `_online_ft_vec()` 当前签名拿不到它

本轮明确采用**方案 A**：
- 修改 `_online_ft_vec(...)` 签名，新增 `reward_scaler_template=None`
- 在 `online_ft(...)` 中调用 vec 分支时，把单 env 已准备好的 scaler 模板传入
- 在 `online_ft(...)` 中，先给 `reward_scaler_template` 一个默认值 `None`
- 只有当 `self.cfg.ppo.scale_strategy == 'dynamic'` 时，才将其赋值为 `scale_dataset.reward_norm`

要求：
- 不要在 `_online_ft_vec()` 内重新复制一套 `scale_dataset` 获取逻辑
- 不要让 `_online_ft_vec()` 依赖外层未显式传入的局部变量
- `scale_strategy != 'dynamic'` 时，`reward_scaler_template=None` 也必须安全运行，避免 `NameError`

在 vec rollout 初始化阶段，只有当 `self.cfg.ppo.scale_strategy == 'dynamic'` 时，显式创建：
- `reward_scalers = [scaler_0, scaler_1, ..., scaler_{train_env_num-1}]`

实现要求：
- 每个并行 env 都必须拥有**独立**的 `RewardScaling` 状态
- 不能让多个 env 共用同一个 scaler 实例
- 复用从 `online_ft(...)` 显式传入的 `reward_scaler_template`
- 具体实现建议为：
  - 对 `reward_scaler_template` 做逐个 `deepcopy`
  - 不要写成同一对象的重复引用

约束：
- 不修改 `RewardScaling` 类签名
- 不把 `RewardScaling` 改成 batch 版本
- v1 继续采用“每 env 一个标量 scaler 对象”的简单方案

如果运行时发现 `scale_dataset.reward_norm` 不能安全复制，则退而求其次：
- 用与单 env 相同参数重新实例化 `RewardScaling(shape=1, gamma=self.cfg.gamma)`

### 3. 在逐 env step 循环中独立计算 scaled reward

在 vec rollout 的 `for i in range(train_env_num)` 循环内部，当前 reward 分支需要扩展为：
- `number`：继续使用 `reward * 0.1`
- `dynamic`：调用该 env 自己的 scaler
  - `scaled_r = reward_scalers[i](reward)[0]`
- 其他默认分支：继续使用原始 `reward`

这里要明确区分两条 reward 路径：
- `replay_buffer`：服务 PPO，可存 `number` 或 `dynamic` 之后的 reward
- `iql_buffer`：服务 IQL，**必须始终存原始 environment reward**

不要这样做：
- 把 `scaled_r` 写进 `iql_buffer`
- 把 `step_rewards[i]` 写进 `iql_buffer`
- 把多个 env 共用同一个 dynamic scaler

原因：
- PPO 的 reward scaling / value scaling 是独立路径
- offline IQL 与 online IQL 的 reward 口径必须保持原始 reward 一致
- dynamic 只应该影响 PPO 的 value / GAE / actor-critic 更新，不应该污染 IQL 的 Q/V 训练目标

### 4. 某个 env 结束时只 reset 它自己的 scaler

在 vec rollout 中，每个 env 都会独立结束。dynamic 分支下必须满足：
- 某个 env `done` 时，只 reset 对应的 `reward_scalers[i]`
- 其他 env 的 scaler 状态保持不变

顺序要求：
- 先用当前 episode 的 reward 计算完这一条 transition 的 `scaled_r`
- 再完成 buffer 存储
- 然后若 `done`，调用 `reward_scalers[i].reset()`
- 最后再 reset 对应环境，并清空该 env 的 `episode_rewards[i]` / `episode_steps_per_env[i]`

不要这样做：
- 在每个 vec step 开头统一 reset 全部 scaler
- 因为一个 env done 就 reset 全部 env 的 scaler

### 5. 保持 PPO / IQL / value 的语义分层

本轮必须保持以下语义不变：

- `replay_buffer`
  - `scale_strategy == 'number'` 时存缩放 reward
  - `scale_strategy == 'dynamic'` 时存 per-env dynamic scaled reward
  - 其他情况存原始 reward
- `iql_buffer`
  - 永远存原始 environment reward
- GAE / value target
  - 继续基于 vec PPO buffer 中的 reward 计算
  - 继续走当前 per-env GAE + `precomputed` 主路径
- 单 env `online_ft`
  - 不做语义改动

换句话说：
- vec dynamic 只是在 PPO reward 路径上补齐单 env 已有能力
- 不是去改 PPO 主公式
- 更不是去改 IQL 的 reward 定义

### 6. 可观测性与最小调试输出

本轮以功能正确为主，不要求新增复杂日志面板。

要求：
- 保留现有 episode reward 打印
- 若需要检查 dynamic 是否生效，可在低频位置打印轻量信息，例如：
  - 最近一次 update 使用的 `step_rewards.mean()`
  - 或最近一批 vec step 的 reward 统计

约束：
- 不新增高频 wandb 指标
- 不改变现有 online / cm / idql 指标口径

### 7. 第三轮执行后的最小验证集

- [ ] `use_vec_env_online=true`, `train_env_num=1`, `ppo.scale_strategy='dynamic'`
  - 至少完成 1 次 PPO update
  - 与原单 env dynamic 路径在行为上无明显偏差
  - 无 reward shape / GAE shape / buffer shape 错误
- [ ] `use_vec_env_online=true`, `train_env_num=4`, `ppo.scale_strategy='dynamic'`
  - 多 env rollout 正常进行
  - PPO update 正常触发
  - actor / critic update 不报错
  - 某个 env 提前 done 不会打断其他 env
- [ ] reset 正确性
  - 某个 env done 后，仅该 env 的 scaler 被 reset
  - 其他 env 的 dynamic 累计状态连续保留
- [ ] vec + `ppo.iql_ft=true`, `ppo.scale_strategy='dynamic'`
  - `replay_buffer` 使用 scaled reward
  - `iql_buffer` 继续使用原始 reward
  - 不出现 mixed reward 口径导致的 batch / shape 错误
- [ ] vec + `ppo.idql_rollout=true`
  - 仍然 assert 阻断
- [ ] vec + `ppo.iql_adv=true`
  - 仍然 assert 阻断
- [ ] vec + `update_phase='outloop'`
  - 仍然 assert 阻断

### 8. 对本轮 work agent 的明确要求

- 只在现有 vec `online_ft` 分支上补 `dynamic`，不要扩到其他 policy variant。
- 不要重构 `RewardScaling` 类；v1 直接使用 per-env scaler 列表。
- `dynamic` 只作用于 PPO reward 路径，不得影响 `iql_buffer` 的原始 reward 语义。
- 某个 env `done` 时，只 reset 它自己的 scaler，不能全局 reset。
- 不要改回旧的 flatten-first PPO / GAE 路径。
- 修改完成后，先按“第三轮执行后的最小验证集”自检，再进入下一轮 review。

---

## 第三轮 Review 结果（剩余修复项）

以下问题来自本轮 work agent 接入 vec `scale_strategy='dynamic'` 之后的代码 review。未修复前，vec dynamic 路径还不能视为与单 env 语义完全对齐。

### 1. per-env `reward_scaler` 在 rollout 开始前必须先 `reset()`

当前实现已经：
- 在 `online_ft(...)` 中把 `reward_scaler_template` 传给 `_online_ft_vec(...)`
- 在 vec 分支里对模板做了逐 env `deepcopy`
- 在某个 env `done` 时只 reset 对应 scaler

但还缺少一个关键步骤：
- 在创建 `reward_scalers = [...]` 之后、开始 rollout 之前，必须先对**每个** scaler 做一次 `reset()`

原因：
- 单 env 路径在每个 episode 开始前都会执行 `reward_scaler.reset()`
- 当前 vec 路径使用的模板来源于 `scale_dataset.reward_norm`
- 这个模板对象在离线数据集构造过程中已经被持续更新过内部状态（尤其是累计回报 `R`）
- 如果 deepcopy 之后直接拿来 rollout，第一个 online episode 会继承离线模板残留状态，导致 dynamic scaled reward 与单 env 口径不一致

work agent 必须修改为：
- 在 `_online_ft_vec(...)` 中创建完 `reward_scalers` 后
- 在 `obs_list = [envs[i].reset() ...]` 之前或紧接其后
- 显式执行一次：

```python
for scaler in reward_scalers:
    scaler.reset()
```

要求：
- 这一步是 rollout 启动时的初始化 reset
- 不能只依赖后续 `done` 时的 reset
- 不能假设 `deepcopy(scale_dataset.reward_norm)` 得到的对象天然处于 episode 初始态

### 2. 本轮 work agent 的明确修复要求

- 保持当前“模板传入 + per-env deepcopy + env 独立 reset”的整体方案不变。
- 只补上 rollout 开始前的初始化 `reset()`，不要再改动 reward 语义。
- `replay_buffer` 继续存 dynamic scaled reward。
- `iql_buffer` 继续存原始 reward。
- 修复后先重新跑：
  - `python3 -m py_compile RL-100/train_cm_mid.py`
  - 至少 1 个 `train_env_num=1`, `scale_strategy='dynamic'` 的 smoke test
  - 至少 1 个 `train_env_num=4`, `scale_strategy='dynamic'` 的 smoke test

---

## 第四轮执行（加入并行 evaluation）

当前 `online_ft` 的 rollout 已经支持 vec env，但 evaluation 仍然是串行：
- `train_cm_mid.py::unio4_eval()` 仍然逐次调用 `env_runner.run()` / `env_runner.idql_run()`
- 各 runner 内部也仍然逐 episode 串行 rollout

这一轮的目标是把 evaluation 并行到 episode 级别，同时保持现有指标定义不变。

### 执行目标

本轮目标有两件事：
- 普通 evaluation 并行化
- `ppo.idql_eval` 对应的 `idql_run()` 也并行化

本轮默认策略：
- 视频只保留一条，来自 `env0`
- 不新增复杂的异步调度；采用固定批次 wave 模式
- `eval_times` 外层循环保持不变，只并行化每次 `run()` / `idql_run()` 内部的 episode rollout

### 1. 在 `unio4_eval()` 中加入 eval 并行入口

在 `train_cm_mid.py::unio4_eval()` 中新增：

```python
eval_env_num = getattr(self.cfg.ppo, 'eval_env_num', 1)
```

要求：
- `eval_env_num <= 1` 时保持现有串行行为完全不变
- `eval_env_num > 1` 时，把 `eval_env_num` 透传给 runner

统一调用方式改为：
- 普通 eval：
  - `env_runner.run(policy, use_cm=use_cm, distill2mean=distill2mean, eval_env_num=eval_env_num)`
- `idql_eval`：
  - `env_runner.idql_run(policy, dynamics=dynamics, first_action=first_action, get_np=get_np, use_gae=use_gae, iql=iql, Q=Q, repeat_num=repeat_num, use_cm=use_cm, distill2mean=distill2mean, eval_env_num=eval_env_num)`

注意：
- 不要改写 `eval_times` 的平均逻辑
- 不要在 `train_cm_mid.py` 里自己聚合 success / reward 细节，指标定义仍然交给 runner

### 2. 扩展 runner 接口，但不改指标口径

本轮只修改当前主路径实际使用的 runner：
- `adroit_runner.py`
- `metaworld_runner.py`
- `dmc_runner.py`
- `dexart_runner.py`

要求：
- `run(..., eval_env_num=1)` 新增可选参数
- `idql_run(..., eval_env_num=1)` 新增可选参数
- `eval_env_num == 1` 时完全复用现有串行逻辑
- `eval_env_num > 1` 时走新的并行 episode rollout

注意：
- 不要改当前 success rate / goal achieved / mean_returns 的定义
- 不要在这一轮顺手扩散到所有历史 runner

### 3. `make_env()` 增加视频开关

为了避免并行 eval 同时记录多路视频，`BaseRunner.make_env()` 需要改为：

```python
def make_env(self, record_video=True):
    ...
```

对应要求：
- `record_video=True` 时保持当前行为
- `record_video=False` 时不要包 `SimpleVideoRecordingWrapper`
- 现有 training vec rollout 调用 `make_env()` 不传参，默认行为不变

本轮需要同步修改：
- `base_runner.py`
- `adroit_runner.py`
- `metaworld_runner.py`
- `dmc_runner.py`
- `dexart_runner.py`

并行 eval 时：
- 第一个 eval env 使用 `record_video=True`
- 其余 eval env 使用 `record_video=False`

### 4. 并行 eval 采用固定批次 wave 模式

每个 runner 在 `eval_env_num > 1` 时，都按下面的固定批次逻辑执行：

- 将 `eval_episodes` 切成若干个 wave
- 每个 wave 最多创建 `eval_env_num` 个 env
- 一个 wave 内所有 env 同时 rollout，直到这一批 episode 全部结束
- wave 结束后再创建下一批 env

不要实现：
- 某个 env 提前结束后立即补位新的 episode
- 复杂的异步任务队列
- `SubprocVecEnv` 版本的 evaluation

原因：
- wave 模式足够简单
- 和当前串行逐 episode 语义最接近
- 更容易保证统计口径不漂移

### 5. wave 内 rollout 逻辑

在每个 wave 内，runner 需要：

- 创建一组 env：
  - `env0` 用 `record_video=True`
  - 其余 env 用 `record_video=False`
- reset 得到 `obs_list`
- 维护每个 env 的：
  - `done`
  - `episode_reward`
  - success / goal 统计（若该 runner 需要）

每个 env step：
- 对当前仍未完成的 env，收集 obs
- 在 **runner 内部** 实现该 runner 自己的 `stack_obs_dicts()` / obs batching helper，把 obs stack 成 batched obs
- 普通 eval：
  - 调用一次 batched `policy.predict_action(..., deterministic=True, ...)`
- `idql_eval`：
  - 调用一次 batched `policy.sample_action(...)`
- 按 env 分别执行 `step()`
- 某个 env 完成后：
  - 记录该 episode 的 reward / success
  - 标记 inactive
  - 不再参与后续 batched policy 推理

当前 wave 全部 env 完成后：
- 关闭这批 env
- 进入下一个 wave

关于 obs batching helper，这一轮明确要求：
- 每个支持并行 eval 的 runner 内部自己实现
- helper 只处理该 runner 自己实际返回的 obs key
- 不要依赖 `train_cm_mid.py` 传入通用 `stack_obs_dicts()`
- 不要为了复用而做一个“全任务通用 obs stacker”

原因：
- 当前 `stack_obs_dicts()` 是 `_online_ft_vec()` 内部闭包，runner 直接复用不到
- 不同 runner 的 obs key 不完全一致，例如 dexart 还有 `imagin_robot`
- obs batching 属于 runner 自己的 rollout 语义，应与 env step / success 统计放在同一层维护

### 6. adroit 的 seed 语义必须保留

`adroit_runner.py` 当前 eval 有明确的 seed 逻辑：
- `seed = eval_seed + episode_idx`

并行后必须保持这层语义，不要改掉。

要求：
- 将 `_seed_eval_episode()` 改为可对指定 env 生效，例如：
  - `_seed_eval_episode(env, episode_idx)`
- 串行路径和并行路径都复用这同一个 seed 逻辑

不要这样做：
- 并行 eval 时完全丢掉 seed
- 用 env index 代替 episode index

### 7. 视频只保留 `env0`

本轮明确要求：
- 并行 eval 最终只记录一条 eval 视频
- 视频来源固定为当前 wave 中 `env0`
- 不做多路视频拼接
- 不把视频全部关闭

要求：
- 普通 eval 和 `idql_eval` 都遵循同一视频策略
- `log_data['sim_video_eval']` 的 key 保持不变

### 8. 配置与默认值

新增最小配置项：

```yaml
ppo:
  eval_env_num: 1
```

要求：
- 代码里用 `getattr(self.cfg.ppo, 'eval_env_num', 1)` 兜底
- v1 可以不改所有历史 config
- 如果要把它写进当前 vec 在线 config，只改当前实际用到的入口 config

注意：
- 本轮不新增 `use_vec_env_eval` 开关
- 是否并行 evaluation 只由 `eval_env_num > 1` 决定

### 9. 第四轮执行后的最小验证集

- [ ] `ppo.eval_env_num=1`
  - 普通 eval 行为完全不变
  - `idql_eval` 行为完全不变
- [ ] `ppo.eval_env_num=4`, 普通 eval
  - 至少在一个主路径任务上跑通
  - `test_mean_score` / `mean_returns` 正常产出
  - `sim_video_eval` 仍存在
- [ ] `ppo.eval_env_num=4`, `ppo.idql_eval=true`
  - `idql_success_rates` / `idql_returns` 正常产出
  - 不退回串行路径
- [ ] adroit eval seed
  - 并行后仍按 `eval_seed + episode_idx` 生效
- [ ] 视频策略
  - 仅 `env0` 记录视频
  - 非 `env0` env 不录视频

### 10. 对本轮 work agent 的明确要求

- 不要在 `train_cm_mid.py` 里直接硬编码各 runner 的 success 聚合逻辑。
- 不要改 `eval_times` 外层逻辑。
- 不要顺手把训练 rollout 切到 `SubprocVecEnv`。
- `run()` 和 `idql_run()` 都要并行化，不要只改普通 eval。
- `eval_env_num=1` 必须保持严格向后兼容。
- 修改完成后，先跑最小验证集，再进入下一轮 review。

---

## 第五轮执行（补齐共享主路径的 2D vec env 兼容）

这轮不再把 `PushT` / `Rotate` / `Pour` 当作本次需要扩展的 2D runner。它们可以忽略，不要求新增 `make_env()`、vec rollout 或并行 evaluation。

本轮真正目标改为：
- 只补齐当前共享 vec `online_ft` 主路径中与 2D / image-heavy 任务相关的输入与运行时兼容约束
- 重点排查并修复“共享主路径在并行环境下 collect reward 直接归零”的真实原因
- 所有修改必须保持对现有功能零干扰，尤其不能影响当前已跑通的 3D 主路径、单环境路径、vec rollout、vec evaluation、IQL/IDQL 与 dynamic reward scaling

### 1. 先明确本轮 scope

本轮 in-scope：
- `RL-100/train_cm_mid.py` 中 `_online_ft_vec()` 的训练 env 创建方式
- `RL-100/train_cm_mid.py` 中 `stack_obs_dicts()` 的 dtype / key 处理
- 最小量的诊断输出，用于确认 batched obs 在并行环境下是否正常

本轮 out-of-scope：
- `PushT` / `Rotate` / `Pour` runner 自身的 vec rollout 改造
- 为 2D 单独新增一套 PPO / GAE / replay buffer 主路径
- 为 2D 单独补并行 evaluation runner 接口

### 2. 训练 vec env 默认不能录视频

当前 `_online_ft_vec()` 里训练 env 是通过 `env_runner.make_env()` 批量创建的。对于 image-heavy 任务，这里必须显式要求：
- 训练用 env 一律 `record_video=False`
- 不允许继续走 `make_env()` 的默认视频录制行为
- 只有 evaluation 才保留视频录制逻辑

原因：
- 多训练 env 同时携带视频 wrapper / offscreen render context，本身就是并行图像任务下的高风险项
- 本轮优先排除这一运行时干扰，再看 collect reward 归零是否消失

### 3. 2D 兼容只改 obs 入口，不改 PPO / IQL 数学路径

本轮必须保持以下路径完全不变：
- PPO replay buffer shape 与存储语义
- per-env GAE 计算
- PPO / BPPO ratio 公式
- `iql_buffer` 与 `replay_buffer` 的 reward 口径分离
- 当前 3D 任务的 obs 组织方式与 key 语义
- 单环境 `online_ft` 主路径
- 已完成的 vec evaluation / `idql_eval` / `dynamic` 支持

换句话说：
- 不允许因为 2D 任务兼容而去改 `uni_ppo.py`
- 不允许为 2D 单独新增一套 advantage / value target 逻辑
- 不允许把“2D 特殊 case”扩散成 PPO 主更新分支

### 4. `stack_obs_dicts()` 先补诊断，再做 dtype 对齐

当前 `_online_ft_vec()` 的 `stack_obs_dicts()` 只对 `image` 做了 `.to(torch.float)`，其余 key 沿用 numpy 默认 dtype。

本轮要求先做最小诊断，再做最小修复：
- 在 vec rollout 开始后，仅对首批 batched obs 打印一次诊断信息：
  - obs keys
  - `image` shape / dtype / min / max
  - `point_cloud` shape / dtype
  - `agent_pos` shape / dtype
- 在确认不影响默认行为的前提下，把 `stack_obs_dicts()` 中进入 policy 的所有 key 都统一 `.to(torch.float)`，而不是只处理 `image`

这一节的目的不是扩展新的 key 映射，而是确认并修正共享主路径的 batched obs 质量。

### 5. 如果诊断暴露的是 obs 缺失或异常，优先做防御性处理

如果诊断发现某个共享路径任务在并行环境下出现以下问题：
- `image` 全零 / 极值异常
- `point_cloud` 或 `agent_pos` 的 dtype / shape 异常
- 关键 key 缺失或 shape 异常

则优先做最小防御性处理：
- 对缺失 key 给出明确报错，避免静默训练到 reward=0
- 不要在 `_online_ft_vec()` 里引入大规模 task-specific key 分支
- 如果最终定位到是 runner / env 侧输出异常，再把修复下沉到 runner / env；不要先修改 PPO 主循环

### 6. 第五轮执行后的最小验证集

- [ ] 选择一个当前走共享主路径、且出现“并行 collect reward 归零”的任务
  - `ppo.use_vec_env_online=True`
  - `train_env_num=1`
  - 行为应与旧单 env 路径一致，不出现 success 直接掉零
- [ ] 同任务做 `train_env_num=4` smoke test
  - 至少完成一次 rollout
  - 至少完成一次 PPO update
  - 首批 batched obs 诊断信息正常
  - success / return 不再无条件归零
- [ ] 负向检查
  - 不应要求 `PushT` / `Rotate` / `Pour` 新增 `make_env()` 或 vec eval 接口
  - 不应在 `uni_ppo.py` 新增任何 2D 特化逻辑

### 7. 对本轮 work agent 的明确要求

- 不要把这轮工作理解成“补齐所有 2D runner 的 vec 能力”。
- 不要新增 `env_runner2d` 之类的新抽象。
- 不要再围绕 `agent_xy -> agent_pos` 做修复；这不在本轮共享路径问题里。
- 这轮的核心是：训练 env 禁视频、首批 obs 诊断、`stack_obs_dicts()` 全 key float 对齐，以及必要的最小防御性检查。
- 所有修改必须以“默认行为不变”为前提；如果会触碰 3D、单 env 或现有 vec 主路径的默认语义，必须先停下并单独汇报。
- 修改完成后，先跑最小验证集，再进入下一轮 review。

---

## 2026-04-01 补充：2D Adroit 并行 env 排查与修复记录

### 背景
- 3D / point-cloud 路径的 online vec rollout 已验证可用
- 2D / `dp_image_unet` 路径出现了明显问题：
  - 单 env evaluate 正常
  - 并行 evaluate 性能异常
  - 并行 online rollout 性能异常

### 最终根因

这次不是 PPO ratio / GAE 主逻辑的问题，而是 **2D Adroit 的并行环境与 2D policy batching 都没有和 serial 行为对齐**，具体有两层：

1. **同进程多 Adroit env 不隔离**
- 原来的 vec evaluate / 2D vec rollout 本质上还是手动维护多个 env 对象
- 在 2D Adroit 上，把多个 env 放在同一进程里会互相污染
- 现象是：同一个 seed、同样的动作序列，serial 的 env0 和 batched 的 env0 从第 2 步左右就开始观测分叉，后续回报和成功率都会被拉坏

2. **`dp_image_unet` 的 batched diffusion 噪声不是 batch-invariant**
- 原来直接 `torch.randn(full_shape)` 按整 batch 采样
- 这会导致同一个样本在 batch size 改变时使用到不同噪声
- 结果是 single-env 和 vec-env 下，即使 obs 相同，policy action 也不会严格对齐

### 修复内容

#### 1. 2D evaluate 改为子进程隔离

文件：[RL-100/rl_100/env_runner/adroit_runner.py](/cephfs/lk/check_rl100_eval/vec_env_test/RL-100/rl_100/env_runner/adroit_runner.py)

- 新增 `_make_adroit_env(...)`
- 新增 `make_env_fn(...)`
- 新增 `make_subproc_vec_env(...)`
- `run(...)` 和 `idql_run(...)` 的并行 evaluate 改为 `SubprocVecEnv`

目的：
- 每个 env 放在独立子进程，消除 2D Adroit 的同进程互扰

#### 2. 2D online vec rollout 只在 2D 路径切到 subproc vec env

文件：[RL-100/train_cm_mid.py](/cephfs/lk/check_rl100_eval/vec_env_test/RL-100/train_cm_mid.py#L1485)

关键守卫：

```python
use_subproc_vec_rollout = (
    getattr(self.cfg, 'feature_type', None) == '2D'
    and hasattr(env_runner, 'make_subproc_vec_env')
)
```

含义：
- 只有 `feature_type == '2D'` 才启用新的 subproc vec rollout
- 3D / point-cloud online vec rollout 仍走原来的 manual env list 路径

同时补了：
- `vec_env.reset()` / `vec_env.step()` 的 batched obs 处理
- `terminal_observation` 处理
- auto-reset 后 next obs / buffer store 对齐
- `vec_env.close()` 清理

#### 3. `dp_image_unet` 改为 batch-invariant noise sampling

文件：[RL-100/rl_100/policy/dp_image_unet.py](/cephfs/lk/check_rl100_eval/vec_env_test/RL-100/rl_100/policy/dp_image_unet.py#L1173)

新增：
- `_sample_initial_trajectory(...)`
- `_sample_step_variance_noise(...)`

改动点：
- 不再整 batch 一次性 `torch.randn(full_shape)`
- deterministic evaluate 路径复用同一个 seed-0 sample
- stochastic rollout 路径按 env 逐个采样，保证 vec 和 serial 的采样顺序一致
- `scheduler.step_logprob(...)` 显式传入 `variance_noise`

目的：
- 保证 single-env / vec-env 下同一个样本的 diffusion 初始噪声与 step noise 对齐

#### 4. 修复 online 阶段 policy / critic 设备放置

文件：
- [RL-100/rl_100/unidpg/ppo.py](/cephfs/lk/check_rl100_eval/vec_env_test/RL-100/rl_100/unidpg/ppo.py)
- [RL-100/rl_100/unidpg/uni_ppo.py](/cephfs/lk/check_rl100_eval/vec_env_test/RL-100/rl_100/unidpg/uni_ppo.py)

修复内容：
- `PPO.__init__` 中 `_policy` / `_old_policy` 显式 `.to(device)`
- `load(...)` 后再次 `.to(self._device)`
- `set_old_policy(...)` 后显式 `.to(self._device)`
- `transfer2online(...)` 中 `_policy` / `_old_policy` / `critic` 都显式移到 `self._device`

原因：
- 不修这个时，vec online rollout 会在 GPU obs 和 CPU policy/critic 之间触发 device mismatch

#### 5. 修复脚本入口，避免“看起来在跑并行，实际没开 vec 分支”

文件：[scripts/train_policy_image_unet_online.sh](/cephfs/lk/check_rl100_eval/vec_env_test/scripts/train_policy_image_unet_online.sh#L15)

问题：
- 原脚本没有把第 5 个参数接到 `ppo.use_vec_env_online / train_env_num / eval_env_num`
- 用户即使传了 env 数，也不一定真的进入新的 vec 路径

修复后：
- `./scripts/train_policy_image_unet_online.sh ... 1` 表示单 env
- `./scripts/train_policy_image_unet_online.sh ... 4` 表示 4-env vec

### 验证结果

#### 1. 2D 并行 evaluate 恢复正常

使用同一个 `adroit_door_medium-dp3-0112_seed300` checkpoint：

- 单 env evaluate：
  - `test_mean_score = 0.5000`
  - `mean_returns = 46.8790`

- 4-env 并行 evaluate：
  - `test_mean_score = 0.5000`
  - `mean_returns = 46.4477`

说明：
- 2D parallel evaluate 已和 single-env 对齐到正常波动范围

#### 2. 2D vec online rollout + PPO update 已走通

用真实入口 `train_cm_mid.py` 跑 2D `dp_image_unet` vec smoke，显式开启：
- `ppo.use_vec_env_online=True`
- `ppo.train_env_num=4`
- `ppo.eval_env_num=4`

结果：
- 初始并行 eval：
  - `test_mean_score = 1.0`
  - `mean_returns = 111.7132`
- 成功进入 `_online_ft_vec()`
- batched obs 打印正常：
  - `image (4, 3, 3, 84, 84)`
  - `point_cloud (4, 3, 512, 6)`
  - `agent_pos (4, 3, 24)`
- 完成一轮并行 rollout + PPO update：
  - `step 40; collecting data time: 1.04; update time: 3.74`
- 成功保存：
  - `online_ft/.../online_last`

#### 3. `DP3` 共享 device 路径 smoke 正常

虽然这轮主修的是 2D，但共享的 `ppo.py` / `uni_ppo.py` device 修复也单独用真实 `DP3` 类做了 smoke：

- `policy_device = cuda:0`
- `old_policy_device = cuda:0`
- `critic_device = cuda:0`

说明：
- point-cloud `DP3` 在 `transfer2online(...)` 后设备放置正常

### 影响范围说明

#### 已确认不会伤到的部分

1. **3D online vec rollout / PPO update 主路径**
- `train_cm_mid.py` 的 subproc vec rollout guard 只在 `feature_type == '2D'` 时触发
- 3D rollout/update 仍走旧逻辑，没有被这次 2D 修复切换实现

2. **`dp3.py` 本体**
- 这轮没有修改 `RL-100/rl_100/policy/dp3.py`
- 2D 的 batch-noise 修复仅在 `dp_image_unet.py`

3. **共享 device 修复对 `DP3` 正常**
- 已用真实 `DP3` 类验证 `transfer2online()` 后 policy / old_policy / critic 都在目标 GPU 上

#### 仍建议单独回归的部分

1. **3D parallel evaluate**
- `adroit_runner.py` 的并行 evaluate 现在统一使用 `SubprocVecEnv`
- 从代码逻辑看这是更安全的实现
- 但这轮没有拿一份“确认是 3D point-cloud DP3”的现成 checkpoint 做完整端到端 evaluate 回归
- 因此如果要对外宣称“3D evaluate 也完全回归过”，仍建议补跑一次真实 3D checkpoint

### 这轮 debug 的核心结论

- 2D `image_unet` 的并行问题，本质上是 **环境隔离问题 + batched diffusion noise 对齐问题**
- 修好后，2D 的 parallel evaluate 和 online vec rollout 都恢复正常
- 3D rollout/update 主路径没有被动到
- 共享层里唯一需要继续保守看待的是“3D parallel evaluate 仍建议做一次真实 checkpoint 回归”
