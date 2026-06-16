# vdelta 方案 A + B 执行文档

## 背景

`per_step_vdelta` 在训练时 crash：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x28 and 448x384)
```

crash 位置：`uni_ppo.py:485` → `dynamics.step()` → `_action_encoder(action)`

根因：当 `chunk_as_single_action=True + use_action_embed=True` 时，dynamics 的 `_action_encoder` 输入维度是 `n_action_steps * action_dim = 16 * 28 = 448`（整个 chunk flatten）。但 `_compute_chunk_step_advantages_vdelta` 逐步传入单步 action（dim=28），导致 shape mismatch。

chunk dynamics 是 chunk-level world model，输入整个 chunk action，输出 chunk 执行完之后的 next state + discounted reward sum。它不支持 single-step rollout。

---

## 方案 A：额外训练 single-step dynamics，做真正的 per-step rollout

### 思路

stage1 额外训练一个 `chunk_as_single_action=False` 的 single-step dynamics。`per_step_vdelta` 加载这个 single-step dynamics 做逐步 rollout，现有 `_compute_chunk_step_advantages_vdelta` 的逻辑基本不变，只是换了 dynamics 对象。

### adv_mode 名称

保持 `per_step_vdelta`

### 需要改的文件和具体改动

#### 1. `scripts/train_policy_chunk_two_stage.sh`

stage1 结束后，额外跑一次 dynamics 训练：
- `chunk_as_single_action=False`
- `prediction_mode` 保持与主 dynamics 一致（当前是 `full`）
- 保存到独立路径，如 `saved_models_{prediction_mode}_singlestep/`
- 只训练 dynamics，不需要重新训练 BC/IQL/value

#### 2. `RL-100/train_ddp.py`

- 当 `offline_chunk_adv_mode == 'per_step_vdelta'` 时，额外加载 single-step dynamics
- 加载路径：`{stage1_run_dir}/saved_models_{prediction_mode}_singlestep/`
- 传给 `update_distribution()` 一个新参数 `singlestep_dynamics`

参考当前 dynamics 加载逻辑（`train_ddp.py` 中搜索 `train_dynamics` 和 `dynamics.load`）。

#### 3. `RL-100/rl_100/unidpg/uni_ppo.py`

- `update_distribution()` 签名新增 `singlestep_dynamics: EnsembleDynamics = None`
- 在 `chunk_adv_mode == 'per_step_vdelta'` 分支中，把 `dynamics` 替换为 `singlestep_dynamics`：
  ```python
  advantages = self._compute_chunk_step_advantages_vdelta(
      nobs_features,
      chunk_actions,
      singlestep_dynamics,  # 用 single-step dynamics，不是 chunk dynamics
      value, iql, gamma, lamda, use_gae,
  )
  ```
- `_compute_chunk_step_advantages_vdelta()` 本身不需要改，它的逐步 rollout 逻辑对 single-step dynamics 是正确的
- `predict_r` assert 仍然有效（single-step dynamics 也需要 `predict_r=True`）

#### 4. config

- 可选：在 yaml 中新增 `offline_chunk_singlestep_dynamics_path` 字段
- 或者在 `train_ddp.py` 中自动推导路径（`stage1_run_dir + /saved_models_{prediction_mode}_singlestep/`）

### 边界约束

- 不影响 chunk dynamics 的训练和使用
- 不影响 `scalar_iql` 路径
- 不影响 online 路径
- 不影响 non-chunk 路径
- single-step dynamics 的 `predict_r` 必须为 True

### 验证

1. stage1 跑完后确认 `saved_models_{prediction_mode}_singlestep/dynamics.pth` 存在
2. stage2 `per_step:per_step_vdelta` 不 crash，loss 正常下降
3. 非 `per_step_vdelta` 的 sweep job 不受影响（`singlestep_dynamics=None`）

---

## 方案 B：chunk-level vdelta（不需要额外 dynamics）

### 思路

用现有 chunk dynamics 做一次 forward pass，得到 chunk 执行完之后的 next state 和 discounted reward sum，计算：

```
advantage = discounted_reward_sum + γ^T · V(s_final) - V(s_0)
```

作为 scalar advantage，broadcast 给每个 step 的 ratio。

chunk dynamics 训练时 reward target 就是 discounted sum（见 `ensemble_dynamics_for_batch.py` `format_samples_for_training` line 336-345）：

```python
gamma_weights = torch.pow(gamma, torch.arange(n_action_steps))
rewards = torch.sum(reward_chunk * gamma_weights, dim=-1, keepdim=True)  # [B, 1]
```

所以 `dynamics.step()` 返回的 reward 就是 `Σ γ^t · r_t`，语义干净。

### adv_mode 名称

`chunk_vdelta`

### 需要改的文件和具体改动

#### 1. `RL-100/rl_100/unidpg/uni_ppo.py`

新增方法 `_compute_chunk_vdelta_scalar()`：

```python
@torch.no_grad()
def _compute_chunk_vdelta_scalar(
    self, nobs_features, chunk_actions, dynamics, value, iql, gamma,
):
    if dynamics is None:
        raise ValueError("dynamics is required for chunk_vdelta")
    if not getattr(dynamics, 'predict_r', False):
        raise ValueError("chunk_vdelta requires predict_r=True")

    batch_size = nobs_features.shape[0]
    feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
    policy_features = nobs_features.reshape(batch_size, self.cfg.n_obs_steps, feature_dim)
    single_nob_features = policy_features[:, -1, :]

    # V(s_0)
    state_features = policy_features.reshape(batch_size, -1)
    value_init = self._evaluate_value_function(state_features, value, iql).reshape(batch_size, -1)[:, 0]

    # chunk dynamics forward: input is flattened chunk action
    chunk_actions_flat = chunk_actions.reshape(batch_size, -1)
    next_obs, reward, terminal, _ = dynamics.step(
        single_nob_features, chunk_actions_flat, policy_features,
    )

    # reward 已经是 discounted sum: Σ γ^t · r_t
    reward_discounted = torch.from_numpy(reward).to(
        device=self._device, dtype=value_init.dtype,
    ).reshape(batch_size, -1)[:, 0]

    terminal_t = torch.from_numpy(terminal).to(
        device=self._device, dtype=value_init.dtype,
    ).reshape(batch_size, -1)[:, 0]

    # V(s_final)
    if dynamics.prediction_mode == "full":
        next_policy_features = torch.from_numpy(next_obs).to(
            device=self._device, dtype=policy_features.dtype,
        )
    else:
        next_single = torch.from_numpy(next_obs).to(
            device=self._device, dtype=policy_features.dtype,
        )
        next_policy_features = torch.cat(
            (policy_features[:, 1:, :], next_single.unsqueeze(1)), dim=1,
        )
    next_state_features = next_policy_features.reshape(batch_size, -1)
    value_final = self._evaluate_value_function(next_state_features, value, iql).reshape(batch_size, -1)[:, 0]

    gamma_T = gamma ** self.cfg.n_action_steps
    advantage = reward_discounted + gamma_T * (1 - terminal_t) * value_final - value_init

    # normalize
    advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
    return advantage
```

修改 `_get_offline_chunk_modes()`：

```python
valid_adv_modes = {'scalar_iql', 'per_step_vdelta', 'chunk_vdelta'}
# 放宽约束：scalar ratio 允许配 chunk_vdelta
if ratio_mode == 'scalar' and adv_mode not in ('scalar_iql', 'chunk_vdelta'):
    raise ValueError(...)
```

修改 `update_distribution()` 中 advantage 计算分支（`final_reward=True` 和 `False` 两处）：

```python
if chunk_adv_mode == 'scalar_iql':
    advantages = self._compute_advantage_actor_only(...)
elif chunk_adv_mode == 'chunk_vdelta':
    advantages = self._compute_chunk_vdelta_scalar(
        nobs_features, chunk_actions, dynamics, value, iql, gamma,
    )
else:  # per_step_vdelta
    advantages = self._compute_chunk_step_advantages_vdelta(...)
```

loss 部分：`chunk_vdelta` 返回 scalar advantage，走现有 `scalar_iql` 的 loss 写法（line 797-804，scalar adv broadcast 给 per-step ratio）。不需要改 loss 代码，只要 advantage shape 跟 `scalar_iql` 一致即可。

#### 2. `RL-100/rl_100/config/dp3_cm_epsilon.yaml`

`offline_chunk_adv_mode` 注释加上 `chunk_vdelta`：

```yaml
offline_chunk_adv_mode: scalar_iql  # scalar_iql | per_step_vdelta | chunk_vdelta
```

#### 3. `scripts/train_policy_chunk_two_stage.sh`

`CHUNK_LOSS_MODE_COMBOS` 加入 `per_step:chunk_vdelta`：

```bash
CHUNK_LOSS_MODE_COMBOS=${CHUNK_LOSS_MODE_COMBOS:-"per_step:scalar_iql scalar:scalar_iql per_step:per_step_vdelta per_step:chunk_vdelta"}
```

#### 4. `RL-100/chunk_ranking_diagnostic.py`

`compute_rollout_proxy_score()` 当前是逐步 rollout，对 chunk dynamics 会 crash。需要改成用 chunk dynamics 一次 forward（跟 `_compute_chunk_vdelta_scalar` 同样的逻辑）。

### 边界约束

- 不影响 `scalar_iql` 路径
- 不影响 `per_step_vdelta` 路径（方案 A 的改动）
- 不影响 online 路径
- 不影响 non-chunk 路径
- `chunk_vdelta` 返回的 advantage shape 必须与 `scalar_iql` 一致（`[B]` 或 `[B, 1]`）

### 验证

1. `per_step:chunk_vdelta` 不 crash，loss 正常下降
2. `scalar_iql` 对照组行为不变
3. diagnostic 脚本能正常输出 correlation

---

## 修改文件汇总

| 文件 | 方案 A | 方案 B |
|------|--------|--------|
| `uni_ppo.py` | `update_distribution` 加 `singlestep_dynamics` 参数，vdelta 分支用它 | 新增 `_compute_chunk_vdelta_scalar()`，`_get_offline_chunk_modes` 加 `chunk_vdelta`，`update_distribution` 加分支 |
| `train_ddp.py` | 加载 single-step dynamics 并传入 | 不改 |
| `train_policy_chunk_two_stage.sh` | stage1 额外训练 single-step dynamics | combos 加 `per_step:chunk_vdelta` |
| `dp3_cm_epsilon.yaml` | 可选：加 singlestep dynamics path | `offline_chunk_adv_mode` 注释加 `chunk_vdelta` |
| `chunk_ranking_diagnostic.py` | proxy score 用 single-step dynamics 逐步 rollout | proxy score 改成 chunk dynamics 一次 forward |

---

## 推荐实验集

| Exp | ratio_mode | adv_mode | 说明 |
|-----|-----------|----------|------|
| 1 | per_step | scalar_iql | 对照组（已有） |
| 2 | per_step | chunk_vdelta | 方案 B 主实验（优先跑，不需要额外 dynamics） |
| 3 | per_step | per_step_vdelta | 方案 A 主实验（需要先训练 single-step dynamics） |

---

## 执行顺序建议

1. **方案 B 先行**：改动面小，不需要重训 dynamics，可以立即验证 chunk-level vdelta 是否能缓解 offline drop
2. **方案 A 并行准备**：stage1 开始训练 single-step dynamics（耗时较长），等 dynamics 训练完再跑 stage2
3. 两个方案都跑通后，对比 Exp 1/2/3 的训练曲线，判断哪种 advantage signal 更有效
