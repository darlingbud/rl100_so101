# Offline RL Debug TODO — Visual vs State 版本对比

## Encoder 架构概览 (visual 版本独有)

`joint_opt_encoder=False` 时，存在 3 份独立的 encoder 拷贝：
1. **Policy encoder** — `self.model.obs_encoder` (BC 训练时更新)
2. **Dynamics encoder** — `deepcopy(obs_encoder)` via `get_dynamics_encoder()` (dp3_cm.py:282)
3. **Critic encoder** — `deepcopy(obs_encoder)` via `initialize_critic()` (dp3_cm.py:308)

Offline RL 阶段要求：所有 encoder 冻结，只更新 critic MLP / dynamics MLP / policy UNet。

---

## 发现的问题

### BUG 1 (低优先级): `policy.obs2latent()` 在 evaluation 路径缺少 encoder fix 保护

**位置**: `dp3_cm.py:622-638`
```python
def obs2latent(self, nobs, eval_policy: bool = False):
    ...
    if eval_policy:          # <-- 默认 False!
        self.obs_encoder.eval()
        nobs_features = self.obs_encoder(this_nobs)
    else:
        nobs_features = self.obs_encoder(this_nobs)  # <-- 没有 no_grad, 没有 eval()
```

**受影响的调用链 (仅 evaluation/inference 路径)**:
- `dynamics_batch.rollout()` (L483) → `policy.obs2latent(batch['obs'])` — 没传 `eval_policy=True`
- `policy.sample_action()` (L1372) → `self.obs2feature(state_dict)` — 没传 `fix_encoder=True`

**不受影响的调用链 (loss 计算路径均已正确 fix)**:
- ✅ Critic `update()` (critic.py L592-601): `fix_encoder` 时显式 `obs_encoder.eval()` + `torch.no_grad()`
- ✅ Dynamics `obs2latent()` (ensemble_dynamics_for_batch.py L277-280): `fix_encoder` 时 `encoder.eval()` + `torch.no_grad()`
- ✅ BPPO `update_distribution()` → `obs2feature()` (dp3_cm.py L450-452): `fix_encoder` 时 `obs_encoder.eval()` + `torch.no_grad()`

**影响**: 仅影响 OPE rollout 评估和推理时的 BatchNorm running stats，不影响 loss 计算和梯度更新。可能导致 OPE 评估不够准确。

**对比 state 版本**: state 版本没有 encoder，不存在此问题。

---

### BUG 2 (高优先级): `IQL_Q_V_no.minQ()` 在 `is_share_encoder` 时缺少 chunk action reshape

**位置**: `critic.py:542-558`
```python
def minQ(self, s, a):
    if not self.is_share_encoder:
        if isinstance(s, dict):
            if len(s['point_cloud'].shape) != 3:
                s = self.obs2nobs(s)
        # <-- 缺少 chunk_as_single_action 时的 action reshape!
    else:
        if isinstance(s, dict):
            ...
            s = self.obs_encoder(s).reshape(batch_size, -1)
    Q1, Q2 = self._Q(s, a)  # <-- a 可能维度不对
```

**对比**: `get_advantage()` (L854) 和 `update()` (L710-718) 中都有 `a = a.reshape(-1, self.action_dim)` 处理，但 `minQ()` 在 `is_share_encoder=True` 路径下缺少此处理。

**对比 state 版本**: `minQ(s, a)` 直接 `Q1, Q2 = self._Q(s, a)`，action 维度始终正确。

---

### ~~BUG 3~~ (非问题): `value_loss.backward(retain_graph=True)` 是为 online 阶段设计的

**位置**: `critic.py:754`

当 `fix_encoder=False` 且 `is_share_encoder=True` (online finetuning) 时，V-loss 和 Q-loss 共享 encoder 计算图，`retain_graph=True` 是必须的，否则 Q-loss backward 会报错。offline 阶段 `fix_encoder=True` 时 encoder 输出是 `no_grad`，retain_graph 无害但多占一点显存。**不需要修复。**

---

### BUG 4 (中优先级): `IQL_Q_V_no.get_advantage()` 中 `minQ` 和 `V` 使用不同 encoder 路径

**位置**: `critic.py:826-850` (double_q 路径)
```python
def get_advantage(self, s, a):
    if self._is_double_q:
        q = self.minQ(s, a)          # <-- minQ 内部会调用 encoder
        if isinstance(s, dict):
            ...
            if self.fix_encoder:
                with torch.no_grad():
                    s = self.obs_encoder(s)  # <-- 又调用一次 encoder!
            v = self._value(s)
        return q - v
```

当 `is_share_encoder=True` 且 `s` 是 dict 时，encoder 被调用了两次：一次在 `minQ()` 内，一次在 `get_advantage()` 计算 V 时。虽然结果正确（因为 fix_encoder 下 encoder 是确定性的），但浪费计算。

**对比 state 版本**: `return self.minQ(s, a) - self._value(s)`，一行搞定，无冗余。

---

### BUG 5 (中优先级): `advantage_computation` 缺少 `@torch.no_grad()` 对 encoder 的保护

**位置**: `uni_ppo.py:121-148`
```python
@torch.no_grad()
def advantage_computation(self, s, action, value, Q=None, iql=None):
    ...
    advantage = iql.get_advantage(s, action)
```

虽然外层有 `@torch.no_grad()`，但 `iql.get_advantage()` 内部的 `minQ()` 在 `is_share_encoder=True` 路径下会调用 `self.obs_encoder(s)`。由于外层 `no_grad`，这里不会产生梯度，但 encoder 可能仍在 training mode（BatchNorm 会更新 running stats）。

**关键**: `advantage_computation` 被 BPPO `update_distribution` 调用时，传入的 `s` 是 `nobs_features`（已经是 latent），所以实际上不会走 `isinstance(s, dict)` 分支。但如果 `s` 是 dict（如 rollout 场景），就会有问题。

---

### 问题 6 (低优先级): Dynamics `format_samples_for_training` 差异

**Visual 版本** (`ensemble_dynamics_for_batch.py:305`): 接收预计算的 `nobs_features` 和 `next_nobs_features`
**原始版本** (`ensemble_dynamics.py:240`): 内部调用 `self.obs2latent()` 并 `.detach()`

实际使用的是 batch 版本，训练循环 (`train_cm_mid.py:548`) 在外部调用 `dynamics.obs2latent()` 预计算 latent，然后传入 `dynamics.learn(batch, nobs_features, next_nobs_features)`。这个流程是正确的。

---

### 问题 7 (确认正确): IQL loss 计算与 state 版本一致

| 组件 | State 版本 | Visual 版本 | 一致性 |
|------|-----------|------------|--------|
| V-loss | `expectile_loss(target_q - value).mean()` | 同左 | ✅ |
| Q-loss (double) | `((Q1-tq)^2 + (Q2-tq)^2).mean()` | 同左 | ✅ |
| Q-target | `r + not_done * γ * V(s')` | `r + not_done * γ^n_action_steps * V(s')` | ⚠️ chunk 时 γ 指数不同 |
| target update | `τ * Q + (1-τ) * Q_target` | 同左 | ✅ |
| expectile | `ω if loss>0 else 1-ω` | 同左 | ✅ |

**注意**: Visual 版本 Q-target 使用 `γ^n_action_steps` (critic.py:765)，这在 chunk 模式下是正确的（因为 reward 已经是 discounted sum）。

---

### 问题 8 (确认正确): BPPO update 逻辑与 state 版本基本一致

核心 PPO loss 计算一致：
```python
ratio = (new_logprob - old_logprob).exp()
loss1 = ratio * advantage
loss2 = clamp(ratio, 1-ε, 1+ε) * advantage
loss = -min(loss1, loss2).mean()
```

**差异点**:
- Visual 版本在 chunk 模式下聚合 logprob: `sum(logprob[action_start:action_end])` → scalar ratio
- State 版本每个 denoising step 独立 backward (`retain_graph=True`)，visual 版本同样如此

---

## 修复优先级

1. **BUG 2** — `minQ()` 缺少 chunk action reshape (高优先级)
   - 影响: chunk 模式下 Q 值计算可能维度错误
   - 修复: 在 `minQ()` 的 `is_share_encoder` 路径添加 action reshape

2. **BUG 4/5** — `get_advantage()` 中 encoder 重复调用 + training mode 问题 (中优先级)
   - 影响: 计算浪费 + 潜在 BatchNorm 问题
   - 修复: 统一 encoder 调用路径

3. **~~BUG 3~~** — retain_graph=True 是为 online 阶段设计的，不需要修复

4. **BUG 1** — `obs2latent` / `sample_action` evaluation 路径 encoder 未 fix (低优先级)
   - 影响: 仅影响 OPE 评估准确性，不影响 loss 计算
   - 修复: rollout/inference 时传入 `eval_policy=True` 或 `fix_encoder=True`

---

## 关键文件索引

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `RL-100/rl_100/policy/dp3_cm.py` | L414, L622, L1075, L1352 | obs2feature, obs2latent, all_step_logprob, sample_action |
| `RL-100/rl_100/unidpg/critic.py` | L387, L542, L591, L826 | IQL_Q_V_no init, minQ, update, get_advantage |
| `RL-100/rl_100/unidpg/uni_ppo.py` | L121, L257, L353 | advantage_computation, update_distribution, chunk PPO |
| `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py` | L67, L267, L457 | step, obs2latent, rollout |
| `RL-100/train_cm_mid.py` | L480, L536, L688, L772 | critic训练, dynamics训练, finetune_dp3, BPPO loop |
| `RL-100/rl_100/unidpg/dynamics_eval_batch.py` | L84 | train_dynamics |
| `third_party/rl100-state/critic.py` | L237-363 | IQL_Q_V (参考实现) |
| `third_party/rl100-state/uni_ppo.py` | L275-358 | BPPO update (参考实现) |
| `third_party/rl100-state/main.py` | L351-500 | 训练流程 (参考实现) |
