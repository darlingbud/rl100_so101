# KL-based LR Scheduler 索引 Bug

## 位置
同类 bug 目前至少有两处：

- `RL-100/rl_100/unidpg/uni_ppo.py:781`
- `RL-100/rl_100/unidpg/uni_ppo.py:912`

两处都属于 online PPO 更新路径中的 KL-based LR scheduler 统计。

## 问题
```python
log_ratio = (a_logprob_now[:self.cfg.n_action_steps].sum(-1, keepdim=True)
             - a_logprob_old[i][:self.cfg.n_action_steps].sum(-1, keepdim=True)).squeeze(1)
```

`a_logprob_now` shape 为 `(B, H, A)`（B=mini_batch, H=horizon, A=action_dim）。

- `a_logprob_now[:self.cfg.n_action_steps]` — 按**第 0 维（batch）**切了前 `n_action_steps` 个样本
- 应该是 `a_logprob_now[:, :self.cfg.n_action_steps]` — 按**第 1 维（action step）**切

对比同方法内 `ratios` 的正确写法（line 895）：
```python
ratios = torch.exp(a_logprob_now[:, :self.cfg.n_action_steps].sum(-1, keepdim=True)
                   - a_logprob_old[i][:, :self.cfg.n_action_steps].sum(-1, keepdim=True))
```

## 影响
- KL 估计只用了前 `n_action_steps` 个 batch 样本（而非全部样本的前 `n_action_steps` 个 action step）
- 导致 KL 估计不准，LR 调度偏差
- 不会崩溃（只要 `B >= n_action_steps`）
- 仅影响 `use_lr_decay=False`（即使用 KL-based LR scheduler）的 online 训练路径

## 需要额外确认的一点
这份修复不能只停留在“补 `:`”。

当前 KL scheduler 统计应该和 actor surrogate 使用**完全相同的 action span**。  
如果这条路径的真实优化区间需要考虑：

- `action_start = 0 if no_pre_action else n_obs_steps - 1`
- `action_end = action_start + n_action_steps`

那么 KL 统计也必须统一使用：

```python
a_logprob_now[:, action_start:action_end]
a_logprob_old[i][:, action_start:action_end]
```

而不是只局部改成：

```python
[:, :self.cfg.n_action_steps]
```

否则会出现：

- surrogate 用一段 action span
- KL scheduler 却统计另一段 action span

最终 learning rate 调度仍然和真实 PPO 更新不一致。

## 修复
```python
# 修前
log_ratio = (a_logprob_now[:self.cfg.n_action_steps].sum(-1, keepdim=True)
             - a_logprob_old[i][:self.cfg.n_action_steps].sum(-1, keepdim=True)).squeeze(1)

# 修后（最低限度）
log_ratio = (a_logprob_now[:, :self.cfg.n_action_steps].sum(-1, keepdim=True)
             - a_logprob_old[i][:, :self.cfg.n_action_steps].sum(-1, keepdim=True)).squeeze(1)
```

更稳妥的修法是和 actor loss 完全共用 action span：

```python
action_start = 0 if self.cfg.no_pre_action else self.n_obs_steps - 1
action_end = action_start + self.cfg.n_action_steps

log_ratio = (
    a_logprob_now[:, action_start:action_end].sum(-1, keepdim=True)
    - a_logprob_old[i][:, action_start:action_end].sum(-1, keepdim=True)
).squeeze(1)
```
