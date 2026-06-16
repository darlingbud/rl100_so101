# chunk_as_single_action=True Offline BPPO Collapse

## 结论

当前问题不是 `chunk` 控制模式整体不可用，而是：

- `chunk_as_single_action=True` 下，online PPO 还能提升；
- 但 offline BPPO 会快速把真实性能打到 0；
- 同时 OPE score 单调上升。

这说明当前 offline chunk BPPO 在稳定地 exploit 一个有偏信号，而不是在提升真实策略质量。

最可疑、最值得优先修复的点是：

- offline BPPO 的 chunk ratio 聚合方式，和 online PPO 不一致；
- offline chunk 用 joint chunk scalar ratio；
- online PPO 用 per-step ratio。

`critic` 和 `dynamics` 在 old commit `19c53562dd155147dd2831f9d2990adfbffcd70a` 到当前版本之间，没有看到足以解释这次退化的目标级变化；主要变化集中在 offline BPPO chunk loss。

## 现象证据

已确认一条代表性 run：

- `RL-100/data/outputs_two_stage_chunk/adroit_door_medium-dp3-0112_seed100/relu/dp3vib/dp3/2026-04-05-01-10-35-lr_1e-6_rollout_5_clip_0.8/2026-04-05-01-10-44/each_scores.csv`
- `RL-100/data/outputs_two_stage_chunk/adroit_door_medium-dp3-0112_seed100/relu/dp3vib/dp3/2026-04-05-01-10-35-lr_1e-6_rollout_5_clip_0.8/2026-04-05-01-10-44/each_ope_score.csv`

观测到：

- actual score: `0.866667 -> 0.600000 -> 0.233333 -> 0.100000 -> 0.066667 -> 0.033333 -> 0.000000`
- OPE score: `246.6 -> 265.3` 持续上升

另外抽查的其他 run 也表现出同样模式：

- `each_ope_score.csv` 上升；
- `each_scores.csv` 快速掉到 0。

这不是普通训练波动，更像典型的 advantage exploitation / policy collapse。

## 当前代码里的关键不一致

### 1. offline chunk BPPO

文件：

- `RL-100/rl_100/unidpg/uni_ppo.py`

当前 `update_distribution()` 在 `chunk_as_single_action=True` 分支中：

- 截取整个 optimized chunk span；
- 将 `old/new logprob` 展平；
- 对 `action_steps * action_dims` 全部求和；
- 得到一个 scalar `ratio_scalar`；
- 用一个 scalar chunk advantage 做 PPO clipped objective。

核心形式是：

```python
old_logprob_scalar = logprob_old.reshape(B, -1).sum(dim=1)
new_logprob_scalar = logprob_new.reshape(B, -1).sum(dim=1)
ratio_scalar = (new_logprob_scalar - old_logprob_scalar).exp()
```

也就是每个 sample 一个整段 chunk 的 joint ratio。

### 2. online PPO

文件：

- `RL-100/rl_100/unidpg/uni_ppo.py`

当前 online `dp_align_update_no_share()` 中：

- 只对最后一维 `action_dim` 求和；
- 保留 `n_action_steps` 这一维；
- 得到的是 per-step ratio，而不是整个 chunk 一个 ratio。

核心形式是：

```python
ratios = torch.exp(
    a_logprob_now[:, action_start:action_end].sum(-1, keepdim=True)
    - a_logprob_old[i][:, action_start:action_end].sum(-1, keepdim=True)
).squeeze(1)
```

所以当前实际不一致是：

- offline chunk BPPO: joint chunk scalar ratio
- online PPO: per-step ratio

## 为什么这是首要嫌疑点

在当前配置下：

- `n_action_steps = 16`
- `action_dim = 28`
- `no_pre_action = true`

也就是说，offline chunk scalar ratio 会把 `16 * 28 = 448` 个 log-prob 项直接聚合到一个标量里。

这会带来两个风险：

- `log-ratio` 的方差被显著放大，ratio 更容易快速偏离 1 并频繁撞到 clip 边界；
- 所有 action steps 共用一个 ratio，梯度无法像 online PPO 那样按 step 细分。

从实验结果看，这种 scalar joint ratio 在 offline chunk 上数值上明显更脆弱。

注意：

- 这不等于“joint ratio 在数学上绝对错误”；
- 但它和当前 online 实现不一致；
- 并且它和这次 collapse 的经验现象高度吻合。

## old commit 对比结论

对比 old commit `19c53562dd155147dd2831f9d2990adfbffcd70a` 后：

### critic

- chunk critic 仍然把整段 chunk 当 macro-action；
- action 维仍然是 `action_dim * n_action_steps`；
- reward 仍然是 chunk reward 的 discounted sum；
- 没看到目标定义层面的显著变化。

### dynamics / OPE

- `chunk_evaluation()` 仍然直接走 macro-Q；
- `rollout()` 在 chunk 模式中仍然是 sample 整段 chunk 后用 critic 打分；
- 也没看到足以解释本次回归的变化。

### uni_ppo

主要差异在这里。

旧版 offline chunk 更新没有当前这套 dedicated chunk scalar ratio 分支；当前版本新增了整段 chunk joint ratio 聚合，这和 online PPO 的 per-step ratio 拉开了差异。

因此本轮修复优先级应放在：

- `uni_ppo.py` 的 offline chunk BPPO 分支；
- 不是先大改 `critic` 或 `dynamics`。

## 修复方案

### First Fix

仅修改 offline chunk BPPO 的 ratio 聚合方式，使其与 online PPO 对齐。

目标：

- 保留 chunk advantage 仍然是 scalar；
- 只改 ratio 聚合；
- 不动 critic；
- 不动 dynamics；
- 不动 online PPO；
- 尽量最小改动，方便 A/B。

### 具体改法

文件：

- `RL-100/rl_100/unidpg/uni_ppo.py`

函数：

- `update_distribution()`

仅修改当前这段：

```python
if self.cfg.chunk_as_single_action:
    action_start = 0 if self.cfg.no_pre_action else self.n_obs_steps - 1
    action_end = action_start + opt_steps
    logprob_old = old_all_logprob[i][:, action_start:action_end]
    logprob_new = new_log_prob[:, action_start:action_end]

    old_logprob_scalar = logprob_old.reshape(logprob_old.shape[0], -1).sum(dim=1)
    new_logprob_scalar = logprob_new.reshape(logprob_new.shape[0], -1).sum(dim=1)
    ratio_scalar = (new_logprob_scalar - old_logprob_scalar).exp()

    adv = advantages.detach().reshape(advantages.shape[0], -1)
    assert adv.shape[1] == 1
    adv = adv[:, 0]

    loss1 = ratio_scalar * adv
    loss2 = torch.clamp(ratio_scalar, 1 - self._clip_ratio, 1 + self._clip_ratio) * adv
    loss = -(torch.min(loss1, loss2)).mean()
```

改成与 online PPO 聚合语义对齐的 per-step ratio 版本：

```python
if self.cfg.chunk_as_single_action:
    action_start = 0 if self.cfg.no_pre_action else self.n_obs_steps - 1
    action_end = action_start + opt_steps

    logprob_old = old_all_logprob[i][:, action_start:action_end]
    logprob_new = new_log_prob[:, action_start:action_end]

    ratio_perstep = torch.exp(
        logprob_new.sum(-1) - logprob_old.sum(-1)
    )  # [B, opt_steps]

    adv = advantages.detach().reshape(advantages.shape[0], -1)
    assert adv.shape[1] == 1, f"chunk_as_single_action expects scalar advantage, got {adv.shape}"
    adv = adv[:, 0].unsqueeze(-1)  # [B, 1], broadcast to all steps

    self._record_ratio_stats(
        "offline_multi",
        i,
        ratio_perstep,
        logprob_old.sum(-1),
        logprob_new.sum(-1),
    )

    loss1 = ratio_perstep * adv
    loss2 = torch.clamp(ratio_perstep, 1 - self._clip_ratio, 1 + self._clip_ratio) * adv
    loss = -(torch.min(loss1, loss2)).mean()
```

重点：

- 只对 `action_dim` 求和；
- 保留 `opt_steps` 维度；
- 让一个 chunk advantage 广播到所有 steps；
- 这和 online PPO 当前“shared scalar advantage + per-step ratio”的结构一致；
- 不再把整个 chunk 的所有 step 和 dim 压成一个 joint scalar ratio。

### ratio logging 也要一起对齐

当前 chunk 分支前面还有一行：

```python
ratio = (new_log_prob - old_all_logprob[i]).exp()
self._record_ratio_stats("offline_multi", i, ratio, old_all_logprob[i], new_log_prob)
```

这在 non-chunk 路径里没问题，但在 `chunk_as_single_action=True` 下记录的是原始 per-element ratio，和真正参与 loss 的量不一致，容易误导分析。

建议：

- non-chunk 路径保留原来的 `ratio_stats`；
- chunk 路径把 `ratio_stats` 移到 chunk 分支内部；
- 记录 `ratio_perstep`、`logprob_old.sum(-1)`、`logprob_new.sum(-1)`。

也就是：

- chunk 看 per-step ratio；
- non-chunk 继续看原有 ratio。

### 不要做的事

本轮 first fix 不要同时做以下改动，否则会混淆归因：

- 不要修改 advantage 定义；
- 不要修改 `critic.py`；
- 不要修改 `ensemble_dynamics_for_batch.py`；
- 不要同时引入新的 reward aggregation；
- 不要顺手调很多超参；
- 不要把 old policy update、OPE 逻辑、ratio 聚合一起改。

## 建议的辅助 ablation

### Ablation 1

只改 offline chunk ratio 聚合，其他保持不变。

这是主实验。

### Ablation 2

在主实验之外，再做一个低成本稳定性对照：

- 保持 `chunk_as_single_action=True`
- 将 `use_action_embed=True`

原因：

- 当前 two-stage 脚本显式把 `use_action_embed=False`；
- 但默认配置很多是 `True`；
- 在 chunk 模式下，开启后会先把整段 raw chunk action 映射到 feature 维，再送入 critic / dynamics；
- 这可能降低 macro-action Q 的外推难度。

但要明确：

- `use_action_embed=True` 是 secondary ablation；
- 它不是这次 collapse 的 first fix；
- 不能替代 offline chunk ratio 修复。

## 验收标准

work agent 完成修复后，至少需要验证以下几点。

### 1. 代码级检查

- `python3 -m py_compile RL-100/rl_100/unidpg/uni_ppo.py`

### 2. 训练行为检查

重跑一条已知会塌的 chunk offline BPPO 配置，重点看：

- `each_scores.csv`
- `each_ope_score.csv`
- 如果有 ratio logging，也记录 chunk ratio 分布

期望看到的最低标准：

- `each_scores.csv` 不再在前几次迭代内快速掉到 0；
- `each_ope_score.csv` 即使继续上涨，也不能再与真实 score 完全背离；
- 最好能看到 offline chunk 至少维持初始性能或小幅提升，而不是立刻崩掉。

### 3. ratio 检查

如果配置里有 ratio logging 开关，建议打开，例如：

- `enable_ratio_logging: true`

重点检查：

- chunk `ratio_perstep` 是否大多数时间仍在 1 附近；
- 是否不再出现大面积长期贴在 clip 边界的情况；
- 与修复前相比，ratio 分布是否明显更温和。

### 4. 回归检查

确认以下路径不被这次修改破坏：

- `chunk_as_single_action=False`
- `n_action_steps=1`
- online PPO chunk update

本轮修改原则上只影响：

- offline BPPO
- `chunk_as_single_action=True`

## 如果 first fix 仍然不够

如果改成 per-step ratio 后，offline chunk 仍有明显 `ope 上升 / real score 下跌`，下一步再做以下诊断，按顺序来：

### Step 2

临时禁止 old policy 自更新相关逻辑，检查是否存在 biased OPE 驱动的自强化。

### Step 3

给 offline chunk 分支补日志：

- per-step ratio 均值 / 方差 / clip 比例
- chunk advantage 分布
- macro-Q 分布
- 真实 score 与 OPE 的差值轨迹

### Step 4

再考虑是否要进一步改：

- chunk advantage 形式；
- chunk OPE 定义；
- macro-Q 的 regularization。

但这些都应放在 first fix 之后。

## Work Agent 执行清单

1. 只修改 `RL-100/rl_100/unidpg/uni_ppo.py` 中 offline `update_distribution()` 的 `chunk_as_single_action=True` ratio 聚合逻辑。
2. 不修改 `critic.py`、`ensemble_dynamics_for_batch.py`、online PPO 路径。
3. 保持 chunk advantage 仍为 scalar，不要同时改 advantage 定义。
4. 跑 `python3 -m py_compile RL-100/rl_100/unidpg/uni_ppo.py`。
5. 选一条已知会 collapse 的 offline chunk run 做 A/B 验证。
6. 记录 `each_scores.csv` 和 `each_ope_score.csv` 的变化结论。
7. 如有余力，再单独做 `use_action_embed=True` 的 secondary ablation，但不要和 first fix 混在同一实验里。
