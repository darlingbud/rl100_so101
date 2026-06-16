# 分析：为什么 DDIM 在 offline BPPO 中表现好，而 Flow Matching 不行

## Context

在 RL-100 项目中，DP3CM 策略进行两阶段训练：
1. Stage 1: BC 预训练 + IQL Critic + Dynamics Model
2. Stage 2: BPPO (Behavior Proximal Policy Optimization) 微调

**关键事实**：Flow Matching 在 **online DPPO** 中与 DDIM 表现接近，但在 **offline BPPO** 中表现不佳。

这意味着 log-prob 本身的质量不是根本问题（online 也用同样的 log-prob）。问题出在 **offline BPPO 特有的使用方式**上。

**实验现象补充**：把协议更接近 DDIM 的 flow runs 单独看，flow 仍然明显差，但差距更像 0.49 vs 0.64，不是"一修就追平"的程度。这说明问题是多因素叠加，不是单一 bug。

## 作用范围约束

**此次修改只能涉及 offline flow 的 BPPO 调试与修复路径，不能改变 DDIM 或 online PPO/DPPO 的现有行为。**

具体约束：
- 只能修改 **offline flow** 相关逻辑，例如：
  - `uni_ppo.py` 中 offline `update_distribution` 的 flow 分支插桩/防御性处理
  - flow scheduler 的 **log-prob / debug 返回路径**
  - flow-only 配置项（如 `flow_noise_on_final_step`、`flow_noise_level`）
- **不能** 修改会影响公共采样语义的模块行为，尤其不能改到：
  - DDIM scheduler 的 sample / step / step_forward_logprob 逻辑
  - flow scheduler 的通用 `step()` / `step_mean()` 推理行为，如果该修改会影响 online rollout 或常规采样结果
  - `predict_action()`、online rollout、online PPO/DPPO 路径的默认行为
- 如果需要增加诊断信息，优先采用：
  - flow / DDIM 各自的 `step_forward_logprob(..., return_debug=True)` 这类**默认关闭**的调试返回
  - 仅在 offline BPPO 调用点启用
- 任何会改变默认 DDIM 行为、online 行为或公共 sampler 数值语义的修改，都应视为超出此次范围，不应纳入本轮工作。

---

## 嫌疑排序

### 嫌疑 1（最强）：最后一步密度错配 — rollout 确定性 vs log-prob 假设 Gaussian

**代码证据**：

Rollout 路径 `step_logprob` 调用 `_compute_step(prev_sample=None)`（`flow_match_scheduler.py:383`），在最后一步走到 line 232-244：

```python
is_final = self.step_index >= len(self.sigmas) - 2
should_add_noise = not is_final or self.flow_noise_on_final_step  # flow_noise_on_final_step 默认 False
if should_add_noise:
    prev_sample = prev_sample_mean + std * noise
else:
    prev_sample = prev_sample_mean  # ← 最后一步：确定性转移，delta distribution
```

但紧接着 line 386：
```python
log_prob = self._compute_logprob(prev_sample_out, prev_sample_mean, std, ...)
#                                 ↑ prev_sample_out == prev_sample_mean
#                                                       ↑ std 仍然非零！
```

这意味着 old_log_prob 在最后一步 = `log N(mean, std)` 在 mean 处求值 = 一个跟 action 内容无关的常数（仅取决于 std 的值）。

BPPO 路径 `step_forward_logprob`（line 412-413）也用同样的 std 做 Gaussian 密度：
```python
_, prev_sample_mean, std = self._compute_step(model_output, sample, generator=generator)
log_prob = self._compute_logprob(next_sample.float(), prev_sample_mean, std, ...)
#                                 ↑ next_sample 来自 old policy 的确定性最后一步
```

**错配的本质**：
- 真实 transition 是 delta distribution（确定性）
- log-prob 计算假设它是 `N(mean, std)` 
- 这两个密度不对应

**为什么 online PPO 能容忍**：
- Online 的 old_log_prob 和 new_log_prob 用的是同一组 trajectory states（从 buffer 取）
- Policy 变化小时 new_mean ≈ old_mean → ratio ≈ 1 → PPO clip 直接兜底
- 真实环境 reward 不断校正

**为什么 offline BPPO 放大这个错配**：
- 每次 `update_distribution` 都重新从 old_policy rollout，最后一步是确定性的
- 随着 new_policy 偏离 old_policy，最后一步的 `new_mean - old_mean` 差异变大
- 但密度 kernel 是 Gaussian 而非 delta，ratio 的波动不反映真实的概率比
- 错误的 ratio 乘以离线 advantage → 梯度方向错误 → 训练不稳定

**DDIM 没有这个问题**：DDIM 的所有步骤（含最后一步）在 eta > 0 时都是真正的随机转移，`std_dev_t` 来自理论推导且非零，密度 kernel 和实际 transition 自洽。

---

### 嫌疑 2（较强）：Flow 在 offline support 上更容易触发 Q/OPE 外推误差

#### Online DPPO 的闭环

```
policy → 环境交互 → reward → advantage → 更新 policy
log-prob 来自同一个 policy 的 rollout
```

- old_log_prob 和 trajectory 来自 **同一次 rollout**
- Advantage 来自 **真实环境 reward**
- Policy 改善 → 更好的 action → 更高的 reward → 正反馈循环

#### Offline BPPO 的开环

```
old_policy → 随机噪声 → 生成 trajectory → Q(s, a_generated) → advantage → 更新 policy
                                                ↑ 离线 Q 函数，只在数据集 support 内可靠
```

- old_policy 从 **随机噪声** 生成 trajectory（不是从数据集取 action）
- Advantage 来自 **离线训练的 Q 函数**，Q 只在数据集的 support 内可靠
- 如果 flow matching 生成的 action 分布与 DDIM 不同，**Q 的外推误差可能不同**

Flow Matching 的 denoising 动力学使得生成的 action 可能更分散：
- **Flow (CPS)**：`std = σ_next · sin(noise_level·π/2)`，`noise_level=0.7` → `std ≈ 0.89·σ_next`
- 较大的中间过程方差 → 生成 action 更分散 → 更容易落入 Q 函数外推误差大的区域
- 在 BPPO 中只微调 model（UNet），不重新训练 Q 函数，无法纠正这个偏差

---

### 嫌疑 3（防御性改进，非根因）：Scheduler 状态管理

**之前的分析过度强调了这一点**。实际上：
- BPPO 开始前有 policy eval（`train_cm_mid.py:884/919`），会走采样路径触发 `set_timesteps`
- `step_forward_logprob` 在 `num_inference_steps is None` 时会直接报错（line 404），不会静默给错值
- `_init_step_index(timestep)` 每次调用都会按 timestep 值重新设置 index（line 410）

因此 `set_timesteps` 未调用和 `step_index` 递增都不是实际的 crash 级 bug。但显式传 `step_index=i` 仍然是合理的防御措施。

---

## 实验层面的前置问题

**在做任何代码修改之前，需要先对齐实验协议。**

目前 DDIM 强 run 和 flow 强 run 的超参不一致：
- DDIM：`eval_episodes=30`, `use_aug=True`, `num_epochs=600`
  - 见 `data/outputs_2d/adroit_door_medium-dp3-0112_seed300/.../overrides.yaml`
- Flow：`eval_episodes=1`, `use_aug=False`, `num_epochs=800`
  - 见 `data/outputs_2d_flow_two_stage/adroit_door_medium-dp3-flow-01121_seed600/.../overrides.yaml`

这些差异（特别是 `eval_episodes` 1 vs 30）会严重影响报告的性能数字。必须在相同协议下重新跑一轮才能可靠对比。

---

## 总结

| | DDIM | Flow Matching |
|---|---|---|
| 最后一步 transition | 随机（eta > 0 时 std > 0） | 确定性（`flow_noise_on_final_step=False`）但 log-prob 仍用 Gaussian |
| 密度 kernel 与实际 transition | 自洽 | **错配**（delta vs Gaussian） |
| 离线 Q 外推 | action 分布由 alpha schedule 控制 | noise_level=0.7 使 action 更分散，Q 外推误差更大 |
| scheduler 状态 | 无状态 | 有状态但有保护（非根因） |

---

## 建议的诊断和修复步骤

### Step 0: 对齐实验协议（最优先）

在 flow 的 BPPO sweep 中使用和 DDIM 完全一致的 `eval_episodes`、`use_aug`、`num_epochs` 等超参，重新跑一轮基线。

### Step 1: 开启 ratio logging + std/noise 对比

在 flow 和 DDIM 的 offline BPPO 运行中都开启 `enable_ratio_logging=True`，并额外记录每个 denoising step 的：

- **`std`**：scheduler 计算出的转移方差
- **`‖prev_sample_mean‖`**：mean 的 norm（衡量信号量级）
- **`‖next_sample - prev_sample_mean‖`**：实际偏差（衡量噪声量级）
- **`‖next_sample - prev_sample_mean‖ / std`**：有效信噪比（SNR）

对比要点：
- DDIM 和 flow 在每一步的 std 量级差多少 → 如果 flow 的 std 系统性偏大或偏小，ratio 的灵敏度就不同
- flow 最后一步的 std 是否和前面步骤一样大 → 如果是，但 transition 其实是确定性的（`prev_sample == prev_sample_mean`），则 old_log_prob 在最后一步退化为常数，验证密度错配假说
- 两个 scheduler 的 SNR 曲线形状是否一致 → 如果 flow 的 SNR 在某些步骤异常大，ratio 在那些步骤就会剧烈波动

**实现方式**：不要在 `uni_ppo.py` 里直接调用 scheduler 的私有方法（`_compute_step` 只有 flow 有，DDIM 没有同样接口，而且对 flow 重复调用会重新采噪声）。正确做法是给两个 scheduler 分别暴露一个 `debug_stats` 返回路径：

- **Flow**：在 `step_forward_logprob` 内部，`_compute_step` 已经算出了 `prev_sample_mean` 和 `std`，增加一个 `return_debug=True` 参数把它们附带返回
- **DDIM**：在 `step_forward_logprob` 内部，已经有 `prev_sample`（mean）和 `std_dev_t`，同样通过 `return_debug=True` 附带返回

这样 `uni_ppo.py` 的插桩代码只需要：
```python
new_log_prob, debug = self._policy.noise_scheduler.step_forward_logprob(
    ..., return_debug=True
)
# debug = {'mean': ..., 'std': ...}
residual = (old_all_next_x[i] - debug['mean']).norm(dim=-1).mean()
snr = residual / debug['std'].clamp(min=1e-12).mean()
```

每个 scheduler 各自负责返回自己内部的 mean/std，调用方不需要知道实现细节。

**TODO（当前实现限制）**：
- 在 `chunk_as_single_action=True` 且使用 `chunk-level ratio` 时，当前 `ratio_stats.csv` 已经按实际优化的 `action_start:action_end` 窗口统计，口径是对的。
- 但当前 `debug_stats.csv` 仍然基于整段 `next_sample / mean / std` 计算 `std_mean`、`residual_norm`、`snr`，没有同步裁剪到相同的 chunk 窗口。
- 这意味着目前的 `debug_stats.csv` 仍可用于看 scheduler 的整体噪声水平和 DDIM/flow 的趋势差异，但**不是**严格对应 `chunk-level ratio` 的同口径统计。
- 本轮先保持现状；如果后续需要做严格的 chunk-level apples-to-apples 对比，再把 `debug_stats` 也改成按 `action_start:action_end` 窗口统计。

### Step 2: 修复最后一步密度错配

两个方向：
- **方案 A（首选）**：设置 `flow_noise_on_final_step=True`，使最后一步也注入噪声，transition 和密度自洽
- **方案 B（仅用于诊断对照，不作为修复）**：在最后一步用极小的 std（如 1e-6）计算 log-prob。注意这会使最后一步 ratio 极端敏感（微小的 mean 偏差 → ratio 爆炸），本质上是重新定义了 BPPO 的密度模型，不是数值修补。只适合用来确认"最后一步确实是问题源"，不适合作为正式修复

### Step 3: 防御性改进（可选）

在 `uni_ppo.py:update_distribution` 的 BPPO 循环中显式传 `step_index=i`：
```python
new_log_prob = self._policy.noise_scheduler.step_forward_logprob(
    model_output, timesteps, old_all_x[i],
    next_sample=old_all_next_x[i], eta=eta,
    step_index=i
)
```

### Step 4: 如果 Step 1-2 不够，考虑调整 noise_level

尝试减小 `noise_level`（如 0.3, 0.5），使 flow 生成的 action 更集中在 Q 函数的可靠区域内。

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `rl_100/unidpg/uni_ppo.py:424-578` | Offline BPPO update_distribution |
| `rl_100/unidpg/uni_ppo.py:837-972` | Online DPPO dp_align_update_no_share |
| `rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py:82,232-248,364-415` | Flow scheduler：final step 逻辑 + log-prob |
| `rl_100/policy/diffusion_policy/diffusers_patch/ddim_with_logprob_dpok.py:388-547` | DDIM log-prob（无状态，所有步骤随机） |
| `rl_100/policy/dp3_cm.py:1284-1347` | all_step_logprob |
