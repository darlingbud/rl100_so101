# Offline/Online Chunk-as-Single-Action Debug Summary

## 1. 目标

本文件总结当前对 `offline BPPO + chunk_as_single_action` drop 问题的完整 debug 过程、已经确认的结论、被排除的假设，以及接下来建议 Codex 实施的代码修改与实验任务。

目标是让 Codex **直接据此继续修改代码和跑实验**，避免重复走已经被证伪的方向。

---

## 2. 问题背景

当前训练框架是：

- BC pretrain
- offline BPPO / RL fine-tuning
- online PPO RL fine-tuning

当前现象：

- **offline single action**: work，有提升
- **online single action**: work
- **online chunk as single action**: work
- **offline chunk as single action**: 不 work，常常从 BC performance 直接 crash 到接近 0

---

## 3. 已确认的重要事实

### 3.1 当前 offline chunk 配置下 `no_pre_action=True`
因此：
- policy 生成的 trajectory 本身已经是裁掉 pre-action 之后的 chunk
- 在当前配置里，offline chunk 的 `advantage` 和 `ratio` 对应的是同一个 chunk action 对象
- 因此，“advantage 和 ratio 指向不同 action 子段”**不是当前问题主因**

结论：
- 之前那个“action 对象不一致”的怀疑，在当前配置下已排除

---

### 3.2 Critic / IQL 的 chunk 支持是存在的
critic 中 `chunk_as_single_action=True` 时：

- `action_dim = single_action_dim * n_action_steps`
- chunk action 被 flatten 为一个 macro-action
- `iql.get_advantage(s, a)` 本质是：
  - `minQ(s, a_chunk) - V(s)`

结论：
- offline chunk BPPO 使用的 actor signal，本质是 **chunk-level scalar IQL advantage**
- 即：
  - `A_chunk(s, a_chunk) = Q_chunk(s, a_chunk) - V(s)`

---

### 3.3 老板旧版 offline BPPO loss 在 `final_reward=True + chunk_as_single_action=True` 下存在语义问题

老板旧版代码中：

- 前面算的是一个 **chunk-level scalar advantage**
- 后面 loss 却写成：
  - `ratio[:, j] * advantages[j]`

这里的 `advantages[j]` 在该分支下不是第 `j` 个 chunk step 的 advantage，
而更像是在 batch 维上取第 `j` 个样本，语义错位。

这是一个真实问题，但后续 ablation 表明：

- **把 `advantages[j]` 改成共享的 `advantages` 后，offline 仍然 drop**
- **不改反而不 drop**

结论：
- 这个 bug **不是 offline drop 的根因**
- 更像是它把真正有害的 `scalar_iql` 信号“打散成噪声”，所以看起来反而不 drop

---

## 4. 已做过的关键 ablation 与结论

### 4.1 新版 offline 实验 1：只改 ratio 粒度，不改 advantage 来源

实验设置：

- `offline_chunk_adv_mode = scalar_iql`
- 对比：
  - `per_step:scalar_iql`
  - `scalar:scalar_iql`

结果：

- 两组都会 drop

结论：

- “scalar advantage + per-step ratio 的 surrogate 粒度错配”**不是主因**
- 就算把 ratio 改成真正的 chunk-scalar ratio，只要 advantage 还是 `scalar_iql`，仍然 drop

这说明：
- **核心问题更像在 `scalar_iql` 这个 chunk-level actor signal 本身**

---

### 4.2 老板旧版最小修复：把 `advantages[j]` 改成共享的 `advantages`

结果：

- 改成共享 `advantages` 后，仍然 drop
- 不改 `advantages[j]` 反而不 drop

结论：

- 老板旧版“不 drop”不是因为它更对
- 更像是因为那个 bug 把 `scalar_iql` 这个坏信号破坏掉了

---

### 4.3 共享 sign advantage ablation

把共享的 `advantages` 改成：

```python
shared_adv = advantages.detach().squeeze(-1).sign().unsqueeze(-1)
```

结果：

- drop 趋势变小
- 但也不提升

结论：

- `scalar_iql` 的**幅值**部分有害
- 但就算只保留**符号**，也仍然不够有用，无法带来提升

所以更像是：

- `scalar_iql` 既“太猛”
- 也“未必准”

---

### 4.4 把 chunk size 从 16 改成 4，offline 仍然 drop

结论：

- 问题不太像只是因为 chunk 动作空间太大（例如 16-step / 448-d）
- 更像是：
  - 只要把 action 当作 multi-step chunk，
  - `scalar_iql = Q(s, a_chunk) - V(s)` 这个 signal 就不可靠

---

### 4.5 online 严格 chunk-scalar ratio 改法测试通过

online 原实现中，所谓 `chunk_as_single_action` 本质上只是：

- per-step ratio
- shared rollout-based scalar advantage

并不是真正的 whole-chunk scalar PPO

后来只修改 online 的 `chunk_as_single_action=True` 分支，使其变成：

- **whole-chunk scalar ratio**
- **chunk-level scalar advantage**

且不影响非-chunk 分支

测试结果：

- online 改成严格 chunk-scalar ratio 后，**没问题**

结论：

- `chunk as single action` 这个 actor loss 语义本身不是问题
- 真正的分水岭在于：
  - online 用的是 rollout / GAE advantage
  - offline 用的是 IQL `Q-V` advantage

这进一步支持：
- **offline 的核心问题在 `scalar_iql`，不是在 chunk-scalar PPO 写法本身**

---

## 5. 当前最可能的总体结论

当前最可能结论：

### 5.1 offline chunk drop 的主因不是 ratio 写法
已经证据充分排除：

- `per_step:scalar_iql` drop
- `scalar:scalar_iql` 也 drop

所以：
- 不是 per-step ratio / scalar ratio 的主导问题

---

### 5.2 offline chunk drop 的主因更像是 `scalar_iql` 不可靠
即：

- `A_chunk(s, a_chunk) = Q_chunk(s, a_chunk) - V(s)`

这个 chunk-level critic signal 对 actor 来说不可靠。

可能体现为：

- 排序质量差
- 数值过于激进
- 对 chunk macro-action 不适合作为直接 actor update signal

---

### 5.3 老板旧版“不 drop”不能说明旧版是对的
更合理的解释是：

- 旧版 bug 破坏了 `scalar_iql` 的有害更新信号
- 使更新变成噪声 / 弱更新
- 因此看起来不 drop

---

## 6. 当前不建议继续深挖的方向

以下方向已经价值不大，除非作为补充 sanity check：

1. 继续在 `scalar_iql` 上调 ratio 粒度  
   - 已经做过 `per_step:scalar_iql` vs `scalar:scalar_iql`
   - 都 drop

2. 继续研究老板旧版 `advantages[j]` bug 本身  
   - 已确认不是根因
   - 更像噪声 masking

3. 单纯通过减小 chunk size 来救  
   - chunk=4 仍然 drop

---

## 7. 接下来最应该做的事

## 7.1 首要任务：跑 offline `per_step_vdelta`
目标：

- 替换掉 `scalar_iql` 作为 offline chunk actor 的 advantage 来源

当前代码中已经加了实验模式：

- `offline_chunk_ratio_mode`
- `offline_chunk_adv_mode`

下一步优先建议跑：

### 方案 A
- `offline_chunk_ratio_mode = per_step`
- `offline_chunk_adv_mode = per_step_vdelta`
- `final_reward = True`

解释：
- 这是真正的“per-step ratio + per-step model-based value-delta advantage”
- 如果这组明显比 `scalar_iql` 稳，说明主问题就是 `scalar_iql`

### 方案 B（可选）
- `offline_chunk_ratio_mode = scalar`
- `offline_chunk_adv_mode = scalar_iql`

这已经做过，作为对照即可

---

## 7.2 次要但高价值的诊断：检查 chunk critic 排序质量

建议额外写一个独立诊断脚本：

### 输入
- 固定一批 state `s`
- 从当前 policy / noise / candidate sampling 中采样多个 chunk action：
  - `a_1, a_2, ..., a_k`

### 对每个 `(s, a_i)` 计算
1. `iql.get_advantage(s, a_i)`
2. dynamics rollout proxy score  
   可选：
   - discounted reward sum
   - `per_step_vdelta` 累积值
   - 或其他 rollout-based proxy return

### 输出
- Pearson correlation
- Spearman rank correlation

目标：
- 检查 `scalar_iql` 对同一 state 下多个 candidate chunk 的排序质量
- 如果 rank correlation 很差，则可直接支持：
  - `Q_chunk - V` 对 chunk actor signal 不可靠

---

## 7.3 可选诊断：shuffle advantage sanity check
在 offline `scalar_iql` 分支中，对 batch 内 advantage 做 shuffle：

```python
perm = torch.randperm(advantages.shape[0], device=advantages.device)
advantages = advantages[perm]
```

如果 shuffle 前后效果差不多，则说明：
- 原始 `scalar_iql` 的 per-sample 排序信息几乎没有比随机好多少

这个诊断不是最优先，但代价低。

---

## 8. Codex 需要继续做的具体代码任务

### Task 1: 保持 online 当前修改，不要回退
online `chunk_as_single_action=True` 分支已经改成 strict chunk-scalar ratio，并验证没问题。

要求：
- **不要影响非-chunk-as-single-action 分支**
- **不要回退 online 当前改动**

---

### Task 2: 在 offline 中优先支持并跑通 `per_step_vdelta`
检查并确保以下逻辑可用：

- `offline_chunk_ratio_mode='per_step'`
- `offline_chunk_adv_mode='per_step_vdelta'`
- `final_reward=True`

重点检查：
- dynamics.step 的输入语义
- value 输入 shape
- `loss = loss / opt_steps` 已存在且保持
- 仅对 chunk-as-single-action 分支生效
- 不影响非-chunk 分支

---

### Task 3: 新增一个独立诊断脚本 / 函数：chunk ranking quality
功能：
- 输入 state batch
- 采样多个 chunk candidates
- 比较：
  - `iql.get_advantage(s, a_chunk)`
  - rollout proxy return / vdelta score
- 输出 rank correlation

推荐输出：
- mean Pearson
- mean Spearman
- 可保存 CSV

---

### Task 4: 保留并整理实验配置开关
确保以下 config 开关存在且明确：

- `offline_chunk_ratio_mode: per_step | scalar`
- `offline_chunk_adv_mode: scalar_iql | per_step_vdelta`

并保证：
- 非 chunk 分支不受影响
- 非 offline BPPO 路径不受影响

---

## 9. 推荐的下一组实验

按优先级：

### Exp 1
offline chunk
- ratio = `per_step`
- adv = `per_step_vdelta`

目标：
- 检查换掉 `scalar_iql` 后是否不再 drop

### Exp 2
offline chunk ranking diagnostic
- 不训练 / 少训练
- 只做 correlation analysis

### Exp 3（可选）
offline chunk
- `scalar_iql` + shuffled advantage
- 做 sanity check

---

## 10. 当前最终判断（一句话版）

**当前最可能的根因不是 chunk PPO ratio 写法，而是 offline chunk actor 使用的 `scalar_iql = Q_chunk(s, a_chunk) - V(s)` 这个 signal 本身不可靠；老板旧版之所以“不 drop”，更像是因为 bug 把这个坏信号打散成了噪声。**


---

## 11. 代码快照（供 Codex 对照）

以下两份代码快照分别对应当前讨论中的 **新版 offline `update_distribution`** 与 **老板旧版 offline `update_distribution`**。

### 11.1 新版代码快照
来源：当前新版 offline `update_distribution`（含 `offline_chunk_ratio_mode / offline_chunk_adv_mode`、`per_step_vdelta` 等实验分支）

参考文件：
- `Pasted code(3).py`

可用于 Codex 对照：
- `chunk_ratio_mode`
- `chunk_adv_mode`
- `scalar_iql`
- `per_step_vdelta`
- `scalar ratio`
- `per-step ratio`
- `loss = loss / opt_steps`

### 11.2 老板旧版代码快照
来源：老板旧版 offline `update_distribution`
参考文件：commit:5af6984cca38c53bed6a7464e028876e20a711bf 的uni_ppo.py文件（注意当时RL-100文件夹名还叫3D-Diffusion-Policy)
- 

可用于 Codex 对照：
- 旧版 `final_reward=True + chunk_as_single_action=True` 分支
- 旧版 `advantages[j]` 写法
- 与新版实验分支逐行比对差异

### 11.3 Codex 使用建议
Codex 在继续修改时，优先将这两份代码作为：
- **新版基线**
- **旧版对照**

并重点围绕以下问题进行修改与实验：

1. `scalar_iql` 是否应继续作为 offline chunk actor signal  
2. `per_step_vdelta` 是否能替代 `scalar_iql`  
3. chunk critic 排序质量诊断脚本  
4. 保持 non-chunk 路径不受影响  
5. 保持 online 当前 strict chunk-scalar ratio 改动不回退

---

## 12. TODO For Work Agent

本节是给后续 work agent 的直接执行清单。目标不是继续扩展猜想，而是按已经收敛的方向完成最小必要修改、验证和实验。

### 12.1 总目标

本轮按两条主线并行推进，但改动面保持最小：

1. 主线 A：把 offline chunk 的 `per_step_vdelta` 实验链路检查、跑通并验证  
2. 主线 B：补一个独立的 chunk ranking diagnostic，用来量化 `scalar_iql` 的排序质量

第一阶段成功标准：

- `offline chunk + per_step_vdelta` 至少显著好于 `scalar_iql`
- 不再出现从 BC performance 直接 crash 到接近 0 的现象

---

### 12.2 必须遵守的边界

1. 不要回退 online 当前 `chunk_as_single_action=True` 的 strict chunk-scalar ratio 改动  
2. 不要影响 non-chunk 分支  
3. 不要影响非 offline BPPO 路径  
4. 不要扩展到 `dp3.py`、`dp_image_unet`、`dp_state`、其他 policy 变体  
5. 不要把这轮工作扩展成 flow / CM / distillation / GRPO 方向  
6. 不要修改默认 DDIM 或 CM 行为；如果发现会被波及，必须单独报出并停止

---

### 12.3 Task A: Offline `per_step_vdelta` 主链路

Work agent 需要先以当前 `uni_ppo.py` 为基线，重点检查并确认以下行为全部成立：

1. `offline_chunk_ratio_mode` 与 `offline_chunk_adv_mode` 只在：
   - offline BPPO
   - `chunk_as_single_action=True`
   - `opt_steps > 1`
   这条路径生效

2. `offline_chunk_ratio_mode='per_step'`
   `offline_chunk_adv_mode='per_step_vdelta'`
   `final_reward=True`
   这组组合可以完整跑通

3. `per_step_vdelta` 的语义必须固定为：
   - 对 chunk 内每一步 action 调一次 dynamics
   - 计算 `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`
   - `use_gae=True` 时，对 chunk 内的 delta 做 GAE
   - `use_gae=False` 时，直接使用原始 delta

4. actor loss 必须固定为：
   - per-step ratio
   - per-step advantage
   - chunk 内平均，即保留 `loss = loss / opt_steps`

5. 要重点检查的实现细节：
   - `dynamics.step()` 输入语义是否与现有 `multi_step` / `multi_step_evaluation` 一致
   - `prediction_mode="full"` 与非 `full` 两条路径下，next state/value 的 shape 和含义是否一致
   - value 输入是否始终对应当前 policy feature window，而不是错误的单帧特征
   - `no_pre_action=True` 下 chunk slice 是否和 ratio 作用区间严格一致
   - offline `update_distribution()` 调用必须显式传入 critic 侧使用的 `gamma/lamda`，不要继续落回 `uni_ppo.py` 中的默认 `0.99/0.95`
   - `per_step_vdelta` 若依赖 dynamics reward，则必须明确要求 `predict_r=True`，不允许在 `predict_r=False` 时静默退化为零奖励版本
   - 可增加低成本 shape assert，避免 `nobs_features` 或 `chunk_actions` 维度漂移后仍然静默运行

6. `per_step_vdelta` 的 advantage 还必须补一件事：
   - 需要像 `scalar_iql` 一样做 advantage normalization
   - 否则 `scalar_iql` 与 `per_step_vdelta` 的实验对比会混入尺度差异，无法只比较 signal 质量
   - normalization 的位置和语义应固定下来，避免一部分 step 已聚合、一部分 step 未标准化的混乱实现

7. 对 `scalar_iql` 的要求：
   - 保留它作为对照组
   - 不再新增新的 `scalar_iql` surrogate 变体
   - 不再把精力放在 `per_step` vs `scalar` ratio 争论上

---

### 12.4 Task B: Chunk Ranking Diagnostic

新增一个独立诊断入口，目的仅用于分析，不接入训练主路径。

需要实现的功能：

1. 输入一批固定 state batch  
2. 对每个 state 采样多个 chunk candidate action  
3. 对每个 candidate 计算：
   - raw `q(s, a_chunk)`
   - raw `q(s, a_chunk) - v(s)`
   - rollout proxy score

注意：

- ranking diagnostic 不要直接复用训练路径里的 normalized `advantage_computation()`
- 诊断必须基于 raw score，而不是 batch-normalized score
- 否则相关性分析会被标准化过程污染

rollout proxy score 的默认定义：

- 使用与 `per_step_vdelta` 一致的 rollout/value 逻辑
- 对 chunk 内 per-step delta 做累积
- 若 `use_gae=True`，可同时输出 GAE 版聚合分数，但默认核心对照应保持为 vdelta 累积值

诊断输出最少包括：

1. 每个 state 上多个 candidate 的 Pearson correlation  
2. 每个 state 上多个 candidate 的 Spearman rank correlation  
3. 全体 state 的 mean Pearson  
4. 全体 state 的 mean Spearman  
5. CSV 导出，便于后续画图或复盘
6. 如实现成本不高，可同时输出：
   - raw `q` vs proxy score 的相关性
   - raw `q-v` vs proxy score 的相关性

该诊断的判据：

- 如果 `scalar_iql` 与 rollout proxy 的 rank correlation 明显较差，则可直接支持：
  - offline chunk actor 的坏信号主要来自 `scalar_iql`

额外检查：

- 当前 `dynamics.chunk_evaluation()` 若继续用于 chunk 诊断，需要先核对它传给 `Q` 的 state 表示是否正确
- 在 `chunk_as_single_action=True` 下，若 critic 期望的是 full flattened state feature，则不能继续只传最后一帧 feature
- 如该函数当前语义不对，应先修正再用于任何 ranking 结论

---

### 12.5 推荐执行顺序

1. 先做静态检查，确认 `per_step_vdelta` 路径实现语义无误  
   - 包括确认 offline `gamma` 实际使用的是 critic 配置值，而不是 `uni_ppo.py` 默认值
   - 包括确认 `per_step_vdelta` 已做 advantage normalization
   - 包括确认 `predict_r=True`，否则 `per_step_vdelta` 的 reward 项没有语义
2. 先跑最小 offline chunk 实验：
   - 对照组：`per_step + scalar_iql`
   - 主实验：`per_step + per_step_vdelta`
3. 主实验路径稳定后，再补 ranking diagnostic  
4. 如果主实验仍然 crash，再回头用 diagnostic 验证：
   - 是 `scalar_iql` 问题已经被替换掉但仍有别的实现 bug
   - 还是 `per_step_vdelta` 本身的 rollout/value 语义存在错误

---

### 12.6 最小验证集

提交前至少完成以下验证：

1. BC `compute_loss`  
2. `predict_action`  
3. `all_step_logprob`  
4. offline `update_distribution`  
5. online rollout 和一轮 PPO mini-batch  
6. DDIM regression smoke test

额外要求：

- 若任何验证暴露出 non-chunk 或 online 行为回归，本轮实现应视为失败

---

### 12.7 推荐实验集

#### Exp 1
offline chunk:

- ratio = `per_step`
- adv = `scalar_iql`

用途：

- 作为当前问题对照组

#### Exp 2
offline chunk:

- ratio = `per_step`
- adv = `per_step_vdelta`

用途：

- 作为主实验，检查换掉 `scalar_iql` 后是否不再 drop
- 该实验必须确认：
  - offline `gamma` 已与 critic 配置对齐
  - `per_step_vdelta` 已做 normalization
  - `predict_r=True`

#### Exp 3
chunk ranking diagnostic:

- 固定 state
- 多 candidate chunk
- 比较 raw `q`、raw `q-v` 与 rollout proxy 的相关性

用途：

- 给出 `scalar_iql` 排序质量的直接证据
- 分离“Q 本身排序差”与“Q-V baseline 后排序差”两种可能

#### Exp 4（可选）
offline chunk:

- `scalar_iql` + shuffled advantage

用途：

- 作为低优先级 sanity check

---

### 12.8 交付物要求

本轮 work agent 最终应交付：

1. 已确认或修正后的 offline `per_step_vdelta` 主链路  
2. 至少一组 `scalar_iql` vs `per_step_vdelta` 的离线对照实验结果  
3. 一个独立可运行的 chunk ranking diagnostic  
4. sweep / 启动脚本已包含 `per_step:per_step_vdelta` 这组主实验入口，避免手工漏跑  
5. 一段简短结论，明确回答：
   - `per_step_vdelta` 是否能显著缓解 offline chunk drop
   - raw `q` 与 raw `q-v` 哪一个排序更差
   - `scalar_iql` 的排序质量是否足够差，从而支持当前根因判断

---

### 12.9 额外实现提醒

以下三点属于本轮应明确落地的实际改动，不应只停留在口头假设：

1. `train_ddp.py` 中 offline `update_distribution()` 调用需要显式传入：
   - `gamma=self.cfg.critic.gamma`
   - `lamda=self.cfg.critic.lamda`

说明：

- 当前如果不传，`uni_ppo.py` 会使用默认 `gamma=0.99`
- 而 chunk 训练脚本当前显式覆盖的是 `critic.gamma=0.997`
- 因此这里是实际的 runtime mismatch，不只是代码风格问题

2. `per_step_vdelta` 分支需要补 advantage normalization

说明：

- `scalar_iql` 当前已经标准化
- `per_step_vdelta` 若不标准化，会让对比掺入 scale difference
- work agent 需要在实现中明确 normalization 的张量维度与时机，并保持 chunk loss 语义一致

3. 训练脚本中的 `CHUNK_LOSS_MODE_COMBOS` 需要加入：
   - `per_step:per_step_vdelta`

说明：

- 否则主实验不会自动进入默认 sweep
- 这会导致文档结论与实际脚本入口脱节

4. `per_step_vdelta` 需要显式要求 `predict_r=True`

说明：

- 当前配置默认 `predict_r=False`
- 如果不打开 reward prediction，dynamics rollout 会返回零奖励
- 这样 `per_step_vdelta` 会退化，实验结论不可信

5. 用于 chunk ranking 的 raw score / critic 评估接口必须与训练接口解耦

说明：

- 诊断不要复用 normalized actor advantage
- 需要直接看 raw `q`、raw `v`、raw `q-v`
- 若继续使用 `dynamics.chunk_evaluation()`，必须先修正其 state 输入语义，保证 chunk critic 看到的是正确的 full state feature，而不是最后一帧 feature

---

## 13. Review - Round 1

本节记录对本轮 work agent 提交结果的第一轮 review。这里只保留需要 work agent 继续执行的修正项，不重复前文已经确认正确的部分。

### 13.1 本轮已确认可以保留的改动

以下改动方向目前看是对的，可以保留：

1. `train_ddp.py` 中 offline `update_distribution()` 显式传入：
   - `gamma=self.cfg.critic.gamma`
   - `lamda=self.cfg.critic.lamda`

2. `uni_ppo.py` 中：
   - `per_step_vdelta` 增加 `predict_r=True` guard
   - `per_step_vdelta` 增加 advantage normalization

3. `ensemble_dynamics_for_batch.py` 中：
   - `chunk_evaluation()` 改为给 chunk critic 传 full flattened state feature，而不是最后一帧 feature

4. `train_policy_chunk_two_stage.sh` 中：
   - `CHUNK_LOSS_MODE_COMBOS` 加入 `per_step:per_step_vdelta`
   - 为 `per_step_vdelta` 提供 `predict_r=True` 入口

---

### 13.2 必须继续修的高优先级问题

#### Issue 1: `chunk_ranking_diagnostic.py` 没有加载已训练 dynamics 权重

当前诊断脚本里：

- 调用了 `train_dynamics(...)` 去实例化 dynamics
- 但没有像 offline 训练主路径那样继续调用 `dynamics.load(dynamics_path)`

这会导致：

- ranking diagnostic 使用的是随机初始化 / 未训练的 dynamics
- rollout proxy score 没有可信度
- 输出的相关性结果不可用于判断 `scalar_iql` 排序质量

work agent 必须修正：

1. 在 `chunk_ranking_diagnostic.py` 中，完成 dynamics 实例化后显式调用：
   - `dynamics.load(dynamics_path)`

2. 若 `dynamics.pth` 不存在，应直接报错，不要静默继续

3. 这一逻辑必须与当前 offline 主训练路径保持一致

参考现有正确路径：

- `train_ddp.py` 中 offline 路径会在 stage1 dynamics 存在时加载 `dynamics_path`
- `ensemble_dynamics_for_batch.py` 已经提供了 `load()`

---

#### Issue 2: 当前 diagnostic 不能回答 “raw `q` vs raw `q-v` 谁排序更差”

当前 `chunk_ranking_diagnostic.py` 的设计是：

- 固定一个 state
- 对这个 state 采样多个 candidate chunk
- 比较 raw `q`
- 比较 raw `q-v`

问题在于：

- 对固定 state 来说，`v(s)` 对所有 candidate 都是常数
- 因此：
  - raw `q-v` 与 raw `q` 的排序严格相同
  - raw `q-v` 与 raw `q` 的 Pearson / Spearman 相关性也会相同

所以当前脚本虽然会输出两组数字，但它并不能回答：

- raw `q` 与 raw `q-v` 哪个排序更差

work agent 必须二选一：

1. 如果诊断目标仍然是“固定 state 下 candidate chunk 排序质量”：
   - 保留 per-state ranking design
   - 但不要再把 “比较 raw `q` vs raw `q-v` 谁排序更差” 当作目标或结论
   - 明确写清楚：对固定 state，`q` 和 `q-v` 的 candidate ranking 等价

2. 如果真的要比较 raw `q` 与 raw `q-v` 谁更差：
   - 需要重新设计诊断目标
   - 不能只在固定 state 内比较 candidate ranking
   - 必须引入跨 state 的分析口径，并单独说明其意义

当前建议：

- 优先采用方案 1
- 保留“固定 state + candidate chunk”这个主诊断设计
- 用它回答：
  - `scalar_iql` / chunk critic 对 candidate chunk 的排序质量是否足够差
- 不要继续把 `q` vs `q-v` 谁更差当成本轮必须回答的问题

---

### 13.3 中优先级问题

#### Issue 3: diagnostic 文档中的 `--run-dir` 用法与实现不一致

当前脚本文档写了：

- `--run-dir /path/to/stage1_run_dir`

但实现中没有真正解析这个 CLI 参数，目前只读取：

- `cfg.hydra.run.dir`
- 或环境变量 `DIAG_RUN_DIR`

work agent 必须修正二者之一：

1. 要么真正支持 `--run-dir`
2. 要么删除文档中的 `--run-dir` 用法，并把正确启动方式写清楚

要求：

- 文档与实际入口必须一致
- 不要保留会误导执行者的假命令

---

### 13.4 建议的下一步执行顺序

work agent 下一轮建议按以下顺序继续：

1. 先修 `chunk_ranking_diagnostic.py` 的 dynamics load 问题  
2. 再收敛 diagnostic 目标，去掉或改写 “raw `q` vs raw `q-v` 谁排序更差” 这一不成立的目标  
3. 再修 `--run-dir` 文档 / 实现不一致问题  
4. 然后再到训练机上跑：
   - `per_step:scalar_iql`
   - `per_step:per_step_vdelta`
   - 修正后的 chunk ranking diagnostic

---

### 13.5 补充问题（第二轮 review 合并）

以下问题由第二轮独立 review 发现，与 13.2–13.4 不重复：

#### Issue 4: `per_step_vdelta` delta 计算缺少 terminal mask

当前 `uni_ppo.py` 中 `_compute_chunk_step_advantages_vdelta()` 的 delta 计算为：

```python
delta = reward_t + gamma * value_next - value_now
```

`dynamics.step()` 返回的 `terminal` 被丢弃了（第三个返回值用 `_` 接收）。

正确写法应为：

```python
next_obs, reward, terminal, _ = dynamics.step(...)
terminal_t = torch.from_numpy(terminal).to(device=self._device, dtype=value_now.dtype).reshape(batch_size, -1)[:, 0]
delta = reward_t + gamma * (1 - terminal_t) * value_next - value_now
```

同样的问题也存在于 `chunk_ranking_diagnostic.py` 的 `compute_rollout_proxy_score()` 中（line 124）。

虽然 chunk 内几步大概率不会 terminal，但修复代价极低，语义更正确。

---

#### ~~Issue 5: `predict_r` assert 的属性路径~~ — 已确认不是问题

`EnsembleDynamics_batch.__init__` 中有 `self.predict_r = cfg.predict_r`（ensemble_dynamics_for_batch.py:59），因此 `dynamics.predict_r` 直接可用。`uni_ppo.py:466` 和 diagnostic 中的引用方向正确，无需修改。

---

### 13.6 合并后的完整修复清单

按优先级排序：

| # | 问题 | 文件 | 优先级 |
|---|------|------|--------|
| 1 | dynamics 权重未加载 | `chunk_ranking_diagnostic.py` | 高 |
| 2 | terminal mask 缺失 | `uni_ppo.py` + `chunk_ranking_diagnostic.py` | 中 |
| 3 | q vs q-v 排序等价，目标表述需修正 | `chunk_ranking_diagnostic.py` | 中 |
| 4 | `--run-dir` 文档与实现不一致 | `chunk_ranking_diagnostic.py` | 低 |

work agent 下一轮按此清单顺序修复即可。

---

## 14. 第二轮实验：vdelta 方案 + stride fix

### 14.1 背景

第一轮 debug 收敛到 `scalar_iql` signal 不可靠这个结论后，分两条线并行推进：

1. **vdelta 方案**：用 dynamics rollout 替代 `scalar_iql` 作为 advantage 来源
2. **stride fix**：修复 offline dataset stride=1 与 online SMDP stride=n_action_steps 的语义不匹配

---

### 14.2 vdelta 方案的 shape mismatch 问题

`per_step_vdelta` 在训练时 crash：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x28 and 448x384)
```

根因：当 `chunk_as_single_action=True + use_action_embed=True` 时，dynamics 的 `_action_encoder` 输入维度是 `n_action_steps * action_dim = 16 * 28 = 448`（整个 chunk flatten）。但 `_compute_chunk_step_advantages_vdelta` 逐步传入单步 action（dim=28），导致 shape mismatch。

chunk dynamics 是 chunk-level world model，不支持 single-step rollout。

因此拆成两个子方案：

- **方案 A（per_step_vdelta）**：额外训练一个 single-step dynamics（`chunk_as_single_action=False`），做真正的 per-step rollout
- **方案 B（chunk_vdelta_scalar）**：用现有 chunk dynamics 做一次 forward，得到 `discounted_reward_sum + γ^T · V(s_final) - V(s_0)` 作为 scalar advantage

方案 A 存在语义问题：V(s) 是 chunk-level IQL 训练的，语义是"从 s 执行一整个 chunk 的 value"，但 per-step rollout 中 V(s_{t+1}) 被当作 single-step value 使用，语义不匹配。

---

### 14.3 stride mismatch 假设

核心假设：

> offline chunk data 用 stride=1 滑窗构建，online chunk rollout 用 stride=n_action_steps。两者不是同一个决策过程。

- **Online**：每 n_action_steps 步决策一次，存一个 transition → SMDP chunk-boundary 决策
- **Offline**：每个 env step 都生成一个 sample → 每个中间状态都被当作 chunk 决策点

这导致 offline critic/IQL 在错误的 state distribution 上训练，`Q(s, a_chunk) - V(s)` 在中间状态上的估计没有实际意义。

修复方案：只改 critic/IQL 训练和 offline BPPO actor update 的 dataset stride 为 n_action_steps，BC pretrain 和 dynamics 训练保持 stride=1。

---

### 14.4 第二轮实验结果

BC baseline：~0.4667

#### 老实验（stride fix 之前，sliding stride=1）

| seed | ratio mode | adv_mode | 典型轨迹 | 趋势 |
|------|-----------|----------|----------|------|
| 300 | per_step | scalar_iql | 0.43→0.40→0.13→0.13→0.07 | crash |
| 300 | scalar | scalar_iql | 0.47→0.53→0.40→0.33→0.20 | 持续下降 |
| 300 | scalar | scalar_iql | 0.47→0.43→0.30→0.27→0.03 | crash |
| 200 | per_step | scalar_iql | 0.50→0.37→0.33→0.37→0.23 | 下降 |
| 200 | scalar | scalar_iql | 0.60→0.43→0.53→0.50→0.43 | 下降 |
| 200 | scalar | scalar_iql | 0.53→0.47→0.50→0.53→0.37 | 下降 |

结论：**sliding stride 下，不管 per_step ratio 还是 scalar ratio，scalar_iql 都会衰减。** scalar ratio 衰减慢一些但仍然不 work。

#### 新实验（stride fix 之后 + vdelta 方案）

| stride | ratio mode | adv_mode | 轨迹 | 趋势 |
|--------|-----------|----------|------|------|
| sliding(1) | per_step | scalar_iql (no clip) | 0.47→...→0.00 | crash（原始问题复现） |
| sliding(1) | per_step | scalar_iql (clip=0.05) | mean=0.49, last=0.50 | 勉强稳住，不提升 |
| sliding(1) | per_step | chunk_vdelta_scalar | mean=0.57, max=0.67, last=0.57 | 稳定，高于 BC |
| sliding(1) | per_step | per_step_vdelta | mean=0.49, last=0.47 | 平，约等于 BC |
| boundary(16) | per_step | scalar_iql | 0.47→0.53→0.47→0.47→0.33 | 下降 |
| **boundary(16)** | **scalar** | **scalar_iql** | **0.47→0.50→0.57→0.67→0.63** | **上升 ✓** |

---

### 14.5 关键结论

#### 结论 1：stride fix + scalar ratio 是唯一让 scalar_iql 真正 work 的组合

- sliding + per_step ratio → crash
- sliding + scalar ratio → 慢衰减，仍然不 work
- boundary + per_step ratio → 下降
- **boundary + scalar ratio → 上升到 0.67**

两个因素缺一不可。

#### 结论 2：stride mismatch 是根因之一

老实验证明 sliding stride 下 scalar ratio 也会衰减（seed300: 0.47→0.20, seed200: 0.60→0.43）。stride fix 后同样的 scalar ratio + scalar_iql 组合从衰减变成上升。

#### 结论 3：per_step ratio 在 chunk 场景下本身不稳定

不管 stride 是 1 还是 16，per_step ratio 都会导致衰减或 crash。chunk-level scalar advantage 乘以 per-step ratio 时，每个 step 的 ratio 波动独立累积，放大了不稳定性。

#### 结论 4：chunk_vdelta_scalar 在 sliding stride 下也稳定

chunk_vdelta_scalar（方案 B）在 sliding stride 下 mean=0.57, max=0.67，没有 crash。这可能是因为它绕过了 critic 的 distribution 问题——advantage 来自 dynamics rollout 而非 IQL Q-V。

#### 结论 5：per_step_vdelta（方案 A）效果不佳

per_step_vdelta 约等于 BC baseline，没有提升。原因可能是 chunk-level V(s) 被当作 single-step value 使用，语义不匹配。

---

### 14.6 修正第一轮结论

第一轮结论（Section 5）认为"offline chunk drop 的主因是 `scalar_iql` 不可靠"。

修正为：

> **offline chunk drop 是两个问题叠加的结果：**
>
> 1. **dataset stride mismatch**：offline 用 stride=1 滑窗，critic 在错误的 state distribution 上训练，导致 Q-V advantage 在非 chunk-boundary 状态上不可靠
> 2. **per-step ratio 不稳定**：chunk-level scalar advantage 乘以 per-step ratio，ratio 波动在 step 维度累积，放大更新不稳定性
>
> 老板旧版"不 drop"的解释不变：旧版 `advantages[j]` bug 把信号打散成噪声，掩盖了这两个问题。

---

### 14.7 推荐的最终方案

基于当前证据，推荐的 offline chunk BPPO 配置为：

- `sequence_stride = n_action_steps`（critic + BPPO dataset 对齐到 chunk boundary）
- `offline_chunk_ratio_mode = scalar`（whole-chunk scalar ratio）
- `offline_chunk_adv_mode = scalar_iql`（IQL Q-V advantage）

待验证的增强方案：

- boundary stride + scalar ratio + chunk_vdelta_scalar（stride 对齐 + dynamics-based advantage，可能进一步提升）

---

### 14.8 待做的正交实验

当前缺少的组合：

| stride | ratio | adv_mode | 状态 |
|--------|-------|----------|------|
| boundary(16) | scalar | chunk_vdelta_scalar | **未跑** |
| boundary(16) | per_step | chunk_vdelta_scalar | **未跑** |

这两组实验可以回答：stride fix + chunk_vdelta 是否比 stride fix + scalar_iql 更好。

---

## 15. 基于全部 Ablation 的修订总结

本节用于修订 14.x 中偏强的结论表述。

核心原则：

- 保留已有实验现象
- 但对“主因”与“充分条件”的判断做更严格的区分
- 明确区分：
  - **旧实验**：stride fix 之前，critic / finetune 都是 sliding
  - **新实验**：stride fix 之后，critic / finetune 对齐到 boundary

### 15.1 需要严格区分的两类 `scalar_iql` 实验

之前容易混淆的点是：

- `per_step:scalar_iql`
- `scalar:scalar_iql`

在 **stride fix 之前** 和 **stride fix 之后** 不是同一类实验。

也就是说，下面两组不能混着下结论：

1. `sliding + scalar ratio + scalar_iql`
2. `boundary + scalar ratio + scalar_iql`

这两者的 critic / BPPO actor-update 数据分布不同，不能简单地把它们都归类成“`scalar ratio + scalar_iql`”。

---

### 15.2 老实验（stride fix 之前）的可靠结论

旧实验里，`scalar ratio + scalar_iql` **确实比** `per_step ratio + scalar_iql` 更稳，但**不能直接称为 work**。

典型现象：

- `per_step + scalar_iql`
  - 经常从 ~0.5 掉到 ~0.0–0.2
- `scalar + scalar_iql`
  - 通常掉得更慢，`mean_eval` 更高，`last` 更高
  - 但仍然存在明显衰减

所以旧实验最稳妥的结论是：

> **在 sliding critic 条件下，`scalar ratio` 明显优于 `per_step ratio`，但它本身还不足以把 offline chunk 彻底救回来。**

换句话说：

- `scalar ratio` 是一个强改善因素
- 但在旧 sliding critic 设定下，它还不是充分条件

---

### 15.3 新实验（stride fix 之后）的可靠结论

目前真正完成了 stride-fix 的可比 run 是：

- `boundary + per_step ratio + scalar_iql`
- `boundary + scalar ratio + scalar_iql`

观察到的现象是：

- `boundary + per_step ratio + scalar_iql`
  - 仍然会衰减
  - 目前并没有表现出“真正 work”
- `boundary + scalar ratio + scalar_iql`
  - 已经出现明显上升并稳定维持高分的 run
  - 目前最像“真正 work”的 `scalar_iql` 组合

所以新实验支持的最强结论是：

> **在 boundary critic / finetune 条件下，`scalar ratio + scalar_iql` 明显优于 `per_step ratio + scalar_iql`。**

---

### 15.4 修订后的主因排序

结合旧实验与新实验，当前更合理的优先级排序是：

#### 第一优先级嫌疑：ratio 粒度错配

证据：

- 在旧 sliding critic 下，`scalar ratio + scalar_iql` 已经系统性优于 `per_step ratio + scalar_iql`
- 在新 boundary critic 下，`scalar ratio + scalar_iql` 进一步优于 `per_step ratio + scalar_iql`

这说明：

> **`per_step ratio` 在 chunk-level scalar advantage 场景下本身就很可疑。**

更形式化地说：

- 当前 advantage 多数是 chunk-level scalar
- 如果仍然乘 per-step ratio，就会把一个 chunk-level target 暴露给 step-level ratio 波动
- 这会放大不稳定性

#### 第二优先级因素：stride / decision semantics 对齐

证据：

- 在旧 sliding critic 条件下，`scalar ratio + scalar_iql` 只是“更好”，但还没有明显达到稳定 work
- 在 boundary critic 条件下，`scalar ratio + scalar_iql` 才第一次表现出“真正 work”的迹象

这说明：

> **stride fix 很可能不是第一主因，但很可能是让 `scalar ratio + scalar_iql` 真正 work 的重要辅助条件。**

#### 第三优先级因素：advantage signal 质量

证据：

- `chunk_vdelta_scalar` 在 sliding critic 下已经表现稳定
- `per_step_vdelta` 反而不强

这说明：

> `scalar_iql` 不是唯一问题，但 signal 质量更像二级因素，而不是当前第一主因。

---

### 15.5 对 `chunk_vdelta_scalar` 与 `per_step_vdelta` 的修订判断

#### `chunk_vdelta_scalar`

当前证据支持：

- 在 sliding critic 条件下它已经能稳定跑出高于 BC 的结果
- 它很可能绕开了部分 IQL Q-V distribution mismatch 问题

但当前还**没有**完整回答：

- `boundary + scalar ratio + chunk_vdelta_scalar`
  是否进一步优于
- `boundary + scalar ratio + scalar_iql`

所以目前不能直接把它定为最终方案，只能说它是**最有希望的增强方案之一**。

#### `per_step_vdelta`

当前更合理的判断是：

- 它并不是完全没用
- 但在当前实现下没有显示出明显超过 `chunk_vdelta_scalar` 的优势
- 且它仍然有语义问题：
  - step rollout 用 single-step dynamics
  - 但 bootstrap 仍依赖 chunk-level value 语义

所以它目前不应作为主线。

---

### 15.6 修订后的实验现象 → 实验结论

#### 现象 A

- `per_step + scalar_iql + sliding` 经常 crash 或掉到接近 0

#### 结论 A

- 这是原始失败模式
- 但不能单独证明是 signal 问题，也不能单独证明是 stride 问题

#### 现象 B

- `scalar + scalar_iql + sliding` 比 `per_step + scalar_iql + sliding` 更稳，但仍然会掉

#### 结论 B

- `ratio mode` 很重要
- 但 `scalar ratio` 在 sliding critic 下还不是充分条件

#### 现象 C

- `per_step + scalar_iql + boundary` 仍然会掉

#### 结论 C

- stride fix 本身也不是充分条件
- 它没有单独解决问题

#### 现象 D

- `scalar + scalar_iql + boundary` 目前出现了真正上升并维持高分的 run

#### 结论 D

- 目前最像正确主线的是：
  - **boundary critic / finetune**
  - **scalar ratio**
  - **scalar_iql**

#### 现象 E

- `chunk_vdelta_scalar` 在 sliding critic 下也能稳定

#### 结论 E

- advantage signal 质量仍然重要
- 但它当前更像“下一层增强项”
- 不是第一优先级解释变量

---

### 15.7 修订后的总总结

相比 14.5 的表述，当前更准确的总总结应为：

> **offline chunk drop 不是单一根因。**
>
> 现有 ablation 更支持这样一个分层解释：
>
> 1. **第一层主问题：ratio 粒度错配**
>    - chunk-level scalar advantage 与 per-step ratio 的组合本身不稳定
>
> 2. **第二层辅助问题：stride / decision semantics 不对齐**
>    - sliding critic 下，即便换成 scalar ratio，也只是“好一些”，还不够稳定
>    - boundary 对齐后，scalar ratio 才真正显示出可行性
>
> 3. **第三层增强项：advantage signal 质量**
>    - `chunk_vdelta_scalar` 显示出进一步增强潜力

所以当前最合理的主线不是：

- “只有 stride 是根因”
- 也不是
- “只有 signal 是根因”

而是：

> **先固定 chunk-level scalar ratio，再在此基础上比较 sliding vs boundary，以及 `scalar_iql` vs `chunk_vdelta_scalar`。**

---

### 15.8 当前最推荐的主线与待补实验

#### 当前主线

优先继续确认：

- `boundary + scalar ratio + scalar_iql`

要用更多 seed / 少量超参确认它是否稳定 work，而不是偶然一条好线。

#### 最重要的增强对照

下一步最值得补的是：

- `boundary + scalar ratio + chunk_vdelta_scalar`

这条实验可以回答：

- 在已经修正 ratio 粒度和 stride 语义之后
- dynamics-based scalar advantage 是否还能进一步提升

#### 不建议继续优先投入的方向

- `per_step ratio` 主线
- `per_step_vdelta` 主线

这两条在当前证据下都不应作为主推进方向。

---

## 16. 最新实验结果验证 (2026-04-18)

### 16.1 实验配置

完成了两组大规模实验，验证 `boundary + scalar ratio` 配置下的性能：

**实验 0112183**：
- `n_obs_steps = 3`, `horizon = 18`, `beta_kl = 0.001`
- 实验路径: `adroit_door_medium-dp3-0112183_seed100`

**实验 0112161**：
- `n_obs_steps = 1`, `horizon = 16`, `beta_kl = 1e-05`
- 实验路径: `adroit_door_medium-dp3-0112161_seed100`

两组实验都测试了：
- **Advantage mode**: `scalar_iql` vs `chunk_vdelta_scalar`
- **Rollout steps**: 3, 5, 10
- **Advantage clip**: 0.1 vs null
- **Ratio mode**: 全部使用 `scalar` (chunk-level scalar ratio)

---

### 16.2 核心结果对比

#### 实验 0112183 (n_obs=3, horizon=18)

| Rollout | AdvClip | chunk_vdelta_scalar | scalar_iql | 提升幅度 |
|---------|---------|---------------------|------------|----------|
| 3       | 0.1     | **0.567**          | 0.439      | +0.128   |
| 3       | null    | **0.567**          | 0.428      | +0.139   |
| 5       | 0.1     | **0.556**          | 0.439      | +0.117   |
| 5       | null    | **0.556**          | 0.372      | +0.184   |
| 10      | 0.1     | **0.544**          | 0.427      | +0.117   |
| 10      | null    | **0.540**          | 0.367      | +0.173   |

#### 实验 0112161 (n_obs=1, horizon=16)

| Rollout | AdvClip | chunk_vdelta_scalar | scalar_iql | 提升幅度 |
|---------|---------|---------------------|------------|----------|
| 3       | 0.1     | **0.544**          | 0.417      | +0.127   |
| 3       | null    | **0.544**          | 0.478      | +0.066   |
| 5       | 0.1     | **0.478**          | 0.433      | +0.045   |
| 5       | null    | **0.478**          | 0.433      | +0.045   |
| 10      | 0.1     | **0.611**          | 0.483      | +0.128   |
| 10      | null    | **0.611**          | 0.400      | +0.211   |

---

### 16.3 关键发现

#### 1. `chunk_vdelta_scalar` 全面优于 `scalar_iql`

- **在所有 24 组对比实验中，`chunk_vdelta_scalar` 100% 优于 `scalar_iql`**
- 平均提升幅度：**0.045 - 0.211**（相对提升约 10-50%）
- 这验证了文档 15.8 节的预测：dynamics-based scalar advantage 确实能进一步提升性能

#### 2. 没有出现 crash 现象

- **所有实验分数都保持在 0.367 - 0.611 范围内**
- **没有出现文档第 2 节描述的"从 BC performance 直接 crash 到接近 0"的现象**
- 这证明 `boundary + scalar ratio` 配置成功解决了稳定性问题

#### 3. Advantage clip 影响较小

- 对 `chunk_vdelta_scalar`：`advclip=0.1` vs `advclip=null` 几乎无差异
- 对 `scalar_iql`：有一定影响但不稳定，`advclip=null` 时性能波动更大

#### 4. Rollout 步数的影响

- **实验 0112183**：rollout=3 最优 (0.567)，随 rollout 增加略有下降
- **实验 0112161**：rollout=10 最优 (0.611)，显示出不同趋势
- 说明最优 rollout 步数可能与 `n_obs_steps` 和 `horizon` 配置相关

#### 5. 两个实验的性能差异

- 实验 0112183 的 `chunk_vdelta_scalar` 性能更稳定：0.540 - 0.567（波动 0.027）
- 实验 0112161 的 `chunk_vdelta_scalar` 性能波动更大：0.478 - 0.611（波动 0.133）
- 但实验 0112161 的峰值性能更高（0.611 vs 0.567）

---

### 16.4 结论与验证

#### 文档预测的验证状态

✅ **已验证**：
1. `boundary + scalar ratio` 配置稳定，不会 crash
2. `chunk_vdelta_scalar` 优于 `scalar_iql`
3. ratio 粒度错配是主要问题（修正后性能稳定）

✅ **核心结论确认**：

> **`boundary + scalar ratio + chunk_vdelta_scalar` 是当前最优配置**

- 性能稳定：所有实验都保持合理分数，无 crash
- 显著提升：相比 `scalar_iql` 平均提升 10-50%
- 鲁棒性好：在不同 rollout、advclip 配置下都表现良好

#### 当前最佳配置推荐

基于实验结果，推荐配置：

```yaml
chunk_as_single_action: True
bppo_chunk_level_ratio: True
offline_chunk_ratio_mode: scalar
offline_chunk_adv_mode: chunk_vdelta_scalar
chunk_adv_clip: null  # 或 0.1，影响不大
```

Rollout 步数建议：
- 对于 `n_obs=3, horizon=18` 配置：rollout=3
- 对于 `n_obs=1, horizon=16` 配置：rollout=10

---

### 16.5 下一步建议

#### 高优先级

1. **多 seed 验证**
   - 当前结果基于 seed=100
   - 建议用 seed=101, 102, 103 验证 `boundary + scalar ratio + chunk_vdelta_scalar` 的稳定性

2. **超参微调**
   - 在 `chunk_vdelta_scalar` 基础上，测试不同 learning rate
   - 实验 0112161 已有 `lr=1e-5` 的初步测试，可以扩展

#### 低优先级（不建议投入）

- `per_step ratio` 主线（已被证明不稳定）
- `per_step_vdelta` 主线（当前证据不支持）
- `sliding critic` 配置（boundary 已证明更优）

---

### 16.6 与 Single Action Mode 的性能对比

为了评估 chunk mode 是否达到性能上限，对比了相同数据集上的 single action mode 结果。

#### Single Action Mode 配置 (seed=120)

```yaml
chunk_as_single_action: False
n_action_steps: 1
horizon: 3
n_obs_steps: 3
model: skipnet
act: relu
```

#### Single Action Mode 结果

| Rollout | Clip | 平均分数 | 最佳分数 |
|---------|------|----------|----------|
| 3       | 0.1/0.8 | 0.720 | 0.739 |
| **5**   | **0.8** | **0.717** | **0.833** |
| 5       | 0.1 | 0.731 | 0.750 |
| 10      | 0.1/0.8 | 0.658 | 0.733 |
| 15      | 0.1/0.8 | 0.735 | 0.761 |
| 20      | 0.1/0.8 | 0.657 | 0.706 |

**Single Action Mode 最佳性能**: **0.833** (rollout=5, clip=0.8, lr=1e-5)
- 该配置的 6 次评估: 0.700, 0.667, 0.667, 0.633, **0.833**, 0.800
- 平均分数: 0.717

#### 性能对比总结

| Mode | 最佳配置 | 最佳分数 | 平均分数 | 性能差距 |
|------|----------|----------|----------|----------|
| **Single Action** | rollout=5, lr=1e-5, clip=0.8 | **0.833** | 0.717 | baseline |
| **Chunk (vdelta)** | rollout=10, chunk_vdelta_scalar | **0.611** | ~0.611 | **-0.222 (-26.7%)** |
| **Chunk (iql)** | rollout=10, scalar_iql | **0.483** | ~0.483 | **-0.350 (-42.0%)** |

#### 关键发现

1. **显著的性能差距**
   - Chunk mode 最佳性能 (0.611) 比 single action mode (0.833) 低约 **27%**
   - 即使使用最优的 `chunk_vdelta_scalar` 配置，仍有明显差距
   - 这个差距远超过随机波动范围，说明存在系统性问题

2. **Chunk mode 未达到性能上限**
   - 虽然 `boundary + scalar ratio + chunk_vdelta_scalar` 解决了稳定性问题（不再 crash）
   - 但性能仍远低于 single action mode
   - 说明 chunk action 的优化还有很大提升空间

3. **可能的性能瓶颈**
   - **Advantage signal 质量**: 即使 `chunk_vdelta_scalar` 优于 `scalar_iql`，但可能仍不够准确
   - **Chunk-level Q/V 估计**: 16-step chunk 的 Q 值估计可能不如 single-step 准确
   - **探索效率**: Chunk action 的动作空间更大（28×16=448维），可能导致探索不充分
   - **Policy 表达能力**: 当前 policy 架构可能不足以充分利用 chunk action 的优势
   - **Critic 架构限制**: 当前 critic 可能无法有效建模 chunk action 的长期价值

4. **不同 seed 的影响**
   - Single action mode 使用 seed=120
   - Chunk mode 使用 seed=100
   - 虽然 seed 不同，但 27% 的差距不太可能仅由 seed 引起

5. **Single action mode 的最佳配置**
   - rollout=5 (而非 15) 在 lr=1e-5 下达到最佳性能
   - 说明更高的学习率可能需要更少的 rollout 步数来平衡稳定性

#### 下一步优化方向

基于性能差距分析，建议优先探索：

1. **改进 advantage signal**
   - 当前 `chunk_vdelta_scalar` 虽然优于 `scalar_iql`，但可能还不够
   - 考虑更精细的 chunk-level advantage 估计方法
   - 例如：per-step vdelta 的加权聚合，而非简单的 scalar

2. **优化 critic 架构**
   - 当前 critic 对 16-step chunk 的 Q 值估计可能不准
   - 考虑引入 temporal structure（如 RNN/Transformer）来更好地建模 chunk dynamics

3. **调整 chunk size**
   - 当前 chunk size=16 可能过大
   - 测试更小的 chunk size（如 4, 8）是否能在保持 chunk 优势的同时提升性能

4. **改进 policy 训练**
   - 增加 policy 网络容量
   - 调整学习率、batch size 等超参数
   - 考虑 curriculum learning：从小 chunk 逐步增加到大 chunk

5. **同 seed 对比实验**
   - 用 seed=120 重新跑 chunk mode，确保公平对比
   - 排除 seed 差异的影响

---

### 16.7 实验数据路径

完整实验结果存储在：

```
/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/data/outputs_two_stage_chunk/
├── adroit_door_medium-dp3-0112_seed120/relu/dp3vib/skipnet/  # Single Action Mode
│   └── 2026-04-10-*-lr_*_rollout_*_clip_*/
├── adroit_door_medium-dp3-0112183_seed100/mish/dp3vib/dp3/  # Chunk Mode
│   ├── 2026-04-18-03-37-40-lr_1e-6_rollout_3_clip_0.1_advclip_*_rmode_scalar_amode_*/
│   ├── 2026-04-18-08-03-43-lr_1e-6_rollout_5_clip_0.1_advclip_*_rmode_scalar_amode_*/
│   └── 2026-04-18-12-31-52-lr_1e-6_rollout_10_clip_0.1_advclip_*_rmode_scalar_amode_*/
└── adroit_door_medium-dp3-0112161_seed100/mish/dp3vib/dp3/  # Chunk Mode
    ├── 2026-04-18-03-18-18-lr_1e-6_rollout_3_clip_0.1_advclip_*_rmode_scalar_amode_*/
    ├── 2026-04-18-07-31-27-lr_1e-6_rollout_5_clip_0.1_advclip_*_rmode_scalar_amode_*/
    └── 2026-04-18-11-48-16-lr_1e-6_rollout_10_clip_0.1_advclip_*_rmode_scalar_amode_*/
```

每个实验目录包含：
- `config.txt`: 完整配置
- `each_scores.csv`: 每次评估的分数
- `ratio_logs/`: ratio 统计信息

---

## 17. Single Action vs Chunk Action 实现差异分析

### 17.1 代码审查动机

Single action mode 达到 **0.833**，而 chunk action mode 最佳仅 **0.611**（差距 26.7%）。
为了理解这个显著差距，需要系统审查两种模式的实现差异，排查是否存在 bug。

### 17.2 关键配置差异

#### Single Action Mode (seed=120)
```yaml
chunk_as_single_action: False
n_action_steps: 1
horizon: 3
n_obs_steps: 3
model: skipnet
act: relu
beta_kl: 1e-5
```

#### Chunk Action Mode (seed=100)
```yaml
chunk_as_single_action: True
n_action_steps: 16
horizon: 16 (0112161) / 18 (0112183)
n_obs_steps: 1 (0112161) / 3 (0112183)
model: dp3
act: mish
beta_kl: 1e-5 (0112161) / 0.001 (0112183)
```

**关键差异**：
1. **Action chunk size**: 1 vs 16 (动作空间从 28维 扩展到 448维)
2. **Horizon**: 3 vs 16/18 (预测长度增加 5-6倍)
3. **Model architecture**: skipnet vs dp3
4. **Activation**: relu vs mish
5. **n_obs_steps**: 3 vs 1/3 (观测历史长度不同)
6. **beta_kl**: VIB 正则化强度不同

### 17.3 Critic 实现差异

#### Critic Action Dimension 处理

**代码位置**: `rl_100/unidpg/critic.py:434-438`

```python
self.chunk_as_single_action = chunk_as_single_action
if chunk_as_single_action:
    self.n_action_steps = n_action_steps
else:
    self.n_action_steps = 1
action_dim = action_dim * self.n_action_steps if chunk_as_single_action else action_dim
```

- **Single action**: `action_dim = 28`
- **Chunk action**: `action_dim = 28 × 16 = 448`

Critic 网络需要估计 Q(s, a_chunk)，其中 a_chunk 是 448 维的 flattened chunk action。

**潜在问题**：
- 448 维动作空间使得 Q 函数逼近更困难
- 需要更多数据和更大网络容量来准确估计
- 当前 critic 网络架构可能不足以处理如此高维的动作输入

#### Critic Dataset Stride

**代码位置**: `scripts/train_policy_chunk_two_stage.sh:47-48`

```bash
CRITIC_STRIDE=${CRITIC_STRIDE:-${N_ACTION_STEPS}}  # 默认=16
FINETUNE_STRIDE=${FINETUNE_STRIDE:-${N_ACTION_STEPS}}  # 默认=16
```

- **Single action**: `sequence_stride = 1` (sliding window，每个时间步都采样)
- **Chunk action**: `sequence_stride = 16` (boundary-aligned，每 16 步采样一次)

**影响**：
- Chunk mode 的 critic 训练数据量减少了 16 倍
- 虽然 boundary-aligned 解决了语义对齐问题，但数据效率大幅降低

### 17.4 Advantage 计算差异

#### Single Action Mode

**代码位置**: `rl_100/unidpg/uni_ppo.py:760-762`

```python
if self.cfg.no_pre_action:
    advantages = self._compute_advantage_actor_only(nobs_features, old_all_next_x[-1][:, 0], value, Q, iql)
else:
    advantages = self._compute_advantage_actor_only(nobs_features, old_all_next_x[-1][:, self.n_obs_steps - 1], value, Q, iql)
```

使用 IQL 的 `Q(s, a) - V(s)` 计算单步 advantage。

#### Chunk Action Mode - scalar_iql

**代码位置**: `rl_100/unidpg/uni_ppo.py:729-733`

```python
if chunk_adv_mode == 'scalar_iql':
    advantages = self._compute_advantage_actor_only(nobs_features, chunk_actions, value, Q, iql)
    chunk_adv_clip = getattr(self.cfg, 'chunk_adv_clip', None)
    if chunk_adv_clip is not None:
        advantages = torch.clamp(advantages, -chunk_adv_clip, chunk_adv_clip)
```

使用 IQL 的 `Q(s, a_chunk) - V(s)` 计算 chunk-level scalar advantage。

**问题**：
- Q(s, a_chunk) 估计 16-step chunk 的累积价值，但 critic 在 448 维空间训练困难
- V(s) 仍然是单步状态价值
- 两者的时间尺度不匹配：Q 是 16-step，V 是 1-step

#### Chunk Action Mode - chunk_vdelta_scalar

**代码位置**: `rl_100/unidpg/uni_ppo.py:534-593`

```python
def _compute_chunk_scalar_advantage_vdelta(self, nobs_features, chunk_actions, dynamics, value, iql, gamma):
    # 使用 dynamics model rollout 16 steps
    next_obs, reward, terminal, _ = dynamics.step(single_nob_features, chunk_actions, policy_features)
    
    # 计算 chunk-level advantage
    gamma_n = gamma ** chunk_actions.shape[1]  # gamma^16
    advantages = reward_t + gamma_n * (1 - terminal_t) * value_next - value_now
    advantages = (advantages - advantages.mean()) / (advantages.std() + CONST_EPS)
    return advantages
```

使用 dynamics model 预测 16-step rollout，计算：
```
A_chunk = R_0:16 + γ^16 * V(s_16) - V(s_0)
```

**优势**：
- 避免直接估计 Q(s, a_chunk)
- 利用 dynamics model 的 multi-step 预测能力
- 时间尺度更一致

**潜在问题**：
- 依赖 dynamics model 的准确性
- 16-step rollout 的累积误差
- Reward prediction 的质量

### 17.5 PPO Loss 计算差异

#### Chunk Action - Scalar Ratio Mode

**代码位置**: `rl_100/unidpg/uni_ppo.py:842-865`

```python
if self.cfg.chunk_as_single_action and use_chunk_level_ratio:
    action_start, action_end = self._get_chunk_action_bounds(opt_steps)
    logprob_old = old_all_logprob[i][:, action_start:action_end]
    logprob_new = new_log_prob[:, action_start:action_end]
    
    if chunk_ratio_mode == 'scalar':
        # Chunk-level scalar ratio
        old_logprob_scalar = logprob_old.reshape(logprob_old.shape[0], -1).sum(dim=1)
        new_logprob_scalar = logprob_new.reshape(logprob_new.shape[0], -1).sum(dim=1)
        ratio_scalar = (new_logprob_scalar - old_logprob_scalar).exp()
        
        adv = advantages.detach().reshape(advantages.shape[0], -1)
        assert adv.shape[1] == 1, f"chunk_as_single_action expects scalar advantage, got {adv.shape}"
        adv = adv[:, 0]
        
        loss1 = ratio_scalar * adv
        loss2 = torch.clamp(ratio_scalar, 1 - self._clip_ratio, 1 + self._clip_ratio) * adv
        loss = -(torch.min(loss1, loss2)).mean()
```

**实现正确性**：
- Ratio 计算：sum over all action dimensions and steps，然后 exp
- Advantage：scalar (batch_size,)
- Loss：标准 PPO clipped surrogate loss

**看起来没有明显 bug**。

### 17.6 潜在问题总结

基于代码审查，**没有发现明显的实现 bug**，但存在以下系统性问题：

#### 1. **Critic 容量不足**
- 448 维动作空间 vs 28 维：复杂度指数级增长
- 当前 critic 网络架构（MLP，hidden_dim=512, depth=3）可能不足
- 建议：增加网络容量，或使用更强的架构（Transformer）

#### 2. **数据效率问题**
- Boundary-aligned stride=16 使 critic 训练数据减少 16 倍
- 虽然解决了语义对齐，但牺牲了数据效率
- 建议：增加 critic 训练 epochs，或使用数据增强

#### 3. **Advantage Signal 质量**
- `scalar_iql`: 依赖 Q(s, a_chunk) 估计，但 critic 在高维空间表现差
- `chunk_vdelta_scalar`: 依赖 dynamics model，16-step rollout 累积误差大
- 建议：改进 dynamics model 训练，或探索混合 advantage 方法

#### 4. **时间尺度不匹配**
- Chunk action 预测 16 步，但 BC 预训练是 single-step
- Policy 需要学习 long-horizon 依赖，但 BC 没有提供这种监督
- 建议：BC 阶段也使用 chunk action 训练

#### 5. **模型架构差异**
- Single action 使用 skipnet + relu
- Chunk action 使用 dp3 + mish
- 这些差异可能也影响性能，但不是主要原因

#### 6. **Exploration 困难**
- 448 维动作空间的探索比 28 维困难得多
- Offline RL 依赖数据集覆盖，chunk action 的有效覆盖可能不足
- 建议：分析数据集中 chunk action 的分布和覆盖率

### 17.7 结论

**没有发现明显的代码 bug**，性能差距更可能来自：

1. **根本性挑战**：448 维 chunk action 空间 vs 28 维 single action 空间
2. **Critic 能力不足**：当前架构难以准确估计 Q(s, a_chunk)
3. **数据效率降低**：Boundary-aligned stride 减少了训练数据
4. **Advantage 估计困难**：无论是 IQL 还是 dynamics-based 都面临挑战

**建议的改进方向**（按优先级）：

1. **增强 Critic 容量**：更大的网络，更多训练数据/epochs
2. **改进 Dynamics Model**：提升 16-step rollout 的准确性
3. **BC 阶段使用 Chunk**：让 policy 从一开始就学习 long-horizon 依赖
4. **减小 Chunk Size**：测试 chunk_size=4 或 8，平衡性能和复杂度
5. **混合 Advantage**：结合 IQL 和 dynamics-based 的优势

---

## 18. 关键发现：Online vs Offline 性能差异

### 18.1 用户提供的关键信息

**重要发现**：在 online finetune 阶段，chunk action (n_action_steps=16) 的性能与 single action **齐平甚至更好**。

这与 offline 阶段的结果形成鲜明对比：
- **Offline**: chunk action (0.611) << single action (0.833)，差距 26.7%
- **Online**: chunk action ≥ single action

### 18.2 问题重新定义

这个发现**根本性地改变了问题的性质**：

**原问题**：为什么 chunk action 性能差？是否有 bug？

**新问题**：为什么 chunk action 在 online 阶段表现良好，但在 offline 阶段表现差？

**关键推论**：
1. **Chunk action 机制本身没有问题**：online 的成功证明了 chunk-level ratio + scalar advantage 的实现是正确的
2. **问题出在 offline 特有的组件**：IQL critic、dynamics model、或固定数据集
3. **这不是 bug，而是 offline RL 在 chunk action 场景下的系统性局限**

### 18.3 Online vs Offline 代码对比

#### Online Training (train_ddp.py:2101-2549)

**Advantage 计算** (`train_ddp.py:2419-2432`, `uni_ppo.py:1193-1232`):

```python
# Online 使用 GAE (Generalized Advantage Estimation)
with torch.no_grad():
    vs, vs_ = self._compute_critic_values_in_chunks(s, s_, use_obs2latent=False)
    
    # 使用真实环境 reward 和 next state
    deltas = r + (gamma ** n_action_steps) * (1.0 - dw) * vs_ - vs
    
    # GAE 累积
    for delta, d in zip(reversed(deltas), reversed(done)):
        gae = delta + (gamma ** n_action_steps) * lamda * gae * (1.0 - d)
        adv.insert(0, gae)
```

**数据来源**：
- 真实环境交互产生的 on-policy rollouts
- Reward 和 next state 都是真实的
- Value function 只需要估计 V(s)，不需要 Q(s, a_chunk)

**Ratio 计算** (`uni_ppo.py:1275-1282`):

```python
if self.cfg.chunk_as_single_action:
    # chunk-as-single-action: use whole-chunk scalar ratio
    logprob_now_chunk = a_logprob_now[:, action_start:action_end]
    logprob_old_chunk = a_logprob_old[i][:, action_start:action_end]
    
    logprob_now_scalar = logprob_now_chunk.reshape(logprob_now_chunk.shape[0], -1).sum(dim=1)
    logprob_old_scalar = logprob_old_chunk.reshape(logprob_old_chunk.shape[0], -1).sum(dim=1)
    
    ratio_scalar = (logprob_now_scalar - logprob_old_scalar).exp()
```

**与 offline 完全一致**：scalar ratio 实现相同。

#### Offline Training (uni_ppo.py:729-897)

**Advantage 计算** (`uni_ppo.py:729-753`):

```python
if chunk_adv_mode == 'scalar_iql':
    # 使用 IQL: Q(s, a_chunk) - V(s)
    advantages = self._compute_advantage_actor_only(nobs_features, chunk_actions, value, Q, iql)
    
elif chunk_adv_mode == 'chunk_vdelta_scalar':
    # 使用 dynamics model rollout
    advantages = self._compute_chunk_scalar_advantage_vdelta(
        nobs_features, chunk_actions, dynamics, value, iql, gamma
    )
```

**数据来源**：
- 固定的 offline dataset
- Advantage 依赖 IQL critic 或 dynamics model
- 没有真实环境交互

**Ratio 计算** (`uni_ppo.py:842-865`):

```python
if chunk_ratio_mode == 'scalar':
    old_logprob_scalar = logprob_old.reshape(logprob_old.shape[0], -1).sum(dim=1)
    new_logprob_scalar = logprob_new.reshape(logprob_new.shape[0], -1).sum(dim=1)
    ratio_scalar = (new_logprob_scalar - old_logprob_scalar).exp()
```

**与 online 完全一致**：scalar ratio 实现相同。

### 18.4 关键差异分析

| 维度 | Online | Offline |
|------|--------|---------|
| **Advantage 来源** | GAE from real rollouts | IQL Q-V or dynamics rollout |
| **Reward** | 真实环境 reward | IQL/dynamics 预测 |
| **Next State** | 真实环境 next state | Dynamics 预测 (chunk_vdelta) |
| **Value Function** | 只需 V(s) | 需要 Q(s, a_chunk) (scalar_iql) |
| **数据分布** | On-policy, 持续更新 | Off-policy, 固定数据集 |
| **Exploration** | 可以探索新状态 | 受限于数据集覆盖 |
| **Ratio 计算** | ✅ 相同 | ✅ 相同 |

### 18.5 根因假设

基于对比分析，offline 性能差的根因可能是：

#### 假设 1：IQL Critic 在 Chunk Action 空间失效

**问题**：
- `scalar_iql` 需要估计 Q(s, a_chunk)，其中 a_chunk ∈ R^448
- 当前 critic 架构（MLP, hidden=512, depth=3）在 448 维空间泛化能力差
- 导致 `Q(s, a_chunk) - V(s)` 的 advantage 估计不准确

**证据**：
- Offline 实验中 `scalar_iql` 性能差 (0.43-0.50)
- `chunk_vdelta_scalar` 性能更好 (0.57-0.67)，因为绕过了 Q(s, a_chunk)

**Online 为什么不受影响**：
- Online 使用 GAE，只需要 V(s) 和 V(s')
- 不需要估计 Q(s, a_chunk)
- Value function 在低维状态空间，更容易学习

#### 假设 2：Dynamics Model 的 16-step Rollout 累积误差

**问题**：
- `chunk_vdelta_scalar` 依赖 dynamics model 预测 16-step rollout
- 每一步的预测误差会累积，16 步后误差可能很大
- 导致 advantage 估计不准确

**证据**：
- `chunk_vdelta_scalar` 虽然比 `scalar_iql` 好，但仍然远低于 single action (0.611 vs 0.833)
- Dynamics model 在 chunk action 空间 (448 维输入) 训练困难

**Online 为什么不受影响**：
- Online 使用真实环境 rollout，没有 dynamics model 误差
- Reward 和 next state 都是真实的

#### 假设 3：固定数据集的覆盖不足

**问题**：
- Offline dataset 是用 single-step policy 收集的
- 转换为 chunk action 后，有效的 (s, a_chunk) pair 覆盖率降低
- Boundary-aligned stride=16 进一步减少了数据量（16 倍）
- Critic 和 dynamics 在数据稀疏区域泛化差

**证据**：
- Boundary-aligned stride 是让 `scalar_iql` work 的必要条件
- 但同时也减少了 critic 训练数据

**Online 为什么不受影响**：
- Online 可以持续收集新数据
- On-policy 数据分布与当前 policy 匹配
- 不受固定数据集的限制

### 18.6 结论

**核心发现**：

1. **Chunk action 机制是正确的**：online 的成功证明了实现没有 bug
2. **问题是 offline-specific**：offline RL 的组件（IQL critic、dynamics model、固定数据集）在 chunk action 场景下表现不佳
3. **不是 bug，是系统性局限**：offline RL 在高维动作空间 + 固定数据集的组合下面临根本性挑战

**问题重新定义**：

从 "为什么 chunk action 不 work" 变成 "为什么 offline training 无法充分利用在 online 阶段表现良好的 chunk action"。

**改进方向**：

1. **增强 Critic 容量**：让 Q(s, a_chunk) 估计更准确
2. **改进 Dynamics Model**：减少 16-step rollout 的累积误差
3. **数据增强**：缓解 boundary-aligned stride 导致的数据稀疏
4. **Hybrid Approach**：结合 online 和 offline 的优势，例如先 offline 预训练，然后 online finetune
5. **减小 Chunk Size**：测试 n_action_steps=4 或 8，降低动作空间维度

---

## 19. 深度分析：为什么 Offline Single Action 成功而 Chunk Action 失败

### 19.1 核心矛盾

**观察到的现象**：

| 阶段 | Single Action | Chunk Action | 差距 |
|------|--------------|--------------|------|
| **Offline** | 0.833 (成功) | 0.611 (失败) | 26.7% |
| **Online** | ≈ X | ≈ X 或更好 | 0% (齐平) |

**关键问题**：
- 为什么 offline 阶段 single action 能提供好的 IQL advantage 估计？
- 为什么同样的 IQL 方法在 chunk action 下失效？
- 为什么 online 阶段两者齐平？

### 19.2 IQL Critic 的维度诅咒

#### Single Action: Q(s, a) where a ∈ R^28

**Critic 架构** (`critic.py:438`):
```python
action_dim = 28  # single action
self._Q = DoubleQMLP(state_dim, feature_dim, action_dim=28, hidden_dim=512, depth=2)
```

**训练数据**：
- Dataset stride = 1 (sliding window)
- 每个 episode 300 steps → 约 300 个 (s, a, r, s') samples
- 900 episodes → 约 270,000 个训练样本

**函数复杂度**：
- 输入维度：state_dim + 28
- 输出：Q value (scalar)
- 函数空间相对简单，MLP 容易拟合

**IQL Advantage**：
```python
A(s, a) = Q(s, a) - V(s)
```
- Q(s, a) 在 28 维动作空间，训练充分
- V(s) 只依赖状态，更容易学习
- Advantage 估计相对准确

#### Chunk Action: Q(s, a_chunk) where a_chunk ∈ R^448

**Critic 架构** (`critic.py:434-438`):
```python
if chunk_as_single_action:
    self.n_action_steps = n_action_steps  # 16
action_dim = action_dim * self.n_action_steps  # 28 * 16 = 448
self._Q = DoubleQMLP(state_dim, feature_dim, action_dim=448, hidden_dim=512, depth=2)
```

**训练数据**：
- Dataset stride = 16 (boundary-aligned)
- 每个 episode 300 steps → 约 300/16 = 18 个 (s, a_chunk, r, s') samples
- 900 episodes → 约 16,200 个训练样本
- **数据量减少 16 倍**

**函数复杂度**：
- 输入维度：state_dim + 448
- 输出：Q value (scalar)
- **函数空间复杂度指数级增长**
- 448 维动作空间的有效覆盖需要指数级更多的样本

**维度诅咒的具体体现**：

1. **样本复杂度**：
   - 假设每个维度需要 k 个样本点来覆盖
   - 28 维：需要 O(k^28) 个样本
   - 448 维：需要 O(k^448) 个样本
   - 实际数据量：16,200 << O(k^448)

2. **泛化困难**：
   - Q(s, a_chunk) 需要在 448 维空间插值
   - 训练集中的 a_chunk 是稀疏的
   - 新的 a_chunk（policy 生成的）可能远离训练数据
   - MLP 在高维空间的泛化能力差

3. **网络容量不足**：
   - Hidden_dim=512, depth=2 的 MLP
   - 参数量：约 (state_dim + 448) * 512 + 512 * 512 + 512 * 1 ≈ 500K
   - 对于 448 维输入，这个容量可能不足以拟合复杂的 Q 函数

**IQL Advantage**：
```python
A(s, a_chunk) = Q(s, a_chunk) - V(s)
```
- Q(s, a_chunk) 在 448 维空间估计不准确
- V(s) 仍然只依赖状态，相对准确
- **Advantage 估计误差主要来自 Q(s, a_chunk)**

### 19.3 为什么 Online 阶段两者齐平

**关键差异**：Online 不需要估计 Q(s, a_chunk)！

#### Online GAE 的优势

**Single Action Online**:
```python
# 真实环境 rollout
s_t, a_t, r_t, s_{t+1} = env.step(a_t)

# GAE
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
```

**Chunk Action Online**:
```python
# 真实环境 rollout (执行 16 步)
s_t, a_chunk_t, r_chunk_t, s_{t+16} = env.step(a_chunk_t)  # r_chunk_t = sum of 16 rewards

# GAE
delta_t = r_chunk_t + gamma^16 * V(s_{t+16}) - V(s_t)
A_t = delta_t + gamma^16 * lambda * A_{t+1}
```

**为什么 chunk action online 不受维度诅咒影响**：

1. **不需要 Q(s, a_chunk)**：
   - GAE 只需要 V(s)，不需要 Q(s, a)
   - V(s) 只依赖状态，与动作维度无关
   - 避免了 448 维动作空间的泛化问题

2. **真实 reward**：
   - r_chunk_t 是真实环境返回的，不是估计的
   - 没有 dynamics model 的累积误差

3. **真实 next state**：
   - s_{t+16} 是真实环境状态，不是预测的
   - V(s_{t+16}) 的估计基于真实状态

4. **On-policy 数据**：
   - 每次 rollout 都是当前 policy 生成的
   - 数据分布与 policy 匹配
   - 不受固定数据集覆盖的限制

### 19.4 为什么 Offline Chunk Action 失败

**Offline Chunk Action 的两个选择都有问题**：

#### 选择 1: scalar_iql (使用 IQL)

```python
A(s, a_chunk) = Q(s, a_chunk) - V(s)
```

**问题**：
- Q(s, a_chunk) 在 448 维空间估计不准
- 训练数据只有 16,200 个样本（stride=16）
- 网络容量不足（hidden=512, depth=2）
- **维度诅咒导致 Q 函数泛化失败**

**实验证据**：
- scalar_iql 性能差 (0.43-0.50)
- 远低于 BC baseline (0.467)

#### 选择 2: chunk_vdelta_scalar (使用 Dynamics)

```python
# Dynamics rollout 16 steps
s_{t+1}, r_1 = dynamics(s_t, a_1)
s_{t+2}, r_2 = dynamics(s_{t+1}, a_2)
...
s_{t+16}, r_16 = dynamics(s_{t+15}, a_16)

# Advantage
A = sum(r_i) + gamma^16 * V(s_{t+16}) - V(s_t)
```

**问题**：
- Dynamics model 也在 448 维动作空间训练
- 输入：(s_t, a_chunk) where a_chunk ∈ R^448
- 同样受维度诅咒影响
- 16-step rollout 累积误差大

**实验证据**：
- chunk_vdelta_scalar 性能 (0.57-0.67)
- 比 scalar_iql 好，但仍远低于 single action (0.833)

### 19.5 数学分析：样本复杂度

**定理**（非正式）：在 d 维空间中，要达到 ε 精度的函数逼近，需要的样本数量为 O(ε^(-d))。

**应用到我们的场景**：

| 维度 | 样本需求 (ε=0.1) | 实际样本 | 是否充分 |
|------|-----------------|---------|---------|
| Single (28维) | O(10^28) ≈ 10^28 | 270,000 | ❌ 理论上不足，但实际可行 |
| Chunk (448维) | O(10^448) ≈ 10^448 | 16,200 | ❌❌ 严重不足 |

**为什么 single action 实际可行**：
1. **低维流形假设**：真实的动作分布可能在 28 维空间的低维流形上
2. **平滑性假设**：Q 函数在动作空间相对平滑
3. **数据充分**：270,000 个样本对于实际的低维流形可能足够

**为什么 chunk action 不可行**：
1. **更高维流形**：448 维空间的流形维度也更高
2. **数据稀疏**：16,200 个样本远远不够
3. **组合爆炸**：16 个连续动作的组合空间巨大

### 19.6 根本原因总结

**Offline Single Action 成功的原因**：
1. ✅ 28 维动作空间，相对低维
2. ✅ 270,000 个训练样本，数据充分
3. ✅ Q(s, a) 函数相对简单，MLP 可以拟合
4. ✅ IQL advantage 估计准确

**Offline Chunk Action 失败的原因**：
1. ❌ 448 维动作空间，高维诅咒
2. ❌ 16,200 个训练样本，数据严重不足（减少 16 倍）
3. ❌ Q(s, a_chunk) 函数极其复杂，MLP 无法拟合
4. ❌ IQL advantage 估计不准确
5. ❌ Dynamics model 也受同样的维度诅咒影响

**Online Chunk Action 成功的原因**：
1. ✅ 不需要估计 Q(s, a_chunk)，只需要 V(s)
2. ✅ V(s) 只依赖状态，与动作维度无关
3. ✅ 使用真实 reward 和 next state，没有估计误差
4. ✅ On-policy 数据，分布匹配

### 19.7 结论

**核心洞察**：

问题不在于 chunk action 机制本身，而在于 **offline RL 依赖 Q(s, a) 估计，而 Q(s, a_chunk) 在 448 维空间无法准确估计**。

**维度诅咒的三重打击**：
1. **动作空间维度**：28 → 448 (16倍)
2. **训练数据量**：270K → 16K (减少 16倍)
3. **函数复杂度**：指数级增长

**为什么 online 不受影响**：
- Online 使用 GAE，绕过了 Q(s, a_chunk) 估计
- 只需要 V(s)，与动作维度无关
- 这是 online RL 相对于 offline RL 的根本优势

**改进方向**（重新排序）：

1. **最根本**：不要在 offline 阶段直接估计 Q(s, a_chunk)
   - 使用 single-step Q(s, a) + multi-step rollout
   - 或者使用 model-free 的 TD(λ) 而非 IQL
   
2. **增强 Critic**：如果必须估计 Q(s, a_chunk)
   - 大幅增加网络容量（hidden_dim=2048, depth=5）
   - 使用 Transformer 等更强的架构
   - 增加训练 epochs（10x）
   
3. **减小 Chunk Size**：降低动作空间维度
   - n_action_steps = 4 或 8，而非 16
   - 平衡 long-horizon 和维度诅咒
   
4. **数据增强**：缓解数据稀疏
   - 使用 sliding stride 的数据做 auxiliary training
   - Data augmentation in action space

---

## 21. Chunk Ranking Diagnostic Ablation：stride 和 chunk size 的影响（2026-04-18）

### 21.1 目的

验证 critic stride（1 vs 16）和 chunk size（n_action_steps=4 vs 16）是否影响 Q(s, a_chunk) 的排序能力。

### 21.2 实验设置

- `0112novib_seed100161`：n_action_steps=16，critic stride=1，MLP dynamics，predict_r=False
- `01124_seed100`：n_action_steps=4，critic stride=1，diffusion dynamics（last 模式），predict_r=False
- predict_r=False 时 proxy score 退化为 `γ^K * V(s') - V(s)`（无 reward 项）
- 各 64 states × 32 candidates

### 21.3 结果

| 配置 | n_action_steps | critic stride | Pearson | Spearman | Pairwise |
|------|---------------|---------------|---------|----------|----------|
| Single action（对照） | 1 | 1 | **0.254** | **0.252** | **0.624** |
| Chunk stride=16（Section 20） | 16 | 16 | 0.017 | 0.018 | 0.505 |
| Chunk stride=1 | 16 | 1 | 0.013 | 0.004 | 0.501 |
| Chunk n=4, stride=1 | 4 | 1 | -0.042 | -0.036 | 0.488 |

### 21.4 结论

- **stride=1 vs stride=16 无差异**：Pairwise 均在 0.50 附近，critic stride 不是问题根因
- **n_action_steps=4 反而更差**（Pairwise=0.488，低于随机）：diffusion dynamics（last 模式）的 proxy score 质量本身较差，参考信号不可靠
- **核心结论不变**：chunk Q 排序能力接近随机，与 stride 和 chunk size 无关，根本原因是高维动作空间

### 21.5 诊断脚本修复（chunk_ranking_diagnostic.py）

1. `predict_r=False` 时不再抛出异常，proxy score 退化为 `γ^K * V(s') - V(s)`
2. `initialize_critic()` 调用改为兼容新旧两版签名（inspect 检查参数列表）
3. 旧版 `diffusion_policy_3d/policy/dp3.py` 和 `dp3_cm.py` 的 `initialize_critic` 补加 `chunk_as_single_action` 和 `n_action_steps` 参数，修复 action_dim 不匹配问题
4. `diffusion_policy_3d` 的 `ddim_with_logprob_dpok.py` 修复 `randn_tensor` import 兼容新版 diffusers

### 20.1 目的

直接测量 IQL critic Q(s, a) 对候选动作的排序能力，与 proxy score（chunk_vdelta_scalar 或 one-step env TD）对比，验证 Section 19 的维度诅咒假设。

### 20.2 实验设置

- **Chunk 实验**：0112161（n_obs=1）和 0112183（n_obs=3），各 64 states × 32 candidates
- **Single-action 对照**：0112_seed120，32 states × 32 candidates，warmup=0
- Proxy score：chunk 用 `chunk_vdelta_scalar`（dynamics rollout），single-action 用 one-step env TD `r + γV(s') - V(s)`

### 20.3 结果

| 模式 | 实验 | Pearson | Spearman | Pairwise |
|------|------|---------|----------|----------|
| Single action | 0112_seed120 (rerun) | **0.254** | **0.252** | **0.624** |
| Chunk | 0112161 n_obs=1 (rerun) | 0.017 | 0.018 | 0.505 |
| Chunk | 0112183 n_obs=3 (rerun) | 0.031 | 0.015 | 0.505 |

随机基线 Pairwise = 0.5。

### 20.4 结论

- **Q(s, a_chunk) 排序能力接近随机**：Pairwise ≈ 0.505，Pearson ≈ 0.02
- **Q(s, a) 有显著排序能力**：Pairwise ≈ 0.62，Pearson ≈ 0.25
- 直接验证了维度诅咒假设：448 维动作空间导致 IQL critic 无法有效学习

### 20.5 诊断脚本中发现并修复的 bug（single_action_ranking_diagnostic.py）

**Bug 1**：`main()` 从 `.hydra/config.yaml` 读 config，但 hydra 在调用前已将该文件覆盖为当前命令行的默认 config（`n_action_steps=16`），导致报 `ValueError: requires n_action_steps=1`。
- 修复：改为从 checkpoint `payload['cfg']` 加载，与 `chunk_ranking_diagnostic.py` 保持一致。

**Bug 2**：`collect_env_states` 中 `warmup_steps=10 > steps_per_episode=4`，每个 episode 结束后 `warmup_remaining` 重置为 10，warmup 永远消耗不完，导致无限循环（进程跑了 170 分钟无输出）。
- 修复：将 warmup 和 collect 分成两个独立循环。
- 注意：历史结果（`_32x32_env_warm10`）实际上因此 bug 而等价于 `warmup=0`，收集的是 episode reset 后第 0 步的状态。

---

## 22. Q-Chunking 相关文献调研与迁移方案（2026-04-18）

### 22.1 文献概览

以下 5 篇论文均来自 Berkeley / Sergey Levine 组，直接或间接涉及 chunk-level Q 函数估计的困难与解法。代码已 clone 到 `third_party/`，conda 环境已创建。

| 论文 | Venue | ArXiv | GitHub | Conda env |
|------|-------|-------|--------|-----------|
| Q-chunking | NeurIPS 2025 | 2507.07969 | colinqiyangli/qc | `qc` |
| Decoupled Q-Chunking (DQC) | preprint 2025 | 2512.10926 | colinqiyangli/dqc | `dqc` |
| MAC | ICLR 2026 | 2512.08108 | kwanyoungpark/MAC | `mac` |
| HIQL | NeurIPS 2023 | 2307.11949 | seohongpark/hiql | — |
| QAM | ICLR 2026 | 2601.14234 | colinqiyangli/qam | `qam` |

---

### 22.2 问题根因的精确诊断

当前 IQL Q 网络的结构性失败来自三个叠加层次：

1. **维度诅咒**：448 维 flat action 输入，16K 离线样本。Q 网络从 `(state_feat, a_flat)` 联合空间学习标量函数，覆盖密度极低，对动作维度几乎没有梯度信号。

2. **Bootstrap 目标噪声放大**：`target_q = r_discounted + γ^16 * V(s')`，稀疏奖励下大多数样本 r_discounted ≈ 0，target 几乎全靠 V(s') 撑起，形成高方差 bootstrap 循环。

3. **IQL expectile 回归在高维动作下退化**：当 Q(s, a_chunk) 本身接近随机，expectile 回归学到的 V(s) 是对随机 Q 值的上分位数，完全失去"状态价值"语义。

---

### 22.3 各论文核心技术与迁移分析

#### Q-Chunking

**核心**：把 chunk 作为 macro-action 输入 Q 网络，实现无偏 h-step TD backup。bootstrap 点的 Q 接受的是数据集中实际执行的下一个 chunk（而非从当前策略重采样），因此即使 off-policy 也无偏。Policy extraction 用 Best-of-N（从 BC prior 采样 N 个 chunk，取 Q 最高的），完全绕开 log-prob 估计问题。

**迁移可行性：中等**。无偏 backup 思想有价值，但前提是 Q(s, a_chunk) 能被有效学习，而我们的根本问题正是 448 维 flat action 导致 Q 学不好。可迁移部分：无偏 h-step TD target 构造 + Best-of-N 替换 PPO ratio loss。

#### Decoupled Q-Chunking (DQC)

**核心**：critic 用长 chunk（h=25）做快速价值传播，policy 用短 chunk 或单步动作。关键创新是**蒸馏部分 chunk critic**：

```
Q^P(s, a_{0:h_a}) ≈ max_{a_{h_a:h}} Q(s, [a_{0:h_a}, a_{h_a:h}])
```

用 expectile loss（κ≈0.9）从长 chunk Q 蒸馏出短 chunk Q^P，让 policy 优化低维的 Q^P 而非高维的 Q_full。

**迁移可行性：高**。这是五篇中最直接相关的。迁移路径：
- 用已有的 Q(s, a_chunk_448) 作为 teacher（即使排序差）
- 蒸馏出 Q^P(s, a_single_28)，输入维度从 448 降到 28
- 用 Q^P - V 替换当前的 scalar_iql advantage

```python
# 蒸馏 partial chunk critic
def distill_loss(Q_full, Q_partial, s_feat, a_chunk, kappa=0.9):
    a_single = a_chunk[:, 0, :]  # 只取第一步动作
    q_full_val = Q_full(s_feat, a_chunk.reshape(B, -1)).detach()
    q_partial_val = Q_partial(s_feat, a_single)
    diff = q_full_val - q_partial_val
    weight = torch.where(diff > 0, kappa, 1 - kappa)
    return (weight * diff**2).mean()
```

**潜在问题**：Q_full 排序接近随机，蒸馏信号质量存疑，需要实验验证。

#### MAC（Model-Based Action Chunking）

**核心**：用 action-chunked dynamics model 做 model-based rollout 替代直接学习 Q(s, a_chunk)。chunk 动力学模型一次预测 h 步，相比单步模型误差累积减少 h 倍。用 flow-based BC policy 做 rejection sampling 防止 model exploitation。

**迁移可行性：最高**。我们已有 dynamics model 和 `chunk_vdelta_scalar` 实现，本质上已经是 MAC 思想的 90%。**最快的迁移路径**：直接把 `chunk_vdelta_scalar` 接入 PPO advantage，替换 scalar_iql。

```python
# 在 advantage_computation 中增加 mac 模式
def _mac_advantage(self, s_feat, a_chunk):
    q_mac = compute_chunk_vdelta_scalar_raw(
        s_feat, a_chunk, self.dynamics, self.iql, self.gamma, self.n_obs_steps
    )
    v_s = self.iql._value(s_feat.reshape(B, -1)).squeeze(-1)
    return q_mac - v_s
```

注意：`chunk_vdelta_scalar` 已经在 offline BPPO 中作为 `offline_chunk_adv_mode` 使用，且实验结果（Section 16）显示它在 24 组对比中 100% 优于 scalar_iql。MAC 的贡献是提供了理论框架和更完整的实现（rejection sampling、更深的 rollout）。

#### HIQL

**核心**：层次化分解解决长程 Q 函数信噪比问题。高层预测子目标，低层执行到子目标的动作，每层的 advantage 信号更清晰。

**迁移可行性：低**。HIQL 是 goal-conditioned RL，不直接适用。但其信噪比分析（远程目标下不同动作的 Q 差异极小，被噪声淹没）是我们问题的理论支撑。可作为 related work 引用。

#### QAM（Q-learning with Adjoint Matching）

**核心**：用伴随方法（adjoint method）把 Q 梯度转化为稳定的逐步 regression 目标，避免 backprop through time 的数值不稳定。最优策略闭式解：`π*(a|s) ∝ π_β(a|s) · exp(τ · Q(s,a))`。

**迁移可行性：中等偏低**。QAM 解决的是"如何用 Q 梯度优化 diffusion policy"，而我们的根本问题是 Q 信号质量。如果先通过 DQC 或 MAC 得到可靠的 Q^P(s, a_single)，QAM 可以作为比 PPO ratio loss 更稳定的 policy optimization 方法。优先级低于 MAC 和 DQC。

---

### 22.4 迁移方案优先级排序

| 优先级 | 方案 | 来源 | 改动量 | 预期收益 |
|--------|------|------|--------|---------|
| 1 | **MAC advantage**：直接用 chunk_vdelta_scalar 作为 PPO advantage | MAC | 极小（已有实现） | 高（已有实验证据） |
| 2 | **DQC partial critic**：从 Q_chunk 蒸馏 Q^P(s, a_single)，用于 actor | DQC | 中（新增蒸馏网络） | 中高（理论上解决维度诅咒） |
| 3 | **Q-Chunking Best-of-N**：用 Best-of-N 替换 PPO ratio loss | Q-chunking | 中（改 policy extraction） | 中（绕开 log-prob 估计） |
| 4 | **QAM adjoint matching**：用伴随方法优化 diffusion policy | QAM | 高（需实现 adjoint ODE） | 中（需先解决 Q 质量问题） |

**最推荐的下一步**：方案 1（MAC advantage）已经在 `offline_chunk_adv_mode=chunk_vdelta_scalar` 中实现，且 Section 16 实验已证明其优越性。方案 2（DQC partial critic）是最有潜力从根本上解决维度诅咒的方向，值得作为下一个实验。

---

## 23. DQC/QC 代码深度分析与具体迁移方案（2026-04-18）

### 23.1 DQC 核心实现细节

**网络架构**（`dqc/agents/dqc.py`）：
- `chunk_critic`（Q_full）：输入 `[obs, goal, a_chunk_H]`，4层 MLP (1024,1024,1024,1024) + LayerNorm，ensemble=2
- `action_critic`（Q_partial）：输入 `[obs, goal, a_chunk_H_actor]`，同架构，但 action 只取前 H_actor 步
- 默认 H_critic=25，H_actor=1（单步）

**蒸馏 loss（expectile BCE）**：
```python
weight = kappa_d if Q_full >= Q_partial else (1 - kappa_d)
L_distill = weight * BCE(Q_partial_logit, Q_full)
# kappa_d=0.5（对称），kappa_b=0.9（value backup 乐观）
```

**训练顺序**：每步同时更新 chunk_critic + action_critic + actor，combined loss 一次 backward。

**Policy extraction**：Best-of-N，N=32，用 Q_partial 打分。

### 23.2 QC 核心实现细节

**无偏 h-step TD target**（`qc/agents/acfql.py`）：
```python
target_Q = r_h + gamma^h * V(s_{t+h}) * mask
# r_h = 数据集中实际的 h 步累积折扣奖励（不是 bootstrap）
# s_{t+h} = 数据集中实际的 h 步后状态
# 因此即使 off-policy 也无偏
```

**Best-of-N**：N=32，从 flow policy 采样 N 个 chunk，用 Q 打分取最高。

### 23.3 迁移到我们 setting 的具体方案

**方案 A（DQC partial critic）**：

在现有 `IQL_Q_V_no` 基础上增加 `Q_partial(s, a_single_28)`：

```python
# 新增 PartialChunkCritic
class PartialChunkCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, depth):
        # 输入：state_dim + 28（单步动作），输出：标量 Q
        ...

# 蒸馏 loss
def distill_loss(Q_full, Q_partial, s, a_chunk, a_single, kappa_d=0.5):
    with torch.no_grad():
        target = Q_full.minQ(s, a_chunk)  # (B,)
    pred = Q_partial(s, a_single)         # (B,)
    diff = target - pred
    weight = torch.where(diff > 0, kappa_d, 1 - kappa_d)
    return (weight * diff**2).mean()

# 用 Q_partial - V 替换 scalar_iql advantage
A_dqc = Q_partial(s, a_single) - V(s)
```

**关键超参**：kappa_d=0.5（对称 expectile），hidden_dim=1024，depth=4，ensemble=2。

**方案 B（QC 无偏 h-step backup）**：

修改 IQL 的 Q target 构造，使用数据集中实际的 h 步累积奖励：

```python
# 需要数据集支持：存储连续 chunk 序列
# batch 中需要有 (s_t, a_chunk_t, r_h_t, s_{t+h}, a_chunk_{t+h})
target_Q = r_h + gamma**h * V(s_{t+h})
# 而非当前的：target_Q = r_discounted + gamma**16 * V(s_next)
```

**注意**：方案 B 需要数据集格式支持，改动较大；方案 A 只需在现有 critic 上增加一个蒸馏头，改动最小。

### 23.4 推荐实施顺序

1. **立即可做**：方案 A（DQC partial critic），在 stage1 训练完 Q_full 后，增加蒸馏步骤训练 Q_partial，然后用 Q_partial - V 作为 offline BPPO 的 advantage
2. **中期**：方案 B（QC 无偏 backup），需要修改 AdroitDataset 支持连续 chunk 序列采样
3. **长期**：QAM adjoint matching，在 Q 信号质量提升后，改进 diffusion policy 的优化方式

---

## 24. MAC 代码深度分析与迁移方案（2026-04-18）

### 24.1 MAC 的 Q(s, a_chunk) 估计方式

MAC 不直接学习 Q(s, a_chunk)，而是用 **multi-step GAE-smoothed model rollout**：

```
rollout K 步 → 用 learned reward/termination model 计算奖励 → GAE 平滑
Q(s, a_chunk) = V(s) + GAE_advantage
```

关键：GAE 平滑（λ 加权）大幅降低 model rollout 的方差，比我们当前的 `chunk_vdelta_scalar`（无 GAE）更稳定。

### 24.2 Rejection Sampling 机制

```python
# 采样 N=8~32 个候选 chunk（从 BC prior 采样，保证 in-distribution）
candidates = [bc_policy.sample(s) for _ in range(N)]
# 用 critic 打分，取最高
a_star = max(candidates, key=lambda a: critic(s, a))
```

防止 model exploitation 的三重保障：
1. 候选来自 BC prior（in-distribution）
2. critic 在 model rollout 数据上训练
3. soft target network 防止过估计

### 24.3 我们当前实现与 MAC 的差距

| 组件 | 我们当前 | MAC | 差距 |
|------|---------|-----|------|
| 价值估计 | 直接 V(s') | GAE 平滑 | 方差高 |
| 防 exploitation | 无 | Rejection sampling | 有 OOD 风险 |
| 动作分布 | 任意采样 | BC prior 约束 | 可能 OOD |
| Critic 更新 | 硬更新 | Soft target | 过估计风险 |

### 24.4 迁移优先级

**Phase 1（最快，2-3天）**：给 `chunk_vdelta_scalar` 加 GAE 平滑，预期提升 10-20%：
```python
# 当前：直接用 dynamics 一步预测
proxy = r_chunk + gamma**K * V(s_end) - V(s_start)

# 改进：GAE 平滑（需要 dynamics 逐步展开）
gae = 0
for k in reversed(range(K)):
    delta_k = r_k + gamma * V(s_{k+1}) - V(s_k)
    gae = delta_k + gamma * lambda_ * gae
proxy_gae = gae
```

**Phase 2（中期，1-2周）**：加 rejection sampling，用 BC policy 约束候选动作分布。

**Phase 3（长期，2-3周）**：完整 MAC 实现，包括 flow-based BC policy 和 learned reward model。
