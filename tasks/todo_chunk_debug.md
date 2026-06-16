# chunk_as_single_action=False 核对结果

## 结论摘要
当前 `chunk_as_single_action=False` 这条路并不完全严格自洽。

- offline OPE rollout 的 chunk 展开语义本身基本合理：会把一个 action chunk 按 primitive action 逐步展开。
- 但 critic target、所谓的 GAE、以及 BPPO actor update 的部分对齐逻辑，并没有完全和这种 primitive-step 语义一致。
- 因此这条路更像“可运行的启发式实现”，不是一条严格正确、定义干净的基线路径。

## 1. OPE rollout 语义
在 `EnsembleDynamics_batch.rollout()` 中：

- `chunk_as_single_action=True` 时，`rollout_length` 直接表示 rollout 多少个 chunk。
- `chunk_as_single_action=False` 时，外层循环是 `int(rollout_length / n_action_steps)`，每次 policy 仍采一个 chunk，但 dynamics 用 `multi_step()` 把 chunk 内的 primitive actions 逐个展开。

对应实现：

- `multi_step()` 会对 `nactions[:, i]` 逐个调用 `self.step(...)`
- 每一步更新 latent / obs window
- 返回的是整个 chunk 执行完后的最终 `next_obs`

因此这条路的 rollout 语义是：

- `o_t + [a_t, ..., a_{t+k-1}] -> o_{t+k}`

也就是说一个 chunk 对应多个 primitive action step 的展开。

相关位置：

- `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py:174`
- `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py:181`
- `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py:507`
- `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py:512`

## 2. critic / value target 不完全合理
这是当前最明显的问题。

在 `critic.py` 中，`chunk_as_single_action=False` 分支取的是：

- `s = nobs_features`
- `a = nactions[:, self.n_obs_steps - 1]`
- `r = batch['reward'][:, self.n_obs_steps - 1]`
- `s_p = next_nobs_features`，且 `next_nobs_features` 来源是 `next_nobs[:, :self.n_obs_steps]`

这说明这条 transition 被按“单步 primitive action”处理。

但后面的 Q target 仍然写成：

- `target_q = r + not_done * (gamma ** n_action_steps) * next_v`

这和前面的单步 `s,a,r,s'` 时间粒度不一致。  
如果是单步 transition，理论上这里更应该是 `gamma`，不是 `gamma ** n_action_steps`。

相关位置：

- `RL-100/rl_100/unidpg/critic.py:627`
- `RL-100/rl_100/unidpg/critic.py:723`
- `RL-100/rl_100/unidpg/critic.py:768`

## 3. 所谓 GAE 不是标准 GAE
`chunk_as_single_action=False` 这条路里，无论 OPE rollout 还是 BPPO actor update，都没有使用标准的：

- `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`

而是直接对 `Q` 或 `Q-V` 做递推。

### OPE rollout
在 `rollout(..., use_gae=True)` 中：

- `Qs.append(Q(...))`
- 然后直接对 `Qs` 做递推
- 没有 reward residual，也没有 done mask

相关位置：

- `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py:526`

### BPPO actor update
在 `NStepValueEstimation()` 中：

- 每一步先算 `advantage = Q(s,a) - V(s)`
- 然后 `GAE_withQ()` 再对这些 advantage 做递推
- 同样没有标准 TD residual 和 done mask

相关位置：

- `RL-100/rl_100/unidpg/uni_ppo.py:316`
- `RL-100/rl_100/unidpg/uni_ppo.py:327`
- `RL-100/rl_100/unidpg/uni_ppo.py:462`

所以这条路里的 “GAE” 更准确地说是：

- 对 `Q` 或 `Q-V` 的启发式递推
- 不是标准 GAE

## 4. BPPO actor ratio / advantage 对齐问题
`chunk_as_single_action=False` 时，actor loss 用的是：

- `ratio[:, j]`
- `advantages[j]`

这里默认 action index 从 0 开始。

但 chunk 分支会显式用：

- `action_start = n_obs_steps - 1`
- 再取真正优化的 action span

所以非 chunk 分支在 `n_obs_steps > 1` 时没有做同样的 offset 对齐。  
当前 two-stage 配置里 `n_obs_steps=1`，所以暂时不会暴露；但这条路本身不是一般正确的。

相关位置：

- `RL-100/rl_100/unidpg/uni_ppo.py:521`
- `RL-100/rl_100/unidpg/uni_ppo.py:541`

## 5. 当前报错的直接原因
这次 `torch.stack expects a non-empty TensorList` 的直接原因不是 indexing，而是参数组合非法：

- `rollout_length = 3`
- `n_action_steps = 4`
- `chunk_as_single_action = False`

于是外层循环次数变成：

- `int(3 / 4) == 0`

最终：

- `Qs = []`
- `gae_advantages = []`
- `torch.stack([])` 报错

相关位置：

- `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py:507`
- `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py:532`

## 总结判断
如果目标语义是“一个 chunk 是一个 macro-action”，那么推荐主路径应当是：

- `chunk_as_single_action=True`

而 `chunk_as_single_action=False` 当前存在以下问题：

- OPE rollout 的 chunk 展开逻辑基本合理
- 但 critic target 折扣和 transition 粒度不一致
- 所谓 GAE 不是标准 GAE
- `n_obs_steps > 1` 时 actor ratio / advantage 对齐不完整
- 对 `rollout_length < n_action_steps` 缺少显式参数检查

因此这条路当前不适合作为严格正确的 chunk offline evaluation / BPPO 基线路径。

## 修复计划（让 chunk_as_single_action=False 严格自洽）

### 目标
把 `chunk_as_single_action=False` 统一成“primitive-step 语义”：

- policy 仍然一次采一个 action chunk
- dynamics rollout 逐 primitive action 展开
- critic / BPPO / OPE 的状态、动作、reward、bootstrap 折扣、advantage 递推全部按同一单步时间粒度对齐

这样这条路的语义是：

- 一个 chunk 只是 policy 的输出形式
- 但学习和 OPE 都在 primitive action 时间尺度上计算

### Reminder
本轮修复只能收敛 `chunk_as_single_action=False` 这条路径，不能干扰其他已工作的链路：

- 不能改变 `chunk_as_single_action=True` 的现有语义
- 不能改变 single action (`n_action_steps=1`) 的现有语义
- 如果某个修复会同时触碰这两条路径的默认行为，必须先拆分成只作用于 `chunk_as_single_action=False` 的条件分支
- 验证时必须显式做回归，确认：
  - `chunk_as_single_action=True` 不回退
  - `n_action_steps=1` 不回退

### 1. 先锁定时间语义，不再混用 chunk 折扣
在 `chunk_as_single_action=False` 分支中，所有训练目标都统一按单步 transition 解释：

- `s_t`
- `a_t`
- `r_t`
- `s_{t+1}`

要求：

- critic target 不能再混入 `gamma ** n_action_steps`
- actor advantage 不能一部分按 chunk，一部分按 single-step
- OPE rollout 的 `use_gae` 也不能继续沿用当前的 `Q` 递推近似

### 2. 修 critic / value update 的 target 对齐
在 `critic.py` 的 `chunk_as_single_action=False` 分支：

- 保留当前 `s = nobs_features`
- 保留当前 `a = nactions[:, self.n_obs_steps - 1]`
- 保留当前 `r = batch['reward'][:, self.n_obs_steps - 1]`
- 保留当前 `s_p = next_nobs_features`

但把 bootstrap target 改成单步一致形式：

- `target_q = r + not_done * gamma * next_v`

同时核对并明确：

- `next_nobs[:, :self.n_obs_steps]` 在这条路里表示的是一步后的 obs window
- 若数据集中的 `next_obs` 实际不是一步后窗口，而是 chunk 末尾窗口，则必须先修 dataset / sampling 对齐，不能只改 `gamma`

验收标准：

- `chunk_as_single_action=False` 下，critic 训练的 `s/a/r/s'` 和折扣是同一时间粒度

### 3. 重写这条路的 advantage / GAE
当前这条路里的 “GAE” 不是标准 GAE，需要统一改掉。

#### 3.1 BPPO actor update
在 `uni_ppo.py`：

- `NStepValueEstimation()` 不再返回 `Q-V` 形式的 pseudo-advantage
- 改为返回逐步 rollout 的单步 TD residual 所需量，最直接的是：
  - `r_t`
  - `V(s_t)`
  - `V(s_{t+1})`
  - `done_t`（如果 dynamics rollout 没有 done，也要明确用 `0`）

- 新增一个标准单步 GAE helper，按：
  - `delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)`
  - `gae_t = delta_t + gamma * lamda * (1-done_t) * gae_{t+1}`

- `chunk_as_single_action=False` 时，actor loss 使用这个标准单步 GAE 输出

#### 3.2 offline OPE rollout
在 `EnsembleDynamics_batch.rollout(..., use_gae=True)`：

- 不再对 `Qs` 直接递推
- 改为和 BPPO actor 保持同一套 primitive-step residual 语义
- 如果 OPE 只需要一个 trajectory-level scalar，也应该先基于单步 delta 算出逐步优势/回报，再聚合

要求：

- BPPO 和 OPE 对 `chunk_as_single_action=False` 的 advantage 定义必须一致
- done mask 要接进去，不能继续省略

### 4. 修 actor ratio / action index 对齐
在 `uni_ppo.py` 的 `chunk_as_single_action=False` 分支：

- 现在直接用 `ratio[:, j]` 和 `advantages[j]`
- 这默认 action index 从 0 开始

需要改成显式按真实优化区间取：

- `action_start = 0 if no_pre_action else n_obs_steps - 1`
- 第 `j` 个 primitive step 实际对应：
  - `ratio[:, action_start + j]`
  - `advantages[j]`

要求：

- `n_obs_steps=1` 时行为与当前实现一致
- `n_obs_steps>1` 时不再发生 action / advantage 错位

### 5. 增加参数合法性检查
在 `EnsembleDynamics_batch.rollout()` 的 `chunk_as_single_action=False` 分支入口加显式检查：

- `rollout_length >= n_action_steps`
- 最好要求 `rollout_length % n_action_steps == 0`

若不满足：

- 直接抛出清晰错误
- 明确告诉调用方当前分支下 `rollout_length` 是 primitive-step horizon，而外层 rollout 次数是 `rollout_length / n_action_steps`

禁止继续落到：

- `Qs = []`
- `torch.stack([])`

### 6. 最小验证集
修完后至少做下面几组验证：

#### 6.1 静态 / 单元级检查

- `chunk_as_single_action=False`
- `n_action_steps=4`
- `n_obs_steps=1`

验证：

- critic target 使用 `gamma` 而不是 `gamma ** 4`
- GAE helper 使用标准 TD residual

#### 6.2 offline OPE smoke test

- `rollout_length=4`
- `n_action_steps=4`
- `chunk_as_single_action=False`

要求：

- 不再出现空 `Qs` / 空 `gae_advantages`
- OPE rollout 能返回正常标量

#### 6.3 非对齐参数负向测试

- `rollout_length=3`
- `n_action_steps=4`
- `chunk_as_single_action=False`

要求：

- 直接抛出可读的参数错误
- 不再出现 `torch.stack expects a non-empty TensorList`

#### 6.4 actor 对齐测试

- 分别测试 `n_obs_steps=1` 和 `n_obs_steps=3`

要求：

- `chunk_as_single_action=False` 下 actor 取到的 ratio index 和真实优化 action 对齐

### 7. 修复完成后的判断标准
只有满足下面条件，才可以认为 `chunk_as_single_action=False` 基本自洽：

- OPE rollout、critic target、BPPO actor advantage 全部按 primitive-step 语义一致
- 折扣项和 `s/a/r/s'` 时间粒度一致
- `n_obs_steps>1` 时 action offset 明确正确
- 不合法参数组合会尽早报清楚错误
