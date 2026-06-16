# Online EMA 闭环方案（单环境 + 并行环境）

## Summary
为 online PPO 增加一套完整的 EMA 闭环，并明确要求同时适配单环境 `online_ft()` 和并行环境 `_online_ft_vec()`。EMA 作为候补分支存在，始终与裸 policy 同步更新、同步评估、同步保存与恢复，但不改变原有主链路。默认训练、rollout、裸 policy eval、默认 checkpoint、默认恢复路径全部保持不变。

## Key Changes
### 1. 单环境与并行环境统一接入 online EMA
- 在 `train_cm_mid.py` 的两条 online 主路径都接入相同语义：
  - 单环境：`online_ft()`
  - 并行环境：`_online_ft_vec()`
- 两条路径都必须具备同样的 EMA 生命周期：
  - online 起点同步
  - PPO update 后 `ema.step(self.unio4._policy)`
  - online 初始 eval
  - 周期 eval
  - 训练结束保存
  - online checkpoint save/load 恢复
- 不允许只接单环境或只接 vec 环境；两条链路功能面必须对齐。

### 2. 在线 EMA update
- 复用现有 `self.ema_model`，不新建第二套 online EMA 对象。
- 进入 online 前，先把 `self.ema_model` 同步到当前 online 裸 policy 起点：
  - offline best / offline last / online resume 后都要执行这一步
- 起点同步采用一次硬复制，不使用 `shadow_params`、`copy_to()`、`store()/restore()` 之类接口：
  - `self.ema_model.load_state_dict(self.unio4._policy.state_dict())`
  - 理由：当前仓库里的 `self.ema_model` 是完整 policy 副本，不是只保存 shadow weights 的 EMA 容器
- EMA update 的唯一合法位置：
  - 每次 online PPO 参数更新完成之后
  - 单环境：放在 `dp_align_update_no_share(...)` 或等价 PPO update 整体返回之后
  - vec 环境：放在对应一次 PPO iteration 整体返回之后
- `ema.step()` 的粒度固定为“每次完整 PPO iteration 一次”，不是每个 mini-batch 一次
- 不在 rollout 前、eval 前、save 时更新 EMA。

### 3. 在线 EMA eval
- 单环境和 vec 环境都增加 `EMA Eval`，显式评 `self.ema_model`。
- `EMA Eval` 不做参数临时切换，不把 EMA 权重临时拷回 `self.unio4._policy`。
- 统一复用现有 `eval(..., policy_override=...)` 入口：
  - `self.eval(..., policy_override=self.ema_model, eval_name='EMA Eval')`
- 理由：当前仓库已经维护独立的 `self.ema_model` 副本，`eval()` 也已经支持显式传入待评 policy，因此不需要额外的 `store()/copy_to()/restore()` 过程。
- online 的评估结构统一为：
  - 裸 policy eval：主指标，沿用当前 `log_data`
  - EMA eval：辅助指标
  - IDQL eval：可选辅助指标
  - CM eval：若在线蒸馏开启则继续保留
- 两条路径都要新增并统一写出：
  - wandb：`online ema success rates`、`online ema returns`
  - CSV：`ema_success_rates.csv`、`ema_returns.csv`
  - 控制台摘要：同时打印 `Policy / EMA / IDQL / CM`
- `EMA Eval` 不接入默认选模或训练闭环，不替代裸 policy eval。

### 4. 在线保存与恢复
- 两条路径都在训练结束时保存：
  - `online_last/`：裸 policy，保持现状
  - `online_last_ema/`：EMA model，新增
- `save_online_checkpoints()` / `load_online_checkpoints()` 增加 EMA 子目录支持，适用于单环境和 vec 环境共用的 checkpoint 逻辑：
  - 保存到 `online_ft/.../ema/update_<n>/`
  - 恢复时若存在 EMA checkpoint，则恢复到 `self.ema_model`
  - 若不存在，则从当前恢复后的裸 policy 同步 EMA，保证兼容旧 checkpoint
- 不修改现有 `update_num.txt`、policy/value/iql/lr/distilled 结构，只新增 EMA 子目录。
- vec 环境的 `EMA Eval` 也必须复用同一个 `eval()` 入口，不单独实现一套 `_online_ft_vec` 专用 EMA eval 分支。
- 这要求实现继续维持 `eval()` 的 `policy_override` 能力，不能回退到硬编码 `self.unio4._policy`。

### 5. 在线 best_ema
- 新增 online EMA best 保存，但单环境和 vec 环境都必须共用同一套 helper 与目录语义。
- 新增：
  - `online_best_ema/`
  - `best_score.csv`
  - `best_meta.txt`
- `best_meta.txt` 需明确写：
  - `eval_name: Online EMA Eval`
  - 来源 run 目录与时间戳
- `online_best_ema` 只按 EMA eval 更新，不影响：
  - 裸 policy 的 `online_last/`
  - 默认 online load
  - rollout / actor-critic 更新
  - 现有主链路 best 逻辑

### 6. 不影响原主链路的硬约束
- 不改变裸 policy 作为 online rollout 模型的事实
- 不改变单环境和 vec 环境的默认主 eval 口径：仍是裸 policy
- 不让 `EMA Eval` 参与：
  - actor/critic 更新
  - old policy 更新
  - 默认 best 保存
  - online 主恢复入口
- `training.use_ema=False` 时，两条链路都必须完全退化为当前行为，EMA 逻辑全部 no-op

## Test Plan
- 单环境 online：
  - 初始阶段同时产出 `Policy Eval`、`EMA Eval`
  - 一次 PPO update 后确认 `ema.step(...)` 生效
  - 结束时同时生成 `online_last/` 和 `online_last_ema/`
- vec online：
  - `ppo.use_vec_env_online=True` 时同样产出 `Policy Eval`、`EMA Eval`
  - 至少完成一次 rollout + 一次 PPO update + 一次评估
  - 确认 vec 路径也写出 `ema_success_rates.csv`、`ema_returns.csv`
- checkpoint 恢复：
  - 新 checkpoint：恢复裸 policy 和 EMA
  - 旧 checkpoint：缺失 EMA 时不报错，自动从裸 policy 初始化 EMA
- 回归：
  - `training.use_ema=False` 时，单环境和 vec 环境行为与当前版本一致
  - `idql_eval` 开关仍可选
  - 不影响 `best/` 仍按裸 policy 管理
  - 不影响 offline 已经加上的 `best_ema/`

## Assumptions
- 使用现有 `self.ema_model` 作为 offline/online 共用 EMA 容器。
- 单环境和 vec 环境共用同一套 EMA save/load helper，不分叉出两套实现。
- online EMA 目录默认采用：
  - `online_last_ema/`
  - `online_ft/.../ema/update_<n>/`
  - `online_best_ema/`
- online 主链路的默认模型始终是裸 policy；EMA 只做同步候补与观测。

## Review Round 1
### Findings
1. online 起点同步当前不能无条件覆盖 EMA。
   - 现实现里在进入 online 时直接执行：
     - `self.ema_model.load_state_dict(self.unio4._policy.state_dict())`
     - `ema.optimization_step = 0`
   - 这会在 `ppo.load_online_cp=True` 的 online resume 场景下，把刚恢复出来的 EMA 轨迹抹掉，并重置 EMA warmup 计数。
   - 修正要求：
     - 只有 fresh offline->online 起点时才做一次 EMA 同步
     - online resume 时如果已从 checkpoint 恢复了 EMA，就必须保留恢复后的 EMA 和 `optimization_step`
     - 不允许无条件 reset `ema.optimization_step`

2. online 初始 eval 还没有接入 EMA eval。
   - 当前实现只在周期性 `evaluate_freq` 评估时跑 `Online EMA Eval`
   - 单环境和 vec 环境的 initial eval 都缺这一项
   - 这会导致：
     - `ema_success_rates.csv` / `ema_returns.csv` 比主 `success_rates.csv` 少一个初始点
     - 初始 console / wandb 没有 EMA 指标
     - 与本文档上面的“online 初始 eval 也包含 EMA Eval”要求不一致
   - 修正要求：
     - 单环境 `online_ft()` 初始 eval 必须补 `Online EMA Eval`
     - vec 环境 `_online_ft_vec()` 初始 eval 也必须补 `Online EMA Eval`
     - 初始 EMA 指标要同步写入 wandb、console、CSV

### Constraints For Fix
- 修复以上两点时，不能影响：
  - 裸 policy 作为 online 主链路模型
  - 现有 `online_last/`、`online_last_ema/`、online checkpoint EMA save/load 结构
  - 单环境和 vec 环境已经接上的周期性 `Online EMA Eval`
