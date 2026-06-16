# train_ddp.py 缺失逻辑修复记录

基于 `train_cm_mid.py`（单卡版本）对 `train_ddp.py`（DDP多卡版本）进行补全，当前累计修复 32 项缺失/偏差。

---

## 一、新增 3 个方法

### 1. `get_distill_optimizer()` (line ~2547)
为 online 阶段的 CM distillation 创建独立的 optimizer 和 lr_scheduler。

### 2. `load_online_checkpoints()` (line ~2565)
加载 online finetuning 的断点续训（policy, value, iql, distilled model, lr）。

### 3. `save_online_checkpoints()` (line ~2598)
保存 online finetuning 各组件的 checkpoint，仅在 rank 0 调用。

---

## 二、`run()` BC 阶段

### 4. KL annealing (line ~575-591)
添加 VIB encoder 的 beta_kl 线性退火逻辑：
```python
if hasattr(model_obs_encoder, 'beta_kl'):
    target_beta_kl = model_obs_encoder.beta_kl
# 每个 epoch:
if cfg.kl_annealing and hasattr(model_obs_encoder, 'beta_kl'):
    progress = local_epoch_idx / max(cfg.training.num_epochs - 1, 1)
    model_obs_encoder.beta_kl = target_beta_kl * progress
```

---

## 三、`finetune_dp3()` 修复

### 5. Encoder fixing (line ~1319-1322)
BPPO 训练前固定 encoder 为 eval 模式：
```python
if self.cfg.unio4.fix_encoder:
    self.unio4._policy.obs_encoder.eval()
    self.unio4._old_policy.obs_encoder.eval()
```

### 6. 双评估模式 (line ~1332-1380)
初始评估和 BPPO 循环中同时运行 `unio4_eval(idql)` 和 `self.eval(normal)`，分别追踪 `idql_scores` 和 `normal_scores`。

### 7. 初始模型保存 (line ~1358-1359)
BPPO 训练开始前保存初始模型到 `output_dir/best`。

### 8. Wandb 日志补全 (line ~1383-1388)
初始日志和 BPPO 循环日志中增加 `idql_eval_scores` 和 `normal_eval_scores`。

### 9. `final_reward` 参数 (line ~1423)
`update_distribution()` 调用中添加 `final_reward=self.cfg.unio4.final_reward`。

### 10. 双评估 CSV 保存 (line ~1508-1511)
训练结束时保存 `last_idql_eval_scores.csv` 和 `last_normal_eval_scores.csv`。

---

## 四、`online_ft()` 修复

### 11. VIB stochastic sampling (line ~1891-1905) ⚠️ 严重
online 阶段开头设置所有 encoder 的 `force_stochastic = True`：
```python
def _set_force_stochastic(encoder, val):
    if hasattr(encoder, 'force_stochastic'):
        encoder.force_stochastic = val
_set_force_stochastic(self.unio4._policy.obs_encoder, True)
_set_force_stochastic(iql.obs_encoder, True)
# ... 以及 iql 的 Q/V 网络和 iql_online
```

### 12. Online distillation 初始化 (line ~1906-1913) ⚠️ 严重
```python
if self.cfg.distill_phase == 'online':
    self.unio4._policy.set_target()
    self.unio4._policy.distilled_model.load_state_dict(...)
    cm_optimizer, cm_lr_scheduler = self.get_distill_optimizer()
```

### 13. `transfer2online` 传递 CM 参数 (line ~2130)
添加 `cm_optimizer=cm_optimizer, cm_lr_scheduler=cm_lr_scheduler`。

### 14. `load_online_cp` 支持 (line ~2126-2129)
添加从 online checkpoint 恢复训练的逻辑。

### 15. `iql_buffer` 创建条件 (line ~2142)
从 `if self.cfg.ppo.iql_ft` 改为 `if self.cfg.ppo.iql_ft or self.cfg.update_phase == 'outloop'`。

### 16. CM/IDQL 追踪列表 (line ~2148-2150)
添加 `cm_all_success_rates`, `cm_all_returns`, `all_idql_success_rates`, `all_idql_returns`。

### 17. `distill_losses` 追踪 (line ~2185)
从 `actor_losses, critic_losses, bc_losses = [], [], []` 改为包含 `distill_losses`。

### 18. `update_num` 和 `idql_log_data` 初始化 (line ~2188-2189)
添加 `update_num = 0` 和 `idql_log_data = None`。

### 19. `idql_rollout` 分支 (line ~2226-2229)
```python
if self.cfg.ppo.idql_rollout:
    action, all_x, a_logprob = self.unio4._policy.sample_action_with_logprob(...)
else:
    action, all_x, a_logprob = self.unio4._policy.all_step_action_logprob(..., fix_encoder=...)
```

### 20. `env.step()` 参数 (line ~2236) ⚠️ 严重
添加 `reward_agg_method='discounted_sum', gamma=self.cfg.gamma`。

### 21. `outloop` update phase (line ~2259-2272)
添加 outloop 分支：online/offline 混合训练 + `distill_update`。

### 22. `iql_buffer.store` 条件 (line ~2279)
从 `if self.cfg.ppo.iql_ft` 改为 `if self.cfg.ppo.iql_ft or self.cfg.update_phase == 'outloop'`。

### 23. `iql.update()` 添加 `online_recon` 参数 (line ~2297)
添加 `online_recon=self.cfg.ppo.online_iql_recon`。

### 24. `dp_align_update_no_share` 返回值 (line ~2312-2316)
从接收 3 个值改为 4 个值 `actor_loss, critic_loss, bc_loss, distill_loss`，并追踪 `distill_loss`。

### 25. `save_online_checkpoints` 调用 (line ~2328-2329)
添加 rank 0 保护的 checkpoint 保存。

### 26. Online 评估补全 (line ~2335-2415)
同时运行 idql_eval + normal_eval + cm_eval，完整记录所有指标到 wandb 和 CSV。

---

## 五、`eval()` 和 `unio4_eval()` 修复

### 27. `eval()` 签名和调用 (line ~2486)
添加 `use_cm=False, distill2mean=False` 参数，`env_runner.run()` 传递这两个参数。

### 28. `unio4_eval()` 签名和调用 (line ~2433)
添加 `use_cm=False, distill2mean=False` 参数，`env_runner.idql_run()` 和 `env_runner.run()` 传递这两个参数。

---

## 六、其他

### 29. `import glob` (line ~27)
新增 `glob` 模块导入，用于 `load_online_cp` 路径查找。

---

## 七、2026-03-19 追加修复

### 30. `resume` 优先恢复 `latest_cm` (line ~483-496) ⚠️ 严重
对齐单卡版恢复逻辑，先查找 `latest_cm.ckpt`，不存在时再回退到 `latest.ckpt`：
```python
if cfg.training.resume:
    latest_ckpt_path = self.get_checkpoint_path(tag='latest')
    latest_cm_ckpt_path = self.get_checkpoint_path(tag='latest_cm')
    if latest_cm_ckpt_path.is_file():
        self.load_checkpoint(path=latest_cm_ckpt_path)
    elif latest_ckpt_path.is_file():
        self.load_checkpoint(path=latest_ckpt_path)
```

### 31. lr scheduler 恢复前补齐 `initial_lr` (line ~498-501)
补回单卡版已有的 optimizer param group 修复，避免 resume 后创建 scheduler 时因缺少 `initial_lr` 导致行为偏差或报错：
```python
for group in self.optimizer.param_groups:
    if 'initial_lr' not in group:
        group['initial_lr'] = group['lr']
```

### 32. `get_checkpoint_path()` 支持 `latest_cm` + 保持 rank-local device (line ~178-181, ~2764-2766) ⚠️ 严重
- `get_checkpoint_path()` 补充 `latest_cm` 分支，修复 `distill2cm(phase='after_dp')` 调用时可能触发的 `NotImplementedError`。
- 删除 `self.device = cfg.training.device` 的错误覆盖，保留 DDP 初始化阶段的 `cuda:{rank}`。
- `BehaviorProximalPolicyOptimization` 初始化时改为直接使用 `self.device`，确保多卡时各 rank 使用各自设备。

---

## 统计

- 文件: `RL-100/train_ddp.py`
- 累计修复项: 32
- 编译检查: 通过
