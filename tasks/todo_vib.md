# VIB Offline RL 崩塌修复 — 消融实验方案

## 问题

在 DP3 encoder 中加入 VIB + Recon loss 的消融实验中：
- Recon + VIB 对 online RL 有效
- Recon 单独对 offline RL 有效
- VIB 加入后 offline RL 崩塌

原因分析：
1. VIB 的 KL 项在 BC 阶段过度压缩表征，丢失了 offline RL critic 需要的细粒度状态信息
2. Encoder fix 住后 `forward()` 仍然每次 reparameterize 采样，同一 observation 产生不同 latent，影响 IQL critic 收敛和 BPPO advantage 估计
3. **VIB 没有做降维**：`nn.Linear(256, 256)` 输入输出维度相同，没有信息瓶颈，KL 项只正则化分布形状而无法过滤冗余信息

---

## 方案 1: 降低 beta_kl

仅改超参数，不改代码。在 shell script 中通过 hydra override 传入：

```bash
# 当前默认值 beta_kl=1e-3，尝试更小值
policy.beta_kl=1e-5
policy.beta_kl=1e-6
```

---

## 方案 2: Offline 阶段 encoder forward 用 mu 不用 sample（通过 train/eval 模式控制）

### 设计

| 阶段 | encoder 模式 | VIB 输出 | 机制 |
|------|-------------|---------|------|
| BC | `.train()` | mu + eps*std | `self.training=True` → stochastic |
| Offline RL | `.eval()` | mu only | `self.training=False` → deterministic |
| Online RL | `.eval()` + `force_stochastic=True` | mu + eps*std | flag 覆盖 eval |

### 已完成 ✅

- `pointnet_extractor.py` 中 `DP3EncoderReconVIB`:
  - `__init__()` line 1059: `self.force_stochastic = False`
  - `forward()` line 1211: `if self.training or self.force_stochastic` 控制采样
  - `Recon_VIB_loss()` line 1127: 同样的采样控制
  - Bug fix: `use_agent_pos=False` 时的 None 检查 (line 1141-1143, 1166, 1180, 1195)

### 待实现: `train_cm_mid.py` 中 online 阶段设置 force_stochastic

#### Step 1: 在 `online_ft()` 方法开头添加 helper 并设置

文件: `RL-100/train_cm_mid.py`
位置: `online_ft()` 方法开头 (~line 874 后，在 online 训练循环开始前)

添加：
```python
# Online RL 需要 VIB stochastic，即使 encoder 在 eval 模式
def _set_force_stochastic(encoder, val):
    if hasattr(encoder, 'force_stochastic'):
        encoder.force_stochastic = val

_set_force_stochastic(self.model.obs_encoder, True)
_set_force_stochastic(self.unio4._policy.obs_encoder, True)
```

#### Step 2: 对 online 阶段创建的 iql_online encoder 副本也设置

文件: `RL-100/train_cm_mid.py`
位置: iql_online 创建后 (~line 650-672 附近，搜索 `iql_online` 的创建位置)

在 iql_online 创建完成后添加：
```python
if iql_online is not None:
    for enc in [iql_online._Q._obs_encoder, iql_online._target_Q._obs_encoder, iql_online._value._obs_encoder]:
        _set_force_stochastic(enc, True)
```

注意：`_set_force_stochastic` helper 需要在这两处都能访问到，建议定义为 `TrainDP3Workspace` 的静态方法或在文件顶层定义。

---

## 方案 3: KL Annealing

### 待实现

#### Step 1: 修改 config

文件: `RL-100/rl_100/config/dp3_cm_epsilon.yaml`

在顶层新增配置项：
```yaml
kl_annealing: False
```

#### Step 2: 修改 BC 训练循环

文件: `RL-100/train_cm_mid.py`
位置: BC 训练循环中 (~line 284-286，已有注释 `# VIB module beta kl anealling`)

在 `for local_epoch_idx in range(cfg.training.num_epochs):` 循环之前，保存原始 beta_kl：
```python
total_steps = cfg.training.num_epochs * len(train_dataloader)
if hasattr(self.model.obs_encoder, 'beta_kl'):
    target_beta_kl = self.model.obs_encoder.beta_kl
```

在循环体内部开头添加 annealing 逻辑：
```python
for local_epoch_idx in range(cfg.training.num_epochs):
    # KL annealing: beta_kl 从 0 线性增长到 target_beta_kl
    if cfg.kl_annealing and hasattr(self.model.obs_encoder, 'beta_kl'):
        progress = local_epoch_idx / max(cfg.training.num_epochs - 1, 1)
        self.model.obs_encoder.beta_kl = target_beta_kl * progress
```

---

## 方案 4: VIB Bottleneck 降维（新增消融）

### 问题

当前 VIB 层没有降维，不是真正的信息瓶颈：
```python
# 当前代码 (pointnet_extractor.py line 1075-1079)
self.latent_mu_pc = nn.Linear(self.out_channel, out_channel)          # 256 → 256 ❌
self.latent_logvar_pc = nn.Linear(self.out_channel, out_channel)      # 256 → 256 ❌
self.latent_mu_state = nn.Linear(self.state_feat_dim, self.state_feat_dim) # 64 → 64 ❌
self.latent_logvar_state = nn.Linear(self.state_feat_dim, self.state_feat_dim) # 64 → 64 ❌
```

### 下游维度依赖分析

所有下游模块的维度都源自 `self.obs_feature_dim = obs_encoder.output_shape()` (dp3_cm.py:141)：

| 模块 | 维度来源 | 代码位置 |
|------|---------|---------|
| Policy UNet | `global_cond_dim = obs_feature_dim` | dp3_cm.py:149-151 |
| IQL Critic (Q/V/target_Q) | `state_dim=self.obs_feature_dim * n_obs_steps` | dp3_cm.py:312 |
| IQL Value | `self.obs_feature_dim` | dp3_cm.py:339 |
| Dynamics model | `self.model.obs_feature_dim` | train_cm_mid.py:535 |
| BPPO | 通过 critic 的 encoder 副本 | 同上 |

**只要 `output_shape()` 返回 VIB 后的正确维度，所有下游自动适配。**
唯一需要手动改的是 encoder 内部的 decoder heads（pc_decoder, state_decoder）。

### 待实现

#### Step 1: 修改 `DP3EncoderReconVIB.__init__()` 签名

文件: `RL-100/rl_100/model/vision/pointnet_extractor.py`
位置: `__init__()` 签名 (~line 1020-1039)

新增两个参数：
```python
def __init__(
    self,
    observation_space: Dict,
    beta_kl: float = 1e-3,
    beta_recon: float = 0.5,
    use_vib: bool = True,
    use_recon: bool = True,
    latent_dim: int = None,        # 新增: VIB pc bottleneck dim; None = 不降维(=out_channel)
    latent_state_dim: int = None,  # 新增: VIB state bottleneck dim; None = 不降维(=state_feat_dim)
    # DP3Encoder kwargs
    img_crop_shape=None,
    out_channel: int = 256,
    ...
):
```

#### Step 2: 修改 VIB 层构建

文件: `RL-100/rl_100/model/vision/pointnet_extractor.py`
位置: VIB layers 构建 (~line 1073-1083)

将原来的同维 Linear 改为可降维：
```python
# ------------------- VIB layers -------------------
if self.use_vib:
    self.latent_pc_dim = latent_dim if latent_dim is not None else out_channel
    self.latent_state_dim_vib = latent_state_dim if latent_state_dim is not None else self.state_feat_dim
    self.latent_mu_pc = nn.Linear(self.out_channel, self.latent_pc_dim)
    self.latent_logvar_pc = nn.Linear(self.out_channel, self.latent_pc_dim)
    if self.use_agent_pos:
        self.latent_mu_state = nn.Linear(self.state_feat_dim, self.latent_state_dim_vib)
        self.latent_logvar_state = nn.Linear(self.state_feat_dim, self.latent_state_dim_vib)
    cprint(
        f"[DP3EncoderVIB] VIB enabled | pc: {out_channel}→{self.latent_pc_dim} | state: {self.state_feat_dim}→{self.latent_state_dim_vib} | beta={beta_kl}",
        "cyan",
    )
else:
    self.latent_pc_dim = out_channel
    self.latent_state_dim_vib = self.state_feat_dim
    cprint("[DP3EncoderVIB] VIB disabled", "cyan")
```

注意：用 `self.latent_state_dim_vib` 避免和父类可能存在的 `state_dim` 冲突。

#### Step 3: 修改 Decoder 输入维度

文件: `RL-100/rl_100/model/vision/pointnet_extractor.py`
位置: decoder heads 构建 (~line 1088-1107)

将 decoder 输入维度从固定的 `out_channel`/`state_feat_dim` 改为 VIB 后的维度：
```python
if self.use_recon:
    pc_decoder_input = self.latent_pc_dim  # VIB 后的维度（如果没 VIB 则等于 out_channel）
    if num_recon_points > 0:
        self.pc_decoder = nn.Sequential(
            nn.Linear(pc_decoder_input, 512),  # 原来是 nn.Linear(out_channel, 512)
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, num_recon_points * 3),
        )
    else:
        self.pc_decoder = None
    if self.use_agent_pos:
        state_decoder_input = self.latent_state_dim_vib  # VIB 后的维度
        self.state_decoder = nn.Sequential(
            nn.Linear(state_decoder_input, 128),  # 原来是 nn.Linear(self.state_feat_dim, 128)
            nn.ReLU(),
            nn.Linear(128, state_dim),
        )
    cprint("[DP3EncoderVIB] Reconstruction heads enabled", "cyan")
else:
    cprint("[DP3EncoderVIB] Reconstruction heads disabled", "cyan")
```

#### Step 4: 重写 `output_shape()`

文件: `RL-100/rl_100/model/vision/pointnet_extractor.py`
位置: 在 `DP3EncoderReconVIB` 类中新增方法（在 `forward()` 之后，`_reparameterize()` 之前）

```python
def output_shape(self):
    if self.use_vib:
        dim = self.latent_pc_dim
        if self.use_agent_pos:
            dim += self.latent_state_dim_vib
        return dim
    return self.n_output_channels  # 无 VIB 时回退到父类行为
```

这确保下游所有模块（Policy UNet, IQL Critic, Dynamics）自动获取正确维度。

#### Step 5: 修改 `DP3CM.__init__()` 传递参数

文件: `RL-100/rl_100/policy/dp3_cm.py`
位置: `__init__()` 签名 (~line 68-75)

在签名中新增参数：
```python
            beta_kl: float = 1e-3,
            beta_recon: float = 0.5,
            use_vib: bool = False,
            use_recon: bool = False,
            latent_dim: int = None,        # 新增
            latent_state_dim: int = None,  # 新增
            eta: float = 1.0,
```

位置: encoder 创建 (~line 116-130, `encoder_type == 'dp3vib'` 分支)

传递给 encoder：
```python
        elif encoder_type == 'dp3vib':
            obs_encoder = DP3EncoderReconVIB(observation_space=obs_dict,
                                                    img_crop_shape=crop_shape,
                                                    out_channel=encoder_output_dim,
                                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                    use_pc_color=use_pc_color,
                                                    pointnet_type=pointnet_type,
                                                    backbone=backbone,
                                                    integrate_strategy=integrate_strategy,
                                                    use_agent_pos=use_agent_pos,
                                                    beta_kl=beta_kl,
                                                    beta_recon=beta_recon,
                                                    use_vib=use_vib,
                                                    use_recon=use_recon,
                                                    latent_dim=latent_dim,            # 新增
                                                    latent_state_dim=latent_state_dim, # 新增
                                                    )
```

#### Step 6: Shell script 中通过 hydra override 传入

不需要改 yaml 默认值（默认 None = 不降维，向后兼容）。实验时在 shell script 中加 override：

```bash
# 在 train_policy_chunk.sh 或 train_policy_online_cm_chunk.sh 中加：
policy.latent_dim=64 \
policy.latent_state_dim=32 \
```

---

## 验证

1. 方案 1: shell script 加 `policy.beta_kl=1e-5` override 跑实验
2. 方案 2: 跑 offline RL，观察 critic loss 是否稳定收敛，BPPO advantage 方差是否降低
3. 方案 3: 跑 BC 训练，观察 wandb 中 kl_loss 从 0 逐渐增长
4. 方案 4: 跑 BC + offline RL，对比 `latent_dim=64` vs `latent_dim=128` vs 无降维(256)
5. 组合实验：方案 2+4、方案 2+3+4 等
6. 可在 forward 中加临时 print 验证维度：`print(f"VIB output: {updated_feat.shape}, output_shape: {self.output_shape()}")`
