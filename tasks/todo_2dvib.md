# 2D Policy Recon + VIB 崩塌修复方案

## 现象

消融实验发现：
- `3D policy`: `recon + vib` 对 online finetune 有明显加成
- `2D policy`: `recon + vib` 在 diffusion BC 训练中基本崩塌，几乎没有 performance

---

## 先说结论

2D 崩塌的主要原因不是“2D 天生不适合 VIB”，而是当前 2D 实现里同时存在几类错配：

1. `train / test` 的 encoder 特征分布不一致
2. reconstruction 的输入视图和目标视图不一致
3. reconstruction target 的数值范围和 decoder 输出范围不一致
4. 2D reconstruction 学的是整张图像像素，和动作控制目标天然更容易冲突
5. 当前 2D VIB 没有真正 bottleneck，只是在同维度上加噪声和 KL

这些问题叠在一起时，diffusion BC 学到的 observation conditioning 在训练和推理阶段发生偏移，因此很容易直接崩掉。

---

## 当前代码里的根因

### 根因 1: `forward()` 和 `Recon_VIB_loss()` 的特征分布不一致

#### 3D encoder 是正确的

`RL-100/rl_100/model/vision/pointnet_extractor.py` 中的 `DP3EncoderReconVIB`：
- `forward()` 经过 VIB head
- train 时采样
- eval 时用 `mu`
- 还能通过 `force_stochastic` 在 eval 下强制采样

也就是说，3D 的训练和推理都走同一条 latent 路径，只是 stochastic / deterministic 不同。

#### 2D encoder 是错误的

`RL-100/rl_100/model/vision/multi_image_obs_encoder.py` 中：
- `forward()` 直接返回 CNN feature
- `Recon_VIB_loss()` 才会走 `mu/logvar` 和 reparameterize

于是 2D policy 中出现下面的错配：

| 阶段 | 调用方法 | 特征来源 |
|------|---------|---------|
| BC 训练 | `Recon_VIB_loss()` | VIB 后的 feature |
| 推理 / eval | `forward()` | 原始 CNN feature |

对 diffusion policy 来说，这基本等价于：
- 训练时学了一套 condition distribution
- 推理时喂了另一套分布

这是最核心的崩塌原因。

---

### 根因 2: reconstruction 的输入 crop 和 target crop 不一致

`multi_image_obs_encoder.py` 中当前逻辑是：
- encoder 输入图像走 `self._apply_transform(..., deterministic=deterministic)`
- reconstruction target 强行用 `self._apply_transform(..., deterministic=True)`

而 2D config 默认：
- `random_crop: True`
- `use_aug: True`

这意味着训练时 encoder 看到的是随机 crop，decoder 却被要求重建中心 crop。

也就是说：
- latent 表示的是 patch A
- reconstruction target 却是 patch B

这个 loss 本身就是冲突目标，会强行把 encoder 往错误方向拉。

3D 没这个问题，因为它重建的是同一个 point cloud / state。

---

### 根因 3: target 图像范围与 decoder 输出范围错配

`multi_image_obs_encoder.py` 里同时存在：
- `imagenet_norm=True`
- decoder 末端 `Sigmoid()`

问题是：
- ImageNet normalization 后的图像不是 `[0, 1]`
- `Sigmoid()` 的输出一定在 `[0, 1]`

于是现在的 reconstruction loss 实际在做：
- 用 `[0, 1]` 的预测
- 拟合一个已经标准化到负值和大于 1 的 target

这会让重建 loss 本身长期处于错误标尺上，持续给 encoder 注入坏梯度。

---

### 根因 4: 2D reconstruction 目标本身更容易伤害控制特征

3D 的 reconstruction 是：
- point cloud chamfer distance
- `agent_pos` MSE

它们和控制相关性很强。

2D 的 reconstruction 是整张图像像素 MSE：
- 背景
- 光照
- 纹理
- 无关外观细节

都会被 encoder 强行保留。

对 offline diffusion BC 而言，这种像素级重建目标更容易和 action prediction 争夺表示能力。

---

### 根因 5: 2D VIB 不是“真正的 bottleneck”

当前 2D VIB head 是：

```python
mu: Linear(D, D)
logvar: Linear(D, D)
```

输入输出同维度，没有显式降维。

所以它更像：
- distribution regularization
- feature noise injection

而不是：
- filtering nuisance information
- learning a compact task-relevant latent

这和你在 3D `todo_vib.md` 里总结的老问题一致。

---

## 必修复项

下面这几项建议按顺序做，前 4 项属于必须修。

---

### Fix 1: 让 2D `forward()` 也经过 VIB head

文件：
- `RL-100/rl_100/model/vision/multi_image_obs_encoder.py`

#### 目标

让 2D encoder 和 3D encoder 的行为对齐：
- train: `mu + eps * std`
- eval: `mu`
- 支持 `force_stochastic`

#### 修改步骤

1. 在 `__init__()` 里新增：

```python
self.force_stochastic = False
self.beta_kl = kl_beta
```

说明：
- `force_stochastic` 对齐 3D encoder
- `beta_kl` 作为 `kl_beta` 的兼容别名，方便训练脚本统一做 annealing

2. 在 `forward()` 的 `share_rgb_model=True` 分支里，`feature = self.key_model_map['rgb'](imgs)` 之后加入：

```python
if self.use_vib:
    mu = self.vib_head['mu'](feature)
    logvar = self.vib_head['logvar'](feature)
    if self.training or self.force_stochastic:
        std = torch.exp(0.5 * logvar)
        feature = mu + std * torch.randn_like(std)
    else:
        feature = mu
```

3. 在 `forward()` 的 `share_rgb_model=False` 分支里，对每个 image key 同样加入 VIB 路径：

```python
if self.use_vib:
    mu = self.vib_heads[key]['mu'](feature)
    logvar = self.vib_heads[key]['logvar'](feature)
    if self.training or self.force_stochastic:
        std = torch.exp(0.5 * logvar)
        feature = mu + std * torch.randn_like(std)
    else:
        feature = mu
```

#### 预期结果

修完后：
- BC 训练用的 feature
- 推理时用的 feature

都会经过同一条 VIB latent 路径，只是 stochastic / deterministic 模式不同。

这是最关键的修复。

---

### Fix 2: 修复 reconstruction target 的 crop 不一致

文件：
- `RL-100/rl_100/model/vision/multi_image_obs_encoder.py`

#### 问题

当前逻辑中：
- feature 使用随机 crop
- target 使用中心 crop

这个必须改成同一视图。

#### 修改原则

decoder 应该重建“和 encoder 实际看到的那张图”。

#### 推荐改法

在 `Recon_VIB_loss()` 里：
- 不要再用 `deterministic=True` 单独构造 `raw_img`
- 直接把经过同一份 transform 后的 `img` 作为 reconstruction target

即把类似下面的逻辑：

```python
raw_img = self._apply_transform(key, img.clone(), deterministic=True)
img = self._apply_transform(key, img, deterministic=deterministic)
```

改成同源版本，例如：

```python
img = self._apply_transform(key, img, deterministic=deterministic)
target_img = img.detach()
```

然后 reconstruction loss 改为拟合 `target_img`。

#### 注意

如果后续想保留”稳定 target”，也应该先解决 transform 同步问题，而不是直接用中心 crop 替代。

#### ⚠️ 实现提醒：Fix 2 和 Fix 3 在代码中是同一处修改

Fix 2（crop 同源）和 Fix 3（值域对齐）虽然逻辑上是两个问题，但改的是 `Recon_VIB_loss()` 中同一段 transform + target 构造代码。建议实现时一起处理：

1. 把 transform 拆成两步：`resize + crop` 和 `normalize`
2. encoder 输入 = crop 后 + normalize 后的图像
3. recon target = crop 后、normalize 前的图像（同一份 crop，`[0,1]` 值域）

这样一次改动同时解决 crop 不一致和值域错配。

---

### Fix 3: 修复 reconstruction target 的值域错配

文件：
- `RL-100/rl_100/model/vision/multi_image_obs_encoder.py`

#### 问题

当前组合是：
- 输入 / target: ImageNet normalized
- decoder 输出: `Sigmoid()` in `[0, 1]`

这是错配的。

#### 两种可选修法

#### 方案 A: reconstruction 分支不使用 ImageNet-normalized target

推荐做法：
- encoder 主干仍然可以吃 normalized image
- 但 reconstruction target 使用未归一化到 `[0, 1]` 的图像

这通常需要把 transform 拆开：
- resize / crop
- normalization

然后：
- backbone 输入用 normalized image
- recon target 用 pre-normalization image

#### 方案 B: 保持 normalized target，但去掉 decoder 末端 `Sigmoid()`

不太推荐，因为会让 decoder 直接拟合标准化值域，训练通常更不稳定。

#### 推荐结论

优先选方案 A。

---

### Fix 4: 修复 `dp_image_unet.py` 中 VIB / recon 的 gate 逻辑

文件：
- `RL-100/rl_100/policy/dp_image_unet.py`

#### 当前问题

`compute_loss()` 里现在用的是：

```python
if self.use_recon:
    vib_recon_loss, loss_items, nobs_features = self.obs_encoder.Recon_VIB_loss(this_nobs)
else:
    nobs_features = self.obs_encoder(this_nobs)
```

这会导致：
- `use_vib=True, use_recon=False` 时
- 完全不会走 `Recon_VIB_loss()`
- KL loss 也不会进总 loss

#### 修改步骤

把 gate 改为：

```python
need_aux_loss = self.use_recon or getattr(self.obs_encoder, 'use_vib', False)
if need_aux_loss:
    vib_recon_loss, loss_items, nobs_features = self.obs_encoder.Recon_VIB_loss(this_nobs)
else:
    nobs_features = self.obs_encoder(this_nobs)
```

同样修改：
- `obs_as_global_cond=True` 分支
- `obs_as_global_cond=False` 分支

并把 loss 加法从：

```python
if self.use_recon:
    loss += vib_recon_loss
```

改为：

```python
if need_aux_loss:
    loss += vib_recon_loss
```

#### 日志也一起修

当前 `kl_loss` 日志被注释掉了，建议统一改成：

```python
loss_dict = {
    'bc_loss': loss.item(),
    'kl_loss': loss_items.get('kl_loss', 0.0),
    'recon_loss': loss_items.get('recon_loss', 0.0),
}
```

---

### Fix 5: 让 KL annealing 能作用到 2D encoder

文件：
- `RL-100/rl_100/model/vision/multi_image_obs_encoder.py`

#### 当前问题

3D encoder 用的是：
- `self.beta_kl`

2D encoder 用的是：
- `self.kl_beta`

训练脚本里通常是检查 `beta_kl` 去做 annealing，因此 2D 这边可能根本没有被 anneal 到。

#### 修改步骤

1. `__init__()` 中保存：

```python
self.beta_kl = kl_beta
```

2. `Recon_VIB_loss()` 中把：

```python
vib_loss = torch.stack(kl_losses).mean() * self.kl_loss_weight * self.kl_beta
```

改为：

```python
vib_loss = torch.stack(kl_losses).mean() * self.kl_loss_weight * self.beta_kl
```

3. 如果还保留 `self.kl_beta`，建议同步：

```python
self.kl_beta = kl_beta
self.beta_kl = kl_beta
```

这样外部无论改哪个字段都不会产生歧义。

---

## 可选增强项

---

### Fix 6: 给 2D VIB 加真正的 bottleneck 降维

文件：
- `RL-100/rl_100/model/vision/multi_image_obs_encoder.py`
- 对应 config 文件

#### 当前问题

现在 `mu/logvar` 是同维映射，不是真 bottleneck。

#### 修改步骤

1. `__init__()` 新增参数：

```python
latent_dim: int = None
```

2. 建 VIB head 时改成：

```python
img_feat_dim = self._last_feature_dims[key]
img_latent_dim = latent_dim if latent_dim is not None else img_feat_dim
mu = nn.Linear(img_feat_dim, img_latent_dim)
logvar = nn.Linear(img_feat_dim, img_latent_dim)
```

3. decoder 输入维度也改成 latent 维度

4. `output_shape()` 要返回 VIB 后的实际维度

#### 说明

如果已经完成 Fix 1，并且 `forward()` 本身走 VIB head，那么 `output_shape()` 通常会自动正确。

---

### Fix 7: 给 2D encoder 也加 `force_stochastic` 的 online 控制

文件：
- `RL-100/rl_100/model/vision/multi_image_obs_encoder.py`
- 训练脚本中 online finetune 入口

#### 目标

和 3D 保持一致：
- BC / offline 阶段 eval 用 `mu`
- online finetune 阶段在 eval 下也可以强制采样

如果 2D 后续也要做 online 路线，这一步建议一并补齐。

---

## 修改文件清单

| 文件 | 修改项 |
|------|--------|
| `RL-100/rl_100/model/vision/multi_image_obs_encoder.py` | Fix 1, Fix 2, Fix 3, Fix 5, Fix 6, Fix 7 |
| `RL-100/rl_100/policy/dp_image_unet.py` | Fix 4 |
| `RL-100/rl_100/config/dp_image_unet*.yaml` | 如启用 Fix 6，需要新增 `latent_dim`；可做减小 `kl_beta` 的消融 |

---

## 第二阶段多卡模式说明

`train_ddp.py` 的第二阶段（`finetune_dp3`）目前支持两类多卡使用方式：

1. 标准 DDP：
   - 同一组参数只启动一次
   - 多张卡共同参与同一个 offline finetune 任务
   - batch size 按 `world_size` 切分，梯度同步后更新同一套参数

2. 多进程 sweep：
   - 每张卡独立启动一组参数
   - 每个进程各自做 offline finetune
   - 多组参数并行扫参
   - 可以共享同一套 stage1 产物，并竞争更新全局 `best`

对应理解可以简化为：
- `BPPO_MODE=False`：多卡共同 finetune 一组参数
- `BPPO_MODE=True`：每张卡独立 finetune 一组参数

需要注意：
- 这两种能力不是 `train_ddp.py` 单文件内部自动切换出来的抽象层，而是 `train_ddp.py + launcher bash` 组合后的两种运行方式
- 如果使用并行 sweep 并共享全局 `best`，就必须保证全局 `best` 更新过程有锁，否则并发写入会不可靠

---

## 推荐的最小修复顺序

不要一上来全部改完再测，建议按下面顺序：

1. 先做 Fix 1
2. 再做 Fix 2
3. 再做 Fix 3
4. 再做 Fix 4
5. 再做 Fix 5
6. 确认不崩后，再做 Fix 6

原因：
- Fix 1 到 Fix 4 是功能正确性问题
- Fix 5 是训练策略兼容问题
- Fix 6 是增强项，不是最先级最高的 bug fix

---

## 第一轮 Review

下面是对 work agent 当前实现的第一轮 review 结论。先修这里，再继续跑实验。

### Finding 1: `use_vib=True` 时 encoder 初始化会直接失败

严重级别：
- High

文件：
- `RL-100/rl_100/model/vision/multi_image_obs_encoder.py`

问题描述：
- 当前实现在 `__init__()` 中先设置了 `self.use_vib = use_vib`
- 然后立刻调用 `self.output_shape()`
- 而 `output_shape()` 会调用 `forward()`
- 新版 `forward()` 在 `self.use_vib=True` 时会访问 `self.vib_head` / `self.vib_heads`
- 但这两个模块是在 `self.output_shape()` 之后才创建的

这会导致：
- 只要 `use_vib=True`
- encoder 在构造阶段就可能因为找不到 `self.vib_head` 或 `self.vib_heads` 而报错

需要修复的位置：
- 初始化顺序

推荐修法：

方案 A：
- 在创建 `vib_head` 之前，`output_shape()` 不要走 VIB 路径
- 例如在 `forward()` 中加保护：只有 `hasattr(self, 'vib_head')` / `hasattr(self, 'vib_heads')` 时才进 VIB

方案 B：
- 先用 backbone feature dim 建好 VIB heads
- 再调用 `output_shape()`

推荐优先：
- 方案 A，更稳，改动更小

work agent 下一步动作：
- 修这个问题后，至少本地验证一次 `use_vib=True` 能成功实例化 policy / encoder

---

### Finding 2: 训练脚本把 KL override 打到了错误的 config key

严重级别：
- Medium

文件：
- `scripts/train_policy_chunk.sh`
- `RL-100/rl_100/config/dp_image_unet*.yaml`

问题描述：
- 当前脚本新增了 `policy.beta_kl=1e-3`
- 但 2D image policy 的 encoder 配置项实际在 `policy.obs_encoder.kl_beta`

如果没有额外 config 桥接：
- 这个 override 可能根本不会作用到 2D encoder
- 或者 Hydra 直接报 unknown key

这样会产生两个风险：
- 你以为已经控制了 KL，实际没有
- 你以为 KL annealing 生效了，实际没有

需要修复的位置：
- shell script 中的 hydra override

推荐修法：
- 如果当前实现保留 `kl_beta` 为主字段，就把 override 改成：

```bash
policy.obs_encoder.kl_beta=1e-3
```

- 如果代码里已经统一切到 `beta_kl`，那就必须同步把 config schema 也改掉，并确认 Hydra 路径存在

work agent 下一步动作：
- 明确 2D 路径最终用哪个字段名
- 脚本、config、代码三处统一

---

### Finding 3: 训练脚本同时改了训练 regime，会污染这轮消融结论

严重级别：
- Medium

文件：
- `scripts/train_policy_chunk.sh`

问题描述：
- 当前脚本除了开 `recon + vib` 外，还把：
- `dynamics_type` 从 `mlp` 改成了 `diffusion`
- `training.num_epochs` 从 `300` 改成了 `400`
- `training.num_critic_epochs` 从 `10` 改成了 `400`
- `dynamics.dynamics_max_epochs` 从 `10` 改成了 `400`

这会导致后续实验结果无法解释：
- 即使性能改善，也不知道是 2D VIB 修复带来的
- 即使性能变差，也不知道是脚本训练 regime 改坏了

这类改动不应该混进“第一轮修复 2D recon+vib 崩塌”里。

推荐修法：
- 第一轮修复只保留和 2D recon / vib 直接相关的 override
- 其它训练策略改动全部回退
- 等功能正确后，再单独做第二轮超参 / 训练 regime 消融

work agent 下一步动作：
- 回退无关的脚本变更
- 保证第一轮实验只比较实现修复前后

---

### 第一轮 Review 通过标准

work agent 修完后，至少满足下面几点再进入下一轮：

1. `use_vib=True` 时 policy 可以成功构造，不会在 encoder 初始化阶段报错
2. KL 相关 override 确实能命中 2D encoder 的真实 config 路径
3. 训练脚本没有混入无关的 regime 改动
4. `python -m py_compile` 继续通过

---

## 验证计划

### 验证 1: train / test feature consistency

目标：
- 确认训练和推理都经过 VIB head

检查方式：
- 在 `forward()` 和 `Recon_VIB_loss()` 中临时打印一次 feature 来源
- 或者打印 `mu.mean()` / `feature.mean()` 的统计

---

### 验证 2: `use_vib=True, use_recon=False`

目标：
- 单独验证 VIB 修完后是否仍崩

期望：
- 不应再出现“训练时用 VIB latent，推理时回退原始 CNN feature”的错配

---

### 验证 3: `use_recon=True, use_vib=False`

目标：
- 单独验证 reconstruction 修完后是否稳定

重点看：
- recon loss 是否正常下降
- BC performance 是否仍保持

---

### 验证 4: `use_recon=True, use_vib=True`

目标：
- 验证组合项是否从“直接崩掉”变为“可训练”

---

### 验证 5: 日志监控

wandb 中至少确认以下曲线存在且合理：
- `bc_loss`
- `kl_loss`
- `recon_loss`

如果 `kl_loss` 一直极大或不下降，先减小 `beta_kl`。

---

### 验证 6: 关闭 augmentation 的排查实验

建议临时跑一个最小对照：
- `use_aug=False`
- `random_crop=False`
- `use_vib=True`
- `use_recon=True`

如果这时不再崩，说明 augmentation / crop mismatch 是主要放大器。

---

### 验证 7: bottleneck 消融

在功能修正确认后再做：
- `latent_dim = 128`
- `latent_dim = 64`

和同维 VIB 做对比。

---

## 超参建议

即使修完实现，2D 上也建议先从更弱的 KL 开始：

```bash
policy.obs_encoder.kl_beta=1e-4
policy.obs_encoder.kl_beta=1e-5
```

或如果统一到 `beta_kl`：

```bash
policy.obs_encoder.beta_kl=1e-4
policy.obs_encoder.beta_kl=1e-5
```

原因：
- 2D 图像表征比 3D 几何表征包含更多 nuisance
- KL 稍大时更容易直接压坏 action-relevant feature

---

## 一句话总结

2D recon + VIB 崩塌的核心不是“VIB 没用”，而是当前实现把：
- 不同的 latent 路径
- 不同的 crop 视图
- 不同的值域

混在一起训练了。先把这几处实现错配修平，再讨论 2D 上 VIB 是否真的带来收益。
