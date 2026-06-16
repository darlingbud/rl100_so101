# Task: 将 Conv1d AE Encoder/Decoder 架构集成到 Critic 和 Dynamics 中

## 背景

当前 critic 和 dynamics 中的 action embedding 是 `Linear(action_dim, feature_dim) + ReLU`，将 16 步 action chunk flatten 成 448 维后直接线性投影到 128 维。实验证明这种方式的 critic 排序能力等同于随机（Spearman ≈ 0）。

目标：用 Conv1d + Downsample 时序卷积架构替换 linear action encoder，加 decoder + reconstruction loss 防止 encoder 退化。Critic 和 dynamics 各自独立一份 encoder+decoder，从零训练。

## 关键参数

- `latent_cz = 32`，`Tz = 4`（H=16 时），flatten 后 = 128 维
- `hidden_dims = [128, 256]`（encoder），`[256, 128]`（decoder）
- `kernel_size = 5`，`n_groups = 8`
- `action_recon_beta = 0.5`（reconstruction loss 权重）
- 新 config flag：`use_conv_action_embed: True`

## 执行步骤

---

### Step 1: 创建 `RL-100/rl_100/model/action_ae/` 目录

创建以下文件：

#### `RL-100/rl_100/model/action_ae/__init__.py`
```python
from .action_chunk_encoder import ActionChunkEncoder
from .action_chunk_decoder import ActionChunkDecoder
```

#### `RL-100/rl_100/model/action_ae/action_chunk_encoder.py`
```python
import torch
import torch.nn as nn
from rl_100.model.diffusion.conv1d_components import Conv1dBlock, Downsample1d


class ActionChunkEncoder(nn.Module):
    """
    Temporal Conv1d encoder: action_chunk [B, H, Da] -> z [B, Tz, Cz].
    With default params and H=16: Tz=4, output flatten dim = Tz*Cz = 128.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dims: list = [128, 256],
        latent_cz: int = 32,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_cz = latent_cz

        dims = [action_dim] + hidden_dims + [latent_cz]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(Conv1dBlock(dims[i], dims[i + 1], kernel_size, n_groups))
            if i < len(dims) - 2:
                layers.append(Downsample1d(dims[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_chunk: [B, H, Da]
        Returns:
            z: [B, Tz, Cz]
        """
        x = action_chunk.transpose(1, 2)  # [B, Da, H]
        x = self.net(x)                    # [B, Cz, Tz]
        return x.transpose(1, 2)           # [B, Tz, Cz]
```

#### `RL-100/rl_100/model/action_ae/action_chunk_decoder.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_100.model.diffusion.conv1d_components import Conv1dBlock, Upsample1d


class ActionChunkDecoder(nn.Module):
    """
    Temporal Conv1d decoder: z [B, Tz, Cz] -> action_chunk_hat [B, H, Da].
    Mirrors the encoder architecture.
    """

    def __init__(
        self,
        action_dim: int,
        target_horizon: int,
        hidden_dims: list = [256, 128],
        latent_cz: int = 32,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.target_horizon = target_horizon

        dims = [latent_cz] + hidden_dims + [action_dim]
        layers = []
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layers.append(Conv1dBlock(dims[i], dims[i + 1], kernel_size, n_groups))
                layers.append(Upsample1d(dims[i + 1]))
            else:
                layers.append(
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size, padding=kernel_size // 2)
                )
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, Tz, Cz]
        Returns:
            action_chunk_hat: [B, H, Da]
        """
        x = z.transpose(1, 2)  # [B, Cz, Tz]
        x = self.net(x)        # [B, Da, ~H]
        if x.shape[-1] != self.target_horizon:
            x = F.interpolate(x, size=self.target_horizon, mode='linear', align_corners=False)
        return x.transpose(1, 2)  # [B, H, Da]
```

---

### Step 2: 修改 `RL-100/rl_100/unidpg/net.py` — QMLP 和 DoubleQMLP

在 `QMLP.__init__` 和 `DoubleQMLP.__init__` 中添加新参数和逻辑：

**新增参数**：
```python
def __init__(self, use_action_embed, obs_encoder, state_dim, feature_dim, action_dim, hidden_dim, depth,
             fix_encoder=False,
             use_conv_action_embed=False,  # 新增
             single_action_dim=None,       # 新增：单步 action 维度
             horizon=None,                 # 新增：action chunk 长度
             action_latent_cz=32,          # 新增
             ):
```

**替换 `_action_encoder` 初始化逻辑**：
```python
self.use_conv_action_embed = use_conv_action_embed
if use_action_embed:
    if use_conv_action_embed and single_action_dim is not None and horizon is not None:
        from rl_100.model.action_ae import ActionChunkEncoder, ActionChunkDecoder
        self._action_encoder = ActionChunkEncoder(
            action_dim=single_action_dim,
            hidden_dims=[128, 256],
            latent_cz=action_latent_cz,
            kernel_size=5,
            n_groups=8,
        )
        self._action_decoder = ActionChunkDecoder(
            action_dim=single_action_dim,
            target_horizon=horizon,
            hidden_dims=[256, 128],
            latent_cz=action_latent_cz,
            kernel_size=5,
            n_groups=8,
        )
        self._conv_horizon = horizon
        self._conv_single_action_dim = single_action_dim
        # flatten dim = Tz * Cz, 其中 Tz 由 encoder 动态推断
        with torch.no_grad():
            dummy = torch.zeros(1, horizon, single_action_dim)
            _tz = self._action_encoder(dummy).shape[1]
        embed_dim = _tz * action_latent_cz
        self._net = MLP((state_dim + embed_dim), hidden_dim, depth - 1, 1)
    else:
        self._action_encoder = nn.Sequential(*[nn.Linear(action_dim, feature_dim), nn.ReLU()])
        self._action_decoder = None
        if use_action_embed:
            self._net = MLP((state_dim + feature_dim), hidden_dim, depth - 1, 1)
        else:
            self._net = MLP((state_dim + action_dim), hidden_dim, depth - 1, 1)
else:
    self._action_encoder = nn.Sequential(*[nn.Linear(action_dim, feature_dim), nn.ReLU()])
    self._action_decoder = None
    self._net = MLP((state_dim + action_dim), hidden_dim, depth - 1, 1)
```

**修改 `forward` 方法**：
```python
def forward(self, s, a):
    if isinstance(s, dict):
        batch_size = s['agent_pos'].shape[0]
    else:
        batch_size = s.shape[0]
    if self._obs_encoder is not None:
        s = self._obs_encoder(s).reshape(-1, self.state_dim)
    if self.use_action_embed:
        if self.use_conv_action_embed:
            a_chunk = a.reshape(-1, self._conv_horizon, self._conv_single_action_dim)
            z = self._action_encoder(a_chunk)  # [B, Tz, Cz]
            a = z.reshape(z.shape[0], -1)      # [B, Tz*Cz]
        else:
            a = self._action_encoder(a)
    sa = torch.cat([s, a], dim=1)
    return self._net(sa)
```

**新增 `compute_recon_loss` 方法**（在 QMLP 和 DoubleQMLP 中都加）：
```python
def compute_action_recon_loss(self, a_flat):
    """计算 action reconstruction loss，用于辅助训练 action encoder。"""
    if not self.use_conv_action_embed or self._action_decoder is None:
        return torch.tensor(0.0, device=a_flat.device)
    a_chunk = a_flat.reshape(-1, self._conv_horizon, self._conv_single_action_dim)
    z = self._action_encoder(a_chunk)
    a_recon = self._action_decoder(z)
    return F.mse_loss(a_recon, a_chunk)
```

---

### Step 3: 修改 `RL-100/rl_100/unidpg/critic.py` — IQL_Q_V_no

#### 3.1 `__init__` 新增参数

```python
def __init__(self, ...,
             use_conv_action_embed=False,  # 新增
             single_action_dim=None,       # 新增
             horizon=None,                 # 新增
             action_latent_cz=32,          # 新增
             action_recon_beta=0.5,        # 新增
             ):
```

将这些参数传递给 `DoubleQMLP` / `QMLP` 的构造：
```python
if is_double_q:
    self._Q = DoubleQMLP(use_action_embed, q1_obs_encoder, state_dim, feature_dim, action_dim,
                         q_hidden_dim, q_depth, fix_encoder=fix_encoder,
                         use_conv_action_embed=use_conv_action_embed,
                         single_action_dim=single_action_dim,
                         horizon=horizon,
                         action_latent_cz=action_latent_cz)
    # target_Q 同样
```

保存 beta：
```python
self.action_recon_beta = action_recon_beta
```

#### 3.2 修改 `update` 方法

在计算完 q_loss 和 value_loss 之后，添加 recon loss：

```python
# 在现有 loss 计算之后添加：
action_recon_loss = self._Q.compute_action_recon_loss(a)
total_q_loss = q_loss + self.action_recon_beta * action_recon_loss

# 用 total_q_loss 替代原来的 q_loss 进行 backward
```

注意：recon loss 只加到 Q optimizer 的 backward 中（因为 action encoder 属于 Q 网络的参数）。

#### 3.3 返回 recon_loss 用于 logging

修改 `update` 返回值，增加 `action_recon_loss.item()`。

---

### Step 4: 修改 dynamics model

#### 4.1 `RL-100/rl_100/unidpg/transition_model/models/dynamics_model.py`

在 `EnsembleDynamicsModel.__init__` 中：

```python
if cfg.use_action_embed:
    if getattr(cfg, 'use_conv_action_embed', False):
        from rl_100.model.action_ae import ActionChunkEncoder, ActionChunkDecoder
        self._action_encoder = ActionChunkEncoder(
            action_dim=cfg.single_action_dim,
            hidden_dims=[128, 256],
            latent_cz=cfg.action_latent_cz,
            kernel_size=5,
            n_groups=8,
        )
        self._action_decoder = ActionChunkDecoder(
            action_dim=cfg.single_action_dim,
            target_horizon=cfg.n_action_steps,
            hidden_dims=[256, 128],
            latent_cz=cfg.action_latent_cz,
            kernel_size=5,
            n_groups=8,
        )
        # 动态推断 embed_dim
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.n_action_steps, cfg.single_action_dim)
            _tz = self._action_encoder(dummy).shape[1]
        embed_dim = _tz * cfg.action_latent_cz
        hidden_dims = [obs_dim + embed_dim] + list(hidden_dims)
        self.use_conv_action_embed = True
        self._conv_horizon = cfg.n_action_steps
        self._conv_single_action_dim = cfg.single_action_dim
    else:
        self._action_encoder = nn.Sequential(*[nn.Linear(action_dim, int(obs_dim)), nn.ReLU()])
        self._action_decoder = None
        hidden_dims = [obs_dim + int(obs_dim)] + list(hidden_dims)
        self.use_conv_action_embed = False
else:
    hidden_dims = [obs_dim + action_dim] + list(hidden_dims)
    self.use_conv_action_embed = False
    self._action_decoder = None
```

#### 4.2 `RL-100/rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py`

修改 `step` 和 `format_samples_for_training` 中的 action encoding：

```python
if self.cfg.use_action_embed:
    model = self.model.module if hasattr(self.model, 'module') else self.model
    if getattr(model, 'use_conv_action_embed', False):
        # [B, n_action_steps * action_dim] -> [B, n_action_steps, action_dim] -> encode -> flatten
        a_chunk = action.reshape(batch_size, model._conv_horizon, model._conv_single_action_dim)
        z = model._action_encoder(a_chunk)
        action = z.reshape(batch_size, -1)
    else:
        action = action.reshape(batch_size, -1)
        action = model._action_encoder(action)
```

#### 4.3 Dynamics 训练中加 recon loss

在 `learn` 方法中（或在 `train_cm_mid.py` 的 dynamics 训练循环中）：

```python
# 在 dynamics loss 之后：
if getattr(model, 'use_conv_action_embed', False) and model._action_decoder is not None:
    a_chunk = action.reshape(batch_size, model._conv_horizon, model._conv_single_action_dim)
    z = model._action_encoder(a_chunk)
    a_recon = model._action_decoder(z)
    recon_loss = F.mse_loss(a_recon, a_chunk)
    total_loss = dynamics_loss + 0.5 * recon_loss
```

---

### Step 5: 修改配置文件

#### `RL-100/rl_100/config/dp3_cm_epsilon.yaml`

添加：
```yaml
# Action AE encoder for critic/dynamics
use_conv_action_embed: True
action_latent_cz: 32
action_recon_beta: 0.5
```

---

### Step 6: 修改 `train_cm_mid.py`

在 `model.initialize_critic(...)` 调用处传入新参数：

```python
critic_kwargs = dict(
    ...,
    use_conv_action_embed=cfg.use_conv_action_embed,
    single_action_dim=model.action_dim,  # 单步 action dim (如 28)
    horizon=cfg.n_action_steps,          # chunk 长度 (如 16)
    action_latent_cz=cfg.action_latent_cz,
    action_recon_beta=cfg.action_recon_beta,
)
```

在 `train_dynamics(...)` 调用处，确保 cfg 中包含：
```python
cfg.single_action_dim = model.action_dim
cfg.action_latent_cz = cfg.action_latent_cz  # 已在 yaml 中
```

---

## 维度计算验证

以 `adroit_door_medium` 为例：
- `single_action_dim = 28`
- `n_action_steps = horizon = 16`
- `action_dim (flat) = 28 * 16 = 448`
- Encoder: `[B, 16, 28]` → Conv1d(28→128)+Down → `[B, 128, 8]` → Conv1d(128→256)+Down → `[B, 256, 4]` → Conv1d(256→32) → `[B, 32, 4]`
- Output: `[B, 4, 32]` → flatten → `[B, 128]`
- Q 网络输入: `cat(obs_feature[128], action_embed[128])` = `[B, 256]` → MLP → Q value

## 验证方法

1. 训练完成后，运行 ranking diagnostic 对比：
   - `use_conv_action_embed=False`（baseline）→ 预期 Spearman ≈ 0
   - `use_conv_action_embed=True`（新）→ 预期 Spearman 显著提升
2. 检查训练过程中 `action_recon_loss` 是否持续下降
3. 检查 dynamics prediction error 不退化

## 文件清单

| 文件 | 操作 |
|------|------|
| `rl_100/model/action_ae/__init__.py` | 新建 |
| `rl_100/model/action_ae/action_chunk_encoder.py` | 新建 |
| `rl_100/model/action_ae/action_chunk_decoder.py` | 新建 |
| `rl_100/unidpg/net.py` | 修改 QMLP + DoubleQMLP |
| `rl_100/unidpg/critic.py` | 修改 IQL_Q_V_no init + update |
| `rl_100/unidpg/transition_model/models/dynamics_model.py` | 修改 action encoder |
| `rl_100/unidpg/transition_model/dynamics/ensemble_dynamics_for_batch.py` | 修改 step/format_samples |
| `rl_100/config/dp3_cm_epsilon.yaml` | 添加新 config keys |
| `train_cm_mid.py` | 传递新参数 |
