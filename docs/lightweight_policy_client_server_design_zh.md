# 轻量策略上位机—LeRobot 下位机 C/S 技术设计

> 状态：设计基线（v1）
> 目标读者：策略服务、机器人控制、系统集成开发者
> 适用范围：上位机运行项目自有策略框架，下位机使用 LeRobot 机器人接口
> 参考实现：OpenPI 的 WebSocket Policy Server / Client

## 1. 文档目标

本文定义一套可移植、依赖少、容易调试的远程策略推理架构。它将 GPU 策略环境和机器人控制环境拆开：

- 上位机负责加载模型、预处理观测、推理和后处理动作。
- 下位机负责连接 LeRobot 机器人、采集观测、请求动作并执行动作。
- 两端通过一条 WebSocket 长连接交换 MessagePack 二进制消息。
- 策略一次返回一段动作（action chunk），下位机按控制频率逐帧执行，以降低网络和推理延迟对控制周期的影响。

本文是另一个项目实现 C/S（Client/Server）的开发规格，不要求复用 OpenPI 的策略类、训练框架或机器人实现。

## 2. 术语和角色

| 术语 | 含义 |
|---|---|
| 上位机 | 有 GPU 或较强算力，运行策略框架和 WebSocket Server 的主机 |
| 下位机 | 连接真实机器人，运行 LeRobot、控制循环和 WebSocket Client 的主机 |
| Policy Adapter | 将项目自有策略框架适配为统一 `infer(observation) -> result` 接口的上位机模块 |
| Robot Adapter | 将 LeRobot 机器人接口适配为统一 `get_observation/apply_action/reset` 接口的下位机模块 |
| observation | 下位机采集的机器人状态、图像和任务信息 |
| action chunk | 策略一次预测的连续多步动作，形状通常为 `[H, D]` |
| `H` | 策略预测长度，即 action horizon |
| `D` | 单步动作维度 |
| episode | 从机器人 reset 到任务结束的一段控制过程 |

本文统一使用“C/S”：上位机是 Server，下位机是 Client。

## 3. 设计目标与非目标

### 3.1 设计目标

1. **轻量**：下位机只依赖 `websockets`、`msgpack`、`numpy` 以及自身已有的 LeRobot 环境。
2. **框架隔离**：下位机不安装上位机的 PyTorch/JAX、模型权重和训练依赖。
3. **策略解耦**：上位机通过 Policy Adapter 接入任意项目自有策略。
4. **机器人解耦**：通信层不直接依赖具体 LeRobot 机器人类。
5. **低带宽**：图像在下位机缩放并转换为 `uint8` 后再发送。
6. **延迟容忍**：通过 action chunk 降低远程推理调用频率。
7. **可观测**：响应携带推理耗时，双方记录连接、请求和控制统计。
8. **协议明确**：规定版本、字段、dtype、shape 和错误行为，避免两端依靠隐式约定。

### 3.2 非目标

v1 不试图实现：

- 通用分布式任务调度系统；
- 多模型动态装卸和模型市场；
- 跨请求自动批处理；
- 硬实时控制总线；
- 用 WebSocket 替代机器人本地急停、限位、驱动器 watchdog；
- 复杂 RPC 框架、Protobuf IDL 或服务发现。

## 4. 总体架构

```text
┌──────────────────────── 上位机 ────────────────────────┐
│ Policy Server                                          │
│                                                        │
│ WebSocketServer                                        │
│      │                                                 │
│ ProtocolCodec (MessagePack + NumPy)                    │
│      │                                                 │
│ PolicyAdapter                                          │
│      ├─ validate / map observation                     │
│      ├─ preprocess / normalize                         │
│      ├─ project policy inference                       │
│      └─ postprocess / unnormalize                      │
│                                                        │
│ 输出：actions[H,D] + timing                            │
└─────────────────────────┬──────────────────────────────┘
                          │ WebSocket 长连接
                          │ MessagePack binary frame
┌─────────────────────────┴──────────────────────────────┐
│ LeRobot 下位机                                         │
│                                                        │
│ ControlRuntime                                         │
│      ├─ RobotAdapter → LeRobot robot                   │
│      ├─ ObservationAdapter                             │
│      ├─ PolicyClient                                   │
│      ├─ ActionChunkBuffer                              │
│      └─ SafetyGuard                                    │
│                                                        │
│ 输入：机器人状态/相机                                  │
│ 输出：单步动作 → LeRobot send_action                   │
└────────────────────────────────────────────────────────┘
```

### 4.1 依赖方向

通信层只能依赖通用数据结构，不得 import 项目策略类或具体机器人类：

```text
项目策略框架 → PolicyAdapter → PolicyServer
                                   ↑
                            通用协议数据结构
                                   ↓
LeRobot Robot ← RobotAdapter ← ControlRuntime
```

这种边界允许分别升级策略框架和 LeRobot，只需维护对应 Adapter。

## 5. 推荐目录结构

另一个项目建议采用以下布局；名称可以调整，但职责不要混合：

```text
project/
├── policy_server/
│   ├── main.py                 # 上位机命令行入口
│   ├── server.py               # WebSocket 生命周期与请求循环
│   ├── policy_adapter.py       # 通用接口/协议
│   ├── project_policy.py       # 项目自有策略的具体适配
│   └── config.py               # host、port、checkpoint 等配置
├── robot_client/
│   ├── main.py                 # 下位机命令行入口
│   ├── client.py               # WebSocket 同步客户端
│   ├── runtime.py              # episode 与控制循环
│   ├── chunk_buffer.py         # action chunk 缓冲与逐步取值
│   ├── robot_adapter.py        # LeRobot 机器人适配
│   ├── observation_adapter.py  # key、图像、dtype、shape 转换
│   ├── safety.py               # 本地动作校验与失联处理
│   └── config.py               # server URI、控制频率等
├── protocol/
│   ├── codec.py                # MessagePack + NumPy 编解码
│   ├── schema.py               # 字段常量与运行时校验
│   └── version.py              # PROTOCOL_VERSION
├── tests/
│   ├── test_codec.py
│   ├── test_protocol.py
│   ├── test_chunk_buffer.py
│   ├── test_policy_adapter.py
│   └── test_loopback.py
└── docs/
    └── policy_client_server.md
```

如果希望客户端可以独立安装，可将 `robot_client/` 和 `protocol/` 打成一个小型 Python package；不要让它依赖整个策略项目。

## 6. 组件职责

### 6.1 上位机 `PolicyAdapter`

Policy Adapter 是策略项目与通信层之间唯一的业务边界，建议接口为：

```python
class PolicyAdapter(Protocol):
    @property
    def metadata(self) -> dict: ...

    def infer(self, observation: dict) -> dict: ...

    def reset(self, episode_id: str | None = None) -> None: ...
```

职责：

- 校验 observation 的 key、shape 和 dtype；
- 将线上的通用字典映射到项目策略输入；
- 在上位机完成 normalize、tokenize、device transfer；
- 调用项目策略推理接口；
- 将结果转换为物理机器人动作空间；
- 返回至少包含 `actions` 的字典；
- 维护策略需要的 episode 状态；无状态策略的 `reset` 可以为空。

通信服务不得了解模型 Tensor、checkpoint 结构或训练配置。

### 6.2 上位机 `WebSocketServer`

职责：

- 监听指定地址和端口；
- 建立连接后发送 metadata；
- 接收二进制 observation；
- 解码、调用 `PolicyAdapter.infer()`、编码响应；
- 提供 `/healthz`；
- 记录请求耗时和错误；
- 连接关闭后释放连接级状态。

v1 推荐每个客户端同一时刻最多存在一个在途请求。下位机必须先收到上一个响应，才能发送下一个请求。

### 6.3 下位机 `PolicyClient`

建议接口：

```python
class PolicyClient:
    def connect(self) -> dict: ...       # 返回 server metadata
    def infer(self, observation: dict) -> dict: ...
    def close(self) -> None: ...
```

职责：

- 根据 URI 建立 WebSocket 长连接；
- 连接后读取并校验 metadata；
- 将 observation 编码成一个 binary frame；
- 同步等待一个结果 frame；
- 区分正常二进制响应与文本错误响应；
- 提供连接和请求超时；
- 将网络异常转换为明确的客户端异常。

### 6.4 下位机 `RobotAdapter`

Robot Adapter 封装 LeRobot 版本差异。通信、缓冲和 Runtime 不直接调用具体机器人实现。

建议接口：

```python
class RobotAdapter(Protocol):
    def connect(self) -> None: ...
    def reset(self) -> None: ...
    def get_observation(self) -> dict: ...
    def apply_action(self, action: np.ndarray) -> None: ...
    def disconnect(self) -> None: ...
```

典型映射原则：

- `connect()` 内部调用当前 LeRobot 机器人对象的连接接口；
- `get_observation()` 取得 LeRobot observation，但先保留原始 key；
- `ObservationAdapter` 再将其转换为线上契约；
- `apply_action()` 把统一动作向量转换成 LeRobot 需要的动作字典或 Tensor；
- `disconnect()` 必须放在 `finally` 中执行。

不要在 `PolicyClient` 中 import LeRobot。这样可以单独运行协议单元测试，也能用假机器人测试整个控制循环。

### 6.5 下位机 `ActionChunkBuffer`

职责：

- 保存最近一次响应中的 `[H, D]` 动作；
- 每个控制周期返回下一条 `[D]` 动作；
- chunk 用尽后通知 Runtime 请求新 chunk；
- reset、异常、episode 切换时清空；
- 检查实际 chunk 长度，禁止只依赖本地配置盲目索引。

建议接口：

```python
class ActionChunkBuffer:
    def empty(self) -> bool: ...
    def put(self, actions: np.ndarray) -> None: ...
    def pop(self) -> np.ndarray: ...
    def clear(self) -> None: ...
```

## 7. 线协议 v1

### 7.1 基本规定

- 传输协议：WebSocket。
- 默认端口：`8000`。
- 默认路径：`/`。
- payload：MessagePack binary frame。
- 数组：NumPy 扩展编码。
- 每个 WebSocket frame 对应一条完整应用消息。
- 禁止 pickle。
- v1 不使用 WebSocket 压缩；图像在发送前缩放和转 `uint8`。
- 字符串统一使用 UTF-8。
- map 的业务 key 使用字符串；NumPy 扩展内部 key 可以使用 bytes。

### 7.2 NumPy 编码

`np.ndarray` 编码为：

```python
{
    b"__ndarray__": True,
    b"data": array.tobytes(order="C"),
    b"dtype": array.dtype.str,
    b"shape": array.shape,
}
```

解码时根据 `dtype`、`shape` 和 `data` 重建数组。

约束：

- 必须使用 C contiguous 数据；编码前必要时调用 `np.ascontiguousarray`。
- 禁止 object、void、complex dtype。
- 图像必须为 `uint8`。
- 状态和动作推荐为 `float32`，服务端可以接受 `float64` 后显式转换。
- 解码前必须限制 WebSocket 最大消息大小，避免任意大内存分配。

### 7.3 连接握手

WebSocket 建立后，Server 必须首先发送一条 metadata 二进制消息。Client 收到并校验成功后，连接才进入 READY 状态。

metadata 最小结构：

```python
{
    "message_type": "metadata",
    "protocol_version": 1,
    "server_name": "my-policy-server",
    "policy_name": "my-policy",
    "action_horizon": 25,
    "action_dim": 14,
    "control_hz": 50.0,
    "observation_spec": {
        "state": {"shape": [14], "dtype": "float32"},
        "images": {
            "camera.main": {"shape": [224, 224, 3], "dtype": "uint8"},
            "camera.wrist": {"shape": [224, 224, 3], "dtype": "uint8"},
        },
    },
    "action_spec": {
        "shape": [25, 14],
        "dtype": "float32",
        "representation": "absolute_joint_position",
    },
    "reset_pose": None,
}
```

规则：

- Client 必须拒绝不支持的 `protocol_version`。
- `action_horizon` 和 `action_dim` 以服务端 metadata 为准。
- 下位机配置若声明期望维度，应与 metadata 做一致性检查。
- `control_hz` 是推荐值，下位机可以覆盖，但必须记录告警。
- metadata 可以增加字段；Client 必须忽略未知字段，以便向后兼容。

### 7.4 推理请求

为保持轻量，WebSocket 连接只用于推理；v1 请求采用一个显式 envelope：

```python
{
    "message_type": "infer_request",
    "protocol_version": 1,
    "request_id": 42,
    "episode_id": "20260714-001",
    "step_id": 125,
    "client_time_ns": 1784012345678900000,
    "observation": {
        "state": np.ndarray(shape=(14,), dtype=np.float32),
        "images": {
            "camera.main": np.ndarray(shape=(224, 224, 3), dtype=np.uint8),
            "camera.wrist": np.ndarray(shape=(224, 224, 3), dtype=np.uint8),
        },
        "task": "pick up the object",
    },
}
```

字段说明：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `message_type` | 是 | 固定为 `infer_request` |
| `protocol_version` | 是 | 固定为 `1` |
| `request_id` | 是 | 当前连接内单调递增整数 |
| `episode_id` | 是 | episode 唯一标识，便于日志关联和策略 reset |
| `step_id` | 是 | 生成该观测时的控制步编号 |
| `client_time_ns` | 否 | 下位机采集/发送时间，使用 Unix time ns |
| `observation` | 是 | 策略输入字典 |

`observation` 的业务 key 必须由双方项目配置固定，不要在通信层硬编码为某个模型的命名。建议在 `observation_adapter.py` 中集中维护 LeRobot key 到协议 key 的映射。

### 7.5 推理响应

```python
{
    "message_type": "infer_response",
    "protocol_version": 1,
    "request_id": 42,
    "episode_id": "20260714-001",
    "actions": np.ndarray(shape=(25, 14), dtype=np.float32),
    "server_time_ns": 1784012345740000000,
    "timing": {
        "preprocess_ms": 3.2,
        "policy_ms": 82.4,
        "postprocess_ms": 1.1,
        "total_ms": 87.3,
    },
}
```

Client 必须校验：

- `message_type == "infer_response"`；
- 协议版本匹配；
- `request_id` 等于当前在途请求；
- `episode_id` 等于当前 episode；
- `actions` 是二维数值数组；
- `actions.shape[1] == action_dim`；
- `1 <= actions.shape[0] <= metadata.action_horizon`，或按项目约定要求严格等于；
- 所有动作值均为有限数：`np.isfinite(actions).all()`。

### 7.6 错误响应

业务错误应发送结构化二进制响应，不应把完整 traceback 直接返回下位机：

```python
{
    "message_type": "error",
    "protocol_version": 1,
    "request_id": 42,
    "code": "INVALID_OBSERVATION",
    "message": "state shape must be [14]",
    "retryable": False,
}
```

建议错误码：

| 错误码 | 是否重试 | 含义 |
|---|---:|---|
| `BAD_MESSAGE` | 否 | MessagePack 或 envelope 不合法 |
| `UNSUPPORTED_VERSION` | 否 | 协议版本不支持 |
| `INVALID_OBSERVATION` | 否 | key、shape、dtype 不符合策略输入 |
| `POLICY_NOT_READY` | 是 | 模型尚未加载完成 |
| `INFERENCE_FAILED` | 视情况 | 策略内部推理失败 |
| `SERVER_BUSY` | 是 | 服务端暂时无法接受请求 |

服务端本地日志保留 traceback；线上只返回可公开的摘要和错误码。

## 8. Observation 数据契约

### 8.1 原则

- LeRobot 原始 observation 和策略输入不一定同名，必须通过适配层映射。
- 下位机负责与带宽有关、且不依赖训练统计的处理。
- 上位机负责必须与训练保持一致的模型处理。

### 8.2 推荐处理位置

下位机处理：

- 从 LeRobot 读取最新状态和相机帧；
- 选择需要上传的字段；
- 图像颜色空间统一；
- resize with pad/crop；
- 图像转 `uint8`；
- HWC/CHW 按 metadata 约定转换；
- 状态转 NumPy，去掉框架 Tensor 依赖；
- 增加任务文本或任务 ID。

上位机处理：

- 字段重命名为项目策略输入；
- state/action normalize 和 unnormalize；
- 训练时定义的关节方向、夹爪和坐标系转换；
- tokenize、batch 维度、Tensor/device 转换；
- 策略输出到物理动作空间的后处理。

### 8.3 图像约定

双方必须在 metadata 中明确：

- 相机 key；
- shape 是 HWC 还是 CHW；
- RGB 还是 BGR；
- resize 方式；
- 缺失相机是否允许，以及填零/mask 规则。

推荐线上格式为 `uint8 HWC RGB`，因为它直观且与常见 LeRobot 相机 observation 接近。策略需要 CHW 时在上位机转换。

### 8.4 时间一致性

轻量 v1 可以使用“每路设备最新值”，但至少应：

- 记录观测生成的 `step_id`；
- 不重复发送旧图像而不自知；
- 可选记录各相机和机器人状态时间戳；
- 若相机帧过旧或状态未初始化，应跳过本次推理而不是发送空值。

## 9. Action 数据契约

必须明确以下属性：

1. `action_dim`；
2. action key 或向量各维的顺序；
3. 单位（rad、m、归一化值等）；
4. absolute position、delta position、velocity 或 torque；
5. 夹爪开合的定义和范围；
6. 控制频率；
7. action chunk 长度；
8. 合法上下限和单步最大变化。

推荐由服务端返回“已经反归一化到机器人语义空间”的动作。下位机只进行：

- dtype/shape/finite 校验；
- 机器人硬件限位；
- 速度或单步变化限制；
- 从统一向量映射为 LeRobot `send_action` 所需格式。

不要让下位机读取策略训练 normalization stats，否则会重新耦合两端环境。

## 10. 下位机控制循环

### 10.1 基本流程

```python
robot.connect()
metadata = policy_client.connect()
validate_compatibility(robot, metadata)

try:
    for episode in episodes:
        robot.reset()
        chunk.clear()
        safety.reset()

        for step_id in range(max_episode_steps):
            loop_start = monotonic()

            raw_obs = robot.get_observation()

            if chunk.empty():
                request = observation_adapter.encode(raw_obs, task, episode, step_id)
                response = policy_client.infer(request)
                chunk.put(response["actions"])

            action = chunk.pop()
            safe_action = safety.validate_and_limit(action)
            robot.apply_action(safe_action)

            rate_limiter.sleep_remaining(loop_start)
finally:
    robot.disconnect()
    policy_client.close()
```

### 10.2 action chunk 语义

假设：

```text
control_hz = 50
action_horizon = 25
```

则一次推理得到约 `25 / 50 = 0.5 s` 的动作。正常情况下每 0.5 秒请求一次上位机。

优点：

- 大幅降低请求频率和 GPU 调用次数；
- chunk 中间的控制周期不受网络抖动影响；
- 实现简单。

代价：

- chunk 执行期间是开环的；
- 上位机只看到每个 chunk 边界的观测；
- horizon 越长，响应环境变化越慢。

初始实现建议完整执行 chunk；后续如需更强闭环，可增加“执行 K 步后重新推理”的 `replan_interval`，其中 `K <= H`。

### 10.3 频率控制

- 使用 `time.monotonic()` 计算周期，不使用系统墙钟控制 sleep。
- 若本周期已超时，不额外 sleep，并增加 overrun 计数。
- 远程推理发生的周期允许超时，但必须统计 `client_infer_ms`。
- 机器人驱动自身若已有节拍，不要叠加两次固定 sleep。

## 11. 连接与状态机

建议 Client 状态机：

```text
DISCONNECTED
    │ connect
    ▼
CONNECTING
    │ WebSocket established
    ▼
WAITING_METADATA
    │ metadata validated
    ▼
READY ── infer ──> WAITING_RESPONSE
  ▲                    │
  └──── valid response ┘

任意状态发生连接错误 → DISCONNECTED → 本地安全处理
```

规则：

- 未收到 metadata 前禁止发送 observation。
- `WAITING_RESPONSE` 状态禁止发送第二个请求。
- episode 切换时清空 chunk，并更新 `episode_id`。
- 断线后旧 chunk 默认作废，不继续盲目执行。
- 自动重连只能恢复网络连接，不得自动恢复机械臂运动；重连后由 Runtime 明确开始新 episode 或进入人工确认流程。

## 12. 超时、异常与轻量安全边界

即使目标是轻量，也建议保留以下最小保护：

### 12.1 网络保护

- 连接超时，例如 10 秒；
- 单次推理超时，例如 5 秒，具体值按模型调整；
- 首次启动可每 2～5 秒重试连接；
- 运行中断线立即停止继续取旧 chunk；
- 限制单帧最大大小，例如按相机数量设置 16～64 MiB；
- 对失败次数和最后成功响应时间打日志。

### 12.2 动作保护

下位机执行动作前至少检查：

```python
actions.ndim == 1
actions.shape == (action_dim,)
np.isfinite(actions).all()
lower_limits <= actions <= upper_limits
abs(actions - previous_action) <= max_step_delta
```

超出限制时采用项目明确选择的一种行为：拒绝整个 chunk、裁剪并告警，或触发停止。真实机器人优先拒绝异常 chunk，不建议静默裁剪严重异常值。

### 12.3 失联行为

失联行为由下位机决定，因为上位机无法保证网络中断时发出停止命令。建议顺序：

1. 停止发送新动作；
2. 调用机器人项目已有的 safe stop/hold；
3. 清空 action chunk；
4. 记录 episode、step、最后动作和异常；
5. 不自动继续原 episode。

WebSocket 不是急停通道。硬件急停、关节限位和驱动器 watchdog 必须保留在机器人本地。

## 13. 并发模型

最轻量实现建议：

- 下位机使用同步 WebSocket Client；
- 一个机器人 Runtime 对应一条连接；
- 每条连接最多一个在途请求；
- 上位机可以使用 asyncio 接受连接，但模型推理应按策略框架能力串行化或放入专用执行器；
- v1 不做跨客户端 batch。

若只支持一台机器人，服务端可以显式限制为一个活跃客户端，避免两个客户端并发访问有状态策略。

注意：不要在 asyncio event loop 中直接执行耗时且会阻塞 Python 的推理工作；若项目推理调用无法让出 event loop，可用单线程 executor。单客户端原型可以先同步实现，但要在文档和日志中明确限制。

## 14. 配置设计

### 14.1 上位机配置

```yaml
server:
  host: 0.0.0.0
  port: 8000
  max_message_bytes: 33554432
  health_path: /healthz

policy:
  name: my_policy
  checkpoint: /path/to/checkpoint
  device: cuda:0
  default_task: null
```

### 14.2 下位机配置

```yaml
server:
  uri: ws://192.168.1.10:8000
  connect_timeout_s: 10
  infer_timeout_s: 5
  retry_interval_s: 3

runtime:
  control_hz: 50
  max_episode_steps: 1000
  replan_interval: null       # null 表示完整执行服务端 chunk

observation:
  image_size: [224, 224]
  image_layout: HWC
  color_space: RGB
  camera_map:
    observation.images.main: camera.main
    observation.images.wrist: camera.wrist

safety:
  reject_non_finite: true
  stop_on_disconnect: true
```

命令行参数可以覆盖配置文件，但启动时必须打印最终生效配置，敏感字段除外。

## 15. 日志与指标

### 15.1 上位机日志

每次请求至少记录：

- remote address；
- `episode_id`、`request_id`、`step_id`；
- 输入校验结果；
- preprocess、policy、postprocess、total 耗时；
- actions shape；
- 错误码。

不要默认打印完整图像、状态或动作数组。

### 15.2 下位机日志

至少记录：

- 连接、重试、断开；
- 服务端 metadata；
- 每次远程推理往返耗时；
- chunk 长度和消费进度；
- 控制周期 overrun；
- 动作保护触发原因；
- episode 开始、结束和异常退出。

推荐统计：

- `client_round_trip_ms`；
- `server_total_ms`；
- `network_and_codec_ms ≈ round_trip - server_total`；
- `control_loop_ms`；
- `control_overrun_count`；
- `reconnect_count`。

## 16. 健康检查和启动顺序

服务端提供：

```http
GET /healthz
200 OK
```

`/healthz` 至少表示进程和 WebSocket 服务已启动。如果模型加载较慢，可扩展 `/readyz` 表示策略已经可推理，但这不是 v1 必需项。

推荐启动顺序：

1. 启动上位机，加载策略；
2. 检查 `/healthz`；
3. 启动 LeRobot 机器人驱动或服务；
4. 启动下位机 Client；
5. Client 校验 metadata；
6. 人工确认后进入 episode；
7. episode 结束后清空 chunk 并安全复位/停止。

## 17. 测试计划

### 17.1 编解码单元测试

- 各种 shape 的 `float32` 数组往返一致；
- `uint8` 图像往返一致；
- NumPy scalar 往返一致；
- object/complex 数组被拒绝；
- 非连续数组编码后数值正确；
- 超大消息被拒绝。

### 17.2 协议测试

- metadata 版本匹配/不匹配；
- request/response ID 匹配；
- observation 缺 key、shape 错、dtype 错；
- action shape 错和 NaN/Inf；
- 服务端返回结构化 error；
- Client 在 READY 之外不能 infer。

### 17.3 Chunk 测试

- `[H,D]` 放入后恰好 pop H 次；
- 空 buffer pop 报明确异常；
- reset 清空；
- 实际 H 小于 metadata 时按约定拒绝或安全处理；
- chunk 中非数组辅助字段不会被误切片。

### 17.4 Loopback 集成测试

使用假策略和假机器人：

```text
FakeRobot → Client → localhost Server → FakePolicy
    ↑                                      │
    └──────────── action chunk ────────────┘
```

验证：

- 握手和首次推理；
- 连续执行多个 chunk；
- episode reset；
- 服务端延迟；
- 服务端异常；
- 推理中断线；
- 下位机进入 safe stop 且不继续执行旧动作。

### 17.5 真机上线前检查

- 首先禁用电机或抬起机器人测试数据链路；
- 打印并核对 state/action 每维语义；
- 用固定动作验证方向、单位和夹爪范围；
- 将控制频率降到安全值测试；
- 验证网络断开后的本地停止；
- 验证 Ctrl-C 和异常退出都会 disconnect/stop；
- 最后再启用完整频率和策略输出。

## 18. 最小实现里程碑

### M1：协议与假数据

- 完成 `codec.py`；
- 完成 metadata、request、response 校验；
- FakePolicy Server 和简单 Client 可通信；
- 完成 codec/protocol 测试。

### M2：接入项目策略

- 实现 `ProjectPolicyAdapter`；
- 固定 observation/action spec；
- 完成预处理、归一化和后处理；
- 使用录制 observation 在上位机离线验证动作结果。

### M3：接入 LeRobot

- 实现 `RobotAdapter`；
- 确定 LeRobot observation key 映射；
- 完成图像预处理；
- 完成 action 到机器人接口的映射；
- 使用 FakePolicy 做低速闭环测试。

### M4：action chunk 与异常处理

- 接入 `ActionChunkBuffer`；
- 加入请求超时、断线停止和动作校验；
- 完成 loopback/断线测试；
- 再连接真实策略并逐步提高控制频率。

## 19. 关键设计决策

### 19.1 为什么使用 WebSocket

- Python 两端实现短小；
- 一条 TCP 长连接即可双向传输；
- 支持 binary frame；
- 容易经过现有网络设施；
- 比自定义 TCP 分包更少样板代码。

### 19.2 为什么使用 MessagePack

- 比 JSON 更适合图像和数组；
- 无需生成 IDL；
- 多语言可实现；
- 不执行任意反序列化代码；
- 保持协议足够轻量。

### 19.3 为什么 normalize 放在上位机

- normalization stats 属于 checkpoint/训练配置；
- 更换模型时下位机无需升级统计参数；
- 避免两份预处理实现漂移；
- 下位机保持只理解机器人语义。

### 19.4 为什么图片预缩放放在下位机

- 原始多路相机图像占用大量带宽；
- resize 和 `uint8` 转换不需要 GPU；
- 减少序列化、传输和服务端内存压力；
- 图像几何规则仍需由 metadata/项目配置固定，保证与训练一致。

### 19.5 为什么返回 action chunk

- 远程模型推理通常远慢于机器人控制周期；
- 每周期请求会导致控制频率等于网络加推理频率；
- chunk 允许本地连续执行，同时保持通信实现简单。

## 20. 与 OpenPI 参考实现的关系

本文保留的设计：

- WebSocket 长连接；
- MessagePack + NumPy；
- Server 首帧发送 metadata；
- Client 同步 `infer`；
- 上位机完成策略转换和归一化；
- 下位机图像预缩放；
- action chunk 本地逐步执行；
- `/healthz` 和推理耗时。

为另一个项目落地时建议补充、但仍保持轻量的内容：

- `protocol_version`；
- `message_type`、`request_id`、`episode_id`；
- observation/action spec；
- 连接和推理超时；
- 最大消息限制；
- 结构化错误；
- 动作有限值和维度检查；
- 下位机断线停止。

这些字段不会引入重型框架，却能显著减少联调时的隐式不一致。

## 21. 实施前必须由新项目确定的清单

开始编码前，应填写并冻结以下内容：

```text
[ ] 项目策略加载入口和 infer 接口
[ ] 策略是否有 episode 状态，何时 reset
[ ] LeRobot 版本/commit
[ ] 具体机器人类及 connect/disconnect/reset/send_action 接口
[ ] LeRobot observation 原始 key、shape、dtype
[ ] 上传到策略的 observation key 映射
[ ] 相机名称、颜色空间、layout、目标尺寸和 resize 规则
[ ] state 各维顺序、单位和范围
[ ] action 各维顺序、单位、表示形式和范围
[ ] action_dim 和模型 action_horizon
[ ] control_hz 和 replan_interval
[ ] task/prompt 的来源
[ ] normalize/unnormalize 所在位置
[ ] 网络地址、端口、超时和最大消息大小
[ ] 断线、超时、NaN、越界动作的处理策略
[ ] 真机 safe stop/hold 接口
```

这份清单完成后，协议层、策略 Adapter 和 LeRobot Adapter 可以并行实现，并通过 FakePolicy/FakeRobot 在接触真机前完成大部分联调。
