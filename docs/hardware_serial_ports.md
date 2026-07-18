# Leader/Follower 串口设备固定映射

本文记录 Leader 和 Follower 硬件的串口身份，以及在 Ubuntu/Linux 上通过 udev 创建固定设备链接的方法。

## 设备信息

`/dev/ttyACM0`、`/dev/ttyACM1` 由系统按照设备发现顺序分配，重启或重新插拔后可能互换，程序中不应依赖这两个名称。

当前识别结果：

| 角色 | 当前临时端口 | USB VID:PID | USB 序列号 | 固定链接 |
| --- | --- | --- | --- | --- |
| Follower | `/dev/ttyACM0` | `1a86:55d3` | `5B42134696` | `/dev/follower` |
| Leader | `/dev/ttyACM1` | `1a86:55d3` | `5B42134767` | `/dev/leader` |

两个设备也有系统自动生成的稳定链接：

```text
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B42134696-if00  # Follower
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B42134767-if00  # Leader
```

## 安装固定链接规则

创建规则文件：

```bash
sudo tee /etc/udev/rules.d/99-robot-serial.rules >/dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B42134696", SYMLINK+="follower", GROUP="dialout", MODE="0660"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B42134767", SYMLINK+="leader", GROUP="dialout", MODE="0660"
EOF
```

重新加载并触发 udev 规则：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

如果链接没有立即出现，拔出并重新插入两个 USB 设备。

## 验证映射

```bash
ls -l /dev/follower /dev/leader
readlink -f /dev/follower
readlink -f /dev/leader
```

预期结果是 `/dev/follower` 指向序列号 `5B42134696` 的设备，`/dev/leader` 指向序列号 `5B42134767` 的设备。它们实际指向 `ttyACM0` 还是 `ttyACM1` 并不重要。

检查设备的 udev 属性：

```bash
udevadm info --query=property --name=/dev/follower | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL'
udevadm info --query=property --name=/dev/leader   | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL'
```

## 串口访问权限

设备设置为 `root:dialout`、权限 `0660`。检查当前用户是否属于 `dialout` 组：

```bash
id -nG | tr ' ' '\n' | grep '^dialout$'
```

如果没有输出，执行：

```bash
sudo usermod -aG dialout "$USER"
```

随后注销并重新登录，使组权限生效。

## 程序中的端口配置

程序应使用固定名称：

```text
Follower: /dev/follower
Leader:   /dev/leader
```

不要把 `/dev/ttyACM0` 或 `/dev/ttyACM1` 写死到配置或代码中。

## 摄像头 ID 与画面预览

进入安装了 LeRobot 的 Python/conda 环境后，查找 OpenCV 摄像头：

```bash
lerobot-find-cameras opencv
```

命令输出中的 `Id` 是 LeRobot/OpenCV 使用的摄像头 ID。例如 `Id: 0` 对应配置中的 `index_or_path: 0`；在 Linux 上通常对应 `/dev/video0`。该命令还会在 `outputs/captured_images` 中保存各摄像头的测试帧。

当前机器检测到的图像采集节点：

```text
/dev/video0  # 第一个摄像头的视频流
/dev/video2  # 第二个摄像头的视频流
```

`/dev/video1` 和 `/dev/video3` 不是正常图像采集节点，不用于预览。

使用低延迟模式预览第一个摄像头：

```bash
ffplay \
  -f v4l2 \
  -input_format mjpeg \
  -video_size 640x480 \
  -framerate 30 \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  -i /dev/video0
```

预览第二个摄像头时，把输入设备改为 `/dev/video2`：

```bash
ffplay \
  -f v4l2 \
  -input_format mjpeg \
  -video_size 640x480 \
  -framerate 30 \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  -i /dev/video2
```

按 `q` 退出预览。如果没有安装 `ffplay`，执行：

```bash
sudo apt update
sudo apt install ffmpeg
```

为减小延迟和 USB 带宽占用，LeRobot 摄像头配置建议使用 `640x480`、`30 FPS` 和 `MJPG`：

```text
{type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}
```

## 排障命令

列出 ACM 串口和稳定链接：

```bash
ls -l /dev/ttyACM* /dev/serial/by-id/* 2>/dev/null
```

查看内核最近识别到的串口设备：

```bash
sudo dmesg --ctime | grep -E 'ttyACM|cdc_acm' | tail -n 30
```

实时观察设备插拔事件：

```bash
udevadm monitor --udev --property --subsystem-match=tty
```

查看规则文件：

```bash
cat /etc/udev/rules.d/99-robot-serial.rules
```

## 删除固定规则

如需撤销自定义链接：

```bash
sudo rm /etc/udev/rules.d/99-robot-serial.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

删除规则不会影响 `/dev/ttyACM*` 或系统自带的 `/dev/serial/by-id/*` 链接。
