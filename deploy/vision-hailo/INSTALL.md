# Hailo-8 安装指南 (Raspberry Pi 5)

本指南记录在 Raspberry Pi 5 (Raspberry Pi OS bookworm) 上安装 Hailo-8 AI HAT
的完整步骤。

## 硬件要求

- Raspberry Pi 5 (推荐 8GB 内存)
- Hailo-8 AI HAT (M.2 接口)
- M.2 HAT 或兼容的 PCIe 扩展板
- 充足电源 (推荐 5V/5A)

## 安装步骤

### 1. 系统更新

```bash
sudo apt update
sudo apt full-upgrade
```

### 2. 安装 Hailo 软件栈

Raspberry Pi OS 从 2024 年起包含 Hailo 官方仓库，可直接安装：

```bash
sudo apt install -y hailo-all hailo-models python3-hailort hailo-tappas-core
```

这个 metapackage 会安装：
- `hailort` — HailoRT 运行时库 (4.23.0+)
- `hailort-pcie-driver` — PCIe 驱动
- `hailo-models` — 预编译 HEF 模型
- `python3-hailort` — Python 绑定
- `hailo-tappas-core` — TAPPAS 应用框架核心

### 3. 验证硬件识别

```bash
hailortcli --version
hailortcli fw-control identify
```

预期输出：
```
HailoRT-CLI version 4.23.0
Executing on device: 0001:01:00.0
Board Name: Hailo-8
Device Architecture: HAILO8
```

### 4. 检查设备节点

```bash
ls /dev/hailo*
```

应显示 `/dev/hailo0`。如果没有：
1. 检查 PCIe HAT 是否正确安装
2. 运行 `sudo dmesg | grep -i hailo` 查看驱动加载情况
3. 可能需要重启 (`sudo reboot`) — **如需重启，请手动操作**

### 5. Python 环境

由于 `hailo_platform` 是系统包，Python 虚拟环境需要继承：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install pyzmq msgpack fastapi uvicorn opencv-python-headless numpy
```

或使用 uv：

```bash
uv venv --system-site-packages
source .venv/bin/activate
uv sync
```

验证 Hailo Python 绑定：

```bash
python3 -c "from hailo_platform import HEF, VDevice; print('OK')"
```

### 6. 测试推理

使用 hailo-models 中的预编译模型测试：

```bash
# 检查可用模型
ls /usr/share/hailo-models/*.hef

# 运行简单推理测试
hailortcli run /usr/share/hailo-models/scrfd_2.5g_h8l.hef
```

## 预编译模型

`hailo-models` 包提供以下 HEF：

| 模型 | 文件 | 用途 |
|------|------|------|
| SCRFD 2.5g | `scrfd_2.5g_h8l.hef` | 人脸检测 |
| YOLOv8s | `yolov8s_h8l.hef` | 通用目标检测 |
| YOLOv5 personface | `yolov5s_personface_h8l.hef` | 人脸+行人检测 |
| YOLO pose | `yolov8s_pose_h8l.hef` | 姿态估计 |

## 故障排除

### Hailo 设备未识别

```bash
# 查看 PCIe 设备
lspci | grep Hailo

# 检查驱动加载
lsmod | grep hailo

# 查看 dmesg
sudo dmesg | grep -i hailo
```

### 权限问题

```bash
# 添加用户到 hailo 组
sudo usermod -a -G hailo $USER
newgrp hailo
```

### Python 导入失败

确保虚拟环境使用 `--system-site-packages`：

```bash
# 检查是否继承系统包
python3 -c "import sys; print(sys.path)"
# 应包含 /usr/lib/python3/dist-packages
```

## 性能调优

### PCIe 性能模式

```bash
# 禁用 PCIe 电源管理
echo "performance" | sudo tee /sys/module/pcie_aspm/parameters/policy
```

### CPU 性能模式

```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## 参考资料

- [Hailo 开发者中心](https://hailo.ai/developer-zone/)
- [Raspberry Pi Hailo 官方文档](https://www.raspberrypi.com/documentation/computers/processors.html#hailo-ai)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)

---
*最后更新: 2026-04-26 (验证于 Pi 5 + Hailo-8, HailoRT 4.23.0)*