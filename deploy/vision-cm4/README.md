# vision-cm4 — CPU-only Vision Service for Raspberry Pi CM4

MediaPipe 人脸检测 + ONNX 情绪识别（跳帧），无需 GPU/NPU。

作为 `vision-trt`（Jetson CUDA）和 `vision-hailo`（Hailo-8）的替代方案，
与 reachy-claw 和 dashboard 完全协议兼容。

## 性能预期

| 硬件 | 人脸检测 FPS | 情绪识别 | 备注 |
|-----|------------|---------|------|
| CM4 (4GB) | 8-12 FPS | 跳帧（每 5 帧） | 无 GPU/NPU |
| Pi 5 | 12-15 FPS | 跳帧 | 略快于 CM4 |

**跳帧策略**：情绪识别每 N 帧（默认 5）运行一次，期间沿用缓存的上一帧结果。
这大幅降低 CPU 负载，同时保持情绪响应的实时性。

## 快速开始

### 1. 本地运行（测试）

```bash
cd deploy/vision-cm4

# 创建虚拟环境（继承系统包）
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .

# 或用 uv:
uv venv --system-site-packages
uv sync

# 启动服务
uv run python producer.py
```

预期输出：
```
[vision-cm4] ZMQ PUB on tcp://0.0.0.0:8631
[vision-cm4] Camera /dev/video0 open at 640x480
[vision-cm4] Emotion model loaded: /app/models/emotion-ferplus-8.onnx
[vision-cm4] Emotion skip-frame: every 5 frames
[vision-cm4] HTTP on 0.0.0.0:8630
```

### 2. Docker 运行（部署）

```bash
cd deploy/vision-cm4
docker compose up -d

# 查看日志
docker logs -f vision-cm4
```

### 3. 验证

```bash
# ZMQ 连接
nc -z localhost 8631

# HTTP API
curl http://localhost:8630/
curl http://localhost:8630/api/captures/count

# MJPEG stream
curl -I http://localhost:8630/stream
```

## 与 reachy-claw 集成

reachy-claw 配置 `vision.tracker: remote` 会自动连接 ZMQ。

### 本地 vision（同一设备）

```yaml
# reachy-claw.yaml
vision:
  tracker: remote
  zmq_url: tcp://127.0.0.1:8631
```

### 远程 vision（跨设备）

```yaml
# reachy-claw.yaml（在 Jetson 上）
vision:
  tracker: remote
  zmq_url: tcp://<cm4-ip>:8631
```

Dashboard 会自动反代 vision-cm4 的 `/stream` 和 `/api/captures`。

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
|-----|-------|------|
| `ZMQ_PUB_PORT` | 8631 | ZMQ 发布端口 |
| `HTTP_PORT` | 8630 | HTTP API 端口 |
| `CAMERA_DEVICE` | /dev/video0 | 摄像头设备 |
| `CAMERA_W` / `CAMERA_H` | 640 / 480 | 采集分辨率 |
| `TARGET_FPS` | 10 | 目标帧率（CM4 默认较低） |
| `EMOTION_SKIP_FRAMES` | 5 | 情绪跳帧间隔 |
| `EMOTION_CONFIDENCE_THRESHOLD` | 0.6 | 情绪置信度阈值 |
| `SMILE_THRESHOLD` | 0.75 | Happy 置信度触发截图 |
| `CAPTURE_DIR` | /app/data/captures | Smile 截图目录 |
| `FACE_DB_DIR` | /app/data/faces | 人脸数据库目录 |
| `PER_IDENTITY_COOLDOWN` | 30.0 | 同一人截图间隔 (秒) |
| `ANONYMOUS_COOLDOWN` | 5.0 | 匿名截图间隔 (秒) |
| `REST_CTRL_URL` | tcp://reachy:18791 | Rest 控制订阅地址 |

## 跳帧策略详解

**问题**：情绪识别（ONNX FERPlus）在 CPU 上较慢（单次约 50-100ms），
每帧运行会严重拖慢整体 FPS。

**方案**：
1. 人脸检测：每帧运行（MediaPipe CPU 约 20-30ms）
2. 情绪识别：每 N 帧（默认 5）运行一次
3. 缓存策略：按人脸位置 ID 缓存上一次情绪结果，跳帧期间沿用

**效果**：
- 整体 FPS 从 3-4 提升到 8-12
- 情绪响应延迟 ≈ N 帧 × 帧间隔（5 × 100ms = 500ms）
- 对机器人交互影响很小（情绪变化通常持续数秒）

**调整**：
- 增大 `EMOTION_SKIP_FRAMES` → 更高 FPS，更慢情绪响应
- 减小 → 更快响应，更低 FPS
- CM4 推荐 5-10，Pi 5 推荐 3-5

## 模型说明

### MediaPipe Face Detection

- 模型选择：`model_selection=0`（短距离，<2m）
- 输出：bbox（归一化坐标）、6 关键点（眼、鼻、嘴、耳）
- 特点：轻量、CPU 优化、无需 GPU

### FERPlus-8 ONNX (emotion-ferplus-8.onnx)

- 来源：`recamera_convert/face-analysis/onnx/`
- 输入：64×64 灰度人脸 crop
- 输出：8 类情绪概率

**FERPlus 8 类映射**：

| Index | FERPlus | Reachy-claw |
|-------|---------|-------------|
| 0 | neutral | neutral |
| 1 | happiness | happy |
| 2 | surprise | surprised |
| 3 | sadness | sad |
| 4 | anger | angry |
| 5 | disgust | disgust |
| 6 | fear | fear |
| 7 | contempt | neutral |

## 已知限制

- **无人脸特征提取**：CPU 版本不包含 ArcFace embedding，
  无法实现人脸注册 `/api/faces/enroll`（返回 503）
- **情绪响应延迟**：跳帧策略导致约 0.5s 延迟
- **单摄像头**：不支持多摄像头源
- **夜间性能下降**：MediaPipe 对低光照敏感，可能漏检

## 文件结构

```
deploy/vision-cm4/
├── producer.py           # 主服务
├── emotion_onnx.py       # ONNX 情绪分类封装
├── face_db.py            # 人脸数据库（来自 vision-stub）
├── pyproject.toml        # 依赖
├── Dockerfile            # 容器构建
├── docker-compose.yml    # 部署配置
├── README.md             # 本文档
└── models/
    └── emotion-ferplus-8.onnx  # 情绪模型 (35MB)
```