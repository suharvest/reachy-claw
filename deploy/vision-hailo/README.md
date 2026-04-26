# vision-hailo — Hailo-8 加速的人脸/情绪识别服务

在 Raspberry Pi 5 + Hailo-8 AI HAT 上运行的视觉推理服务，作为 `vision-trt` 的
替代品，与 reachy-claw 和 dashboard 完全兼容。

## 协议兼容

- **ZMQ**: `tcp://0.0.0.0:8631`，topic `"vision"`，msgpack 格式
- **HTTP**: `0.0.0.0:8630`，提供 MJPEG stream + face DB API
- **输出**: 与 vision-trt 完全相同的 face dict 结构

## 依赖模型 (HEF)

| 模型 | HEF 文件 | 输入尺寸 | 来源 |
|------|----------|---------|------|
| 人脸检测 (SCRFD) | `/usr/share/hailo-models/scrfd_2.5g_h8l.hef` | 640×640 | hailo-models 包 |
| 情绪分类 | `/tmp/hsemotion_b0.hef` 或 `models/hsemotion_b0.hef` | 224×224 | 用户提供 |
| 人脸特征 | ❌ 未提供 | — | `/api/faces/enroll` 返回 503 |

## 快速开始

### 1. 安装 Hailo

见 [INSTALL.md](INSTALL.md) 详细步骤。简要版：

```bash
sudo apt update
sudo apt install -y hailo-all hailo-models python3-hailort
hailortcli fw-control identify  # 确认 Hailo-8 硬件识别
```

### 2. 准备情绪模型

```bash
# 从本 repo 复制 HEF 到 Pi
scp models/hsemotion_b0.hef harvest-pi:/tmp/
# 或在 Pi 上：
fleet push harvest-pi models/hsemotion_b0.hef --dst /tmp/
```

### 3. 创建虚拟环境 (继承系统 hailo_platform)

```bash
cd deploy/vision-hailo
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
# 或用 uv:
uv venv --system-site-packages
uv sync
```

### 4. 启动服务

```bash
uv run python producer.py
```

预期输出：
```
[vision-hailo] ZMQ PUB on tcp://0.0.0.0:8631
[vision-hailo] Camera /dev/video0 open at 640x480
[vision-hailo] Detection model shape: (640, 640, 3)
[vision-hailo] Emotion model shape: (224, 224, 3)
[vision-hailo] HTTP on 0.0.0.0:8630
```

### 5. 验证

```bash
# ZMQ 连接
nc -z harvest-pi 8631

# HTTP API
curl http://harvest-pi:8630/api/captures/count
curl -I http://harvest-pi:8630/stream  # Content-Type: multipart/x-mixed-replace
```

## 配置 (环境变量)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ZMQ_PUB_PORT` | 8631 | ZMQ 发布端口 |
| `HTTP_PORT` | 8630 | HTTP 服务端口 |
| `CAMERA_DEVICE` | `/dev/video0` | 摄像头设备 |
| `CAMERA_W` / `CAMERA_H` | 640 / 480 | 采集分辨率 |
| `CAPTURE_DIR` | `/var/lib/vision-hailo/captures` | smile 截图目录 |
| `FACE_DB_DIR` | `/var/lib/vision-hailo/faces` | 人脸数据库目录 |
| `TARGET_FPS` | 15 | 目标帧率 |
| `HEF_DETECT` | `/usr/share/hailo-models/scrfd_2.5g_h8l.hef` | 检测模型路径 |
| `HEF_EMOTION` | `/tmp/hsemotion_b0.hef` | 情绪模型路径 |
| `PER_IDENTITY_COOLDOWN` | 30.0 | 同一人微笑截图间隔 (秒) |
| `ANONYMOUS_COOLDOWN` | 5.0 | 匿名人脸截图间隔 (秒) |

## HTTP API

| Method | Path | 说明 |
|--------|------|------|
| GET | `/` | 服务状态 `{service, fps}` |
| GET | `/api/captures/count` | 截图数量 |
| GET | `/api/captures/list` | 截图文件列表 |
| GET | `/api/captures/image/{filename}` | 获取截图图片 |
| DELETE | `/api/captures` | 清空所有截图 |
| GET | `/stream` | MJPEG 实时预览 |
| GET | `/api/faces` | 已注册人脸列表 |
| POST | `/api/faces/enroll?name=X` | 注册人脸 (返回 503，缺少 embedding 模型) |
| DELETE | `/api/faces/{name}` | 删除人脸注册 |

## ZMQ 消息格式

每帧发布一条 msgpack 消息，结构：

```python
{
    "frame_id": int,
    "faces": [
        {
            "center": [float, float],  # [-1, 1] 归一化，0 = 画面中心
            "bbox": [x1, y1, x2, y2],  # [0, 1] 彛一化
            "confidence": float,       # 检测置信度
            "landmarks": [[x,y], ...], # 5 点 SCRFD landmarks
            "emotion": str,            # happy/sad/angry/...
            "emotion_confidence": float,
            "identity": str | None,    # 身份名称 (无 embedding 模型时为 None)
            "identity_distance": float,
            "embedding": None,
        },
        ...
    ],
    "capture": {  # 可选，微笑截图时出现
        "event": "smile",
        "count": int,
        "file": str,
    },
}
```

## 性能预期

- **Pi 5 + Hailo-8**: ≥10 FPS (单人脸)
- **检测**: SCRFD 2.5g ≈ 100 FPS 理论上限
- **情绪**: HSEmotion ≈ 400 FPS 理论上限

实际性能受摄像头帧率、人脸数量、预处理开销影响。

## 已知限制

- **无人脸特征提取**: 缺少 ArcFace HEF，无法实现身份识别或 `/api/faces/enroll`
- **情绪模型固定**: 8 类 HSEmotion，不支持自定义情绪模型
- **单摄像头**: 不支持多摄像头源

## 文件结构

```
deploy/vision-hailo/
├── producer.py        # 主服务
├── hailo_pipeline.py  # Hailo 推理封装
├── face_db.py         # 人脸数据库 (JSON + npy)
├── pyproject.toml     # 依赖
├── README.md          # 本文档
├── INSTALL.md         # 安装指南
├── HEADS_UP.md        # 开发历史备注
└── models/
    ├── README.md      # HEF 说明
    └── hsemotion_b0.hef  # 情绪模型 (7.9 MB)
```