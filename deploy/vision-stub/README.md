# vision-stub — 自带推理后端的 vision-trt 替代

`vision-trt` 是 NVIDIA-only（TensorRT 引擎 + Jetson CUDA），在非 NVIDIA 硬件
上跑不了。这个 stub 是**协议兼容**的最小实现：摄像头采集、ZMQ 发布、HTTP API、
MJPEG 流、smile 截图、cooldown 全部包好，**你只填 3 个推理函数**就能让
reachy-claw + dashboard 在任何硬件上工作（Hailo / Coral / RKNN / MediaPipe /
ONNX-CPU / ……）。

## 1. 它解决什么问题

reachy-claw 默认配置 `vision.tracker: remote`，启动时会连 `tcp://127.0.0.1:8631`
等一个 vision producer 推送人脸数据。原来这个 producer 必须是 vision-trt 容器，
现在可以换成你自己的进程。

```
┌──────────────┐  ZMQ msgpack  ┌──────────────┐
│ vision 推理  │──────────────▶│ reachy-claw  │──▶ 头部追踪 / 情绪联动 / 对话
│ (你的代码)    │   tcp:8631    │              │
└──────┬───────┘                └──────────────┘
       │ HTTP                          ▲
       │                               │
       ▼ :8630                         │ HTTP 反代
┌──────────────┐                ┌──────────────┐
│ MJPEG /stream│◀───────────────│  dashboard   │
│ /api/captures│                │   :8640      │
└──────────────┘                └──────────────┘
```

## 2. 三步用起来

### 步骤 1：填三个 TODO

打开 `producer.py`，找到三个 `TODO_*` 函数：

```python
def TODO_init_models() -> dict
    # 启动时跑一次。加载你的模型、初始化加速卡，返回的 dict 会传给下面两个函数
    return {"detector": ..., "emotion": ...}

def TODO_infer_frame(frame_bgr, models) -> list[dict]
    # 每帧调用一次（默认 15 FPS）。frame_bgr 是 numpy BGR，shape (h, w, 3)
    # 返回人脸列表，schema 见下面"face dict 协议"
    return [{"center": [...], "bbox": [...], "emotion": "happy", ...}]

def TODO_should_capture_smile(faces, frame_bgr) -> bool
    # 每帧调用。返回 True 时，stub 会把当前帧存为 smile JPG 推到 dashboard gallery
    # 已经有全局 2s cooldown，你只判断"现在是不是该截"就行
    return False
```

剩下的（摄像头、ZMQ、HTTP、文件落盘、并发锁、FPS）**不用改**。

### 步骤 2：起 stub

```bash
cd deploy/vision-stub
uv sync                    # 没装 uv 的话: pip install -e .
uv run python producer.py
```

控制台应看到：
```
[vision-stub] ZMQ PUB on tcp://0.0.0.0:8631
[vision-stub] camera /dev/video0 open at 640x480
[vision-stub] HTTP on 0.0.0.0:8630
```

### 步骤 3：起 reachy-claw（**不要带 `--vision`**）

```bash
cd deploy/jetson
./deploy.sh                # 默认就跳过 vision-trt
```

reachy-claw 配置里 `vision.tracker: remote` 已经写好了，会自动连上 stub。
打开 dashboard `http://<host>:8640`，"What I see" 面板看视频流，机器人头部跟你转。

---

## 3. face dict 协议（**这一节决定你的代码对不对**）

`TODO_infer_frame` 返回的 list 里，每个 dict 是一张人脸：

| 字段 | 类型 | 说明 | 必填? |
|---|---|---|---|
| `center` | `[float, float]` | 人脸中心，**归一化到 [-1, 1]**，0 = 画面中心 | **必填**（头部追踪靠它） |
| `bbox` | `[x1, y1, x2, y2]` | 人脸框，**归一化到 [0, 1]** | **必填**（多脸排序按面积） |
| `confidence` | `float` | 检测置信度 0-1 | 推荐 |
| `emotion` | `str` | `happy` / `sad` / `angry` / `surprised` / `fear` / `neutral` / `disgust` | 推荐（情绪联动靠它） |
| `emotion_confidence` | `float` | 0-1 | 推荐（< 阈值会被忽略） |
| `landmarks` | `[[x,y], ...]` | SCRFD 5 点，**归一化 [0,1]**，`[0]=左眼 [1]=右眼` | 可选（用于头部 roll） |
| `identity` | `str` 或 `None` | 已识别的人名 | 可选（VLM 模式会用） |
| `identity_distance` | `float` | 0-1，越小越像 | 可选 |
| `embedding` | `[128 floats]` 或 `None` | 人脸特征向量 | 可选（仅 enroll 接口用） |

> ⚠️ **坐标系坑**：reachy-claw 严格要求归一化坐标，**不接受像素**。如果你的检测器
> 输出像素，转换公式：
> ```python
> h, w = frame_bgr.shape[:2]
> cx_px, cy_px = (x1+x2)/2, (y1+y2)/2
> center = [(cx_px/w)*2 - 1, (cy_px/h)*2 - 1]      # [-1, 1]
> bbox_n = [x1/w, y1/h, x2/w, y2/h]                # [0, 1]
> ```

返回空 list = 看不到人脸（合法状态，机器人会回正头部）。

---

## 4. 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `ZMQ_PUB_PORT` | `8631` | reachy-claw 连这个端口 |
| `HTTP_PORT` | `8630` | dashboard 反代到这里取 stream 和图片 |
| `CAMERA_DEVICE` | `/dev/video0` | 传给 `cv2.VideoCapture` |
| `CAMERA_W` / `CAMERA_H` | `640` / `480` | 采集分辨率 |
| `CAPTURE_DIR` | `/app/data/captures` | smile JPG 存这 |
| `TARGET_FPS` | `15` | 推理上限帧率 |

---

## 5. 容器化（可选）

`Dockerfile` 已经写好。如果想塞进 docker-compose，把 `deploy/jetson/docker-compose.yml`
里的 `vision-trt` 整段换成：

```yaml
vision-stub:
  build: ../vision-stub          # 相对于 docker-compose.yml
  container_name: vision-stub
  restart: unless-stopped
  network_mode: host
  devices:
    - /dev/video0:/dev/video0
  volumes:
    - ${DATA_DIR:-~/reachy-data}/vision:/app/data
  environment:
    - CAMERA_DEVICE=/dev/video0
    - ZMQ_PUB_PORT=8631
    - HTTP_PORT=8630
  # 用 CUDA / Hailo / Coral 时按需加 runtime / devices
```

不容器化也行，直接 `python producer.py` 当系统服务跑（systemd unit 自己写）。

---

## 6. 验证清单

跑起来之后逐项确认：

- [ ] `nc -z localhost 8631` → 连得上（ZMQ）
- [ ] `curl http://localhost:8630/api/captures/count` → `{"count": N}`
- [ ] `curl http://localhost:8630/stream` → `multipart/x-mixed-replace` 流
- [ ] dashboard 的 "What I see" 面板能看到你的画面
- [ ] 站在摄像头前 → 机器人头部跟着你转（说明 `center` 正确）
- [ ] dashboard 顶部的情绪图标会切换（说明 `emotion` 正确）
- [ ] 微笑 → ❤️ 计数 +1，gallery 能打开看到截图（说明 capture 链路通）

任何一项不对，先看 `producer.py` 的 console，然后看 reachy-claw 容器日志：
```bash
docker logs reachy-claw 2>&1 | grep -iE "vision|zmq"
```

---

## 7. 已知限制 / 与原版差异

- **smile 触发逻辑不一样**：原版 `vision-trt` 用「连续 5 帧高置信度 happy +
  按身份 30s 冷却」，stub 用「你的布尔判断 + 按身份 30s 冷却」。dashboard 只看
  `count` 和 `file`，所以**功能正常**，但触发"什么时候算微笑"由你决定。
- **face enroll 接口**：stub 现在有 `/api/faces/*` 端点，但默认返回 503（需要
  你的 TODO_infer_frame 返回 `embedding` 字段才能启用）。见下文"人脸注册"。
- **按身份去重**：已内置 per-identity 30s 冷却，同一个人不会连续触发多张截图。

---

## 7.1 人脸注册 API

stub 现在提供 `/api/faces/*` 端点，兼容 dashboard：

| Method | Path | 说明 |
|--------|------|------|
| GET | `/api/faces` | 已注册人脸列表 |
| POST | `/api/faces/enroll?name=X` | 注册当前人脸 (需要 embedding) |
| DELETE | `/api/faces/{name}` | 删除人脸注册 |

**默认返回 503**：因为 stub 的 `TODO_infer_frame` 不返回 `embedding` 字段。

要启用人脸注册，你的实现需要：
1. 在 `TODO_infer_frame` 返回的 face dict 中添加 `"embedding": [128 floats]`
2. 在 `TODO_init_models` 加载 embedding 模型 (如 ArcFace)
3. 自己在 `TODO_infer_frame` 中实现"取最近一帧的 embedding 存入 face_db"

示例（假设你有 embedding 模型）：
```python
def TODO_infer_frame(frame_bgr, models):
    faces = models["detector"].detect(frame_bgr)
    for face in faces:
        crop = extract_face_crop(frame_bgr, face["bbox"])
        face["embedding"] = models["embedder"].infer(crop)  # [128 floats]
    return faces
```

---

## 8. 调试 tip

想先验证 stub 整条链路通不通（不写真的推理），把 `TODO_infer_frame` 改成假数据：

```python
def TODO_infer_frame(frame_bgr, models):
    return [{
        "center": [0.0, 0.0],          # 假装人脸在画面正中
        "bbox": [0.4, 0.3, 0.6, 0.7],
        "confidence": 0.95,
        "emotion": "happy",
        "emotion_confidence": 0.9,
    }]
```

跑起来后机器人头应该回正不动（center=0），dashboard 显示 happy 表情。链路通了
再去接真的检测器。
