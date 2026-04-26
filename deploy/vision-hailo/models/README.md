# HEF 模型说明

## 已提供的模型

### hsemotion_b0.hef

- **来源**: 用户预编译
- **大小**: 7.9 MB
- **md5**: `fb155867327611d9a38ae038899463e5`
- **输入**: NHWC(224×224×3) UINT8
- **输出**: NC(8) UINT8 (8 类 softmax)
- **类别**: `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`

使用方式：
```bash
hailortcli parse-hef hsemotion_b0.hef  # 查看模型结构
```

## 系统预装模型 (hailo-models 包)

安装 `hailo-models` 后，以下 HEF 位于 `/usr/share/hailo-models/`：

| HEF | 用途 | 输入尺寸 |
|-----|------|---------|
| `scrfd_2.5g_h8l.hef` | 人脸检测 + 5 landmarks | 640×640 |
| `yolov8s_h8l.hef` | 通用目标检测 | 640×640 |
| `yolov5s_personface_h8l.hef` | 人脸 + 行人 | 640×640 |
| `yolov8s_pose_h8l.hef` | 姿态估计 | 640×640 |

## 缺失的模型

### ArcFace (人脸特征提取)

本项目未提供 ArcFace HEF，因此：
- `/api/faces/enroll` 返回 503
- 无法进行身份识别

如需人脸识别功能，可从 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
下载 `arcface_mobilefacenet.hef` 或类似模型。

## 模型搜索

```bash
# 在 Hailo Model Zoo 搜索人脸相关模型
pip install hailomz
hailomz download scrfd_2.5g
```

或直接访问 GitHub：
https://github.com/hailo-ai/hailo_model_zoo/tree/master/models