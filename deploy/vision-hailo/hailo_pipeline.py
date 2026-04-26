"""Hailo inference pipeline for face detection and emotion classification.

Single VDevice shared across both models using Hailo's multi-context feature.
"""

import logging
import queue
import threading
from functools import partial
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
except ImportError:
    raise ImportError("hailo_platform not installed. Install: sudo apt install hailo-all")


EMOTION_CLASSES = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise"
]


class HailoSharedInference:
    """Single-VDevice inference wrapper that can run multiple HEFs.

    Uses the same VDevice for all models (multi-context scheduling).
    """

    def __init__(self, hef_paths: list[str], batch_size: int = 1) -> None:
        """Initialize shared device with multiple HEFs.

        Args:
            hef_paths: List of HEF file paths to load.
            batch_size: Batch size for all models.
        """
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.target = VDevice(params)
        self.hefs = {path: HEF(path) for path in hef_paths}
        self.models = {path: self.target.create_infer_model(path) for path in hef_paths}
        self.batch_size = batch_size

        for model in self.models.values():
            model.set_batch_size(batch_size)

        # Configure each model ONCE at init (not per-frame) — major perf win
        self._configured_models = {}
        self._configure_ctxs = {}
        for path, model in self.models.items():
            ctx = model.configure()
            configured = ctx.__enter__()
            self._configured_models[path] = configured
            self._configure_ctxs[path] = ctx

        # Cache output buffers and quantization info per HEF (one-time allocation)
        self._output_buffers = {}
        self._quant_info = {}
        for path, hef in self.hefs.items():
            model = self.models[path]
            self._output_buffers[path] = {}
            self._quant_info[path] = {}
            for info in hef.get_output_vstream_infos():
                # Pre-allocate UINT8 output buffer (Hailo quantized outputs)
                self._output_buffers[path][info.name] = np.empty(
                    model.output(info.name).shape, dtype=np.uint8
                )
                # Cache quantization parameters for dequantization
                qi_list = model.output(info.name).quant_infos
                if qi_list:
                    self._quant_info[path][info.name] = qi_list[0]

        logger.info(f"Loaded {len(hef_paths)} HEFs on shared VDevice")
        for path, hef in self.hefs.items():
            shape = hef.get_input_vstream_infos()[0].shape
            logger.info(f"  {path}: input shape {shape}")

    def get_input_shape(self, hef_path: str) -> tuple:
        return self.hefs[hef_path].get_input_vstream_infos()[0].shape

    def infer_single(self, hef_path: str, frame: np.ndarray) -> dict:
        """Run synchronous inference on a single frame.

        Args:
            hef_path: Which HEF to use.
            frame: Preprocessed input frame (matches HEF input shape).

        Returns:
            Dict of output_name -> dequantized float32 numpy array.
        """
        configured_model = self._configured_models[hef_path]

        # Bind pre-allocated output buffers (filled in-place by SDK)
        bindings = configured_model.create_bindings(
            output_buffers=self._output_buffers[hef_path]
        )
        bindings.input().set_buffer(frame)

        # Real callback signals completion (fixes 5s timeout bug)
        done = {"info": None}
        def callback(completion_info=None):
            done["info"] = completion_info

        configured_model.wait_for_async_ready(timeout_ms=5000)
        job = configured_model.run_async([bindings], callback)
        job.wait(5000)

        # Dequantize UINT8 outputs to float32
        results = {}
        for name, raw in self._output_buffers[hef_path].items():
            qi = self._quant_info[hef_path].get(name)
            if qi is not None:
                results[name] = (raw.astype(np.float32) - qi.qp_zp) * qi.qp_scale
            else:
                # No quant info (unlikely for quantized HEF) — return raw
                results[name] = raw.astype(np.float32)

        return results

    def shutdown(self) -> None:
        # Exit configure contexts first (clean resource release)
        for path, ctx in self._configure_ctxs.items():
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        self.target.release()


class SCRFDPostprocessor:
    """Decode SCRFD multi-scale outputs into face detections."""

    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def decode(
        self,
        outputs: dict,
        original_shape: tuple,
        model_size: int = 640
    ) -> list[dict]:
        """Decode SCRFD outputs to normalized face detections."""
        h, w = original_shape[:2]

        all_bboxes = []
        all_scores = []
        all_landmarks = []

        # SCRFD outputs: 3 scales with bbox/conf/landmarks
        # Scale names pattern: conv44/42/43 (80x80), conv51/49/50 (40x40), conv57/55/56 (20x20)
        scales = [
            ("scrfd_2_5g/conv44", "scrfd_2_5g/conv42", "scrfd_2_5g/conv43", 80, 8),
            ("scrfd_2_5g/conv51", "scrfd_2_5g/conv49", "scrfd_2_5g/conv50", 40, 16),
            ("scrfd_2_5g/conv57", "scrfd_2_5g/conv55", "scrfd_2_5g/conv56", 20, 32),
        ]

        for bbox_name, conf_name, lm_name, grid_size, stride in scales:
            if bbox_name not in outputs:
                continue

            bbox_data = outputs[bbox_name]  # FCR(grid, grid, num_anchors*4)
            conf_data = outputs[conf_name]  # NHWC(grid, grid, num_anchors)
            lm_data = outputs[lm_name]      # NHWC(grid, grid, num_anchors*10)

            # Vectorized: threshold all anchors at once (was 8400+ Python iterations per scale)
            num_anchors = min(2, conf_data.shape[2] if len(conf_data.shape) == 3 else 1)
            scores_flat = conf_data.reshape(-1)
            mask = scores_flat > self.conf_threshold
            if not mask.any():
                continue

            passing = np.where(mask)[0]
            n = len(passing)

            # Decode flat index → (y, x, anchor)
            if len(conf_data.shape) == 3:
                y_grid = passing // (grid_size * num_anchors)
                rem = passing % (grid_size * num_anchors)
                x_grid = rem // num_anchors
                anchor_idx = rem % num_anchors
            else:
                y_grid = passing // grid_size
                x_grid = passing % grid_size
                anchor_idx = np.zeros(n, dtype=int)

            # Anchor centers
            anchor_x = (x_grid.astype(np.float32) + 0.5) * stride
            anchor_y = (y_grid.astype(np.float32) + 0.5) * stride

            if len(bbox_data.shape) == 3:
                cell_idx = passing // num_anchors
                base = anchor_idx * 4
                bbox_flat = bbox_data.reshape(grid_size * grid_size, -1)
                dx = bbox_flat[cell_idx, base].astype(np.float32)
                dy = bbox_flat[cell_idx, base + 1].astype(np.float32)
                dw = np.clip(bbox_flat[cell_idx, base + 2].astype(np.float32), -2.0, 2.0)
                dh = np.clip(bbox_flat[cell_idx, base + 3].astype(np.float32), -2.0, 2.0)
                cx = anchor_x + dx * stride
                cy = anchor_y + dy * stride
                bw = np.exp(dw) * stride * 4
                bh = np.exp(dh) * stride * 4
            else:
                cx, cy = anchor_x, anchor_y
                bw = bh = np.full(n, stride * 4, dtype=np.float32)

            x1 = np.clip((cx - bw / 2) / model_size, 0.0, 1.0)
            y1 = np.clip((cy - bh / 2) / model_size, 0.0, 1.0)
            x2 = np.clip((cx + bw / 2) / model_size, 0.0, 1.0)
            y2 = np.clip((cy + bh / 2) / model_size, 0.0, 1.0)

            all_bboxes.extend(
                [[float(a), float(b), float(c), float(d)]
                 for a, b, c, d in zip(x1, y1, x2, y2)]
            )
            all_scores.extend(scores_flat[passing].tolist())

            # Landmarks — keep scalar inner loop (small n after threshold; layout tricky to vectorize safely)
            if len(lm_data.shape) >= 2 and lm_data.shape[2] >= 10:
                for i in range(n):
                    yi, xi, ai = int(y_grid[i]), int(x_grid[i]), int(anchor_idx[i])
                    lm_idx = ai * 10
                    pts = []
                    for j in range(5):
                        lx = float(lm_data[yi, xi, lm_idx + j * 2]) * stride / model_size
                        ly = float(lm_data[yi, xi, lm_idx + j * 2 + 1]) * stride / model_size
                        pts.append([lx, ly])
                    all_landmarks.append(pts)
            else:
                all_landmarks.extend([[] for _ in range(n)])

        if not all_bboxes:
            return []

        # NMS
        keep = self._nms(all_bboxes, all_scores)

        results = []
        for idx in keep:
            bbox = all_bboxes[idx]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            center = [cx * 2 - 1, cy * 2 - 1]  # [-1, 1]

            results.append({
                "bbox": bbox,
                "center": center,
                "confidence": all_scores[idx],
                "landmarks": all_landmarks[idx] if idx < len(all_landmarks) else [],
                "crop_bbox_px": [
                    int(bbox[0] * w), int(bbox[1] * h),
                    int(bbox[2] * w), int(bbox[3] * h),
                ],
            })

        return results

    def _nms(self, bboxes: list, scores: list) -> list:
        if not bboxes:
            return []

        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []

        while indices:
            curr = indices.pop(0)
            keep.append(curr)

            remaining = []
            for i in indices:
                iou = self._iou(bboxes[curr], bboxes[i])
                if iou < self.nms_threshold:
                    remaining.append(i)
            indices = remaining

        return keep

    def _iou(self, a: list, b: list) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0


class EmotionClassifier:
    """Simple emotion classification from hsemotion_b0 output."""

    def __init__(self, classes: list = EMOTION_CLASSES):
        self.classes = classes

    def classify(self, output: np.ndarray) -> tuple[str, float]:
        """Return (emotion, confidence) from dequantized softmax output."""
        probs = output.flatten().astype(np.float32)
        # After dequantization, values are already in valid range
        # Clamp to [0, 1] for safety
        probs = np.clip(probs, 0.0, 1.0)
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])


class HailoPipeline:
    """End-to-end detection + emotion pipeline using single VDevice."""

    def __init__(
        self,
        detect_hef: str,
        emotion_hef: str,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        self.inference = HailoSharedInference([detect_hef, emotion_hef])

        self.detect_hef = detect_hef
        self.emotion_hef = emotion_hef

        self.detect_shape = self.inference.get_input_shape(detect_hef)
        self.emotion_shape = self.inference.get_input_shape(emotion_hef)

        self.postprocessor = SCRFDPostprocessor(conf_threshold, nms_threshold)
        self.emotion_classifier = EmotionClassifier()

        logger.info(f"Pipeline ready: detect {self.detect_shape}, emotion {self.emotion_shape}")

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Process one BGR frame through detection + emotion.

        Returns list of face dicts with center, bbox, confidence, landmarks, emotion.
        """
        import cv2

        h, w = frame.shape[:2]

        # 1. Detection
        det_h, det_w, _ = self.detect_shape
        det_frame = self._preprocess(frame, det_w, det_h)

        det_results = self.inference.infer_single(self.detect_hef, det_frame)

        # 2. Decode faces
        faces = self.postprocessor.decode(det_results, frame.shape)

        if not faces:
            return []

        # 3. Emotion for each face
        emo_h, emo_w, _ = self.emotion_shape

        for face in faces:
            x1, y1, x2, y2 = face["crop_bbox_px"]
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                face["emotion"] = "Neutral"
                face["emotion_confidence"] = 0.0
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                face["emotion"] = "Neutral"
                face["emotion_confidence"] = 0.0
                continue

            emo_frame = self._preprocess(crop, emo_w, emo_h)
            emo_results = self.inference.infer_single(self.emotion_hef, emo_frame)

            output_key = list(emo_results.keys())[0]
            emotion, confidence = self.emotion_classifier.classify(emo_results[output_key])

            face["emotion"] = emotion
            face["emotion_confidence"] = confidence

        # Build output dicts (remove internal crop_bbox_px)
        results = []
        for face in faces:
            results.append({
                "center": face["center"],
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "landmarks": face["landmarks"],
                "emotion": face["emotion"],
                "emotion_confidence": face["emotion_confidence"],
                "identity": None,
                "identity_distance": None,
                "embedding": None,
            })

        return results

    def _preprocess(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize with aspect ratio preservation and padding."""
        import cv2

        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return padded

    def shutdown(self) -> None:
        self.inference.shutdown()


def init_pipeline() -> HailoPipeline:
    """Initialize pipeline with default HEF paths."""
    import os

    _models_dir = os.path.join(os.path.dirname(__file__), "models")
    detect_hef = os.environ.get(
        "HEF_DETECT", os.path.join(_models_dir, "scrfd_2.5g.hef")
    )
    emotion_hef = os.environ.get(
        "HEF_EMOTION", os.path.join(_models_dir, "hsemotion_b0.hef")
    )

    return HailoPipeline(detect_hef, emotion_hef)