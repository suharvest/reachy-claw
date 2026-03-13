"""Inference pipeline: SCRFD detection → face align → MobileFaceNet + HSEmotion.

Processes a single BGR frame through the full vision pipeline.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# SCRFD reference landmarks for alignment (112x112)
_ARCFACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# HSEmotion classes
EMOTION_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]


@dataclass
class FaceResult:
    """Detection + recognition result for a single face."""
    bbox: list[float]           # [x1, y1, x2, y2] normalized [0, 1]
    center: list[float]         # [x, y] normalized [-1, 1]
    landmarks: list[list[float]]  # 5 points normalized [0, 1]
    confidence: float
    embedding: list[float] = field(default_factory=list)
    emotion: str = "Neutral"
    emotion_confidence: float = 0.0
    identity: str | None = None
    identity_distance: float = float("inf")


class VisionPipeline:
    """Full face analysis pipeline using TensorRT engines."""

    # Identity recognized → hold for this many frames even if some fail
    IDENTITY_HOLD_FRAMES = 15

    def __init__(self, engines: dict, face_db, config):
        self._engines = engines
        self._face_db = face_db
        self._config = config
        self._emotion_windows: dict[int, deque] = {}  # track_id → recent emotions
        # Identity smoothing: bucket → (name, distance, frames_remaining)
        self._identity_cache: dict[int, tuple[str, float, int]] = {}
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray) -> list[FaceResult]:
        """Run full pipeline on a BGR frame.

        Returns list of FaceResult for all detected faces.
        """
        self._frame_count += 1
        h, w = frame.shape[:2]

        # Step 1: SCRFD face detection
        scrfd = self._engines.get("scrfd")
        if scrfd is None:
            return []

        detections = self._detect_faces(scrfd, frame, w, h)
        if not detections:
            return []

        results = []
        arcface = self._engines.get("arcface")
        emotion_engine = self._engines.get("emotion")

        for det in detections:
            bbox, landmarks_px, conf = det
            x1, y1, x2, y2 = bbox

            # Normalized coordinates
            bbox_norm = [x1 / w, y1 / h, x2 / w, y2 / h]
            cx = ((x1 + x2) / 2 / w) * 2 - 1  # → [-1, 1]
            cy = ((y1 + y2) / 2 / h) * 2 - 1
            landmarks_norm = [[lx / w, ly / h] for lx, ly in landmarks_px]

            result = FaceResult(
                bbox=bbox_norm,
                center=[cx, cy],
                landmarks=landmarks_norm,
                confidence=conf,
            )

            # Spatial bucket for temporal smoothing (shared by identity + emotion)
            bucket = int((cx + 1) * 5) * 10 + int((cy + 1) * 5)

            # Step 2: Face alignment + embedding
            if arcface is not None and len(landmarks_px) == 5:
                aligned = self._align_face(frame, landmarks_px)
                embedding = self._extract_embedding(arcface, aligned)
                result.embedding = embedding.tolist()

                # Identity matching with temporal smoothing
                name, dist = self._face_db.identify(
                    embedding, self._config.RECOGNITION_THRESHOLD
                )
                name, dist = self._smooth_identity(bucket, name, dist)
                result.identity = name
                result.identity_distance = dist

            # Step 3: Emotion classification
            if emotion_engine is not None:
                face_crop = self._crop_face(frame, bbox)
                emotion, emotion_conf = self._classify_emotion(
                    emotion_engine, face_crop
                )
                smoothed = self._smooth_emotion(bucket, emotion)
                result.emotion = smoothed
                result.emotion_confidence = emotion_conf

            results.append(result)

        return results

    def _detect_faces(
        self, engine, frame: np.ndarray, orig_w: int, orig_h: int
    ) -> list[tuple]:
        """Run SCRFD detection. Returns [(bbox_px, landmarks_px, conf), ...]."""
        input_w = self._config.INPUT_WIDTH
        input_h = self._config.INPUT_HEIGHT

        # Preprocess: resize + normalize (skip resize if already correct size)
        h, w = frame.shape[:2]
        if (w, h) == (input_w, input_h):
            img = frame.astype(np.float32)
        else:
            img = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # NCHW

        outputs = engine.infer({"input.1": img})

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        threshold = self._config.DETECTION_THRESHOLD

        detections = self._parse_scrfd_outputs(outputs, scale_x, scale_y, threshold)
        return detections

    def _parse_scrfd_outputs(
        self, outputs: dict, scale_x: float, scale_y: float, threshold: float
    ) -> list[tuple]:
        """Parse SCRFD multi-stride outputs into detections.

        SCRFD outputs 9 tensors in groups of 3 (score, bbox, kps) per stride.
        The output order is always: [score_8, bbox_8, kps_8, score_16, bbox_16,
        kps_16, score_32, bbox_32, kps_32]. Tensor names may be numeric (e.g.
        '443', '471') rather than semantic, so we match by order + shape.
        """
        detections = []
        strides = [8, 16, 32]
        input_h = self._config.INPUT_HEIGHT
        input_w = self._config.INPUT_WIDTH

        # Try semantic name matching first
        has_semantic = any(
            "stride" in k or "score" in k for k in outputs
        )

        if has_semantic:
            groups = self._match_outputs_by_name(outputs, strides)
        else:
            groups = self._match_outputs_by_order(outputs, strides, input_h, input_w)

        for stride, (scores_arr, bbox_arr, kps_arr) in groups.items():
            scores = scores_arr.flatten()
            mask = scores > threshold
            if not mask.any():
                continue

            indices = np.where(mask)[0]
            feat_h = input_h // stride
            feat_w = input_w // stride
            num_spatial = feat_h * feat_w
            num_anchors = max(1, len(scores) // num_spatial)
            bbox_data = bbox_arr.reshape(-1, 4) if bbox_arr is not None else None
            kps_data = kps_arr.reshape(-1, 10) if kps_arr is not None else None

            for idx in indices:
                spatial_idx = idx // num_anchors
                row = spatial_idx // feat_w
                col = spatial_idx % feat_w
                score = float(scores[idx])

                if bbox_data is None or idx >= len(bbox_data):
                    continue
                dl, dt, dr, db = bbox_data[idx]
                cx = (col + 0.5) * stride
                cy = (row + 0.5) * stride
                x1 = (cx - dl * stride) * scale_x
                y1 = (cy - dt * stride) * scale_y
                x2 = (cx + dr * stride) * scale_x
                y2 = (cy + db * stride) * scale_y

                landmarks = []
                if kps_data is not None and idx < len(kps_data):
                    kps = kps_data[idx]
                    for k in range(5):
                        lx = (kps[k * 2] * stride + cx) * scale_x
                        ly = (kps[k * 2 + 1] * stride + cy) * scale_y
                        landmarks.append([lx, ly])

                detections.append(([x1, y1, x2, y2], landmarks, score))

        if len(detections) > 1:
            detections = self._nms(detections, iou_threshold=0.3)

        return detections

    @staticmethod
    def _match_outputs_by_name(outputs, strides):
        """Match output tensors by semantic names (stride/score/bbox/kps)."""
        groups = {}
        for stride in strides:
            score_key = bbox_key = kps_key = None
            for key in outputs:
                if f"stride{stride}" in key or f"_{stride}" in key:
                    if "score" in key:
                        score_key = key
                    elif "bbox" in key:
                        bbox_key = key
                    elif "kps" in key:
                        kps_key = key
            if score_key:
                groups[stride] = (
                    outputs[score_key],
                    outputs.get(bbox_key) if bbox_key else None,
                    outputs.get(kps_key) if kps_key else None,
                )
        return groups

    @staticmethod
    def _match_outputs_by_order(outputs, strides, input_h, input_w):
        """Match output tensors by shape heuristics.

        SCRFD outputs 9 tensors grouped by type (not by stride):
        [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
        Score has last dim 1, bbox has 4, kps has 10.
        We classify by last dimension and sort by size (largest = smallest stride).
        """
        tensors = list(outputs.values())

        # Classify by last dimension
        scored = [t for t in tensors if t.shape[-1] == 1]
        bboxed = [t for t in tensors if t.shape[-1] == 4]
        kpsed = [t for t in tensors if t.shape[-1] == 10]

        # Sort each by number of elements descending (stride 8 has largest feature map)
        scored.sort(key=lambda t: t.size, reverse=True)
        bboxed.sort(key=lambda t: t.size, reverse=True)
        kpsed.sort(key=lambda t: t.size, reverse=True)

        groups = {}
        for i, stride in enumerate(strides):
            s = scored[i] if i < len(scored) else None
            b = bboxed[i] if i < len(bboxed) else None
            k = kpsed[i] if i < len(kpsed) else None
            if s is not None:
                groups[stride] = (s, b, k)

        return groups

    def _nms(self, detections: list, iou_threshold: float = 0.4) -> list:
        """Simple NMS on detection list."""
        if not detections:
            return []

        bboxes = np.array([d[0] for d in detections])
        scores = np.array([d[2] for d in detections])

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def _align_face(
        self, frame: np.ndarray, landmarks: list[list[float]]
    ) -> np.ndarray:
        """Align face to 112x112 using 5-point landmarks."""
        src = np.array(landmarks, dtype=np.float32)
        tform = cv2.estimateAffinePartial2D(src, _ARCFACE_REF)[0]
        aligned = cv2.warpAffine(frame, tform, (112, 112))
        return aligned

    def _extract_embedding(self, engine, aligned: np.ndarray) -> np.ndarray:
        """Extract 128-dim face embedding from aligned 112x112 face."""
        img = aligned.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # NCHW

        outputs = engine.infer({"input.1": img})
        # Get the first output tensor
        embedding = list(outputs.values())[0].flatten()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _crop_face(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Crop and resize face for emotion classification."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (224, 224))
        return crop

    def _classify_emotion(
        self, engine, face_crop: np.ndarray
    ) -> tuple[str, float]:
        """Classify emotion from face crop."""
        img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis]  # NCHW

        outputs = engine.infer({"input": img.astype(np.float32)})
        logits = list(outputs.values())[0].flatten()

        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        idx = int(probs.argmax())
        return EMOTION_LABELS[idx], float(probs[idx])

    def _smooth_identity(
        self, bucket: int, name: str | None, dist: float
    ) -> tuple[str | None, float]:
        """Temporal smoothing for identity: hold recognized name for N frames.

        Once a face is recognized, keep that identity even if a few frames
        fail to match (due to angle/lighting changes). Only drop the identity
        after IDENTITY_HOLD_FRAMES consecutive misses.
        """
        cached = self._identity_cache.get(bucket)

        if name is not None:
            # Fresh recognition — reset hold counter
            self._identity_cache[bucket] = (name, dist, self.IDENTITY_HOLD_FRAMES)
            return name, dist

        if cached is not None:
            cached_name, cached_dist, frames_left = cached
            if frames_left > 0:
                # Hold previous identity
                self._identity_cache[bucket] = (cached_name, cached_dist, frames_left - 1)
                return cached_name, cached_dist
            else:
                # Expired
                del self._identity_cache[bucket]

        return None, dist

    def _smooth_emotion(self, track_id: int, emotion: str) -> str:
        """5-frame sliding window vote for emotion smoothing."""
        window_size = self._config.EMOTION_WINDOW_SIZE
        if track_id not in self._emotion_windows:
            self._emotion_windows[track_id] = deque(maxlen=window_size)

        self._emotion_windows[track_id].append(emotion)

        # Majority vote
        counts: dict[str, int] = {}
        for e in self._emotion_windows[track_id]:
            counts[e] = counts.get(e, 0) + 1

        return max(counts, key=counts.get)
