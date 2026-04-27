"""ONNX Emotion classifier wrapper (FERPlus 8-class).

Wraps emotion-ferplus-8.onnx from recamera_convert/face-analysis/onnx/.
Input: 64x64 grayscale face crop.
Output: 8-class emotion probabilities.

FERPlus 8 classes (index order):
    0: neutral
    1: happiness
    2: surprise
    3: sadness
    4: anger
    5: disgust
    6: fear
    7: contempt
"""
from __future__ import annotations

import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# FERPlus 8-class mapping to reachy-claw expected names
FERPLUS_TO_REACHY = {
    0: "neutral",
    1: "happy",
    2: "surprised",
    3: "sad",
    4: "angry",
    5: "disgust",
    6: "fear",
    7: "neutral",  # contempt → neutral (reachy doesn't have contempt)
}

# Reachy-claw expects these emotion names
REACHY_EMOTIONS = ["happy", "sad", "angry", "surprised", "fear", "neutral", "disgust"]


class EmotionONNX:
    """ONNX-based emotion classifier with 64x64 grayscale input."""

    def __init__(self, model_path: str | None = None):
        if not HAS_ONNX:
            raise ImportError("onnxruntime not installed")

        # Default model path (relative to this file)
        if model_path is None:
            from pathlib import Path
            model_path = str(Path(__file__).parent / "models" / "emotion-ferplus-8.onnx")

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        # Expected: [1, 1, 64, 64]
        self.input_size = (64, 64)

    def infer(self, face_crop_bgr: np.ndarray) -> dict:
        """Classify emotion from a BGR face crop.

        Args:
            face_crop_bgr: BGR image crop of a single face (any size)

        Returns:
            dict with:
                - emotion: str (reachy-claw compatible name)
                - emotion_confidence: float (0-1)
                - probabilities: list[float] (8 values, raw softmax outputs)
        """
        # Convert BGR to grayscale
        if face_crop_bgr.ndim == 3:
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop_bgr

        # Resize to 64x64
        gray_resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and reshape to [1, 1, 64, 64]
        # ONNX model expects float32 normalized input
        normalized = gray_resized.astype(np.float32) / 255.0
        blob = normalized.reshape(1, 1, 64, 64)

        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})
        probs = outputs[0][0]  # [8] probabilities

        # Get top class
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        # Map to reachy-claw emotion name
        emotion_name = FERPLUS_TO_REACHY.get(top_idx, "neutral")

        return {
            "emotion": emotion_name,
            "emotion_confidence": confidence,
            "probabilities": probs.tolist(),
        }