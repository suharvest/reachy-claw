"""Voice Activity Detection backends for Reachy Mini.

All VAD runs locally. Backends differ in compute: CPU (energy/silero) vs NPU (hailo).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from clawd_reachy_mini.backend_registry import register_vad

logger = logging.getLogger(__name__)


class VADBackend(ABC):
    """Abstract base class for VAD backends."""

    @abstractmethod
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio chunk contains speech."""

    def reset(self) -> None:
        """Reset internal state (e.g. between utterances)."""

    def preload(self) -> None:
        """Pre-load model to avoid delay on first call."""


@register_vad("silero")
class SileroVAD(VADBackend):
    """Silero VAD using the official silero-vad package.

    Uses the ONNX backend via silero_vad.load_silero_vad(onnx=True).
    Handles state management internally.
    """

    class Settings:
        threshold: float = 0.5

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._model = None

    def preload(self) -> None:
        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return

        from silero_vad import load_silero_vad

        logger.info("Loading Silero VAD (onnxruntime)...")
        self._model = load_silero_vad(onnx=True)
        logger.info("Silero VAD ready")

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        import torch

        self._load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        # Process in 512-sample chunks (required by Silero VAD at 16kHz)
        chunk_size = 512
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            tensor = torch.from_numpy(chunk)
            prob = self._model(tensor, sample_rate).item()

            if prob > self._threshold:
                return True
        return False

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()

    @property
    def threshold(self) -> float:
        return self._threshold


@register_vad("energy")
class EnergyVAD(VADBackend):
    """Simple energy-based VAD — zero dependencies, works everywhere."""

    class Settings:
        threshold: float = 0.01

    def __init__(self, threshold: float = 0.01):
        self._threshold = threshold

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        return float(np.abs(audio).mean()) > self._threshold


def create_vad_backend(
    backend: str = "energy",
    config=None,
) -> VADBackend:
    """Create a VAD backend by name using the registry."""
    from clawd_reachy_mini.backend_registry import get_vad_info, get_vad_names

    name = backend.lower().strip()

    info = get_vad_info(name)
    if info is None:
        available = ", ".join(get_vad_names())
        raise ValueError(f"Unknown VAD backend: {backend!r}. Choose from: {available}")

    kwargs = {}
    if config:
        for field_name in info.settings_fields:
            config_key = f"{info.name}_{field_name}"
            if hasattr(config, config_key):
                kwargs[field_name] = getattr(config, config_key)

    import inspect

    sig = inspect.signature(info.cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    logger.info(f"Using VAD backend: {name}")
    return info.cls(**filtered)
