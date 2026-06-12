"""Voice Activity Detection backends for Reachy Mini.

All VAD runs locally. Backends differ in compute: CPU (energy/silero) vs NPU (hailo).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

from reachy_claw.backend_registry import register_vad

logger = logging.getLogger(__name__)


class VADBackend(ABC):
    """Abstract base class for VAD backends."""

    @abstractmethod
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio chunk contains speech."""

    def speech_probability(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Return raw speech probability for the audio chunk.

        Default implementation returns 1.0 if speech detected, 0.0 otherwise.
        Backends with native probability support should override this.
        """
        return 1.0 if self.is_speech(audio, sample_rate) else 0.0

    def reset(self) -> None:
        """Reset internal state (e.g. between utterances)."""

    def preload(self) -> None:
        """Pre-load model to avoid delay on first call."""


@register_vad("silero")
class SileroVAD(VADBackend):
    """Silero VAD using pure onnxruntime — no PyTorch dependency.

    Loads the ONNX model from the silero_vad package data directory
    and runs inference with numpy arrays directly.
    """

    class Settings:
        threshold: float = 0.5

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._session = None
        self._state: np.ndarray | None = None
        self._context: np.ndarray | None = None
        self._last_sr = 0
        self._lock = threading.Lock()

    def preload(self) -> None:
        self._load_model()

    def _find_onnx_model(self) -> str:
        """Locate silero_vad.onnx.

        Preference order:
        1. Vendored copy at ``reachy_claw/assets/silero_vad.onnx`` — the
           supported path. Avoids the silero-vad pypi package, which
           declares torch as a runtime dep (~500MB) we never use because
           inference runs through onnxruntime directly.
        2. silero_vad pypi package data dir — fallback for dev envs
           where someone explicitly installed silero-vad.
        """
        import os

        # 1. Vendored
        vendored = os.path.join(
            os.path.dirname(__file__), "assets", "silero_vad.onnx"
        )
        if os.path.isfile(vendored):
            return vendored

        # 2. silero-vad pypi package fallback
        import importlib.resources

        try:
            ref = importlib.resources.files("silero_vad") / "data" / "silero_vad.onnx"
            with importlib.resources.as_file(ref) as p:
                return str(p)
        except Exception:
            pass

        import site

        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            candidate = f"{sp}/silero_vad/data/silero_vad.onnx"
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            "silero_vad.onnx not found. Expected at "
            f"{vendored} (vendored) or via `pip install silero-vad`."
        )

    def _load_model(self):
        if self._session is not None:
            return

        import onnxruntime

        logger.info("Loading Silero VAD (onnxruntime, no PyTorch)...")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        path = self._find_onnx_model()
        self._session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._reset_states()
        logger.info("Silero VAD ready")

    def _reset_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 0), dtype=np.float32)
        self._last_sr = 0

    def _infer(self, x: np.ndarray, sr: int) -> float:
        """Run one 512-sample chunk through the ONNX model. Returns speech probability."""
        # x shape: (1, 512)
        context_size = 64 if sr == 16000 else 32

        if self._context.shape[1] == 0:
            self._context = np.zeros((1, context_size), dtype=np.float32)

        # Prepend context
        x_with_ctx = np.concatenate([self._context, x], axis=1)

        ort_inputs = {
            "input": x_with_ctx,
            "state": self._state,
            "sr": np.array(sr, dtype=np.int64),
        }
        out, state = self._session.run(None, ort_inputs)

        self._state = state
        self._context = x_with_ctx[:, -context_size:]
        self._last_sr = sr

        return float(out[0, 0])

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio to float32 mono in [-1, 1]."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        return audio

    def _max_probability(self, audio: np.ndarray, sample_rate: int) -> float:
        """Process audio and return max speech probability across chunks."""
        self._load_model()
        audio = self._prepare_audio(audio, sample_rate)

        with self._lock:
            if self._last_sr and self._last_sr != sample_rate:
                self._reset_states()

            max_prob = 0.0
            chunk_size = 512
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                chunk_2d = chunk.reshape(1, -1)
                prob = self._infer(chunk_2d, sample_rate)
                if prob > max_prob:
                    max_prob = prob
            return max_prob

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        return self._max_probability(audio, sample_rate) > self._threshold

    def speech_probability(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Return max speech probability across 512-sample chunks."""
        return self._max_probability(audio, sample_rate)

    def reset(self) -> None:
        if self._session is not None:
            with self._lock:
                self._reset_states()

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
    from reachy_claw.backend_registry import get_vad_info, get_vad_names

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
