"""Speech-to-text backends for Reachy Mini interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from clawd_reachy_mini.backend_registry import register_stt
from clawd_reachy_mini.config import Config

logger = logging.getLogger(__name__)


class STTBackend(ABC):
    """Abstract base class for speech-to-text backends."""

    def preload(self) -> None:
        """Preload the model to avoid delay on first transcription."""
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text."""
        pass

    @abstractmethod
    def transcribe_file(self, path: Path) -> str:
        """Transcribe audio file to text."""
        pass


@register_stt("whisper")
class WhisperSTT(STTBackend):
    """Local Whisper model for speech-to-text."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    def preload(self) -> None:
        """Preload the Whisper model."""
        self._load_model()

    def _load_model(self):
        if self._model is None:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        model = self._load_model()

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        result = model.transcribe(audio, fp16=False)
        return result["text"].strip()

    def transcribe_file(self, path: Path) -> str:
        model = self._load_model()
        result = model.transcribe(str(path), fp16=False)
        return result["text"].strip()


@register_stt("faster-whisper")
class FasterWhisperSTT(STTBackend):
    """Faster-Whisper for optimized local transcription."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    def preload(self) -> None:
        """Preload the Faster-Whisper model."""
        self._load_model()

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Faster-Whisper model: {self.model_name}")
            self._model = WhisperModel(self.model_name, compute_type="int8")
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        model = self._load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        segments, _ = model.transcribe(audio)
        return " ".join(segment.text for segment in segments).strip()

    def transcribe_file(self, path: Path) -> str:
        model = self._load_model()
        segments, _ = model.transcribe(str(path))
        return " ".join(segment.text for segment in segments).strip()


@register_stt("openai")
class OpenAISTT(STTBackend):
    """OpenAI Whisper API for cloud transcription."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import tempfile
        import wave

        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        with wave.open(str(temp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Convert to int16
            audio_int = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int.tobytes())

        try:
            return self.transcribe_file(temp_path)
        finally:
            temp_path.unlink()

    def transcribe_file(self, path: Path) -> str:
        client = self._get_client()
        logger.info("Sending audio to OpenAI Cloud Whisper...")
        with open(path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",  # Force English transcription
            )
        logger.info("OpenAI transcription complete")
        return response.text.strip()


@register_stt("sensevoice")
class SenseVoiceSTT(STTBackend):
    """Remote SenseVoice ASR via Jetson Docker service."""

    class Settings:
        language: str = "auto"

    def __init__(self, base_url: str = "http://localhost:8000", language: str = "auto"):
        self._base_url = base_url.rstrip("/")
        self._language = language

    def preload(self) -> None:
        import urllib.request

        try:
            resp = urllib.request.urlopen(f"{self._base_url}/health", timeout=5)
            data = resp.read().decode()
            logger.info(f"SenseVoice service health: {data}")
        except Exception as e:
            logger.warning(f"SenseVoice service not reachable: {e}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import io
        import wave

        # Encode audio as WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            if audio.dtype != np.int16:
                if audio.max() <= 1.0:
                    audio_int = (audio * 32767).astype(np.int16)
                else:
                    audio_int = audio.astype(np.int16)
            else:
                audio_int = audio
            wf.writeframes(audio_int.tobytes())
        buf.seek(0)
        return self._post_asr(buf.read(), "audio.wav")

    def transcribe_file(self, path: Path) -> str:
        with open(path, "rb") as f:
            return self._post_asr(f.read(), path.name)

    def _post_asr(self, wav_bytes: bytes, filename: str) -> str:
        import json
        import urllib.request

        boundary = "----SenseVoiceBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()

        url = f"{self._base_url}/asr?language={self._language}"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        return result.get("text", "").strip()


def create_stt_backend(config: Config) -> STTBackend:
    """Create STT backend by name using the registry."""
    from clawd_reachy_mini.backend_registry import get_stt_info, get_stt_names

    backend = config.stt_backend.lower()

    info = get_stt_info(backend)
    if info is None:
        available = ", ".join(get_stt_names())
        raise ValueError(f"Unknown STT backend: {backend!r}. Choose from: {available}")

    # Build kwargs from config
    kwargs = {}
    for field_name in info.settings_fields:
        config_key = f"{info.name}_{field_name}"
        if hasattr(config, config_key):
            kwargs[field_name] = getattr(config, config_key)
    if hasattr(config, "speech_service_url"):
        kwargs.setdefault("base_url", config.speech_service_url)

    # Special cases for built-in backends
    if backend in ("whisper", "faster-whisper"):
        kwargs.setdefault("model_name", config.whisper_model)
    elif backend == "openai":
        if not config.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI STT backend")
        kwargs["api_key"] = config.openai_api_key

    # Filter to valid constructor params
    import inspect
    sig = inspect.signature(info.cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    logger.info(f"Using STT backend: {backend}")
    return info.cls(**filtered)
