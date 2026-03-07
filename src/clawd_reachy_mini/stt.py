"""Speech-to-text backends for Reachy Mini interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clawd_reachy_mini.backend_registry import register_stt
from clawd_reachy_mini.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PartialResult:
    """A partial or final transcription result from streaming STT."""

    text: str
    is_final: bool = False
    is_stable: bool = False  # stable partial — won't change


class STTBackend(ABC):
    """Abstract base class for speech-to-text backends."""

    supports_streaming: bool = False

    def preload(self) -> None:
        """Preload the model to avoid delay on first transcription."""
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text (batch mode)."""
        pass

    @abstractmethod
    def transcribe_file(self, path: Path) -> str:
        """Transcribe audio file to text."""
        pass

    # ── Streaming interface (override in streaming backends) ──────────

    def start_stream(self, sample_rate: int = 16000) -> None:
        """Begin a new streaming recognition session."""
        raise NotImplementedError("This backend does not support streaming")

    def feed_chunk(self, chunk: np.ndarray) -> PartialResult | None:
        """Feed an audio chunk. Returns a PartialResult if available, else None."""
        raise NotImplementedError("This backend does not support streaming")

    def finish_stream(self) -> str:
        """Signal end of audio. Returns the final transcription."""
        raise NotImplementedError("This backend does not support streaming")

    def cancel_stream(self) -> None:
        """Cancel the current streaming session."""
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
            raise ConnectionError(
                f"SenseVoice service not reachable at {self._base_url}: {e}"
            ) from e

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


@register_stt("paraformer-streaming")
class ParaformerStreamingSTT(STTBackend):
    """Streaming Paraformer ASR via Jetson speech service (sherpa-onnx, CUDA).

    Uses WebSocket for streaming: client sends audio chunks, server returns
    partial/final results in real-time.

    Falls back to batch HTTP /asr endpoint for transcribe() if needed.
    """

    supports_streaming = True

    class Settings:
        language: str = "auto"

    def __init__(self, base_url: str = "http://localhost:8000", language: str = "auto"):
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws = None
        self._sample_rate = 16000
        self._partial_text = ""
        self._final_text = ""

    def preload(self) -> None:
        import urllib.request

        try:
            resp = urllib.request.urlopen(f"{self._base_url}/health", timeout=5)
            data = resp.read().decode()
            logger.info(f"Paraformer streaming service health: {data}")
        except Exception as e:
            raise ConnectionError(
                f"Paraformer streaming service not reachable at {self._base_url}: {e}"
            ) from e

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Batch fallback: POST to /asr like SenseVoice."""
        import io
        import json
        import urllib.request
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            if audio.dtype != np.int16:
                if np.abs(audio).max() <= 1.0:
                    audio_int = (audio * 32767).astype(np.int16)
                else:
                    audio_int = audio.astype(np.int16)
            else:
                audio_int = audio
            wf.writeframes(audio_int.tobytes())
        buf.seek(0)
        wav_bytes = buf.read()

        boundary = "----ParaformerBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
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

    def transcribe_file(self, path: Path) -> str:
        import json
        import urllib.request

        with open(path, "rb") as f:
            wav_bytes = f.read()

        boundary = "----ParaformerBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
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

    # ── Streaming interface ───────────────────────────────────────────

    def start_stream(self, sample_rate: int = 16000) -> None:
        import json

        import websockets.sync.client

        self._sample_rate = sample_rate
        self._partial_text = ""
        self._final_text = ""
        ws_url = f"{self._ws_url}/asr/stream?language={self._language}&sample_rate={sample_rate}"
        self._ws = websockets.sync.client.connect(ws_url)
        logger.debug("Paraformer streaming session started")

    def feed_chunk(self, chunk: np.ndarray) -> PartialResult | None:
        """Send audio chunk, receive partial result if available."""
        import json

        if self._ws is None:
            return None

        # Convert to int16 bytes
        if chunk.dtype != np.int16:
            if np.abs(chunk).max() <= 1.0:
                chunk_int = (chunk * 32767).astype(np.int16)
            else:
                chunk_int = chunk.astype(np.int16)
        else:
            chunk_int = chunk

        self._ws.send(chunk_int.tobytes())

        # Non-blocking receive — check if server sent a result
        try:
            raw = self._ws.recv(timeout=0)
        except (TimeoutError, Exception):
            return None

        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

        text = msg.get("text", "")
        is_final = msg.get("is_final", False)
        is_stable = msg.get("is_stable", False)

        if is_final:
            self._final_text = text
        else:
            self._partial_text = text

        return PartialResult(text=text, is_final=is_final, is_stable=is_stable)

    def finish_stream(self) -> str:
        """Send end-of-stream signal, collect final result."""
        import json

        if self._ws is None:
            return self._final_text or self._partial_text

        try:
            # Short-circuit: if we already have a final result from feed_chunk,
            # skip the EOF round-trip to save ~40-110ms
            if self._final_text:
                return self._final_text

            # Send empty bytes to signal end of audio
            self._ws.send(b"")

            # Wait for final result
            try:
                raw = self._ws.recv(timeout=5)
                msg = json.loads(raw)
                if msg.get("is_final"):
                    self._final_text = msg.get("text", "")
            except Exception:
                pass

            return self._final_text or self._partial_text
        finally:
            self._close_ws()

    def cancel_stream(self) -> None:
        self._close_ws()
        self._partial_text = ""
        self._final_text = ""

    def _close_ws(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


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
    instance = info.cls(**filtered)

    # Fallback: if remote backend is unreachable, fall back to whisper
    if backend in ("paraformer-streaming", "sensevoice"):
        try:
            instance.preload()
        except Exception as e:
            logger.warning(f"STT backend {backend!r} not available ({e}), falling back to whisper")
            from clawd_reachy_mini.backend_registry import get_stt_info as _get_info

            fallback_info = _get_info("whisper")
            if fallback_info is None:
                raise
            fallback_kwargs = {}
            if hasattr(config, "whisper_model"):
                fallback_kwargs["model_name"] = config.whisper_model
            instance = fallback_info.cls(**fallback_kwargs)

    return instance
