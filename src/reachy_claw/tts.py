"""Text-to-speech backends for Reachy Mini interface."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod

from reachy_claw.backend_registry import register_tts

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    """Abstract base class for text-to-speech backends."""

    supports_streaming: bool = False

    @abstractmethod
    async def synthesize(self, text: str) -> str:
        """Synthesize speech from text.

        Returns:
            Path to a temporary audio file (WAV or MP3). Caller is
            responsible for deleting it after playback.
        """

    async def synthesize_streaming(self, text: str):
        """Yield (np.ndarray chunk, sample_rate) tuples as audio is generated.

        Override in backends that support streaming. Default falls back to
        batch synthesize + yield the whole thing at once.
        """
        import numpy as np
        import wave

        path = await self.synthesize(text)
        try:
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio = data.astype(np.float32) / 32768.0
            yield audio, sr
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def cleanup(self) -> None:
        """Release any resources held by the backend."""


@register_tts("elevenlabs")
class ElevenLabsTTS(TTSBackend):
    """ElevenLabs cloud TTS."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
    ):
        from reachy_claw.elevenlabs import load_elevenlabs_config

        self._config = load_elevenlabs_config(
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )

    async def synthesize(self, text: str) -> str:
        from reachy_claw.elevenlabs import elevenlabs_tts_to_temp_audio_file

        return await elevenlabs_tts_to_temp_audio_file(
            text=text,
            config=self._config,
            voice_settings={"use_speaker_boost": True},
        )


@register_tts("say")
@register_tts("macos-say")
class MacOSSayTTS(TTSBackend):
    """macOS built-in `say` command (offline, zero-config)."""

    def __init__(self, voice: str | None = None, rate: int | None = None):
        self._voice = voice  # e.g. "Samantha", "Alex"
        self._rate = rate  # words per minute

    async def synthesize(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            prefix="reachy_claw_say_", suffix=".aiff", delete=False
        )
        tmp.close()
        cmd = ["say"]
        if self._voice:
            cmd += ["-v", self._voice]
        if self._rate:
            cmd += ["-r", str(self._rate)]
        cmd += ["-o", tmp.name, "--", text]
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"say failed: {proc.stderr.decode()}")
        return tmp.name


@register_tts("piper")
class PiperTTS(TTSBackend):
    """Piper TTS — fast local neural TTS via the piper CLI.

    Expects `piper` on PATH and a model file/dir. See https://github.com/rhasspy/piper
    """

    def __init__(self, model: str | None = None, speaker: int | None = None):
        self._model = model or os.getenv("PIPER_MODEL", "")
        self._speaker = speaker

    async def synthesize(self, text: str) -> str:
        if not self._model:
            raise ValueError(
                "Piper model not set. Pass --tts-model or set PIPER_MODEL env var."
            )
        tmp = tempfile.NamedTemporaryFile(
            prefix="reachy_claw_piper_", suffix=".wav", delete=False
        )
        tmp.close()
        cmd = ["piper", "--model", self._model, "--output_file", tmp.name]
        if self._speaker is not None:
            cmd += ["--speaker", str(self._speaker)]
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            input=text.encode(),
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"piper failed: {proc.stderr.decode()}")
        return tmp.name


@register_tts("none")
class NoopTTS(TTSBackend):
    """Dummy backend that prints text instead of speaking."""

    async def synthesize(self, text: str) -> str:
        logger.info(f"[TTS disabled] {text}")
        # Return empty file
        tmp = tempfile.NamedTemporaryFile(
            prefix="reachy_claw_noop_", suffix=".wav", delete=False
        )
        tmp.close()
        # Write a minimal silent WAV header (44 bytes, 0 samples)
        import struct
        with open(tmp.name, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36))  # file size - 8
            f.write(b"WAVE")
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # chunk size
            f.write(struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16))
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", 0))  # 0 bytes of audio
        return tmp.name


@register_tts("matcha")
@register_tts("kokoro")
class KokoroTTS(TTSBackend):
    """Remote TTS via Jetson speech service (sherpa-onnx, CUDA)."""

    class Settings:
        speaker_id: int = 0
        speed: float = 1.0
        pitch_shift: float = 0.0
        clone_seed: int = 42

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        speaker_id: int = 0,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        clone_seed: int = 42,
    ):
        self._base_url = base_url.rstrip("/")
        self._speaker_id = speaker_id
        self._speed = speed
        self._pitch_shift = pitch_shift
        self._clone_seed = clone_seed
        self._cloned_voice_embedding: bytes | None = None
        self._cloned_voice_name: str | None = None
        self._check_streaming_support()

    def set_cloned_voice(self, embedding_path: str | None, name: str | None = None) -> None:
        """Set cloned voice embedding for synthesis.

        Args:
            embedding_path: Path to .bin file (1024 float32 embedding), or None to disable.
            name: Voice name (for logging/debugging).
        """
        if embedding_path:
            import os
            if os.path.exists(embedding_path):
                with open(embedding_path, "rb") as f:
                    self._cloned_voice_embedding = f.read()
                self._cloned_voice_name = name
                logger.info("Kokoro TTS: using cloned voice '%s'", name)
            else:
                logger.warning("Cloned voice file not found: %s", embedding_path)
                self._cloned_voice_embedding = None
                self._cloned_voice_name = None
        else:
            self._cloned_voice_embedding = None
            self._cloned_voice_name = None
            logger.info("Kokoro TTS: switched to speaker_id mode")

    def _check_streaming_support(self) -> None:
        """Probe server health and /tts/stream endpoint at init time.

        Raises ConnectionError if the speech service is not reachable at all,
        allowing the TTS factory to fall back to another backend.
        """
        import urllib.request

        # First check if the service is reachable
        try:
            urllib.request.urlopen(f"{self._base_url}/health", timeout=3)
        except Exception as e:
            raise ConnectionError(
                f"Kokoro speech service not reachable at {self._base_url}: {e}"
            ) from e

        # Then check for streaming support
        try:
            req = urllib.request.Request(
                f"{self._base_url}/tts/stream", method="OPTIONS"
            )
            urllib.request.urlopen(req, timeout=3)
            self.supports_streaming = True
            logger.info("Kokoro TTS: streaming endpoint available")
        except Exception:
            self.supports_streaming = False
            logger.info("Kokoro TTS: using batch mode (no /tts/stream)")

    async def synthesize(self, text: str) -> str:
        import base64
        import json
        import urllib.request

        if self._cloned_voice_embedding:
            # Use /tts/clone endpoint with speaker embedding
            # Use fixed seed for deterministic output with cloned voice
            payload = json.dumps({
                "text": text,
                "speaker_embedding_b64": base64.b64encode(self._cloned_voice_embedding).decode(),
                "speed": self._speed,
                "pitch": self._pitch_shift,
                "seed": self._clone_seed,
            }).encode()
            endpoint = f"{self._base_url}/tts/clone"
        else:
            # Use /tts endpoint with speaker_id
            payload = json.dumps({
                "text": text,
                "sid": self._speaker_id,
                "speed": self._speed,
                "pitch": self._pitch_shift,
            }).encode()
            endpoint = f"{self._base_url}/tts"

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _request_wav() -> bytes:
            resp = urllib.request.urlopen(req, timeout=30)
            try:
                return resp.read()
            finally:
                close = getattr(resp, "close", None)
                if callable(close):
                    close()

        wav_bytes = await asyncio.to_thread(_request_wav)

        tmp = tempfile.NamedTemporaryFile(
            prefix="reachy_claw_kokoro_", suffix=".wav", delete=False
        )
        tmp.write(wav_bytes)
        tmp.close()
        return tmp.name

    async def synthesize_streaming(self, text: str):
        """Stream PCM chunks from Jetson TTS endpoint.

        Supports both speaker_id mode (/tts/stream) and clone mode (/tts/clone/stream).
        """
        import asyncio
        import base64
        import json
        import struct
        import urllib.request

        import numpy as np

        if self._cloned_voice_embedding:
            # Clone streaming mode: /tts/clone/stream
            payload = json.dumps({
                "text": text,
                "speaker_embedding_b64": base64.b64encode(self._cloned_voice_embedding).decode(),
                "speed": self._speed,
                "pitch": self._pitch_shift,
                "seed": self._clone_seed,
            }).encode()
            endpoint = f"{self._base_url}/tts/clone/stream"
        else:
            # Standard streaming with speaker_id
            payload = json.dumps({
                "text": text,
                "sid": self._speaker_id,
                "speed": self._speed,
                "pitch": self._pitch_shift,
            }).encode()
            endpoint = f"{self._base_url}/tts/stream"

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        resp = await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=30))

        # Read sample_rate from first 4 bytes (little-endian uint32)
        sr_bytes = await asyncio.to_thread(resp.read, 4)
        if len(sr_bytes) < 4:
            return
        sample_rate = struct.unpack("<I", sr_bytes)[0]

        # Read 16-bit PCM chunks (1024 samples = ~64ms at 16kHz for low TTFC)
        chunk_bytes_size = 1024 * 2  # 1024 int16 samples
        while True:
            raw = await asyncio.to_thread(resp.read, chunk_bytes_size)
            if not raw:
                break
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            yield audio, sample_rate


class _ReconnectingTTS(TTSBackend):
    """Proxy that starts with a fallback and hot-swaps to the preferred backend."""

    def __init__(self, fallback: TTSBackend, factory, probe_interval: float = 10.0):
        self._backend = fallback
        self._factory = factory  # callable that returns preferred backend or raises
        self._probe_interval = probe_interval
        self._task: asyncio.Task | None = None
        self._probe_pending = False

    @property
    def supports_streaming(self) -> bool:
        return self._backend.supports_streaming

    def start_probing(self) -> None:
        """Schedule background probe loop. Safe to call with or without a running loop."""
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._probe_loop())
        except RuntimeError:
            self._probe_pending = True

    async def _probe_loop(self) -> None:
        while True:
            await asyncio.sleep(self._probe_interval)
            try:
                preferred = await asyncio.to_thread(self._factory)
                old = self._backend
                self._backend = preferred
                logger.info("TTS reconnected to preferred backend")
                old.cleanup()
                return
            except Exception:
                logger.debug("TTS preferred backend still unavailable, retrying...")

    async def synthesize(self, text: str) -> str:
        if self._probe_pending:
            self._probe_pending = False
            try:
                self._task = asyncio.create_task(self._probe_loop())
            except RuntimeError:
                pass
        return await self._backend.synthesize(text)

    async def synthesize_streaming(self, text: str):
        async for chunk in self._backend.synthesize_streaming(text):
            yield chunk

    def cleanup(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._backend.cleanup()


def create_tts_backend(
    backend: str = "elevenlabs",
    voice: str | None = None,
    model: str | None = None,
    config=None,
) -> TTSBackend:
    """Create a TTS backend by name using the registry."""
    from reachy_claw.backend_registry import get_tts_info, get_tts_names

    name = backend.lower().strip()

    info = get_tts_info(name)
    if info is None:
        available = ", ".join(get_tts_names())
        raise ValueError(f"Unknown TTS backend: {backend!r}. Choose from: {available}")

    # Build kwargs from config's backend-specific settings
    kwargs = {}
    if config:
        for field_name in info.settings_fields:
            config_key = f"{info.name}_{field_name}"
            if hasattr(config, config_key):
                kwargs[field_name] = getattr(config, config_key)
        if hasattr(config, "speech_service_url"):
            kwargs.setdefault("base_url", config.speech_service_url)

    # Pass through generic voice/model
    if voice is not None:
        kwargs.setdefault("voice", voice)
        kwargs.setdefault("voice_id", voice)
    if model is not None:
        kwargs.setdefault("model", model)

    # Filter to only params the constructor accepts
    import inspect
    sig = inspect.signature(info.cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    logger.info(f"Using TTS backend: {name}")
    try:
        instance = info.cls(**filtered)
    except Exception as e:
        if name in ("kokoro", "matcha"):
            import platform

            fallback_name = "say" if platform.system() == "Darwin" else "none"
            logger.warning(
                f"TTS backend {name!r} not available ({e}), "
                f"falling back to {fallback_name} (will retry in background)"
            )
            fallback_info = get_tts_info(fallback_name)
            if fallback_info is None:
                raise
            fallback = fallback_info.cls()
            proxy = _ReconnectingTTS(fallback, lambda: info.cls(**filtered))
            proxy.start_probing()
            return proxy
        raise

    return instance
