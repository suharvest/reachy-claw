"""Text-to-speech backends for Reachy Mini interface."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod

from clawd_reachy_mini.backend_registry import register_tts

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    """Abstract base class for text-to-speech backends."""

    @abstractmethod
    async def synthesize(self, text: str) -> str:
        """Synthesize speech from text.

        Returns:
            Path to a temporary audio file (WAV or MP3). Caller is
            responsible for deleting it after playback.
        """

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
        from clawd_reachy_mini.elevenlabs import load_elevenlabs_config

        self._config = load_elevenlabs_config(
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )

    async def synthesize(self, text: str) -> str:
        from clawd_reachy_mini.elevenlabs import elevenlabs_tts_to_temp_audio_file

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
            prefix="clawd_reachy_say_", suffix=".aiff", delete=False
        )
        tmp.close()
        cmd = ["say"]
        if self._voice:
            cmd += ["-v", self._voice]
        if self._rate:
            cmd += ["-r", str(self._rate)]
        cmd += ["-o", tmp.name, "--", text]
        proc = subprocess.run(cmd, capture_output=True)
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
            prefix="clawd_reachy_piper_", suffix=".wav", delete=False
        )
        tmp.close()
        cmd = ["piper", "--model", self._model, "--output_file", tmp.name]
        if self._speaker is not None:
            cmd += ["--speaker", str(self._speaker)]
        proc = subprocess.run(cmd, input=text.encode(), capture_output=True)
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
            prefix="clawd_reachy_noop_", suffix=".wav", delete=False
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


@register_tts("kokoro")
class KokoroTTS(TTSBackend):
    """Remote Kokoro TTS via Jetson speech service (sherpa-onnx, CUDA)."""

    class Settings:
        speaker_id: int = 50
        speed: float = 1.0

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        speaker_id: int = 50,
        speed: float = 1.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._speaker_id = speaker_id
        self._speed = speed

    async def synthesize(self, text: str) -> str:
        import json
        import urllib.request

        payload = json.dumps({
            "text": text,
            "sid": self._speaker_id,
            "speed": self._speed,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/tts",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        wav_bytes = resp.read()

        tmp = tempfile.NamedTemporaryFile(
            prefix="clawd_reachy_kokoro_", suffix=".wav", delete=False
        )
        tmp.write(wav_bytes)
        tmp.close()
        return tmp.name


def create_tts_backend(
    backend: str = "elevenlabs",
    voice: str | None = None,
    model: str | None = None,
    config=None,
) -> TTSBackend:
    """Create a TTS backend by name using the registry."""
    from clawd_reachy_mini.backend_registry import get_tts_info, get_tts_names

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
    return info.cls(**filtered)
