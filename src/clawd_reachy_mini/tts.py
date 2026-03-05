"""Text-to-speech backends for Reachy Mini interface."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

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


def create_tts_backend(
    backend: str = "elevenlabs",
    voice: str | None = None,
    model: str | None = None,
) -> TTSBackend:
    """Create a TTS backend from a name string.

    Args:
        backend: One of "elevenlabs", "macos-say", "piper", "none".
        voice: Backend-specific voice identifier.
        model: Backend-specific model path or ID.
    """
    name = backend.lower().strip()

    if name == "elevenlabs":
        logger.info("Using ElevenLabs cloud TTS")
        return ElevenLabsTTS(voice_id=voice)
    elif name in ("macos-say", "say"):
        logger.info(f"Using macOS say TTS (voice={voice or 'default'})")
        return MacOSSayTTS(voice=voice)
    elif name == "piper":
        logger.info(f"Using Piper TTS (model={model or 'env'})")
        return PiperTTS(model=model)
    elif name == "none":
        logger.info("TTS disabled")
        return NoopTTS()
    else:
        raise ValueError(
            f"Unknown TTS backend: {backend!r}. "
            "Choose from: elevenlabs, macos-say, piper, none"
        )
