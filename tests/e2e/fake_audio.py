"""Deterministic mic/speaker replacement for live Reachy simulation tests."""

from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path
from typing import AsyncIterator

import numpy as np


def _read_wav(path: str | Path, target_sr: int) -> bytes:
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("scripted audio requires PCM16 WAV")
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if sr != target_sr and samples.size:
        size = int(samples.size * target_sr / sr)
        samples = np.interp(
            np.linspace(0, 1, size, endpoint=False),
            np.linspace(0, 1, samples.size, endpoint=False),
            samples.astype(np.float32),
        ).astype(np.int16)
    return samples.tobytes()


class ScriptedAudioIO:
    """Yield one WAV after silence and capture all streamed TTS bytes."""

    def __init__(
        self,
        wav_path: str | Path,
        *,
        delay_ms: int = 1500,
        input_sr: int = 16000,
        chunk_ms: int = 100,
    ) -> None:
        self.wav_path = Path(wav_path)
        self.delay_ms = delay_ms
        self.input_sr = input_sr
        self.output_sr = input_sr
        self.chunk_ms = chunk_ms
        self.captured_tts = bytearray()
        self.tts_first_frame_ts_ms: int | None = None
        self._closed = False
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    async def start_capture(self) -> AsyncIterator[bytes]:
        chunk_bytes = self.input_sr * self.chunk_ms // 1000 * 2
        silence = bytes(chunk_bytes)
        for _ in range(self.delay_ms // self.chunk_ms):
            yield silence
            await asyncio.sleep(self.chunk_ms / 1000)
        pcm = _read_wav(self.wav_path, self.input_sr)
        for pos in range(0, len(pcm), chunk_bytes):
            chunk = pcm[pos : pos + chunk_bytes]
            yield chunk + bytes(chunk_bytes - len(chunk))
            await asyncio.sleep(self.chunk_ms / 1000)
        while not self._closed:
            # Silence must keep flowing so both client/server VAD observe EOS.
            yield silence
            await asyncio.sleep(self.chunk_ms / 1000)

    async def play(self, pcm: bytes) -> None:
        if self.tts_first_frame_ts_ms is None:
            self.tts_first_frame_ts_ms = int(time.time() * 1000)
        self._is_playing = True
        self.captured_tts.extend(pcm)

    async def stop_playback(self) -> None:
        self._is_playing = False

    def mark_playback_done(self) -> None:
        self._is_playing = False

    def set_output_sample_rate(self, sample_rate: int) -> None:
        self.output_sr = int(sample_rate)

    async def close(self) -> None:
        self._closed = True


__all__ = ["ScriptedAudioIO"]
