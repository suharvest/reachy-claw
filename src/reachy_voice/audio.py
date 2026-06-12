"""Full-duplex audio for the Reachy Mini USB sound card.

The Reachy Mini USB audio device does NOT tolerate independent input and
output streams: the moment ovs_agent's stock ``AudioIO`` lazily opens its
output stream for the first TTS, the running input stream degrades (quiet /
garbled capture), and ASR goes silent from the second turn on. The original
reachy-claw used a single duplex stream for exactly this reason.

``DuplexAudioIO`` subclasses ovs_agent's ``AudioIO`` so the public surface that
``BaseApp`` relies on (``start_capture``/``play``/``is_playing``/
``mark_playback_done``/``stop_playback``/``arm_for_next_turn``/
``set_output_sample_rate``/``close``) is inherited unchanged — only the
transport differs: ONE ``sd.RawStream`` whose callback simultaneously captures
the mic and feeds the speaker from the playback buffer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import numpy as np
import sounddevice as sd
from ovs_agent.audio_io import AudioIO

logger = logging.getLogger("reachy_voice.audio")


class DuplexAudioIO(AudioIO):
    def __init__(
        self,
        device: str | int | None = None,
        sr: int = 16000,
        chunk_ms: int = 100,
        input_channel: int = 0,
    ) -> None:
        # input_sr == output_sr == device rate; play() resamples TTS source
        # (set via set_output_sample_rate) to this rate using the inherited path.
        super().__init__(
            input_device=device,
            output_device=device,
            input_sr=sr,
            output_sr=sr,
            chunk_ms=chunk_ms,
        )
        self._duplex_stream: "sd.RawStream | None" = None
        self._in_ch = 1
        self._out_ch = 1
        # The Reachy mic (XMOS XVF3800) exposes 2 capture channels — typically one
        # is the echo-cancelled / processed output and the other is a raw/reference
        # channel. We capture BOTH and keep only this one (NOT an average, which
        # would re-mix the cancelled echo back in). Which index is the clean one is
        # determined from the per-channel RMS diagnostic during TTS playback.
        self._input_channel = int(input_channel)
        self._ch_rms = [0.0, 0.0]  # smoothed per-channel input RMS (diagnostic)
        self._dbg = 0
        # Smoothed RMS (0..1) of the audio currently going to the speaker, so
        # the motion layer can wobble the head in time with TTS. Updated in the
        # audio callback; read from the motion thread (a float read/write is
        # atomic under the GIL, so no lock needed).
        self._play_rms = 0.0

    def play_rms(self) -> float:
        """Smoothed RMS (0..1) of the speaker output right now (0 when idle)."""
        return self._play_rms

    # ── capture (the single duplex stream lives here) ────────────────
    async def start_capture(self) -> AsyncIterator[bytes]:
        self._loop = asyncio.get_running_loop()
        self._in_queue = asyncio.Queue(maxsize=64)
        self._ensure_playback_buffer()
        self._open_duplex()
        try:
            while True:
                chunk = await self._in_queue.get()
                yield chunk
        finally:
            self._close_duplex()

    def _duplex_cb(self, indata, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("duplex status: %s", status)
        # ── capture: pick ONE channel (never average) and queue ──
        try:
            x = np.frombuffer(bytes(indata), dtype=np.int16)
            if self._in_ch > 1:
                m = x.reshape(-1, self._in_ch)
                # diagnostic: smoothed RMS of each of the first two channels, so
                # we can spot which channel is echo-cancelled during TTS.
                for c in range(min(2, self._in_ch)):
                    r = float(np.sqrt(np.mean((m[:, c].astype(np.float32) / 32768.0) ** 2)))
                    self._ch_rms[c] += 0.3 * (r - self._ch_rms[c])
                ch = self._input_channel if self._input_channel < self._in_ch else 0
                x = m[:, ch].copy()
            buf = x.tobytes()
            if self._loop is not None and self._in_queue is not None:
                self._loop.call_soon_threadsafe(self._safe_put, buf)
        except Exception as e:  # pragma: no cover
            logger.warning("duplex capture error: %s", e)
        # ── playback: fill from the inherited mono buffer, upmix ──
        try:
            need_mono = frames * 2  # int16 mono bytes
            with self._playback_lock:
                n = min(need_mono, len(self._playback_buffer))
                mono = bytes(self._playback_buffer[:n])
                del self._playback_buffer[:n]
            # Speech-wobble signal: RMS of the real (pre-pad) samples, smoothed
            # so the head motion eases in/out rather than jittering per-block.
            if n > 0:
                s = np.frombuffer(mono, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
            else:
                rms = 0.0
            self._play_rms += 0.3 * (rms - self._play_rms)  # one-pole smoothing
            # diagnostic: ~every 2s, log per-channel input RMS vs speaker RMS.
            # During TTS (play_rms high), the echo-cancelled channel stays low.
            self._dbg += 1
            if self._dbg % 20 == 0:
                logger.info(
                    "audio: in_ch0=%.4f in_ch1=%.4f play=%.4f sel_ch=%d",
                    self._ch_rms[0], self._ch_rms[1], self._play_rms, self._input_channel,
                )
            if n < need_mono:
                mono += b"\x00" * (need_mono - n)
            if self._out_ch == 1:
                out = mono
            else:
                y = np.frombuffer(mono, dtype=np.int16)
                out = np.repeat(y[:, None], self._out_ch, axis=1).tobytes()
            outdata[: len(out)] = out
        except Exception as e:  # pragma: no cover
            logger.warning("duplex playback error: %s", e)
            outdata[:] = b"\x00" * len(outdata)

    def _open_duplex(self) -> None:
        # Prefer 2 INPUT channels (the device is natively 2-in) so we can pick the
        # echo-cancelled channel instead of letting PortAudio average both — which
        # re-mixes the cancelled echo back in. Fall back to mono only if forced.
        last_err: Exception | None = None
        for in_ch, out_ch in ((2, 2), (2, 1), (1, 2), (1, 1)):
            try:
                stream = sd.RawStream(
                    samplerate=self.input_sr,
                    blocksize=self._chunk_frames,
                    device=(self.input_device, self.output_device),
                    channels=(in_ch, out_ch),
                    dtype="int16",
                    callback=self._duplex_cb,
                )
                stream.start()
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            self._in_ch, self._out_ch = in_ch, out_ch
            self._duplex_stream = stream
            try:
                dev = stream.device
                name = sd.query_devices(dev[0])["name"] if dev else "(default)"
            except Exception:  # noqa: BLE001
                name = "?"
            logger.info(
                "duplex stream open: device=%s sr=%d ch=(%d,%d) chunk=%d",
                name, self.input_sr, in_ch, out_ch, self._chunk_frames,
            )
            return
        raise RuntimeError(f"could not open duplex stream: {last_err!r}")

    def _close_duplex(self) -> None:
        if self._duplex_stream is not None:
            try:
                self._duplex_stream.stop()
                self._duplex_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._duplex_stream = None

    # ── playback overrides: output IS the duplex stream ─────────────
    def _ensure_output(self) -> None:
        # No separate output stream — playback is mixed in _duplex_cb.
        return

    async def close(self) -> None:
        self._close_duplex()
        await super().close()


__all__ = ["DuplexAudioIO"]
