"""V2V (Voice-to-Voice) WebSocket client for seeed-local-voice.

A single WebSocket carries ASR + VAD + TTS for a multi-utterance session.
This module is a thin transport layer: it owns the WS, the recv loop, the
send lock, and the binary-audio parser. Higher-level orchestration (mapping
ASR finals into LLM calls, barge-in, state transitions) lives in
ConversationPlugin (Wave 2).

Wire format (JSON frames + binary PCM audio):

Outgoing (clawd → V2V):
  * first frame: `{"type":"config", ...}`
  * binary PCM16 mono mic chunks
  * `{"type":"text","text": <delta>}` — feed LLM tokens to V2V TTS
  * `{"type":"tts_flush"}` — flush sentence buffer after LLM stream end
  * `{"type":"asr_eos"}` — only at shutdown (with multi_utterance=True)
  * `{"type":"abort"}` — cancel active TTS (barge-in)

Incoming (V2V → clawd):
  * `{"type":"asr_partial","text":..., "is_stable":bool}`
  * `{"type":"asr_endpoint"}`
  * `{"type":"asr_final","text":..., "session_complete":bool, "duplicate_of_streamed":bool}`
  * `{"type":"tts_started","sentence":...}`
  * `{"type":"tts_sentence_done","sentence":...}`
  * `{"type":"tts_done"}`
  * `{"type":"vad_event","event":"speech_start"|"speech_end"}`
  * `{"type":"error","error":<msg>}`
  * binary frames: first binary frame carries 4-byte LE uint32 sample-rate
    header followed by int16 PCM mono. ALL SUBSEQUENT binary frames in the
    same WebSocket session are raw int16 PCM (no header). Upstream sends
    the header once when the first synthesised audio is ready and never
    again for the lifetime of the connection (see
    `seeed-local-voice/app/main.py::tts_out_task` `sr_header_sent` latch).
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

import websockets

logger = logging.getLogger(__name__)


@dataclass
class V2VConfig:
    """Configuration for the unified ASR/TTS/VAD WebSocket service."""

    url: str = "ws://localhost:8621/v2v/stream"
    sample_rate: int = 16000
    asr_language: str = "auto"
    tts_language: str = "auto"
    vad: str = "silero"
    vad_silence_ms: int = 700
    tts_voice: str | None = None
    tts_speed: float | None = None
    multi_utterance: bool = True


_VadEvent = Literal["speech_start", "speech_end"]


class V2VClient:
    """Bidirectional V2V WebSocket client for mic PCM in and ASR/TTS events out.

    Callbacks are externally assigned. All are optional; missing ones drop
    the event silently. The recv loop dispatches them concurrently with
    user sends, guarded by `_send_lock` for outbound serialization.

    Wave 1 reconnect policy: NONE. WS drop raises in the recv loop;
    higher layers must decide whether to reconnect.
    """

    def __init__(self, config: V2VConfig):
        self._config = config
        self._ws: Any = None  # websockets connection
        self._recv_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._connected = False
        # Per-connection TTS sample-rate latch. Upstream sends the 4-byte
        # LE uint32 SR header on the FIRST binary frame of the session
        # only (verified in seeed-local-voice/app/main.py:1214,1246-1253:
        # `sr_header_sent` is scoped to `tts_out_task` lifetime). Subsequent
        # frames are raw int16 PCM. Reset on disconnect.
        self._tts_sample_rate: int | None = None

        # Event callbacks — wire from outside before connect().
        self.on_asr_partial: Callable[[str, bool], Awaitable[None] | None] | None = None
        self.on_asr_final: Callable[[str, bool, bool], Awaitable[None] | None] | None = None
        self.on_asr_endpoint: Callable[[], Awaitable[None] | None] | None = None
        self.on_tts_started: Callable[[str], Awaitable[None] | None] | None = None
        self.on_tts_sentence_done: Callable[[str], Awaitable[None] | None] | None = None
        self.on_tts_done: Callable[[], Awaitable[None] | None] | None = None
        self.on_tts_audio: Callable[[int, bytes], Awaitable[None] | None] | None = None
        self.on_vad_event: Callable[[_VadEvent], Awaitable[None] | None] | None = None
        self.on_error: Callable[[str], Awaitable[None] | None] | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        logger.info("V2VClient connecting to %s", self._config.url)
        self._ws = await websockets.connect(
            self._config.url, max_size=None, ping_interval=20, ping_timeout=20,
        )
        # First frame: config.
        cfg_frame: dict[str, Any] = {
            "type": "config",
            "sample_rate": self._config.sample_rate,
            "asr_language": self._config.asr_language,
            "tts_language": self._config.tts_language,
            "vad": self._config.vad,
            "vad_silence_ms": self._config.vad_silence_ms,
            "multi_utterance": self._config.multi_utterance,
        }
        if self._config.tts_voice is not None:
            cfg_frame["tts_voice"] = self._config.tts_voice
        if self._config.tts_speed is not None:
            cfg_frame["tts_speed"] = self._config.tts_speed
        await self._ws.send(json.dumps(cfg_frame))
        self._connected = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("V2VClient connected; config sent: %s", cfg_frame)

    async def disconnect(self) -> None:
        self._connected = False
        # Reset per-connection state so a reconnect re-reads the SR header.
        self._tts_sample_rate = None
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("V2VClient disconnected")

    async def send_audio(self, pcm16: bytes) -> None:
        if not self.is_connected or self._ws is None:
            raise RuntimeError("V2VClient not connected")
        async with self._send_lock:
            await self._ws.send(pcm16)

    async def send_text_delta(self, text: str) -> None:
        await self._send_json({"type": "text", "text": text})

    async def flush_tts(self) -> None:
        await self._send_json({"type": "tts_flush"})

    async def send_asr_eos(self) -> None:
        await self._send_json({"type": "asr_eos"})

    async def abort(self) -> None:
        await self._send_json({"type": "abort"})

    # ── Internal ──────────────────────────────────────────────────────

    async def _send_json(self, frame: dict[str, Any]) -> None:
        if not self.is_connected or self._ws is None:
            raise RuntimeError("V2VClient not connected")
        async with self._send_lock:
            await self._ws.send(json.dumps(frame))

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    await self._handle_binary(msg)
                else:
                    await self._handle_text(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("V2V recv loop terminated: %s", e)
            self._connected = False
            if self.on_error:
                await _maybe_await(self.on_error(str(e)))

    async def _handle_binary(self, data: bytes) -> None:
        # First binary frame of the session: 4-byte LE uint32 SR header
        # followed by PCM. Subsequent frames are raw PCM (no header).
        if self._tts_sample_rate is None:
            if len(data) < 4:
                logger.warning(
                    "V2V binary frame too short for SR header: %d bytes", len(data),
                )
                return
            self._tts_sample_rate = struct.unpack("<I", data[:4])[0]
            pcm = data[4:]
            logger.debug("V2V SR header parsed: %d Hz", self._tts_sample_rate)
            if pcm and self.on_tts_audio:
                await _maybe_await(self.on_tts_audio(self._tts_sample_rate, pcm))
            return
        if self.on_tts_audio:
            await _maybe_await(self.on_tts_audio(self._tts_sample_rate, data))

    async def _handle_text(self, raw: str) -> None:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("V2V malformed JSON frame: %r", raw[:200])
            return
        ftype = frame.get("type")
        if ftype == "asr_partial":
            if self.on_asr_partial:
                await _maybe_await(self.on_asr_partial(
                    frame.get("text", ""), bool(frame.get("is_stable", False)),
                ))
        elif ftype == "asr_endpoint":
            if self.on_asr_endpoint:
                await _maybe_await(self.on_asr_endpoint())
        elif ftype == "asr_final":
            if self.on_asr_final:
                await _maybe_await(self.on_asr_final(
                    frame.get("text", ""),
                    bool(frame.get("session_complete", False)),
                    bool(frame.get("duplicate_of_streamed", False)),
                ))
        elif ftype == "tts_started":
            if self.on_tts_started:
                await _maybe_await(self.on_tts_started(frame.get("sentence", "")))
        elif ftype == "tts_sentence_done":
            if self.on_tts_sentence_done:
                await _maybe_await(
                    self.on_tts_sentence_done(frame.get("sentence", ""))
                )
        elif ftype == "tts_done":
            if self.on_tts_done:
                await _maybe_await(self.on_tts_done())
        elif ftype == "vad_event":
            event = frame.get("event")
            if event in ("speech_start", "speech_end") and self.on_vad_event:
                await _maybe_await(self.on_vad_event(event))
            else:
                logger.debug("V2V vad_event unknown: %r", event)
        elif ftype == "error":
            err = frame.get("error") or frame.get("message") or "unknown"
            logger.error("V2V error frame: %s", err)
            if self.on_error:
                await _maybe_await(self.on_error(str(err)))
        else:
            logger.debug("V2V unknown frame type: %r", ftype)


async def _maybe_await(result: Any) -> None:
    if asyncio.iscoroutine(result):
        await result
