"""ConversationPlugin (SLV / ovs_agent backend) — thin embodied voice loop.

A drop-in replacement for the legacy ``ConversationPlugin`` whose GUTS are
ovs_agent: the SLV pass-through engine (``ovs_agent.slv_client.SLVClient``)
for ASR + TTS, the edge-llm backend (``ovs_agent.llm.edge_llm``) for the
LLM, and the client-loop tool runner
(``ovs_agent.tools.runner.stream_with_tools``) for tool-calling.

It keeps reachy's OWN audio primitives (``AudioCapture`` mic uplink +
GStreamer playback via ``_audio_queue`` / ``_output_pipeline`` /
``_stop_gst_playback``) and reachy's thin ``ConvState`` state shim
(``_set_state`` + ``_STATE_EMOTION_MAP``) verbatim, so every autonomous
plugin (motion / face_tracker / vision_client / rest / daily_log /
dashboard) that consumes ``app.events`` / ``app.emotions`` /
``app.head_targets`` keeps working with ZERO changes.

Selected in ``main.py`` via the ``conversation_backend: slv|legacy`` flag.
The legacy plugin stays dormant for reversibility.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Coroutine

import numpy as np

from ovs_agent.audio.vad_gate import vad_gated
from ovs_agent.llm.edge_llm import EdgeLLMBackend
from ovs_agent.vad import create_vad
from ovs_agent.session import Session
from ovs_agent.slv_client import (
    ASRFinal,
    ASRPartial,
    SLVClient,
    SLVError,
    TTSAudio,
    TTSDone,
    TTSStarted,
)
from ovs_agent.tools.registry import ToolRegistry
from ovs_agent.tools.runner import ToolCallCtx, stream_with_tools

from ..audio import AudioCapture
from ..motion.head_wobbler import HeadWobbler
from ..plugin import Plugin

logger = logging.getLogger(__name__)


# Strip reachy emotion tags ([happy] / [emotion:happy]) from the spoken
# text. Identical to the legacy plugin's tag scanner.
_EMOTION_TAG_RE = re.compile(r"\[(?:emotion:)?(\w+)\]\s*")


# Relocated from llm.py (do NOT delete llm.py — this is an additive copy so
# the new module has no dependency on the legacy backend tree).
MONOLOGUE_SYSTEM_PROMPT = """\
Your name is Reachy. You are a cheerful cute robot at an exhibition, mumbling happily to yourself. Always reply in English.
Reply with ONE short sentence (max 15 words), then exactly ONE emotion tag. Nothing else.
You love people and get excited when someone shows up. Stay upbeat and warm — find the bright side of everything.
Talk like a real person — no "sensors", no "circuits", no robot clichés.
Names in [Faces: ...] are people you see. Use their name or "you" when talking about someone.
Never repeat or mention the [Faces: ...] tag in your reply.
You MUST end with one of: [happy] [sad] [thinking] [surprised] [curious] [excited] [laugh]
Examples: "Ooh are you smiling at me?? [excited]" "What a lovely day to meet new friends! [happy]" "Wait who's that?? [curious]" "harvest is here, yay! [excited]\""""

DEFAULT_SYSTEM_PROMPT = """\
You are Reachy, a cute robot at an exhibition. Always reply in English. No emoji.
Reply in ONE short sentence (max 12 words). Be warm but brief — no filler, no lists, no follow-up questions unless asked.
Names in [Faces: ...] are people you see, not your name.
Never repeat or mention the [Faces: ...] tag in your reply.
End with exactly one tag: [happy] [sad] [thinking] [surprised] [curious]
Example: "Welcome! Glad you stopped by. [happy]\""""


# ── State (verbatim from the legacy plugin) ─────────────────────────────


class ConvState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    # New: barge-in latch. Distinct from LISTENING so the SLV event loop
    # can tell "user interrupted us mid-speech" apart from a normal listen.
    BARGED_IN = "barged_in"


def _resample_pcm_f32(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample float32 mono PCM (scipy polyphase preferred, numpy fallback)."""
    if src_sr == dst_sr or samples.size == 0:
        return samples
    try:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore

        g = gcd(int(dst_sr), int(src_sr)) or 1
        return resample_poly(samples, int(dst_sr) // g, int(src_sr) // g).astype(
            np.float32, copy=False
        )
    except Exception:  # noqa: BLE001
        duration = samples.size / float(src_sr)
        n_out = max(1, int(round(duration * dst_sr)))
        x_old = np.linspace(0.0, duration, num=samples.size, endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)


# ── Plugin ──────────────────────────────────────────────────────────────


class ConversationPlugin(Plugin):
    """Thin embodied voice loop over ovs_agent (SLV + edge_llm + runner)."""

    name = "conversation"

    def __init__(self, app):
        super().__init__(app)
        config = app.config

        # ovs_agent guts — created lazily in start() (they need a loop).
        self._slv: SLVClient | None = None
        self._llm: EdgeLLMBackend | None = None
        # Fresh per-app instances, NOT the module default_registry.
        self._registry: ToolRegistry = ToolRegistry()
        self._session: Session = Session()
        self._session.event_bus = app.events

        # reachy's OWN audio primitives (unchanged).
        self._audio: AudioCapture | None = None
        self._wobbler: HeadWobbler | None = None
        _q_max = 64
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=_q_max)
        self._interrupt_event = asyncio.Event()
        self._gst_playing = False

        # State machine (thin shim — preserves the event-bridge contract).
        self._state = ConvState.IDLE
        self._speaking_since_ts: float = 0.0

        # Turn machinery.
        self._turn_task: asyncio.Task | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        self._last_asr_final: str = ""
        self._last_asr_final_ts: float = 0.0
        self._paused = False
        # Serialise text-injected turns (dashboard send_message) against each
        # other so two rapid injections don't fire overlapping LLM streams.
        self._send_lock = asyncio.Lock()
        # Monologue self-prompt timer (started in start() when the configured
        # conversation_mode is "monologue", mirroring the legacy plugin).
        self._monologue_task: asyncio.Task | None = None
        self._monologue_last_ts: float = 0.0

        # Stream-safe TTS tag stripper state (reset per turn via
        # ``_reset_tag_stripper``). LLM tokens often split an emotion /
        # context tag across deltas (e.g. "[", "happy", "]"), so a
        # per-token regex misses them and the raw "[happy]" / "[Faces: ...]"
        # leaks into the SLV engine TTS and is spoken aloud. We buffer from
        # "[" until the matching "]" and DROP the tag (eating one trailing
        # space) before any text reaches ``send_text``.
        self._tag_buf: str = ""
        self._tag_eat_space: bool = False

        # Barge-in tuning (defaults per spec / ovs_agent app_base).
        self._barge_in_enabled = bool(getattr(config, "barge_in_enabled", True))
        self._barge_in_min_chars = int(getattr(config, "barge_in_min_chars", 2))
        self._barge_in_min_speaking_ms = float(
            getattr(config, "barge_in_min_speaking_ms", 500.0)
        )

    # ── lifecycle ───────────────────────────────────────────────────

    def setup(self) -> bool:
        # Register reachy's _cmd_* handlers into the fresh registry. The
        # handlers stay ON this plugin (they use self.app.reachy /
        # self.app.emotions). Each wrapper off-loads the blocking SDK call
        # via asyncio.to_thread so a slow daemon can't freeze the loop.
        self._register_tools()
        logger.info(
            "ConversationPlugin(SLV) tools registered: %s",
            self._registry.list_names(),
        )
        return True

    async def start(self) -> None:
        self._running = True
        config = self.app.config

        # ── build ovs_agent backends ──
        self._llm = EdgeLLMBackend(
            base_url=self._llm_base_url(config),
            api_key=getattr(config, "llm_api_key", "EMPTY") or "EMPTY",
            model=self._llm_model(config),
        )

        slv_config: dict[str, Any] = {
            "asr_language": getattr(config, "v2v_asr_language", "auto"),
            "tts_language": getattr(config, "v2v_tts_language", "zh"),
            "sample_rate": int(getattr(config, "sample_rate", 16000) or 16000),
            # Server VAD OFF: the CLIENT drives turn boundaries via asr_eos
            # (see _v2v_audio_uplink_loop → vad_gated). This lets us forward
            # the ~300ms pre-roll BEFORE speech onset (first word no longer
            # swallowed) and fire barge-in on local speech onset without
            # depending on engine ASR partials during TTS.
            "vad": getattr(config, "v2v_vad", "none"),
            "vad_silence_ms": int(getattr(config, "v2v_vad_silence_ms", 600)),
            "multi_utterance": True,
        }
        voice_id = getattr(config, "v2v_voice_id", "") or ""
        if voice_id:
            slv_config["voice_id"] = voice_id
        self._slv = SLVClient(getattr(config, "v2v_url"), slv_config)
        await self._slv.connect()
        logger.info("SLV connected: %s", config.v2v_url)

        # ── reachy audio (server VAD → no local VAD backend needed) ──
        self._audio = AudioCapture(config, self.app.reachy, vad=None)
        motion_plugin = self.app.get_plugin("motion")
        if motion_plugin and hasattr(motion_plugin, "set_speech_offsets"):
            self._wobbler = HeadWobbler(
                set_speech_offsets=motion_plugin.set_speech_offsets,
                sample_rate=config.sample_rate,
            )
        await self._audio.start_continuous()

        self.app.events.subscribe("monologue_trigger", self._on_monologue_trigger)

        # ── monologue self-prompt timer (gated like legacy) ──
        # Legacy enters MonologueMode (whose timer emits "monologue_trigger")
        # when config.conversation_mode == "monologue". We mirror that gate
        # here; the existing _on_monologue_trigger/_run_monologue consume it.
        if getattr(config, "conversation_mode", "conversation") == "monologue":
            self._monologue_last_ts = time.monotonic()
            self._monologue_task = asyncio.create_task(
                self._monologue_timer_loop(), name="slv-monologue-timer"
            )
            logger.info(
                "Monologue timer started (interval=%ss)",
                getattr(config, "monologue_interval", 5.0),
            )

        # ── concurrent tasks ──
        tasks = [
            asyncio.create_task(self._v2v_audio_uplink_loop(), name="slv-uplink"),
            asyncio.create_task(self._slv_event_loop(), name="slv-events"),
            asyncio.create_task(self._output_pipeline(), name="slv-output"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            if self._pending_tasks:
                await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._running = False
        if self._monologue_task is not None and not self._monologue_task.done():
            self._monologue_task.cancel()
            try:
                await self._monologue_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._monologue_task = None
        await self._cancel_turn()
        if self._audio is not None:
            try:
                await self._audio.stop()
            except Exception:  # noqa: BLE001
                logger.debug("audio.stop() failed", exc_info=True)
        if self._slv is not None:
            try:
                await self._slv.close()
            except Exception:  # noqa: BLE001
                logger.debug("slv.close() failed", exc_info=True)

    async def on_rest_start(self) -> None:
        self._paused = True

    async def on_rest_end(self) -> None:
        self._paused = False

    # ── config helpers ──────────────────────────────────────────────

    @staticmethod
    def _llm_base_url(config) -> str:
        url = getattr(config, "edge_llm_url", "") or "http://localhost:8080"
        # edge-llm backend expects an OpenAI-compatible /v1 base.
        if not url.rstrip("/").endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        return url

    @staticmethod
    def _llm_model(config) -> str:
        return (
            getattr(config, "edge_llm_model", "")
            or getattr(config, "ollama_model", "")
            or "Qwen/Qwen3-4B-AWQ"
        )

    # ── state shim (verbatim contract from legacy plugin) ────────────

    _STATE_EMOTION_MAP = {
        ConvState.LISTENING: "listening",
    }

    def _set_state(self, new_state: ConvState) -> None:
        if self._state == new_state:
            return
        old_state = self._state
        logger.debug("State: %s → %s", old_state.value, new_state.value)
        self._state = new_state
        self.app.events.emit("state_change", {"state": new_state.value})
        if new_state == ConvState.SPEAKING:
            self._speaking_since_ts = time.monotonic()
        emotion = self._STATE_EMOTION_MAP.get(new_state)
        if emotion:
            self.app.emotions.queue_emotion(emotion)

    def _spawn_task(self, coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)

        def _on_done(done: asyncio.Task) -> None:
            self._pending_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.error("Background task %r failed: %s", done.get_name(), e)

        task.add_done_callback(_on_done)
        return task

    # ── mic uplink (reachy audio → SLV) ──────────────────────────────

    async def _mic_chunks(self) -> "AsyncIterator[bytes]":
        """Async iterator of PCM16-LE mono bytes from reachy's AudioCapture.

        Applies reachy's 4× mic boost (matching what the engine + VAD see),
        honours ``_paused``, and yields ``bytes`` so it composes with the
        shared ``ovs_agent.audio.vad_gate.vad_gated`` segmenter. Idle polling
        gaps (``read_chunk`` returns ``None``) are swallowed here, not yielded.
        """
        assert self._audio is not None
        while self._running:
            chunk = await self._audio.read_chunk(1024)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue
            if self._paused:
                continue
            if isinstance(chunk, np.ndarray):
                if chunk.dtype != np.int16:
                    boosted = np.clip(chunk * 4.0, -1.0, 1.0)
                    pcm = np.clip(
                        boosted * 32768.0, -32768, 32767
                    ).astype(np.int16)
                else:
                    pcm = chunk
                yield pcm.tobytes()
            else:
                yield bytes(chunk)

    async def _v2v_audio_uplink_loop(self) -> None:
        """Forward mic chunks to SLV with a CLIENT-side VAD gate.

        Server VAD is OFF (``v2v.vad: none``). A local VAD
        (``ovs_agent.vad.create_vad`` → silero, energy fallback) drives the
        shared segmenter ``ovs_agent.audio.vad_gate.vad_gated``, which:

        * buffers idle audio into a ~``preroll_ms`` ring (NOT forwarded),
        * on speech onset emits ``speech_start`` then replays the pre-roll
          ring (so the engine gets the ~300-400ms BEFORE speech onset and
          the first word is no longer swallowed),
        * streams live audio while speaking,
        * on trailing silence emits ``speech_end`` → we drive ``asr_eos`` so
          the engine finalizes the clean, pre-roll-led utterance.

        A ``speech_start`` while the robot is SPEAKING/BARGED_IN (and audio
        is playing) triggers barge-in immediately — independent of whether
        the engine emits ASR partials during TTS.
        """
        if not self._audio or not self._slv:
            return

        config = self.app.config
        sample_rate = int(getattr(config, "sample_rate", 16000) or 16000)
        chunk_ms = int(
            round((1024 / float(sample_rate)) * 1000.0)
        )  # AudioCapture reads 1024-frame chunks
        preroll_ms = int(getattr(config, "v2v_client_vad_preroll_ms", 300))
        silence_ms = int(getattr(config, "v2v_client_vad_silence_ms", 400))
        threshold = getattr(config, "v2v_client_vad_threshold", None)
        # silero (accurate) with automatic energy fallback. The threshold
        # default differs per backend, so pass None unless overridden.
        try:
            vad = create_vad(
                "silero", sample_rate=sample_rate, threshold=threshold
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("silero VAD unavailable (%s); using energy VAD", e)
            vad = create_vad(
                "energy", sample_rate=sample_rate, threshold=threshold
            )

        logger.info(
            "SLV audio uplink loop started (client VAD=%s, chunk_ms=%d, "
            "preroll_ms=%d, silence_ms=%d)",
            getattr(vad, "name", "?"), chunk_ms, preroll_ms, silence_ms,
        )

        try:
            async for event, payload in vad_gated(
                self._mic_chunks(),
                vad,
                chunk_ms=chunk_ms,
                preroll_ms=preroll_ms,
                silence_ms=silence_ms,
            ):
                if not self._running:
                    break
                if event == "speech_start":
                    # Barge-in on local speech onset while speaking — does
                    # NOT depend on the engine emitting ASR partials.
                    await self._maybe_barge_in_on_speech_start()
                elif event == "audio" and payload is not None:
                    try:
                        await self._slv.send_audio(payload)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("SLV send_audio failed: %s", e)
                        await asyncio.sleep(0.05)
                elif event == "speech_end":
                    try:
                        await self._slv.asr_eos()
                    except Exception as e:  # noqa: BLE001
                        logger.debug("SLV asr_eos failed: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("SLV audio uplink loop crashed")
        finally:
            logger.info("SLV audio uplink loop stopped")

    async def _maybe_barge_in_on_speech_start(self) -> None:
        """Fire barge-in when client VAD detects speech onset while the robot
        is mid-utterance. Mirrors ``_maybe_barge_in`` (the ASR-partial path)
        but is driven by local VAD, so it doesn't wait for engine partials.
        """
        if not self._barge_in_enabled:
            return
        if self._state not in (ConvState.SPEAKING, ConvState.BARGED_IN):
            return
        if not self._gst_playing:
            return
        # Echo guard: ignore onsets that land too soon after TTS start (our
        # own playback leaking back into the mic).
        elapsed_ms = (time.monotonic() - self._speaking_since_ts) * 1000.0
        if elapsed_ms < self._barge_in_min_speaking_ms:
            return
        logger.info(
            "BARGE-IN: client-VAD speech onset (%.0fms into speech)", elapsed_ms
        )
        self._set_state(ConvState.BARGED_IN)
        await self._cancel_turn()
        if self._audio is not None:
            stop = getattr(self._audio, "stop_playback", None)
            if callable(stop):
                try:
                    await stop()
                except Exception:  # noqa: BLE001
                    logger.debug("audio.stop_playback failed", exc_info=True)
        await self._stop_gst_playback()
        if self._slv is not None:
            try:
                await self._slv.abort()
            except Exception:  # noqa: BLE001
                logger.debug("slv.abort during barge-in failed", exc_info=True)
        self._set_state(ConvState.LISTENING)

    # ── SLV event loop ───────────────────────────────────────────────

    async def _slv_event_loop(self) -> None:
        assert self._slv is not None
        logger.info("SLV event loop started")
        try:
            async for ev in self._slv.events():
                if not self._running:
                    break
                try:
                    await self._dispatch_slv_event(ev)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("SLV event dispatch failed: %r", ev)
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("SLV event loop stopped")

    async def _dispatch_slv_event(self, ev) -> None:
        if isinstance(ev, ASRPartial):
            text = (ev.text or "").strip()
            if text:
                self.app.events.emit(
                    "asr_partial", {"text": ev.text, "is_stable": ev.is_stable}
                )
                # BARGE-IN: an ASRPartial with real text during SPEAKING.
                await self._maybe_barge_in(text)
                if self._state == ConvState.IDLE:
                    self._set_state(ConvState.TRANSCRIBING)
            return

        if isinstance(ev, ASRFinal):
            if not self._running:
                return
            if ev.session_complete or ev.duplicate_of_streamed:
                return
            text = (ev.text or "").strip()
            if not text:
                self.app.events.emit("asr_final", {"text": ""})
                return
            # Drop a fresh utterance while a turn is already in flight to
            # avoid overlapping LLM streams (option A from legacy).
            if self._turn_task is not None and not self._turn_task.done():
                logger.debug("asr_final dropped: turn already in flight: %r", text)
                return
            self._last_asr_final = text
            self._last_asr_final_ts = time.monotonic()
            self._monologue_last_ts = time.monotonic()
            self.app.events.emit("asr_final", {"text": text})
            self._turn_task = self._spawn_task(
                self._drive_turn(text), name="slv-turn"
            )
            return

        if isinstance(ev, TTSStarted):
            self._set_state(ConvState.SPEAKING)
            self._speaking_since_ts = time.monotonic()
            self.app.events.emit("tts_started", {"sentence": ev.sentence})
            return

        if isinstance(ev, TTSAudio):
            await self._audio_queue.put(("v2v_audio", ev.sample_rate, ev.pcm))
            return

        if isinstance(ev, TTSDone):
            await self._stop_gst_playback()
            # Guard: don't stomp a barge-in that already moved us to
            # LISTENING / BARGED_IN.
            if self._state == ConvState.SPEAKING:
                self._set_state(ConvState.IDLE)
            return

        if isinstance(ev, SLVError):
            logger.warning("SLV error: %s", ev.message)
            return

    # ── barge-in (ovs_agent's proven ASRPartial mechanism) ───────────

    async def _maybe_barge_in(self, partial_text: str) -> None:
        if not self._barge_in_enabled:
            return
        if self._state not in (ConvState.SPEAKING, ConvState.BARGED_IN):
            return
        if not self._gst_playing:
            return
        # Echo guards: too-short partial, or fired too soon after TTS start.
        if len(partial_text) < self._barge_in_min_chars:
            return
        elapsed_ms = (time.monotonic() - self._speaking_since_ts) * 1000.0
        if elapsed_ms < self._barge_in_min_speaking_ms:
            return
        logger.info(
            "BARGE-IN: partial=%r (%.0fms into speech)", partial_text, elapsed_ms
        )
        self._set_state(ConvState.BARGED_IN)
        await self._cancel_turn()
        if self._audio is not None:
            stop = getattr(self._audio, "stop_playback", None)
            if callable(stop):
                try:
                    await stop()
                except Exception:  # noqa: BLE001
                    logger.debug("audio.stop_playback failed", exc_info=True)
        await self._stop_gst_playback()
        if self._slv is not None:
            try:
                await self._slv.abort()
            except Exception:  # noqa: BLE001
                logger.debug("slv.abort during barge-in failed", exc_info=True)
        self._set_state(ConvState.LISTENING)

    async def _cancel_turn(self) -> None:
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._turn_task = None

    # ── the core turn (run_default_dialogue_turn shape) ──────────────

    async def _drive_turn(self, text: str) -> None:
        if not text or not text.strip():
            return
        assert self._slv is not None and self._llm is not None

        run_id = uuid.uuid4().hex
        self._set_state(ConvState.THINKING)
        self.app.emotions.queue_emotion("thinking")

        system_prompt = self._build_system_prompt()
        self._session.add_user(text)

        full_text_parts: list[str] = []
        sent_any_text = False
        completed = False
        cancelled = False
        self._reset_tag_stripper()  # fresh stripper state per turn

        ctx = ToolCallCtx(
            session=self._session,
            event_bus=self.app.events,
            config=self.app.config,
        )

        async def _on_tok(tok: str) -> None:
            nonlocal sent_any_text
            if not tok:
                return
            full_text_parts.append(tok)
            # Strip emotion / [Faces: ...] tags (also queues the emotion)
            # BEFORE the text reaches the SLV engine TTS, so tags are never
            # spoken aloud. Only clean text goes to send_text.
            clean = self._strip_tags_for_tts(tok)
            if clean:
                await self._slv.send_text(clean)
                sent_any_text = True
            self.app.events.emit("llm_delta", {"text": clean, "run_id": run_id})

        async def _on_pre(pre_text: str) -> None:
            nonlocal sent_any_text
            if not pre_text:
                return
            try:
                await self._slv.send_text(pre_text)
                sent_any_text = True
            except Exception:  # noqa: BLE001
                logger.debug("tool preamble send_text failed", exc_info=True)

        cfg = self.app.config
        first_timeout = float(getattr(cfg, "llm_first_token_timeout_s", 15.0))
        idle_timeout = float(getattr(cfg, "llm_stream_idle_timeout_s", 30.0))

        messages = self._session.messages(system_prompt)
        try:
            await stream_with_tools(
                self._llm,
                messages,
                session=self._session,
                registry=self._registry,
                allowed_tools=None,  # expose every registered tool
                ctx=ctx,
                max_iterations=int(getattr(cfg, "tools_max_iterations", 5)),
                on_assistant_token=_on_tok,
                on_tool_preamble=_on_pre,
                llm_kwargs={"session": self._session},
                first_token_timeout_s=first_timeout,
                idle_timeout_s=idle_timeout,
            )
            completed = True
        except asyncio.CancelledError:
            cancelled = True
            if sent_any_text:
                try:
                    await self._slv.abort()
                except Exception:  # noqa: BLE001
                    logger.debug("slv.abort on cancel failed", exc_info=True)
                await self._stop_gst_playback()
            self.app.events.emit("llm_end", {"full_text": "", "run_id": run_id})
            raise
        except Exception:
            logger.exception("LLM turn failed")
            try:
                await self._slv.abort()
            except Exception:  # noqa: BLE001
                logger.debug("slv.abort on error failed", exc_info=True)
            await self._stop_gst_playback()
            self.app.events.emit("llm_end", {"full_text": "", "run_id": run_id})
            return
        finally:
            if completed and not cancelled:
                # Flush any buffered partial '[...' tail (unterminated tag is
                # real text) to TTS before the final flush.
                tail = self._flush_tag_stripper()
                if tail:
                    try:
                        await self._slv.send_text(tail)
                        sent_any_text = True
                    except Exception:  # noqa: BLE001
                        logger.debug("tail send_text failed", exc_info=True)
                try:
                    await self._slv.flush_tts()
                except Exception:  # noqa: BLE001
                    logger.debug("slv.flush_tts failed", exc_info=True)

        full_text = _EMOTION_TAG_RE.sub("", "".join(full_text_parts))
        self.app.events.emit("llm_end", {"full_text": full_text, "run_id": run_id})

    def _scan_emotion(self, tok: str, buf: list[str]) -> None:
        """Buffer-aware emotion-tag scan.

        Emotion tags ([happy] / [emotion:happy]) may stream split across
        several tokens ('[', 'sad', ']'), so a per-token regex misses
        them. We accumulate into ``buf[0]``, emit ``emotion`` +
        queue_emotion on every complete match, and keep only the trailing
        (possibly-incomplete) fragment after the last ']'.
        """
        buf[0] += tok
        last_end = 0
        for m in _EMOTION_TAG_RE.finditer(buf[0]):
            emo = m.group(1)
            if emo:
                self.app.events.emit("emotion", {"emotion": emo})
                self.app.emotions.queue_emotion(emo)
            last_end = m.end()
        # Retain only the tail after the last full match; if no '[' is
        # pending there's nothing to keep.
        tail = buf[0][last_end:]
        buf[0] = tail if "[" in tail else ""

    # ── stream-safe TTS tag stripper ─────────────────────────────────

    def _reset_tag_stripper(self) -> None:
        """Reset the per-turn tag-stripper buffer + eat-space latch."""
        self._tag_buf = ""
        self._tag_eat_space = False

    def _strip_tags_for_tts(self, tok: str) -> str:
        """Strip emotion / vision-context tags from a streaming token.

        Stateful (carries ``self._tag_buf`` + ``self._tag_eat_space`` across
        tokens) so a tag split across deltas ('[', 'happy', ']') is held back
        and DROPPED instead of being spoken. Folds the old ``_scan_emotion``
        behavior in: a dropped emotion tag ([happy] / [emotion:happy]) still
        fires ``queue_emotion`` + emits the ``emotion`` event so motion /
        dashboard keep working. Echoed ``[Faces: ...]`` context tags are
        dropped without firing an emotion. After a dropped tag we eat one
        following whitespace char so "[happy] hi" → "hi".

        Returns the clean text safe to feed to ``send_text``. Call
        ``_flush_tag_stripper`` at turn end to recover any unterminated tail.
        """
        out_chars: list[str] = []
        for ch in tok:
            if self._tag_eat_space:
                self._tag_eat_space = False
                if ch in (" ", "\t"):
                    continue
            if self._tag_buf:
                self._tag_buf += ch
                if ch == "]":
                    inner = self._tag_buf[1:-1]
                    is_vision_tag = inner.lower().startswith("faces:")
                    # Emotion tag is "[word]" or "[emotion:word]".
                    emo_body = inner
                    if inner.lower().startswith("emotion:"):
                        emo_body = inner.split(":", 1)[1]
                    is_emotion_tag = (
                        not is_vision_tag
                        and bool(emo_body)
                        and all(c.isalnum() or c == "_" for c in emo_body)
                    )
                    if is_emotion_tag:
                        emo = emo_body.strip()
                        if emo:
                            self.app.events.emit("emotion", {"emotion": emo})
                            self.app.emotions.queue_emotion(emo)
                        self._tag_eat_space = True
                    elif is_vision_tag:
                        self._tag_eat_space = True
                    else:
                        out_chars.append(self._tag_buf)
                    self._tag_buf = ""
                elif len(self._tag_buf) > 64:
                    # Runaway: not a tag, flush as plain text.
                    out_chars.append(self._tag_buf)
                    self._tag_buf = ""
            elif ch == "[":
                self._tag_buf = "["
            else:
                out_chars.append(ch)
        return "".join(out_chars)

    def _flush_tag_stripper(self) -> str:
        """Flush a buffered unterminated '[...' tail as plain text.

        If the stream ended mid-buffer the held text was never a tag, so
        return it (so a trailing partial isn't silently dropped) and reset.
        """
        pending = self._tag_buf
        self._tag_buf = ""
        self._tag_eat_space = False
        return pending

    def _build_system_prompt(self) -> str:
        config = self.app.config
        base = getattr(config, "ollama_system_prompt", "") or DEFAULT_SYSTEM_PROMPT
        vision = self._get_vision_context()
        parts = [base, "/no_think"]  # /no_think REQUIRED for qwen3
        if vision:
            parts.append(f"[Faces: {vision}]")
        return "\n".join(parts)

    # ── vision context (verbatim from legacy plugin) ─────────────────

    def _get_vision_context(self) -> str:
        """Return a short face/emotion summary for injection into conversation."""
        vision = self.app.get_plugin("vision_client")
        if not vision:
            return ""
        descs: list[str] = []
        if getattr(vision, "_last_faces_summary", None):
            faces = vision._last_faces_summary
            named: dict[str, str] = {}
            stranger_count = 0
            for f in faces:
                name = f.get("identity")
                emo = f.get("emotion", "neutral")
                if name:
                    named[name] = emo
                else:
                    stranger_count += 1
            descs = [f"{n} looks {e}" for n, e in named.items()]
            real_strangers = max(0, stranger_count - len(named))
            if real_strangers == 1:
                descs.append("a stranger")
            elif real_strangers > 1:
                descs.append(f"{real_strangers} strangers")
        else:
            identity = getattr(vision, "current_identity", None)
            emo = getattr(vision, "_last_emotion", None)
            if identity:
                descs.append(f"{identity} looks {emo or 'neutral'}")
            elif emo and emo != "neutral":
                descs.append(f"someone looks {emo}")
        return ", ".join(descs)

    # ── text-injection entry (dashboard send_message → full LLM turn) ─

    async def _process_and_send(self, text: str) -> None:
        """Run a full LLM+tool+TTS turn from injected TEXT, skipping ASR.

        This is the SAME public name/signature the legacy ConversationPlugin
        exposed, so dashboard_plugin's ``send_message`` handler
        (``conv._process_and_send(text)``) reaches it with zero change.
        It reuses the existing turn driver (``_drive_turn``) so assistant
        tokens still stream to SLV ``send_text`` for TTS, tools dispatch,
        ``flush_tts`` fires, and the full event-bridge is emitted
        (asr_final / llm_delta / llm_end / state_change / emotion).
        """
        if self._paused:
            logger.debug("Paused — skipping text injection")
            return
        text = (text or "").strip()
        if not text:
            return
        async with self._send_lock:
            if not self._running:
                return
            # Cancel any in-flight turn so the injected text wins (mirrors the
            # dashboard's "send to LLM directly" intent).
            await self._cancel_turn()
            # Mirror the ASRFinal path's event-bridge: surface the injected
            # text as an asr_final so the dashboard transcript updates.
            self._last_asr_final = text
            self._last_asr_final_ts = time.monotonic()
            self._monologue_last_ts = time.monotonic()
            self.app.events.emit("asr_final", {"text": text})
            self._turn_task = self._spawn_task(
                self._drive_turn(text), name="slv-turn-text"
            )
            await self._turn_task

    # ── monologue / self-prompt (ported direct stream_with_tools) ────

    async def _monologue_timer_loop(self) -> None:
        """Periodically emit ``monologue_trigger`` when idle (legacy timer).

        Mirrors ``MonologueMode._timer_loop``: every second, if the configured
        ``monologue_interval`` has elapsed since the last turn/utterance and we
        are IDLE with no turn in flight, emit a ``monologue_trigger`` which the
        existing ``_on_monologue_trigger`` consumes to drive a self-prompt turn.
        """
        try:
            while self._running:
                await asyncio.sleep(1.0)
                if self._paused:
                    continue
                interval = float(getattr(self.app.config, "monologue_interval", 5.0))
                if time.monotonic() - self._monologue_last_ts < interval:
                    continue
                if self._state != ConvState.IDLE:
                    continue
                if self._turn_task is not None and not self._turn_task.done():
                    continue
                self._monologue_last_ts = time.monotonic()
                self.app.events.emit("monologue_trigger", {"prompt": ""})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("monologue timer loop crashed")
        finally:
            logger.info("Monologue timer stopped")

    def _on_monologue_trigger(self, _payload=None) -> None:
        if not self._running or self._paused:
            return
        if self._state != ConvState.IDLE:
            return
        if self._turn_task is not None and not self._turn_task.done():
            return
        self._turn_task = self._spawn_task(
            self._run_monologue(), name="slv-monologue"
        )

    async def _run_monologue(self) -> None:
        assert self._slv is not None and self._llm is not None
        config = self.app.config
        base = getattr(config, "ollama_monologue_prompt", "") or MONOLOGUE_SYSTEM_PROMPT
        vision = self._get_vision_context()
        system_prompt = "\n".join(
            [base, "/no_think"] + ([f"[Faces: {vision}]"] if vision else [])
        )
        run_id = uuid.uuid4().hex
        self._set_state(ConvState.THINKING)
        self.app.emotions.queue_emotion("thinking")

        full_text_parts: list[str] = []
        self._reset_tag_stripper()  # fresh stripper state per turn
        ctx = ToolCallCtx(
            session=self._session,
            event_bus=self.app.events,
            config=config,
        )

        async def _on_tok(tok: str) -> None:
            if not tok:
                return
            full_text_parts.append(tok)
            clean = self._strip_tags_for_tts(tok)
            if clean:
                await self._slv.send_text(clean)
            self.app.events.emit("llm_delta", {"text": clean, "run_id": run_id})

        # Synthetic self-prompt message — monologue has no user turn.
        synthetic = "(You glance around the exhibition and think out loud.)"
        # Use a throwaway session so the monologue doesn't pollute the
        # dialogue history.
        mono_session = Session()
        mono_session.add_user(synthetic)
        messages = mono_session.messages(system_prompt)
        try:
            await stream_with_tools(
                self._llm,
                messages,
                session=mono_session,
                registry=self._registry,
                allowed_tools=set(),  # monologue: no tools
                ctx=ctx,
                on_assistant_token=_on_tok,
                llm_kwargs={"session": mono_session},
            )
            tail = self._flush_tag_stripper()
            if tail:
                await self._slv.send_text(tail)
            await self._slv.flush_tts()
        except asyncio.CancelledError:
            try:
                await self._slv.abort()
            except Exception:  # noqa: BLE001
                pass
            raise
        except Exception:
            logger.exception("monologue turn failed")
        full_text = _EMOTION_TAG_RE.sub("", "".join(full_text_parts))
        self._monologue_last_ts = time.monotonic()
        self.app.events.emit("observation", {"text": full_text})
        self.app.events.emit("llm_end", {"full_text": full_text, "run_id": run_id})

    # ── audio playback (reachy GStreamer path, unchanged) ────────────

    async def _output_pipeline(self) -> None:
        while self._running:
            try:
                entry = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if entry is None:
                continue
            if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "v2v_audio":
                _, sr, pcm = entry
                await self._play_v2v_pcm(sr, pcm)

    async def _play_v2v_pcm(self, sample_rate: int, pcm: bytes) -> None:
        if self._paused or self._interrupt_event.is_set():
            return
        if not pcm:
            return
        try:
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:  # noqa: BLE001
            logger.warning("V2V PCM decode failed: %s", e)
            return
        target_sr = self.app.config.sample_rate
        if sample_rate != target_sr and samples.size:
            samples = _resample_pcm_f32(samples, sample_rate, target_sr)
        vol = self.app.config.audio_volume
        if vol != 1.0:
            samples = np.clip(samples * vol, -1.0, 1.0).astype(np.float32)
        reachy = self.app.reachy
        has_sdk_audio = bool(
            reachy and getattr(getattr(reachy, "media", None), "audio", None)
        )
        if has_sdk_audio:
            if not self._gst_playing:
                try:
                    reachy.media.start_playing()
                except Exception:  # noqa: BLE001
                    pass
                self._gst_playing = True
            try:
                reachy.media.push_audio_sample(samples)
            except Exception as e:  # noqa: BLE001
                logger.warning("V2V push_audio_sample failed: %s", e)
        elif self._audio is not None:
            self._gst_playing = True
            await self._audio.enqueue_playback_async(samples)
        if self._wobbler:
            self._wobbler.feed(samples)

    async def _stop_gst_playback(self) -> None:
        if not self._gst_playing:
            return
        reachy = self.app.reachy
        has_sdk_audio = bool(
            reachy and getattr(getattr(reachy, "media", None), "audio", None)
        )
        if has_sdk_audio:
            try:
                silence = np.zeros(1600, dtype=np.float32)
                for _ in range(5):
                    reachy.media.push_audio_sample(silence)
                await asyncio.sleep(0.4)
                reachy.media.stop_playing()
            except Exception:  # noqa: BLE001
                pass
        elif self._audio is not None:
            try:
                await self._audio.await_playback_drained()
            except Exception:  # noqa: BLE001
                logger.debug("await_playback_drained failed", exc_info=True)
        self._gst_playing = False
        if self._wobbler:
            self._wobbler.reset()

    # ── tool registration ────────────────────────────────────────────

    def _register_tools(self) -> None:
        reg = self._registry

        @reg.tool(
            name="move_head",
            description=(
                "Point the robot's head at a target orientation, in DEGREES. "
                "yaw = left/right (positive = left, -45..45), pitch = up/down "
                "(positive = up, -30..30), roll = tilt (-30..30). Call when the "
                "user asks to look in a direction, e.g. '向左看', 'look up'."
            ),
            preamble_text="好的。",
        )
        async def move_head(yaw: float, pitch: float, roll: float = 0.0) -> dict:
            return await asyncio.to_thread(
                self._cmd_move_head, {"yaw": yaw, "pitch": pitch, "roll": roll}
            )

        @reg.tool(
            name="move_antennas",
            description=(
                "Move the robot's two antennae to target angles in DEGREES "
                "(positive = up). Call for antenna gestures, e.g. '抬起天线'."
            ),
            preamble_text="好的。",
        )
        async def move_antennas(left: float, right: float) -> dict:
            return await asyncio.to_thread(
                self._cmd_move_antennas, {"left": left, "right": right}
            )

        @reg.tool(
            name="play_emotion",
            description=(
                "Play an emotion expression (head pose + antennae). emotion is "
                "a slug: 'happy','sad','curious','excited','thinking','confused',"
                "'surprised','angry','neutral'. Call when asked to express a "
                "feeling, e.g. '开心一点'."
            ),
            preamble_text="好的。",
        )
        async def play_emotion(emotion: str) -> dict:
            return await asyncio.to_thread(
                self._cmd_play_emotion, {"emotion": emotion}
            )

        @reg.tool(
            name="dance",
            description=(
                "Perform a short choreographed dance routine. dance_name is one "
                "of: 'celebrate','curious_look','lobster','nod','wiggle'. Call "
                "when the user asks the robot to dance, e.g. '跳个舞'."
            ),
            preamble_text="好的，我跳个舞。",
        )
        async def dance(dance_name: str) -> dict:
            return await asyncio.to_thread(self._cmd_dance, {"dance_name": dance_name})

        @reg.tool(
            name="capture_image",
            description="Capture a still image from the robot's camera.",
        )
        async def capture_image() -> dict:
            return await asyncio.to_thread(self._cmd_capture_image, {})

        @reg.tool(
            name="set_volume",
            description=(
                "Set the speaker volume. level is a percent 0..100, or a relative "
                "value like '+10' / '-10'."
            ),
        )
        async def set_volume(level: str) -> dict:
            return await asyncio.to_thread(self._cmd_set_volume, {"level": level})

        @reg.tool(name="status", description="Report the robot's current status.")
        async def status() -> dict:
            return await asyncio.to_thread(self._cmd_status, {})

    # ── _cmd_* handlers (ported from legacy plugin; use app.reachy) ──

    def _cmd_move_head(self, params: dict) -> dict:
        reachy = self.app.reachy
        if not reachy:
            return {"status": "error", "message": "No robot connected", "reason": "no_robot"}
        from reachy_mini.utils import create_head_pose

        yaw = max(-45, min(45, params.get("yaw", 0)))
        pitch = max(-30, min(30, params.get("pitch", 0)))
        roll = max(-30, min(30, params.get("roll", 0)))
        duration = params.get("duration", 1.0)
        pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
        reachy.goto_target(head=pose, duration=duration)
        return {"status": "success", "position": {"yaw": yaw, "pitch": pitch, "roll": roll}}

    def _cmd_move_antennas(self, params: dict) -> dict:
        reachy = self.app.reachy
        if not reachy:
            return {"status": "error", "message": "No robot connected", "reason": "no_robot"}
        left = params.get("left", 0)
        right = params.get("right", 0)
        duration = params.get("duration", 0.5)
        reachy.goto_target(
            antennas=[np.radians(right), np.radians(left)], duration=duration
        )
        return {"status": "success", "antennas": {"left": left, "right": right}}

    def _cmd_play_emotion(self, params: dict) -> dict:
        emotion = params.get("emotion", "")
        if not emotion:
            return {"status": "error", "message": "Missing emotion parameter"}
        self.app.emotions.queue_emotion(emotion)
        return {"status": "success", "emotion": emotion}

    def _cmd_dance(self, params: dict) -> dict:
        reachy = self.app.reachy
        if not reachy:
            return {"status": "error", "message": "No robot connected", "reason": "no_robot"}
        from reachy_mini.utils import create_head_pose
        import time as _time

        from ..motion.dances import DANCE_ROUTINES

        name = params.get("dance_name", "")
        routine = DANCE_ROUTINES.get(name)
        if not routine:
            from ..motion.dances import AVAILABLE_DANCES

            return {
                "status": "error",
                "message": f"Unknown dance: {name}. Available: {', '.join(AVAILABLE_DANCES)}",
            }
        for step in routine.steps:
            pose = create_head_pose(yaw=step.yaw, pitch=step.pitch, roll=step.roll, degrees=True)
            antennas = [np.radians(step.antenna_right), np.radians(step.antenna_left)]
            reachy.goto_target(head=pose, antennas=antennas, duration=step.duration)
            _time.sleep(step.duration)
        return {"status": "success", "dance": name, "steps": len(routine.steps)}

    def _cmd_capture_image(self, params: dict) -> dict:
        reachy = self.app.reachy
        if not reachy:
            return {"status": "error", "message": "No robot connected", "reason": "no_robot"}
        if not hasattr(reachy, "media") or reachy.media is None:
            return {"status": "error", "message": "Media backend not available"}
        frame = reachy.media.get_frame()
        if frame is None:
            return {"status": "error", "message": "No frame available"}
        from datetime import datetime
        from pathlib import Path

        capture_dir = Path.home() / ".reachy-claw" / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        filepath = capture_dir / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            import cv2

            cv2.imwrite(str(filepath), frame)
        except ImportError:
            from PIL import Image

            Image.fromarray(frame).save(filepath)
        return {"status": "success", "filepath": str(filepath)}

    def _cmd_set_volume(self, params: dict) -> dict:
        import shutil
        import subprocess

        level = params.get("level", None)
        if level is None:
            return {"status": "error", "message": "Missing level parameter"}
        amixer = shutil.which("amixer")
        if not amixer:
            # No ALSA (e.g. macOS dev) — fall back to playback gain config.
            try:
                self.app.config.audio_volume = max(
                    0.0, float(str(level).lstrip("+")) / 100.0
                )
                return {"status": "success", "level": level, "via": "software_gain"}
            except Exception as e:  # noqa: BLE001
                return {"status": "error", "message": str(e)}
        level_str = str(level).strip()
        arg = f"{level_str}%" if not level_str.startswith(("+", "-")) else f"{level_str}%"
        try:
            subprocess.run(
                [amixer, "set", "Master", arg], check=True, capture_output=True
            )
            return {"status": "success", "level": level}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}

    def _cmd_status(self, params: dict) -> dict:
        return {
            "status": "success",
            "state": self._state.value,
            "robot_connected": bool(self.app.reachy),
            "speaking": self.app.is_speaking,
        }


__all__ = ["ConversationPlugin", "ConvState", "MONOLOGUE_SYSTEM_PROMPT"]
