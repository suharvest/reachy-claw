"""ConversationPlugin -- dual-pipeline STT/TTS/gateway conversation loop.

Three concurrent tasks connected by queues and an interrupt event:
  _audio_loop:          mic → VAD → state machine → STT → send to AI
  _sentence_accumulator: stream deltas → sentence splitting → sentence_queue
  _output_pipeline:     sentence_queue → TTS → interruptible playback
"""

from __future__ import annotations

import asyncio
import collections
import enum
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Coroutine

import numpy as np

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from ..audio import AudioCapture, WakeWordDetector
from ..gateway import DesktopRobotClient
from ..llm import OllamaClient, OllamaConfig, MONOLOGUE_SYSTEM_PROMPT
from ..motion.head_wobbler import HeadWobbler
from ..plugin import Plugin
from ..stt import create_stt_backend
from ..tts import create_tts_backend
from ..vad import create_vad_backend

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────


class ConvState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class SentenceItem:
    text: str
    is_last: bool = False


# In-band sentinel: when placed in _stream_text_queue, tells the
# accumulator to discard its buffer (stale text from a previous response).
_RESET_BUFFER = object()


# ── Interpreter Sequencer ─────────────────────────────────────────────


class _InterpreterSequencer:
    """Concurrent translator with ordered output for interpreter mode.

    Bypasses OllamaClient — calls Ollama /api/chat directly via httpx.
    Translates utterances concurrently but emits to sentence_queue in
    utterance order (FIFO by submission time, not completion time).
    """

    def __init__(self, config, sentence_queue, events):
        self._config = config
        self._sentence_queue = sentence_queue
        self._events = events
        self._http: httpx.AsyncClient | None = None
        self._slots: list[asyncio.Task] = []
        self._seq = 0
        self._emitter_task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        if httpx is None:
            raise RuntimeError("httpx is required for interpreter mode — pip install httpx")
        self._http = httpx.AsyncClient(
            base_url=self._config.ollama_base_url,
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
        )
        self._running = True
        self._emitter_task = asyncio.create_task(self._emitter_loop())

    async def stop(self):
        self._running = False
        for task in self._slots:
            if not task.done():
                task.cancel()
        self._slots.clear()
        if self._emitter_task:
            self._emitter_task.cancel()
        if self._http:
            await self._http.aclose()
            self._http = None

    def submit(self, text: str):
        """Submit an utterance for translation. Returns immediately."""
        self._seq += 1
        seq = self._seq
        task = asyncio.create_task(self._translate(seq, text))
        self._slots.append(task)

    async def _translate(self, seq: int, text: str) -> tuple[int, str]:
        """Call Ollama /api/chat (non-streaming) and return (seq, translated_text)."""
        from ..llm import INTERPRETER_SYSTEM_PROMPT

        system_prompt = self._config.interpreter_prompt or INTERPRETER_SYSTEM_PROMPT.format(
            source_lang=self._config.interpreter_source_lang,
            target_lang=self._config.interpreter_target_lang,
        )
        payload = {
            "model": self._config.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate: \"{text}\""},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 200},
        }
        try:
            resp = await self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
            translated = resp.json()["message"]["content"].strip()
            self._events.emit(
                "llm_end", {"full_text": translated, "run_id": f"interp-{seq}"}
            )
            return (seq, translated)
        except Exception as e:
            logger.error(f"Interpreter translation failed (seq={seq}): {e}")
            return (seq, "")

    async def _emitter_loop(self):
        """Consume completed translations in order and feed to sentence_queue."""
        while self._running:
            if not self._slots:
                await asyncio.sleep(0.05)
                continue

            first = self._slots[0]
            try:
                seq, translated = await asyncio.shield(first)
            except (asyncio.CancelledError, Exception):
                self._slots.pop(0)
                continue

            self._slots.pop(0)

            if translated:
                logger.info(f"Interpreter emit seq={seq}: \"{translated[:40]}\"")
                await self._sentence_queue.put(
                    SentenceItem(text=translated, is_last=True)
                )


# ── Plugin ────────────────────────────────────────────────────────────


class ConversationPlugin(Plugin):
    """Dual-pipeline conversation: continuous listen + concurrent TTS output."""

    name = "conversation"

    def __init__(self, app):
        super().__init__(app)
        self._client: DesktopRobotClient | OllamaClient | None = None
        self._stt = None
        self._tts = None
        self._vad = None
        self._audio: AudioCapture | None = None
        self._wake_detector: WakeWordDetector | None = None
        self._wobbler: HeadWobbler | None = None

        # Queues connecting the four pipeline stages (bounded to prevent OOM)
        self._stream_text_queue: asyncio.Queue[str | object | None] = asyncio.Queue(maxsize=200)
        self._sentence_queue: asyncio.Queue[SentenceItem | None] = asyncio.Queue(maxsize=50)
        # Audio queue: (SentenceItem, list_of_chunks | None) — TTS worker feeds this
        self._audio_queue: asyncio.Queue[tuple[SentenceItem, list | None] | None] = asyncio.Queue(maxsize=3)
        self._interrupt_event = asyncio.Event()
        self._gst_playing = False  # GStreamer playback pipeline state
        self._gst_drain_s: float = 0.0  # estimated ALSA buffer tail (seconds)

        self._current_run_id: str | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._conversation_active = False
        self._conversation_stopped = False  # STOPPED: still listen+send, but no TTS
        self._state = ConvState.IDLE
        self._speaking_since: float = 0.0
        self._pending_tasks: set[asyncio.Task] = set()

        # Serialise _process_and_send so two concurrent utterances
        # don't fire overlapping LLM streams.
        self._send_lock = asyncio.Lock()

        # Deferred task-completion announcements: task_completed may arrive
        # while we're SPEAKING/LISTENING/THINKING.  We queue them and drain
        # when we transition back to IDLE.
        self._pending_announcements: list[str] = []

        # Client-side watchdog: if we stay in THINKING for too long without
        # receiving any delta, recover to IDLE.
        self._thinking_watchdog_s: float = 30.0

        # Mode state
        self._monologue_mode = False
        self._interpreter_mode = False
        self._narration_mode = False  # True during diary narration — suppresses conversation
        self._interp_sequencer: _InterpreterSequencer | None = None
        self._last_speech_time: float = 0.0  # for monologue auto-trigger
        self._pending_speech: str | None = None  # speech heard while speaking (monologue)
        self._bg_speech_frames: list[np.ndarray] = []  # frames for bg listening
        self._bg_silence_count: int = 0  # silence counter for bg listening

    def setup(self) -> bool:
        return True

    async def start(self):
        self._event_loop = asyncio.get_running_loop()
        config = self.app.config

        # ── Phase 1: create backends (fast, no I/O) ──────────────────
        self._stt = create_stt_backend(config)
        self._tts = create_tts_backend(
            backend=config.tts_backend,
            voice=config.tts_voice,
            model=config.tts_model,
            config=config,
        )
        self._vad = create_vad_backend(backend=config.vad_backend, config=config)
        self._audio = AudioCapture(config, self.app.reachy, vad=self._vad)

        if config.wake_word:
            self._wake_detector = WakeWordDetector(config.wake_word)

        motion_plugin = self._find_motion_plugin()
        if motion_plugin:
            self._wobbler = HeadWobbler(
                set_speech_offsets=motion_plugin.set_speech_offsets,
                sample_rate=config.sample_rate,
            )

        # ── Phase 2: preload + connect + warmup ALL in parallel ──────
        async def _init_stt():
            logger.info("Loading speech recognition model...")
            await asyncio.to_thread(self._stt.preload)
            logger.info("Speech recognition ready")

        async def _init_vad():
            await asyncio.to_thread(self._vad.preload)
            logger.info(f"VAD backend: {config.vad_backend}")

        async def _init_gateway():
            """Connect, wire callbacks, then warmup."""
            if config.standalone_mode:
                logger.info("Running in standalone mode - no server connection")
                return

            # Mode setup
            self._monologue_mode = config.conversation_mode == "monologue"
            self._interpreter_mode = config.conversation_mode == "interpreter"

            if config.llm_backend == "ollama":
                from ..llm import DEFAULT_SYSTEM_PROMPT, INTERPRETER_SYSTEM_PROMPT

                if self._interpreter_mode:
                    system_prompt = config.interpreter_prompt or INTERPRETER_SYSTEM_PROMPT.format(
                        source_lang=config.interpreter_source_lang,
                        target_lang=config.interpreter_target_lang,
                    )
                    history = 0
                    temperature = 0.3
                elif self._monologue_mode:
                    system_prompt = config.ollama_monologue_prompt or MONOLOGUE_SYSTEM_PROMPT
                    history = max(config.ollama_max_history, 5)
                    temperature = max(config.ollama_temperature, 0.9)
                else:
                    system_prompt = config.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT
                    history = config.ollama_max_history
                    temperature = config.ollama_temperature

                ollama_cfg = OllamaConfig(
                    base_url=config.ollama_base_url,
                    model=config.ollama_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_history=history,
                    skip_emotion_extraction=self._interpreter_mode,
                    enable_vlm=config.enable_vlm and not self._interpreter_mode,
                    vlm_model=config.vlm_model,
                    vlm_prompt=config.vlm_prompt,
                )
                self._client = OllamaClient(ollama_cfg)
                self._client.capture_frame = self._capture_frame_b64
                await self._client.connect()
                self._setup_callbacks()
                if config.gateway_warmup:
                    await self._client.warmup_session()
                mode_label = (
                    " (interpreter)" if self._interpreter_mode
                    else " (monologue)" if self._monologue_mode
                    else ""
                )
                logger.info(f"Using Ollama LLM: {config.ollama_model}{mode_label}")
            else:
                self._client = DesktopRobotClient(config)
                await self._client.connect()
                self._setup_callbacks()
                if config.gateway_warmup:
                    await self._client.warmup_session()

            # Start interpreter sequencer if needed
            if self._interpreter_mode:
                self._interp_sequencer = _InterpreterSequencer(
                    config=config,
                    sentence_queue=self._sentence_queue,
                    events=self.app.events,
                )
                await self._interp_sequencer.start()

        async def _init_tts():
            logger.info(f"TTS backend: {config.tts_backend}")
            await self._warmup_tts()

        async def _init_robot():
            if not self.app.reachy:
                return
            logger.info("Waking up Reachy...")
            await asyncio.to_thread(self.app.reachy.wake_up)
            await asyncio.sleep(0.5)
            try:
                self.app.reachy.set_target_antenna_joint_positions([0.7, -0.7])
                await asyncio.sleep(0.2)
                self.app.reachy.set_target_antenna_joint_positions([-0.7, 0.7])
                await asyncio.sleep(0.2)
                self.app.reachy.set_target_antenna_joint_positions([0.0, 0.0])
            except Exception as e:
                logger.debug(f"Startup animation failed: {e}")

        t0 = time.perf_counter()
        results = await asyncio.gather(
            _init_stt(),
            _init_vad(),
            _init_gateway(),
            _init_tts(),
            _init_robot(),
            return_exceptions=True,
        )
        # Log any init failures (non-fatal except gateway in non-standalone)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                name = ["STT", "VAD", "Gateway", "TTS", "Robot"][i]
                logger.warning(f"{name} init failed: {r}")
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"All subsystems initialized in {elapsed:.0f}ms")

        # ── Phase 3: start listening + pipeline immediately ──────────
        await self._audio.start_continuous()

        logger.info("=" * 50)
        if config.wake_word:
            logger.info(f'Say "{config.wake_word}" to activate')
        else:
            logger.info("Speak anytime - always listening!")
        logger.info("=" * 50)

        tasks = [
            self._audio_loop(),
            self._sentence_accumulator(),
            self._tts_worker(),
            self._output_pipeline(),
        ]
        if self._monologue_mode:
            tasks.append(self._monologue_timer())

        async def _guarded(coro, name: str):
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Pipeline task '%s' crashed — stopping plugin", name)
                self._running = False

        guarded = [_guarded(t, t.__name__ if hasattr(t, '__name__') else str(i))
                   for i, t in enumerate(tasks)]
        try:
            await asyncio.gather(*guarded)
        except asyncio.CancelledError:
            logger.info("Conversation pipeline cancelled")

    async def stop(self):
        self._running = False

        if self._pending_tasks:
            for task in list(self._pending_tasks):
                task.cancel()
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

        if self._wobbler:
            self._wobbler.stop()

        if self._audio:
            await self._audio.stop()

        if self._client:
            await self._client.disconnect()

        if self._tts:
            self._tts.cleanup()

    def _find_motion_plugin(self):
        """Find the MotionPlugin instance if registered."""
        from .motion_plugin import MotionPlugin

        for p in self.app._plugins:
            if isinstance(p, MotionPlugin):
                return p
        return None

    async def _warmup_tts(self) -> None:
        """Send a tiny TTS request to warm up the remote service connection."""
        t0 = time.perf_counter()
        try:
            if self._tts.supports_streaming:
                async for _ in self._tts.synthesize_streaming("."):
                    pass
            else:
                path = await self._tts.synthesize(".")
                import os
                os.unlink(path)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"TTS warmup complete ({elapsed:.0f}ms)")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning(f"TTS warmup failed ({elapsed:.0f}ms): {e}")

    # ── State helpers ─────────────────────────────────────────────────

    # State → emotion mapping for antenna animations
    _STATE_EMOTION_MAP = {
        ConvState.LISTENING: "listening",
    }

    def _set_state(self, new_state: ConvState) -> None:
        if self._state != new_state:
            old_state = self._state
            logger.debug(f"State: {old_state.value} → {new_state.value}")
            self._state = new_state
            self.app.events.emit("state_change", {"state": new_state.value})
            if new_state == ConvState.SPEAKING:
                self._speaking_since = time.monotonic()
            # Reset monologue timer when finishing speaking, so there's
            # a full monologue_interval gap for ASR to process user speech
            if old_state == ConvState.SPEAKING and new_state == ConvState.IDLE:
                self._last_speech_time = time.monotonic()
            # Drain deferred task-completion announcements when we go IDLE
            if new_state == ConvState.IDLE and self._pending_announcements:
                self._spawn_task(
                    self._drain_pending_announcements(),
                    name="drain-announcements",
                )
            # Client-side watchdog for THINKING state
            if new_state == ConvState.THINKING:
                self._spawn_task(
                    self._thinking_watchdog(), name="thinking-watchdog"
                )
            # Drive antenna animation from state changes (conversation mode only)
            if not self._monologue_mode:
                emotion = self._STATE_EMOTION_MAP.get(new_state)
                if emotion:
                    self.app.emotions.queue_emotion(emotion)

    def _spawn_task(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        """Track background tasks so they can be cancelled on shutdown."""
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)

        def _on_done(done: asyncio.Task) -> None:
            self._pending_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Background task '{done.get_name()}' failed: {e}")
                # Recovery: reset to IDLE so audio loop doesn't get stuck
                if self._state in (ConvState.TRANSCRIBING, ConvState.THINKING):
                    self._set_state(ConvState.IDLE)

        task.add_done_callback(_on_done)

    # ── Callbacks from desktop-robot protocol ─────────────────────────

    def _setup_callbacks(self) -> None:
        assert self._client is not None
        cb = self._client.callbacks
        cb.on_stream_start = self._on_stream_start
        cb.on_stream_delta = self._on_stream_delta
        cb.on_stream_end = self._on_stream_end
        cb.on_stream_abort = self._on_stream_abort
        cb.on_tool_start = self._on_tool_start
        cb.on_tool_end = self._on_tool_end
        cb.on_task_spawned = self._on_task_spawned
        cb.on_task_completed = self._on_task_completed
        cb.on_emotion = self._on_emotion
        cb.on_robot_command = self._on_robot_command

    async def _on_stream_start(self, run_id: str) -> None:
        logger.debug(f"Stream started: {run_id}")
        self._current_run_id = run_id
        self._set_state(ConvState.THINKING)
        _drain_queue(self._stream_text_queue)
        # In-band reset: accumulator will discard stale buffer when it sees this
        await self._stream_text_queue.put(_RESET_BUFFER)

    async def _on_stream_delta(self, text: str, run_id: str) -> None:
        if run_id != self._current_run_id:
            logger.debug(f"Ignoring stale delta (run {run_id[:8]})")
            return
        if self._state == ConvState.THINKING:
            # First delta = TTFT measurement point.
            # Don't transition to SPEAKING yet — that happens when audio
            # actually starts (_output_pipeline), to avoid premature barge-in
            # cooldown before the robot is audible.
            if hasattr(self, "_t_send") and self._t_send:
                ttft = (time.perf_counter() - self._t_send) * 1000
                logger.info(f"TTFT: {ttft:.0f}ms (send → first delta)")
        try:
            self._stream_text_queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("stream_text_queue full, dropping delta")
        self.app.events.emit("llm_delta", {"text": text, "run_id": run_id})

    async def _on_stream_end(self, full_text: str, run_id: str) -> None:
        if run_id != self._current_run_id:
            logger.debug(f"Ignoring stale stream_end (run {run_id[:8]})")
            return
        if hasattr(self, "_t_send") and self._t_send:
            total = (time.perf_counter() - self._t_send) * 1000
            logger.info(f"Response complete ({len(full_text)} chars, {total:.0f}ms)")
            self._t_send = None
        else:
            logger.info(f"Response complete ({len(full_text)} chars)")
        await self._stream_text_queue.put(None)
        self.app.events.emit("llm_end", {"full_text": full_text, "run_id": run_id})

    async def _on_stream_abort(self, reason: str, run_id: str) -> None:
        # Empty run_id = synthetic abort from disconnect; always accept.
        if run_id and run_id != self._current_run_id:
            logger.debug(f"Ignoring stale stream_abort (run {run_id[:8]})")
            return
        logger.info(f"Stream aborted: {reason}")
        self._current_run_id = None
        await self._stream_text_queue.put(None)
        self.app.events.emit("llm_end", {"full_text": "", "run_id": run_id})
        # Ensure we recover to IDLE so deferred announcements can proceed
        if self._state in (ConvState.THINKING, ConvState.SPEAKING):
            self._set_state(ConvState.IDLE)

    async def _on_tool_start(self, tool_name: str, run_id: str) -> None:
        logger.info(f"Tool started: {tool_name}")

    async def _on_tool_end(self, tool_name: str, run_id: str) -> None:
        logger.info(f"Tool ended: {tool_name}")

    async def _on_task_spawned(self, label: str, task_run_id: str) -> None:
        logger.info(f"Background task started: {label}")

    async def _on_task_completed(self, summary: str, task_run_id: str) -> None:
        logger.info(f"Background task completed: {summary[:100]}")
        prompt = (
            "[system: 后台任务已完成。请用1-2句话简要播报任务结果，"
            "让用户知道任务做了什么、结果如何。语气自然，像是在汇报工作。]"
        )
        if self._client and self._state == ConvState.IDLE:
            # IDLE — announce immediately (serialised via _send_lock)
            self._spawn_task(
                self._announce_task_result(prompt),
                name="announce-task",
            )
        else:
            # Busy — queue for later when we transition back to IDLE
            logger.info("Deferring task announcement (state=%s)", self._state.value)
            self._pending_announcements.append(prompt)

    async def _announce_task_result(self, prompt: str) -> None:
        """Send a task-result announcement through the LLM, serialised."""
        async with self._send_lock:
            if not self._client:
                return
            logger.info("Announcing task result via LLM")
            self._set_state(ConvState.THINKING)
            self._t_send = time.perf_counter()
            try:
                await self._client.send_message_streaming(prompt)
            except Exception as e:
                logger.error(f"Failed to request task summary: {e}")
                await self._sentence_queue.put(
                    SentenceItem(text="后台任务已完成。", is_last=True)
                )
                self._set_state(ConvState.IDLE)

    async def _thinking_watchdog(self) -> None:
        """Recover from stuck THINKING state if no stream events arrive."""
        await asyncio.sleep(self._thinking_watchdog_s)
        if self._state == ConvState.THINKING:
            logger.warning(
                "THINKING watchdog fired after %.0fs — forcing IDLE",
                self._thinking_watchdog_s,
            )
            self._current_run_id = None
            self._set_state(ConvState.IDLE)

    async def _drain_pending_announcements(self) -> None:
        """Process queued task-completion announcements one by one."""
        while self._pending_announcements:
            prompt = self._pending_announcements.pop(0)
            await self._announce_task_result(prompt)
            # Wait for the LLM response + TTS to finish before next one.
            # _announce_task_result sets state to THINKING; the streaming
            # callbacks will move it through SPEAKING → IDLE, which will
            # trigger another drain via _set_state.  So we just do one here.
            break

    # ── Passive emotion channel ──────────────────────────────────────

    async def _on_emotion(self, emotion: str) -> None:
        """Server sends an emotion tag — queue it for immediate expression."""
        self.app.events.emit("emotion", {"emotion": emotion})
        if self.app.config.play_emotions and not self._monologue_mode:
            logger.info(f"Emotion from server: {emotion}")
            self.app.emotions.queue_emotion(emotion)

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

    # ── Monologue mode ─────────────────────────────────────────────────

    def _compose_monologue_prompt(self, transcript: str | None = None) -> str:
        """Build natural-language LLM input for monologue mode from speech + vision."""
        parts = []
        if transcript:
            parts.append(f"heard: \"{transcript}\"")

        vision = self.app.get_plugin("vision_client")
        if vision and getattr(vision, "_last_faces_summary", None):
            faces = vision._last_faces_summary
            # Deduplicate: collect named persons, count remaining strangers
            named: dict[str, str] = {}  # name → emotion
            stranger_count = 0
            for f in faces:
                name = f.get("identity")
                emo = f.get("emotion", "neutral")
                if name:
                    named[name] = emo
                else:
                    stranger_count += 1
            descs = [f"{n} looks {e}" for n, e in named.items()]
            # Only count strangers that aren't duplicate detections of known people
            real_strangers = max(0, stranger_count - len(named))
            if real_strangers == 1:
                descs.append("someone you don't know")
            elif real_strangers > 1:
                descs.append(f"{real_strangers} people you don't know")
            if descs:
                parts.append(f"you see: {', '.join(descs)}")
        elif vision:
            emo = getattr(vision, "_last_emotion", None)
            identity = getattr(vision, "current_identity", None)
            if identity:
                parts.append(f"you see: {identity} looks {emo or 'neutral'}")
            elif emo and emo != "neutral":
                parts.append(f"you see: someone looks {emo}")

        if not parts:
            parts.append("nobody around")
        return ". ".join(parts)

    def _capture_frame_b64(self) -> str | None:
        """Capture camera frame, return base64 JPEG. Called from worker thread."""
        import base64

        # Try SDK camera first (works on macOS dev)
        reachy = self.app.reachy
        if reachy and hasattr(reachy, "media") and reachy.media is not None:
            frame = reachy.media.get_frame()
            if frame is not None:
                import cv2

                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return base64.b64encode(jpg.tobytes()).decode()

        # Fallback: grab a JPEG from vision service MJPEG stream (Jetson)
        return self._capture_from_vision_stream()

    def _capture_from_vision_stream(self) -> str | None:
        """Grab a single JPEG frame from the vision-trt MJPEG stream."""
        import base64

        vision_url = self.app.config.vision_service_url  # tcp://127.0.0.1:8631
        if not vision_url:
            return None
        host = vision_url.replace("tcp://", "").split(":")[0]
        stream_url = f"http://{host}:8630/stream"

        import urllib.request

        try:
            req = urllib.request.Request(stream_url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                buf = b""
                jpeg_start = -1
                while len(buf) < 500_000:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    if jpeg_start < 0:
                        jpeg_start = buf.find(b"\xff\xd8")
                    if jpeg_start >= 0:
                        jpeg_end = buf.find(b"\xff\xd9", jpeg_start + 2)
                        if jpeg_end >= 0:
                            jpg_data = buf[jpeg_start : jpeg_end + 2]
                            return base64.b64encode(jpg_data).decode()
        except Exception as e:
            logger.warning("Vision stream capture failed: %s", e)
        return None

    def switch_mode(self, mode: str) -> None:
        """Hot-switch between conversation, monologue, and interpreter modes."""
        from ..llm import DEFAULT_SYSTEM_PROMPT, INTERPRETER_SYSTEM_PROMPT

        self._monologue_mode = mode == "monologue"
        self._interpreter_mode = mode == "interpreter"
        self.app.config.conversation_mode = mode

        # Conversation mode requires barge-in; interpreter/monologue don't
        if mode == "conversation":
            self.app.config.barge_in_enabled = True
        elif mode == "interpreter":
            self.app.config.barge_in_enabled = False

        if isinstance(self._client, OllamaClient):
            if self._interpreter_mode:
                self._client._config.system_prompt = self.app.config.interpreter_prompt or INTERPRETER_SYSTEM_PROMPT.format(
                    source_lang=self.app.config.interpreter_source_lang,
                    target_lang=self.app.config.interpreter_target_lang,
                )
                self._client._config.skip_emotion_extraction = True
                self._client._config.monologue_mode = False
                self._client._config.max_history = 0
                self._client._config.temperature = 0.3
                self._client._config.enable_vlm = False  # VLM overrides system prompt
            elif self._monologue_mode:
                self._client._config.system_prompt = (
                    self.app.config.ollama_monologue_prompt or MONOLOGUE_SYSTEM_PROMPT
                )
                self._client._config.skip_emotion_extraction = False
                self._client._config.monologue_mode = True
                # Monologue respects dashboard history setting as-is
                self._client._config.temperature = max(self._client._config.temperature, 0.9)
            else:
                self._client._config.system_prompt = (
                    self.app.config.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT
                )
                self._client._config.skip_emotion_extraction = False
                self._client._config.monologue_mode = False
                # Restore settings that interpreter may have overridden
                self._client._config.max_history = self.app.config.ollama_max_history
                self._client._config.temperature = self.app.config.ollama_temperature
                self._client._config.enable_vlm = self.app.config.enable_vlm
            # Reset history on mode switch
            self._client._history.clear()

        # Start/stop monologue timer on mode switch
        if self._monologue_mode:
            # Start timer if not already running
            if not hasattr(self, '_monologue_timer_task') or self._monologue_timer_task.done():
                self._monologue_timer_task = asyncio.ensure_future(self._monologue_timer())
                self._last_speech_time = time.monotonic()
        else:
            # Cancel timer when leaving monologue mode
            task = getattr(self, '_monologue_timer_task', None)
            if task and not task.done():
                task.cancel()

        # Start/stop interpreter sequencer (only if pipeline is running)
        if self._interpreter_mode and hasattr(self, '_sentence_queue'):
            async def _start_sequencer():
                if self._interp_sequencer:
                    await self._interp_sequencer.stop()
                self._interp_sequencer = _InterpreterSequencer(
                    config=self.app.config,
                    sentence_queue=self._sentence_queue,
                    events=self.app.events,
                )
                await self._interp_sequencer.start()
            try:
                asyncio.ensure_future(_start_sequencer())
            except RuntimeError:
                pass  # no event loop — sequencer will start in _init_gateway
        elif not self._interpreter_mode and self._interp_sequencer:
            seq = self._interp_sequencer
            self._interp_sequencer = None
            async def _stop_sequencer(s=seq):
                await s.stop()
                if hasattr(self, '_sentence_queue'):
                    _drain_queue(self._sentence_queue)
                if hasattr(self, '_audio_queue'):
                    _drain_queue(self._audio_queue)
            try:
                asyncio.ensure_future(_stop_sequencer())
            except RuntimeError:
                pass

        if self._interpreter_mode and not isinstance(self._client, OllamaClient):
            logger.warning("Interpreter mode requires Ollama backend; gateway does not support custom system prompts")

        logger.info(f"Conversation mode switched to: {mode}")

    async def switch_backend(self, backend: str, model: str | None = None) -> None:
        """Hot-switch LLM backend (ollama/gateway) or change Ollama model."""
        config = self.app.config

        if backend == config.llm_backend and model is None:
            return  # no change

        # Update model if specified (works live for Ollama without reconnect)
        if model and backend == "ollama" and config.llm_backend == "ollama":
            config.ollama_model = model
            if isinstance(self._client, OllamaClient):
                self._client._config.model = model
                self._client._history.clear()
                logger.info(f"Ollama model switched to: {model}")
            return

        # Full backend switch: disconnect old, connect new
        if self._client:
            await self._client.disconnect()
            self._client = None

        config.llm_backend = backend
        if model:
            config.ollama_model = model

        if backend == "ollama":
            from ..llm import DEFAULT_SYSTEM_PROMPT, INTERPRETER_SYSTEM_PROMPT

            if self._interpreter_mode:
                system_prompt = config.interpreter_prompt or INTERPRETER_SYSTEM_PROMPT.format(
                    source_lang=config.interpreter_source_lang,
                    target_lang=config.interpreter_target_lang,
                )
                history, temperature = 0, 0.3
            elif self._monologue_mode:
                system_prompt = config.ollama_monologue_prompt or MONOLOGUE_SYSTEM_PROMPT
                history = max(config.ollama_max_history, 5)
                temperature = max(config.ollama_temperature, 0.9)
            else:
                system_prompt = config.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT
                history = config.ollama_max_history
                temperature = config.ollama_temperature

            ollama_cfg = OllamaConfig(
                base_url=config.ollama_base_url,
                model=config.ollama_model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_history=history,
                skip_emotion_extraction=self._interpreter_mode,
                enable_vlm=config.enable_vlm and not self._interpreter_mode,
                vlm_model=config.vlm_model,
                vlm_prompt=config.vlm_prompt,
            )
            self._client = OllamaClient(ollama_cfg)
            self._client.capture_frame = self._capture_frame_b64
            await self._client.connect()
            self._setup_callbacks()
            logger.info(f"Switched to Ollama: {config.ollama_model}")
        else:
            self._client = DesktopRobotClient(config)
            await self._client.connect()
            self._setup_callbacks()
            logger.info("Switched to OpenClaw gateway")

    # ── Active robot commands (LLM tool calls) ────────────────────────

    async def _on_robot_command(
        self, action: str, params: dict, command_id: str
    ) -> None:
        """Execute a robot command from LLM tool call and return result."""
        logger.info(f"Robot command: {action} {params}")
        result = await asyncio.to_thread(self._execute_robot_command, action, params)
        if self._client and command_id:
            await self._client.send_robot_result(command_id, result)

    def _execute_robot_command(self, action: str, params: dict) -> dict:
        """Dispatch a robot command to the appropriate handler."""
        handlers = {
            "move_head": self._cmd_move_head,
            "move_antennas": self._cmd_move_antennas,
            "play_emotion": self._cmd_play_emotion,
            "dance": self._cmd_dance,
            "capture_image": self._cmd_capture_image,
            "set_volume": self._cmd_set_volume,
            "status": self._cmd_status,
            "stop_conversation": self._cmd_stop_conversation,
            "resume_conversation": self._cmd_resume_conversation,
        }
        handler = handlers.get(action)
        if not handler:
            return {"status": "error", "message": f"Unknown action: {action}"}
        try:
            return handler(params)
        except Exception as e:
            logger.error(f"Robot command '{action}' failed: {e}")
            return {"status": "error", "message": str(e)}

    def _cmd_move_head(self, params: dict) -> dict:
        reachy = self.app.reachy
        if not reachy:
            return {"status": "error", "message": "No robot connected"}
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
            return {"status": "error", "message": "No robot connected"}
        import numpy as np

        left = params.get("left", 0)
        right = params.get("right", 0)
        duration = params.get("duration", 0.5)

        reachy.goto_target(
            antennas=[np.radians(right), np.radians(left)],
            duration=duration,
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
            return {"status": "error", "message": "No robot connected"}
        from reachy_mini.utils import create_head_pose
        import numpy as np
        import time as _time

        from ..motion.dances import DANCE_ROUTINES

        name = params.get("dance_name", "")
        routine = DANCE_ROUTINES.get(name)
        if not routine:
            from ..motion.dances import AVAILABLE_DANCES
            return {"status": "error", "message": f"Unknown dance: {name}. Available: {', '.join(AVAILABLE_DANCES)}"}

        for step in routine.steps:
            pose = create_head_pose(yaw=step.yaw, pitch=step.pitch, roll=step.roll, degrees=True)
            antennas = [np.radians(step.antenna_right), np.radians(step.antenna_left)]
            reachy.goto_target(head=pose, antennas=antennas, duration=step.duration)
            _time.sleep(step.duration)

        return {"status": "success", "dance": name, "steps": len(routine.steps)}

    def _cmd_capture_image(self, params: dict) -> dict:
        reachy = self.app.reachy
        if not reachy:
            return {"status": "error", "message": "No robot connected"}
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
        """Set system speaker volume using ALSA amixer."""
        import shutil
        import sys

        level = params.get("level", None)
        if level is None:
            return {"status": "error", "message": "Missing level parameter"}

        # Support relative values like "+10" or "-10"
        level_str = str(level).strip()
        if level_str.startswith(("+", "-")):
            amixer_val = f"{level_str}%"
        else:
            try:
                pct = max(0, min(100, int(level_str)))
            except (ValueError, TypeError):
                return {"status": "error", "message": f"Invalid level: {level}"}
            amixer_val = f"{pct}%"

        if sys.platform == "darwin":
            # macOS: use osascript
            try:
                pct = int(level_str.lstrip("+-")) if level_str.startswith(("+", "-")) else int(level_str)
                result = subprocess.run(
                    ["osascript", "-e", f"set volume output volume {max(0, min(100, pct))}"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode != 0:
                    return {"status": "error", "message": result.stderr.strip()}
                return {"status": "success", "volume": max(0, min(100, pct))}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        # Linux (Jetson / Raspberry Pi): use amixer
        if not shutil.which("amixer"):
            return {"status": "error", "message": "amixer not found"}

        # Try the configured audio device first, fall back to default
        card_args: list[str] = []
        device_name = self.app.config.audio_device
        if device_name:
            try:
                import sounddevice as sd
                for i, d in enumerate(sd.query_devices()):
                    if device_name.lower() in d["name"].lower() and d["max_output_channels"] > 0:
                        # Extract ALSA card number from device name
                        # sounddevice names look like "hw:2,0" or "Reachy Mini Audio: ..."
                        # We need the card number for amixer -c
                        info = sd.query_devices(i)
                        # Parse hostapi and device index to find ALSA card
                        card_args = ["-c", str(info.get("index", i))]
                        break
            except Exception:
                pass

            # Alternative: parse /proc/asound/cards for the device name
            if not card_args:
                try:
                    with open("/proc/asound/cards") as f:
                        for line in f:
                            if device_name.lower().replace(" ", "").replace("-", "") in line.lower().replace(" ", "").replace("-", ""):
                                card_num = line.strip().split()[0]
                                card_args = ["-c", card_num]
                                break
                except Exception:
                    pass

        # Try common mixer control names
        for control in ["PCM", "Speaker", "Master", "Playback"]:
            result = subprocess.run(
                ["amixer", *card_args, "sset", control, amixer_val],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # Parse current volume from output
                import re
                match = re.search(r"\[(\d+)%\]", result.stdout)
                current = int(match.group(1)) if match else None
                return {"status": "success", "volume": current, "control": control}

        return {"status": "error", "message": "No suitable mixer control found"}

    def _cmd_status(self, params: dict) -> dict:
        reachy = self.app.reachy
        status = {
            "connected": reachy is not None,
            "conversation_stopped": self._conversation_stopped,
        }
        if reachy:
            try:
                pose = reachy.get_current_head_pose()
                status["head_pose"] = pose.tolist() if hasattr(pose, "tolist") else str(pose)
            except Exception:
                pass
            try:
                pos = reachy.get_present_antenna_joint_positions()
                status["antenna_positions_rad"] = list(pos)
            except Exception:
                pass
        return status

    def _cmd_stop_conversation(self, params: dict) -> dict:
        if self._conversation_stopped:
            return {"status": "ok", "conversation": "already_stopped"}
        self._conversation_stopped = True
        logger.info("Conversation STOPPED by gateway command")
        # Thread-safe interrupt: schedule on the event loop
        if self._event_loop:
            self._event_loop.call_soon_threadsafe(self._interrupt_event.set)
        return {"status": "success", "conversation": "stopped"}

    def _cmd_resume_conversation(self, params: dict) -> dict:
        if not self._conversation_stopped:
            return {"status": "ok", "conversation": "already_active"}
        self._conversation_stopped = False
        logger.info("Conversation RESUMED by gateway command")
        return {"status": "success", "conversation": "active"}

    # ── Pipeline task 1: Audio loop (mic → VAD → STT → send) ─────────

    async def _audio_loop(self) -> None:
        """Continuous mic read → VAD state machine → STT → send to AI."""
        speech_frames: list[np.ndarray] = []
        silence_count = 0
        barge_in_count = 0
        barge_in_chunks: list[np.ndarray] = []  # audio from barge-in detection
        # Pre-roll: ring buffer of recent audio during SPEAKING/THINKING.
        # ~640ms at 1024/16kHz = 10 frames.  On barge-in, pre_roll +
        # barge_in_chunks are replayed to STT so the start of the user's
        # utterance isn't lost.  10 frames ensures Paraformer's 61-frame
        # first-decode threshold (~610ms) is met even without barge_in_chunks.
        pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=10)
        max_silence = int(self.app.config.silence_duration * self.app.config.sample_rate / 1024)
        max_frames = int(self.app.config.max_recording_duration * self.app.config.sample_rate / 1024)
        confirm_frames = self.app.config.barge_in_confirm_frames
        streaming_stt = self._stt.supports_streaming
        # Interpreter mode: track stable text for timeout fallback
        interp_last_stable = ""
        interp_stable_since = 0.0
        # Streaming STT: connect once at start, feed continuously
        if streaming_stt:
            await asyncio.to_thread(
                self._stt.start_stream, self.app.config.sample_rate
            )

        self._set_state(ConvState.IDLE)
        self._last_speech_time = time.monotonic()

        if self._client:
            await self._client.send_state_change("listening")

        if streaming_stt:
            logger.info("Listening... (streaming STT mode)")
        else:
            logger.info("Listening... (speak now)")

        while self._running:
            # Sync streaming_stt with current backend state (handles reconnect/disconnect)
            stt_available = self._stt.supports_streaming
            if stt_available and not streaming_stt:
                streaming_stt = True
                await asyncio.to_thread(
                    self._stt.start_stream, self.app.config.sample_rate
                )
                logger.info("Upgraded to streaming STT after reconnect")
            elif not stt_available and streaming_stt:
                streaming_stt = False
                logger.info("STT disconnected, falling back to batch mode")

            chunk = await self._audio.read_chunk(1024)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            if not isinstance(chunk, np.ndarray):
                chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # ── SPEAKING state ──
            if self._state == ConvState.SPEAKING and not self._interpreter_mode:
                # Monologue mode: always listen in background (no barge-in)
                if self._monologue_mode:
                    await self._bg_listen(chunk, streaming_stt)
                    continue

                # Conversation mode: detect barge-in (3-layer defense)
                elif not self.app.config.barge_in_enabled:
                    continue

                # Layer 1: Cooldown — ignore early frames after TTS starts
                cooldown_s = self.app.config.barge_in_cooldown_ms / 1000.0
                if time.monotonic() - self._speaking_since < cooldown_s:
                    continue

                # Buffer all post-cooldown audio for pre-roll replay
                pre_roll.append(chunk)

                # Layer 2: Energy gate — skip low-energy noise
                energy = float(np.abs(chunk).mean())
                if energy < self.app.config.barge_in_energy_threshold:
                    # Decay instead of hard reset — one quiet frame shouldn't
                    # erase progress when speech is intermittent with echo.
                    barge_in_count = max(0, barge_in_count - 1)
                    if barge_in_count == 0:
                        barge_in_chunks.clear()
                    continue

                # Layer 3: Stricter Silero threshold during playback
                if self._vad:
                    prob = await asyncio.to_thread(
                        self._vad.speech_probability, chunk, self.app.config.sample_rate
                    )
                    if prob < self.app.config.barge_in_silero_threshold:
                        barge_in_count = max(0, barge_in_count - 1)
                        if barge_in_count == 0:
                            barge_in_chunks.clear()
                        continue

                # All layers passed — count toward confirmation
                barge_in_count += 1
                barge_in_chunks.append(chunk)
                if barge_in_count >= confirm_frames:
                    logger.info(
                        f"Barge-in confirmed ({barge_in_count} frames, "
                        f"{len(pre_roll)} pre-roll)"
                    )
                    speech_frames = list(barge_in_chunks)
                    await self._fire_interrupt()
                    self._set_state(ConvState.LISTENING)
                    if streaming_stt:
                        # Restart STT and replay pre-roll (~640ms) + confirmed
                        # barge-in speech (~128ms) so the utterance start and
                        # the VAD-verified speech aren't lost.
                        replay = list(pre_roll) + list(barge_in_chunks)
                        def _restart_stt():
                            self._stt.cancel_stream()
                            self._stt.start_stream(self.app.config.sample_rate)
                            for c in replay:
                                self._stt.feed_chunk(c)
                        await asyncio.to_thread(_restart_stt)
                    silence_count = 0
                    barge_in_count = 0
                    barge_in_chunks.clear()
                    pre_roll.clear()
                continue

            # ── TRANSCRIBING / THINKING states ──
            if self._state in (ConvState.TRANSCRIBING, ConvState.THINKING) and not self._interpreter_mode:
                if self._monologue_mode:
                    await self._bg_listen(chunk, streaming_stt)
                    continue
                else:
                    if streaming_stt:
                        # Keep feeding audio so STT has context when user speaks
                        # next — prevents first-syllable loss after silence gap.
                        await asyncio.to_thread(self._stt.feed_chunk, chunk)
                    continue

            # ── Interpreter continuous mode: feed ASR regardless of state ──
            if self._interpreter_mode and streaming_stt:
                partial = await asyncio.to_thread(self._stt.feed_chunk, chunk)
                if partial and partial.text:
                    logger.debug(f"Partial: \"{partial.text}\" (final={partial.is_final}, stable={partial.is_stable})")
                    self.app.events.emit("asr_partial", {"text": partial.text})

                    should_translate = False
                    if partial.is_final and partial.text.strip():
                        should_translate = True
                    elif partial.is_stable and partial.text.strip():
                        # Timeout fallback: if stable text unchanged for 3s,
                        # treat as final (handles ASR backends that rarely
                        # emit is_final).
                        if partial.text != interp_last_stable:
                            interp_last_stable = partial.text
                            interp_stable_since = time.monotonic()
                        elif time.monotonic() - interp_stable_since > 3.0:
                            should_translate = True
                            logger.info("Interpreter stable timeout fallback")
                    else:
                        interp_last_stable = ""

                    if should_translate:
                        text = partial.text
                        self._stt._partial_text = ""
                        self._stt._final_text = ""
                        interp_last_stable = ""
                        self.app.events.emit("asr_final", {"text": text})
                        logger.info(f"Interpreter translate: \"{text}\"")
                        if self._interp_sequencer:
                            self._interp_sequencer.submit(text)
                continue

            # ── IDLE / LISTENING states: accumulate speech ──
            # Always feed audio to STT when connected (keeps recognition continuous)
            if streaming_stt:
                partial = await asyncio.to_thread(self._stt.feed_chunk, chunk)
                if partial and partial.text:
                    logger.debug(f"Partial: \"{partial.text}\" (final={partial.is_final}, stable={partial.is_stable})")
                    self.app.events.emit("asr_partial", {"text": partial.text})

            has_speech = await asyncio.to_thread(self._audio._detect_speech, chunk)
            if has_speech:
                if self._state == ConvState.IDLE:
                    self._set_state(ConvState.LISTENING)
                    logger.info("Speech detected!")
                speech_frames.append(chunk)
                silence_count = 0

                # If ASR sent is_final, skip silence wait and process immediately.
                # The server already reset its stream after endpoint — no WS
                # restart needed.  Restarting would add 50-200ms latency during
                # which audio is lost, causing first-word drops.
                if streaming_stt and partial and partial.is_final:
                    text = partial.text
                    self._stt._partial_text = ""
                    self._stt._final_text = ""
                    if text.strip():
                        # Real result — reset all state and process.
                        speech_frames = []
                        silence_count = 0
                        barge_in_count = 0
                        if self._vad:
                            self._vad.reset()
                        logger.info(f"ASR is_final received: \"{text}\"")
                        self._spawn_task(
                            self._process_and_send(text),
                            name="conversation.process_and_send",
                        )
                    else:
                        # Empty endpoint from stale server silence — ignore.
                        # Do NOT reset speech_frames/silence_count; real speech
                        # is still in progress and would be lost.
                        logger.debug("ASR empty endpoint — continuing")
                    continue

            elif self._state == ConvState.LISTENING:
                silence_count += 1
                speech_frames.append(chunk)
                # Audio already fed to STT at top of this block

                # Check if STT sent is_final during silence.
                # No WS restart — server already reset its stream.
                if streaming_stt and partial and partial.is_final:
                    text = partial.text
                    self._stt._partial_text = ""
                    self._stt._final_text = ""
                    if text.strip():
                        speech_frames = []
                        silence_count = 0
                        barge_in_count = 0
                        if self._vad:
                            self._vad.reset()
                        logger.info(f"ASR is_final during silence: \"{text}\"")
                        self._spawn_task(
                            self._process_and_send(text),
                            name="conversation.process_and_send",
                        )
                    else:
                        # Empty endpoint — don't reset counters.
                        logger.debug("ASR empty endpoint during silence — continuing")
                    continue

                if silence_count >= max_silence or len(speech_frames) >= max_frames:
                    # End of utterance by silence timeout
                    self._set_state(ConvState.TRANSCRIBING)

                    if streaming_stt:
                        # Get final text and eagerly reconnect STT
                        def _finish_and_restart():
                            text = self._stt.finish_stream()
                            self._stt.start_stream(self.app.config.sample_rate)
                            return text
                        text = await asyncio.to_thread(_finish_and_restart)
                        speech_frames = []
                        silence_count = 0
                        barge_in_count = 0
    
                        if self._vad:
                            self._vad.reset()
                        self._spawn_task(
                            self._process_and_send(text),
                            name="conversation.process_and_send",
                        )
                    else:
                        # Batch mode — concatenate and transcribe
                        audio_data = np.concatenate(speech_frames)
                        speech_frames = []
                        silence_count = 0
                        barge_in_count = 0
    
                        if self._vad:
                            self._vad.reset()
                        duration = len(audio_data) / self.app.config.sample_rate
                        logger.info(f"Captured {duration:.2f}s of audio")
                        self._spawn_task(
                            self._transcribe_and_send(audio_data),
                            name="conversation.transcribe_and_send",
                        )

    def _restart_stt_stream(self) -> None:
        """Cancel current STT stream and open a fresh connection immediately.

        Called from worker thread. Eagerly reconnects so that the next
        feed_chunk doesn't pay WS connection latency.
        """
        self._stt.cancel_stream()
        self._stt.start_stream(self.app.config.sample_rate)

    async def _bg_listen(self, chunk: np.ndarray, streaming_stt: bool) -> None:
        """Background listening during SPEAKING/THINKING in monologue mode.

        Runs VAD on audio and feeds speech to streaming STT without
        interrupting playback. Detected speech is stored in _pending_speech
        for inclusion in the next monologue prompt.
        """
        max_silence_bg = int(
            self.app.config.silence_duration * self.app.config.sample_rate / 1024
        )
        has_speech = await asyncio.to_thread(self._audio._detect_speech, chunk)

        if has_speech:
            if not self._bg_speech_frames:
                # Speech start — begin streaming STT
                logger.debug("BG listen: speech detected while speaking")
                if streaming_stt:
                    self._stt.cancel_stream()  # cancel any stale stream
                    await asyncio.to_thread(
                        self._stt.start_stream, self.app.config.sample_rate
                    )
            self._bg_speech_frames.append(chunk)
            self._bg_silence_count = 0

            if streaming_stt:
                partial = await asyncio.to_thread(self._stt.feed_chunk, chunk)
                if partial and partial.text:
                    self.app.events.emit("asr_partial", {"text": partial.text})
                    if partial.is_final:
                        # STT says utterance complete
                        text = await asyncio.to_thread(self._stt.finish_stream)
                        if text and text.strip():
                            logger.info(f'BG heard: "{text}"')
                            self.app.events.emit("asr_final", {"text": text})
                            self._pending_speech = text
                        self._bg_speech_frames = []
                        self._bg_silence_count = 0
                        if self._vad:
                            self._vad.reset()

        elif self._bg_speech_frames:
            # Silence after speech — keep feeding STT
            self._bg_silence_count += 1
            self._bg_speech_frames.append(chunk)
            if streaming_stt:
                await asyncio.to_thread(self._stt.feed_chunk, chunk)

            if self._bg_silence_count >= max_silence_bg:
                # End of bg utterance
                if streaming_stt:
                    text = await asyncio.to_thread(self._stt.finish_stream)
                    if text and text.strip():
                        logger.info(f'BG heard: "{text}"')
                        self.app.events.emit("asr_final", {"text": text})
                        self._pending_speech = text
                self._bg_speech_frames = []
                self._bg_silence_count = 0
                if self._vad:
                    self._vad.reset()

    async def _transcribe_and_send(self, audio: np.ndarray) -> None:
        """Transcribe audio (batch) and send to AI."""
        try:
            text = await asyncio.to_thread(
                self._stt.transcribe, audio, self.app.config.sample_rate
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.app.events.emit("asr_final", {"text": ""})
            self._set_state(ConvState.IDLE)
            return

        await self._process_and_send(text)

    async def _process_and_send(self, text: str) -> None:
        """Process transcribed text (wake word check etc.) and send to AI.

        Serialised via _send_lock so two rapid utterances don't fire
        overlapping LLM streams.  In interpreter mode, ASR keeps running
        but translations are still serialised to avoid pipeline conflicts.
        """
        async with self._send_lock:
            await self._process_and_send_inner(text)

    async def _process_and_send_inner(self, text: str) -> None:
        if not text or not text.strip():
            logger.info("(no speech detected)")
            self.app.events.emit("asr_final", {"text": ""})
            self._set_state(ConvState.IDLE)
            return

        # During diary narration, suppress conversation/monologue/interpreter responses.
        # ASR still fires events (for daily logging) but we don't send to LLM.
        if self._narration_mode:
            logger.debug("Narration active — skipping LLM for: %s", text[:40])
            self.app.events.emit("asr_final", {"text": text})
            self._set_state(ConvState.IDLE)
            return

        # Filter out background noise transcriptions.
        # STT often produces short repetitive fragments from ambient sound
        # (e.g. "SHE SAYS SHE", "THE THE", "HMM HMM HMM").
        # Real speech like "hi" or "hello how are you" should pass.
        stripped = text.strip()
        alpha_only = re.sub(r"[^\w\s]", "", stripped)
        words = alpha_only.split()
        if words:
            unique_words = set(w.lower() for w in words)
            # Noise pattern: many words but very few unique (repetitive)
            if len(words) >= 3 and len(unique_words) <= 2:
                logger.info(f'Ignoring repetitive noise: "{stripped}"')
                self.app.events.emit("asr_final", {"text": ""})
                self._set_state(ConvState.IDLE)
                return

        logger.info(f'You said: "{text}"')
        self.app.events.emit("asr_final", {"text": text})

        # Check wake word
        if self._wake_detector and not self._conversation_active:
            if not self._wake_detector.detect(text):
                logger.info(f'Waiting for wake word "{self.app.config.wake_word}"...')
                self._set_state(ConvState.IDLE)
                return
            logger.info("Wake word detected!")
            if self.app.reachy and self.app.motor_enabled:
                try:
                    self.app.reachy.set_target_antenna_joint_positions([0.7, -0.7])
                    await asyncio.sleep(0.2)
                    self.app.reachy.set_target_antenna_joint_positions([-0.7, 0.7])
                    await asyncio.sleep(0.2)
                    self.app.reachy.set_target_antenna_joint_positions([0.0, 0.0])
                except Exception as e:
                    logger.error(f"Antenna animation failed: {e}")
            text = text.lower().replace(self.app.config.wake_word.lower(), "").strip()
            self._conversation_active = True

        if not text:
            self._set_state(ConvState.IDLE)
            return

        # Standalone mode
        if self.app.config.standalone_mode:
            response = f"I heard you say: {text}"
            await self._sentence_queue.put(SentenceItem(text=response, is_last=True))
            self._set_state(ConvState.SPEAKING)
            return

        # Send to AI
        if self._monologue_mode:
            text = self._compose_monologue_prompt(text)
        else:
            # Inject face recognition context into conversation
            ctx = self._get_vision_context()
            if ctx:
                text = f"[Faces: {ctx}]\n{text}"
        logger.info("Sending to AI...")
        self._set_state(ConvState.THINKING)
        self._t_send = time.perf_counter()
        self._last_speech_time = time.monotonic()

        if self.app.config.play_emotions and not self._monologue_mode:
            self.app.emotions.queue_emotion("thinking")

        if self._client:
            await self._client.send_state_change("thinking")

        try:
            await self._client.send_message_streaming(text)
        except Exception as e:
            logger.error(f"Error sending to AI: {e}")
            self.app.events.emit("asr_final", {"text": ""})
            if self.app.config.play_emotions and not self._monologue_mode:
                self.app.emotions.queue_emotion("sad")
            self._set_state(ConvState.IDLE)

    async def _process_and_send_raw(self, text: str) -> None:
        """Send pre-composed text directly to LLM (used by monologue timer)."""
        if not text or not self._client:
            return
        logger.info(f"Monologue prompt: {text[:60]}")
        # Emit observation for dashboard monologue area
        self.app.events.emit("observation", {"text": text})
        self._set_state(ConvState.THINKING)
        self._t_send = time.perf_counter()
        try:
            await self._client.send_message_streaming(text)
        except Exception as e:
            logger.error(f"Monologue send error: {e}")
            self._set_state(ConvState.IDLE)

    # ── Monologue timer (independent of audio input) ───────────────────

    async def _monologue_timer(self) -> None:
        """Periodically trigger monologue when idle, independent of audio stream.

        Runs as a separate async task so monologue generation works even when
        the microphone device is missing or no audio chunks are flowing.
        """
        interval = self.app.config.monologue_interval
        logger.info(f"Monologue timer started (interval={interval}s)")

        while self._running:
            await asyncio.sleep(1.0)  # check every second

            if not self._monologue_mode or self._narration_mode:
                continue  # mode switched or narration active — stay alive but do nothing

            if (
                self._state != ConvState.IDLE
                or not self._client
                or time.monotonic() - self._last_speech_time < interval
            ):
                continue

            # Grab any speech captured in background, then clear
            transcript = self._pending_speech
            self._pending_speech = None
            prompt = self._compose_monologue_prompt(transcript)
            self._spawn_task(
                self._process_and_send_raw(prompt),
                name="conversation.monologue_auto",
            )
            # Reset timer so next monologue waits a full interval
            self._last_speech_time = time.monotonic()

    # ── Pipeline task 2: Sentence accumulator ─────────────────────────

    async def _sentence_accumulator(self) -> None:
        """Consume stream text deltas, split into sentences, feed sentence_queue."""
        sentence_ends = {
            ".", "!", "?", ",", ";", "\n",
            "\u3002", "\uff01", "\uff1f", "\uff0c", "\uff1b", "\u3001",
        }
        buffer = ""
        last_chunk_ts = time.monotonic()
        first_sentence_emitted = False

        while self._running:
            # Use aggressive thresholds for first sentence (get TTS started fast)
            flush_timeout_s = 0.15 if not first_sentence_emitted else 0.35
            flush_min_chars = 8 if not first_sentence_emitted else 24

            try:
                chunk = await asyncio.wait_for(
                    self._stream_text_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                if (
                    buffer.strip()
                    and (time.monotonic() - last_chunk_ts) >= flush_timeout_s
                    and len(buffer.strip()) >= flush_min_chars
                ):
                    await self._sentence_queue.put(SentenceItem(text=buffer.strip()))
                    buffer = ""
                    first_sentence_emitted = True
                continue

            # In-band reset: discard stale buffer from previous response
            if chunk is _RESET_BUFFER:
                if buffer.strip():
                    logger.debug(f"Accumulator reset: discarding '{buffer.strip()[:30]}'")
                buffer = ""
                first_sentence_emitted = False
                continue

            if chunk is None:
                # End of stream — flush remaining buffer
                if buffer.strip():
                    await self._sentence_queue.put(
                        SentenceItem(text=buffer.strip(), is_last=True)
                    )
                else:
                    await self._sentence_queue.put(
                        SentenceItem(text="", is_last=True)
                    )
                buffer = ""
                first_sentence_emitted = False
                continue

            buffer += chunk
            last_chunk_ts = time.monotonic()

            # Split on sentence boundaries (min 2 chars to avoid empty splits)
            min_split = 2
            while True:
                idx = -1
                for ch in sentence_ends:
                    pos = buffer.find(ch, min_split)
                    if pos >= 0 and (idx < 0 or pos < idx):
                        idx = pos

                if idx < 0:
                    break

                sentence = buffer[: idx + 1].strip()
                buffer = buffer[idx + 1 :]

                if sentence:
                    await self._sentence_queue.put(SentenceItem(text=sentence))
                    first_sentence_emitted = True

    # ── Pipeline task 3: TTS worker (sentence → synthesized audio) ───

    async def _tts_worker(self) -> None:
        """Continuously synthesize sentences into the audio queue.

        Runs ahead of playback: while the output pipeline plays sentence N,
        this worker is already synthesizing sentences N+1, N+2, etc.
        The bounded audio_queue (maxsize=3) provides natural back-pressure.
        """
        while self._running:
            try:
                item = await asyncio.wait_for(
                    self._sentence_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if item is None:
                continue

            if item.is_last and not item.text:
                await self._audio_queue.put((item, None))
                continue

            clean = _strip_for_tts(item.text)
            if not clean:
                # Forward is_last sentinel even if text was only emoji/markdown
                if item.is_last:
                    await self._audio_queue.put((SentenceItem(text="", is_last=True), None))
                continue

            # Synthesize TTS (streaming or batch)
            if self._tts and self._tts.supports_streaming:
                chunks = []
                try:
                    async for chunk, sr in self._tts.synthesize_streaming(clean):
                        if self._interrupt_event.is_set():
                            break
                        chunks.append((chunk, sr))
                except Exception as e:
                    logger.warning(f"TTS worker synthesis failed: {e}")
                    chunks = None

                # Don't check interrupt_event here — let the output_pipeline
                # handle draining.  Checking here causes ALL new sentences to
                # be silently dropped when the event hasn't been cleared yet
                # (fast responses after barge-in).

                logger.debug(
                    f"TTS ready: \"{clean[:20]}...\" "
                    f"({len(chunks) if chunks else 0} chunks)"
                )
                await self._audio_queue.put((item, chunks if chunks else None))
            else:
                # Batch mode — no pre-synthesized chunks, output pipeline
                # will call synthesize() itself
                await self._audio_queue.put((item, None))

    # ── Pipeline task 4: Output pipeline (play pre-synthesized audio) ─

    async def _output_pipeline(self) -> None:
        """Play pre-synthesized audio from the audio queue.

        TTS synthesis is handled by _tts_worker, so this task only does
        playback — no waiting for TTS between sentences.
        """
        while self._running:
            try:
                entry = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if entry is None:
                continue

            item, prefetched_chunks = entry

            if item.is_last and not item.text:
                await self._finish_speaking()
                continue

            # STOPPED: skip TTS playback, just drain
            if self._conversation_stopped:
                logger.debug(f"Stopped — skipping TTS: \"{item.text[:30]}\"")
                if item.is_last:
                    await self._finish_speaking()
                continue

            # Speak this sentence
            t_speak_start = time.perf_counter()
            self.app.is_speaking = True
            self._set_state(ConvState.SPEAKING)

            interrupted = await self._speak_interruptible(
                item.text, prefetched_chunks
            )
            speak_ms = (time.perf_counter() - t_speak_start) * 1000
            logger.info(
                f"Spoke: {len(item.text)} chars in {speak_ms:.0f}ms"
                f"{' (interrupted)' if interrupted else ''}"
            )

            # Brief pause between sentences for natural rhythm
            if not interrupted and not item.is_last:
                try:
                    await asyncio.wait_for(
                        self._interrupt_event.wait(), timeout=0.15
                    )
                    interrupted = True  # event was set during pause
                except asyncio.TimeoutError:
                    pass  # normal: no interrupt during pause

            if interrupted or self._interrupt_event.is_set():
                _drain_queue(self._audio_queue)
                _drain_queue(self._sentence_queue)
                self._interrupt_event.clear()
                await self._finish_speaking()
                continue

            if item.is_last:
                await self._finish_speaking()

    async def _stop_gst_playback(self) -> None:
        """Stop GStreamer playback pipeline after audio has drained."""
        if self._gst_playing and self.app.reachy:
            try:
                # Push several silence frames to flush ALSA pipeline and
                # avoid the pop/click noise when stopping abruptly.
                silence = np.zeros(1600, dtype=np.float32)
                for _ in range(5):
                    self.app.reachy.media.push_audio_sample(silence)
                await asyncio.sleep(0.4)
                self.app.reachy.media.stop_playing()
            except Exception:
                pass
            self._gst_playing = False

    def _stop_gst_playback_sync(self) -> None:
        """Stop GStreamer playback pipeline (sync, for interrupt path)."""
        if self._gst_playing and self.app.reachy:
            try:
                self.app.reachy.media.stop_playing()
            except Exception:
                pass
            self._gst_playing = False

    async def _finish_speaking(self) -> None:
        """Clean up after speaking is done."""
        await self._stop_gst_playback()
        self.app.is_speaking = False
        if self._wobbler:
            self._wobbler.reset()
        # Skip IDLE transition if barge-in already moved us to LISTENING —
        # otherwise the dashboard briefly flickers LISTENING → IDLE → LISTENING.
        if self._state != ConvState.LISTENING:
            self._set_state(ConvState.IDLE)
        if self._client:
            try:
                state = "stopped" if self._conversation_stopped else "listening"
                await self._client.send_state_change(state)
            except Exception:
                pass
        if self._conversation_stopped:
            logger.info("Conversation stopped — waiting for resume")
        else:
            logger.info("Ready for next turn")

    # ── Public speak API (for diary narration etc.) ──────────────────

    async def speak_text(self, text: str) -> None:
        """Speak arbitrary text via TTS and wait for completion.

        Used by diary narration and other features that need to speak
        without going through the LLM pipeline.
        """
        if not text.strip():
            return

        # Clear interrupt event so TTS pipeline processes the text
        self._interrupt_event.clear()

        # Queue sentences
        await self._sentence_queue.put(SentenceItem(text=text, is_last=True))

        # Wait for speaking to finish.
        # Need to wait for is_speaking to go True first (pipeline picks it up),
        # then wait for it to go False again (done speaking).
        # Allow up to 90s for long diary sections.
        started = False
        for _ in range(900):
            await asyncio.sleep(0.1)
            if self._interrupt_event.is_set():
                break  # Interrupted externally (stop narration)
            if self.app.is_speaking:
                started = True
            elif started and not self.app.is_speaking:
                break  # Was speaking, now done
            # Also break if queue drained and never started speaking (TTS disabled)
            if not started and self._sentence_queue.empty() and self._audio_queue.empty():
                await asyncio.sleep(0.3)  # Small grace period
                if not self.app.is_speaking:
                    break

    def enter_narration_mode(self) -> None:
        """Enter narration mode — suppress conversation, monologue, interpreter."""
        if self._narration_mode:
            return
        self._narration_mode = True
        logger.info("Entering narration mode — conversation suppressed")
        # Interrupt any in-progress speech
        self._interrupt_event.set()
        self._stop_gst_playback_sync()
        self.app.is_speaking = False
        _drain_queue(self._sentence_queue)
        _drain_queue(self._audio_queue)

    def exit_narration_mode(self) -> None:
        """Exit narration mode — resume normal conversation."""
        if not self._narration_mode:
            return
        self._narration_mode = False
        logger.info("Exiting narration mode — conversation resumed")

    async def stop_speaking(self) -> None:
        """Interrupt any ongoing TTS playback immediately."""
        await self._fire_interrupt()

    # ── Interrupt ─────────────────────────────────────────────────────

    async def _fire_interrupt(self) -> None:
        """Fire barge-in interrupt: stop playback, drain queues, notify server.

        Designed to be fast — network calls are fire-and-forget so the audio
        loop resumes listening immediately after barge-in.
        """
        logger.info("Firing interrupt")
        self._interrupt_event.set()
        # Invalidate current run so stale deltas arriving after drain are rejected
        self._current_run_id = None

        # Stop playback and drain queues immediately (no awaits)
        self._stop_gst_playback_sync()
        self.app.is_speaking = False
        if self._wobbler:
            self._wobbler.reset()
        _drain_queue(self._stream_text_queue)
        _drain_queue(self._sentence_queue)
        _drain_queue(self._audio_queue)
        # Tell the accumulator to discard its local buffer so stale text
        # from the interrupted response doesn't flush later.
        await self._stream_text_queue.put(_RESET_BUFFER)

        # Notify server in background — don't block the audio loop
        if self._client:
            self._spawn_task(
                self._client.send_interrupt(),
                name="conversation.send_interrupt",
            )

    # ── TTS + interruptible playback ──────────────────────────────────

    async def _speak_interruptible(
        self, text: str, prefetched_chunks: list | None = None
    ) -> bool:
        """Synthesize and play one sentence. Returns True if interrupted."""
        if not text.strip():
            return False

        clean_text = _strip_for_tts(text)

        # Use streaming TTS if backend supports it
        if self._tts.supports_streaming:
            return await self._speak_streaming_tts(clean_text, prefetched_chunks)

        return await self._speak_batch_tts(clean_text)

    async def _speak_streaming_tts(
        self, text: str, prefetched_chunks: list | None = None
    ) -> bool:
        """Stream TTS: play audio chunks as they arrive from the backend."""
        try:
            n_chunks = len(prefetched_chunks) if prefetched_chunks else 0
            logger.info(f"Playing: \"{text[:20]}\" ({len(text)} chars, {n_chunks} chunks)")

            # Build chunk source: prefetched list or live stream
            if prefetched_chunks:
                async def _chunk_source():
                    for item in prefetched_chunks:
                        yield item
                source = _chunk_source()
            else:
                source = self._tts.synthesize_streaming(text)

            if self.app.reachy and getattr(getattr(self.app.reachy, "media", None), "audio", None) is not None:
                # Accumulate all chunks first, then push as continuous audio
                all_chunks = []
                sample_rate = 16000
                async for chunk, sr in source:
                    if self._interrupt_event.is_set():
                        return True
                    all_chunks.append(chunk)
                    sample_rate = sr

                if not all_chunks:
                    return False

                audio = np.concatenate(all_chunks)

                # Apply software volume gain
                vol = self.app.config.audio_volume
                if vol != 1.0:
                    audio = np.clip(audio * vol, -1.0, 1.0).astype(np.float32)

                # Resample to 16kHz if TTS outputs a different rate
                audio, sample_rate = _resample_if_needed(audio, sample_rate)

                reachy = self.app.reachy

                # Keep pipeline alive across sentences — only start if not already playing
                if not self._gst_playing:
                    reachy.media.start_playing()
                    self._gst_playing = True
                if self._wobbler:
                    self._wobbler.start()

                interrupted = False
                chunk_size = 1600  # 100ms chunks at 16kHz
                prebuf = 4  # push first N chunks without sleeping to fill pipeline buffer
                audio_duration = len(audio) / sample_rate
                t_push_start = time.monotonic()
                try:
                    for idx, i in enumerate(range(0, len(audio), chunk_size)):
                        if self._interrupt_event.is_set():
                            interrupted = True
                            break
                        chunk = audio[i : i + chunk_size]
                        reachy.media.push_audio_sample(chunk)
                        if self._wobbler:
                            self._wobbler.feed(chunk)
                        if idx >= prebuf:
                            await asyncio.sleep(chunk_size / sample_rate * 0.9)
                finally:
                    if self._wobbler:
                        self._wobbler.reset()
                    if self.app.motor_enabled:
                        reachy.set_target_antenna_joint_positions([0.0, 0.0])

                # Push a fade-out tail to avoid pop/click between sentences
                # or when pipeline stops.  Ramps from last sample to zero.
                if not interrupted and len(audio) > 0:
                    fade_len = 800  # 50ms at 16kHz
                    last_val = float(audio[-1])
                    fade = np.linspace(last_val, 0.0, fade_len, dtype=np.float32)
                    reachy.media.push_audio_sample(fade)

                # Wait for ALSA buffer to finish playing before returning.
                # Push runs faster than realtime (0.9x + prebuf), so there's
                # audio still in the buffer.  Wait interruptibly so barge-in
                # can still cut in.
                if not interrupted:
                    remaining = audio_duration - (time.monotonic() - t_push_start)
                    if remaining > 0:
                        try:
                            await asyncio.wait_for(
                                self._interrupt_event.wait(), timeout=remaining
                            )
                            interrupted = True
                        except asyncio.TimeoutError:
                            pass  # audio finished naturally

                # Don't stop pipeline here — it stays open for next sentence
                return interrupted
            else:
                # Local playback via sounddevice (no temp file, no subprocess)
                all_chunks = []
                sample_rate = 16000
                async for chunk, sr in source:
                    if self._interrupt_event.is_set():
                        return True
                    all_chunks.append(chunk)
                    sample_rate = sr

                if not all_chunks:
                    return False

                audio = np.concatenate(all_chunks)
                return await self._play_sounddevice(audio, sample_rate)

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            logger.info(f"[TTS] {text}")
            return False

    async def _speak_batch_tts(self, text: str) -> bool:
        """Batch TTS: synthesize full file, then play."""
        temp_audio_path: str | None = None
        try:
            logger.info(f"TTS: generating speech ({len(text)} chars)...")
            temp_audio_path = await self._tts.synthesize(text)

            if self._interrupt_event.is_set():
                return True

            if self.app.reachy and getattr(getattr(self.app.reachy, "media", None), "audio", None) is not None:
                try:
                    return await self._play_on_reachy_interruptible(temp_audio_path)
                except Exception as e:
                    logger.error(f"Reachy playback failed: {e}")
                    return await self._play_local_interruptible(temp_audio_path)
            else:
                return await self._play_local_interruptible(temp_audio_path)

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            logger.info(f"[TTS] {text}")
            return False
        finally:
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
        """Write float32 audio to a temporary WAV file."""
        import struct
        import tempfile

        audio_int = (audio * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(
            prefix="reachy_claw_stream_", suffix=".wav", delete=False
        )
        data_bytes = audio_int.tobytes()
        # Write WAV header + data
        tmp.write(b"RIFF")
        tmp.write(struct.pack("<I", 36 + len(data_bytes)))
        tmp.write(b"WAVE")
        tmp.write(b"fmt ")
        tmp.write(struct.pack("<I", 16))
        tmp.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        tmp.write(b"data")
        tmp.write(struct.pack("<I", len(data_bytes)))
        tmp.write(data_bytes)
        tmp.close()
        return tmp.name

    async def _play_on_reachy_interruptible(self, audio_path: str) -> bool:
        """Play on Reachy with 100ms chunk granularity interrupt checks."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            temp_wav_path = wf.name

        try:
            await asyncio.to_thread(
                subprocess.run,
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-ac",
                    "1",
                    temp_wav_path,
                ],
                capture_output=True,
                check=True,
            )

            import wave

            with wave.open(temp_wav_path, "rb") as wf:
                sample_rate = wf.getframerate()
                audio_data = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                )
                audio_float = audio_data.astype(np.float32) / 32768.0

            # Apply software volume gain
            vol = self.app.config.audio_volume
            if vol != 1.0:
                audio_float = np.clip(audio_float * vol, -1.0, 1.0).astype(np.float32)

            # Resample to 16kHz if TTS outputs a different rate
            audio_float, sample_rate = _resample_if_needed(audio_float, sample_rate)

            reachy = self.app.reachy
            reachy.media.start_playing()

            if self._wobbler:
                self._wobbler.start()

            chunk_size = 1600  # 100ms chunks
            chunk_duration = chunk_size / sample_rate
            prebuf = 4  # push first N chunks without sleeping to fill pipeline buffer
            interrupted = False

            for idx, i in enumerate(range(0, len(audio_float), chunk_size)):
                if self._interrupt_event.is_set():
                    interrupted = True
                    break

                chunk = audio_float[i : i + chunk_size]
                reachy.media.push_audio_sample(chunk)

                if self._wobbler:
                    self._wobbler.feed(chunk)

                if idx >= prebuf:
                    await asyncio.sleep(chunk_duration * 0.9)

            if self._wobbler:
                self._wobbler.reset()

            if self.app.motor_enabled:
                reachy.set_target_antenna_joint_positions([0.0, 0.0])
            if not interrupted:
                # Wait for ALSA buffer to drain (last chunks + hw buffer)
                await asyncio.sleep(0.8)
            reachy.media.stop_playing()
            return interrupted
        finally:
            try:
                os.unlink(temp_wav_path)
            except FileNotFoundError:
                pass

    async def _play_sounddevice(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Play numpy audio via sounddevice (zero overhead, no subprocess)."""
        try:
            import sounddevice as sd

            # Find the configured audio device for output
            out_device = None
            device_name = self.app.config.audio_device
            if device_name:
                for i, d in enumerate(sd.query_devices()):
                    if device_name.lower() in d["name"].lower() and d["max_output_channels"] > 0:
                        out_device = i
                        break

            # Resample to device rate if needed (Reachy Mini Audio = 16000Hz only)
            play_audio = audio
            play_rate = sample_rate
            if out_device is not None:
                dev_info = sd.query_devices(out_device)
                # Reachy Mini Audio only supports 16000Hz but ALSA may
                # report 44100/48000 as default_samplerate.  Force 16000
                # when we matched the Reachy device by name.
                dev_rate = int(dev_info["default_samplerate"])
                if device_name and "reachy" in device_name.lower():
                    dev_rate = 16000
                if dev_rate != sample_rate:
                    # Simple linear resample
                    ratio = dev_rate / sample_rate
                    n_out = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, n_out)
                    play_audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
                    play_rate = dev_rate

                # Reachy Mini Audio needs stereo
                if dev_info["max_output_channels"] >= 2 and play_audio.ndim == 1:
                    play_audio = np.column_stack([play_audio, play_audio])

            # Apply volume gain
            vol = self.app.config.audio_volume
            if vol != 1.0:
                play_audio = np.clip(play_audio * vol, -1.0, 1.0).astype(np.float32)

            finished = asyncio.Event()
            play_error: list[Exception] = []

            def _play_blocking():
                try:
                    sd.play(play_audio, samplerate=play_rate, device=out_device, blocking=True)
                except Exception as e:
                    play_error.append(e)
                finally:
                    finished.set()

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _play_blocking)

            while not finished.is_set():
                if self._interrupt_event.is_set():
                    sd.stop()
                    return True
                await asyncio.sleep(0.02)  # 20ms poll

            if play_error:
                logger.warning(f"sounddevice play error: {play_error[0]}")
                return False

            return False
        except Exception as e:
            logger.warning(f"sounddevice playback failed: {e}, falling back to afplay")
            # Fallback to afplay
            temp_path = self._write_temp_wav(audio, sample_rate)
            try:
                return await self._play_local_interruptible(temp_path)
            finally:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    async def _play_local_interruptible(self, audio_path: str) -> bool:
        """Play locally via subprocess with interrupt support (fallback)."""
        import sys

        player = "afplay" if sys.platform == "darwin" else "aplay"
        try:
            proc = await asyncio.create_subprocess_exec(
                player, audio_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            while proc.returncode is None:
                if self._interrupt_event.is_set():
                    proc.kill()
                    await proc.wait()
                    return True
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

            return False
        except FileNotFoundError:
            logger.warning("No local audio player available")
            return False



# ── Helpers ───────────────────────────────────────────────────────────

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"  # zero width joiner
    "\U00002640-\U00002642"
    "\U000023CF-\U000023F3"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
)


_EMOTION_TAG_RE = re.compile(r"\[(?:emotion:)?\w+\]\s*")


_GST_RATE: int = 16000  # SDK pipeline is fixed at 16kHz


def _resample_if_needed(audio: np.ndarray, src_rate: int) -> tuple[np.ndarray, int]:
    """Resample audio to 16kHz if TTS outputs a different rate.

    The SDK's GStreamer pipeline is initialised at 16kHz.  Dynamically changing
    appsrc caps at runtime breaks playback, so we resample in software instead.
    """
    if src_rate == _GST_RATE:
        return audio, src_rate
    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(src_rate, _GST_RATE)
        up, down = _GST_RATE // g, src_rate // g
        resampled = resample_poly(audio, up, down).astype(np.float32)
        logger.debug("Resampled %d→%d Hz (%d→%d samples)", src_rate, _GST_RATE, len(audio), len(resampled))
        return resampled, _GST_RATE
    except ImportError:
        # scipy not available — fall back to simple linear interpolation
        n_out = int(len(audio) * _GST_RATE / src_rate)
        indices = np.linspace(0, len(audio) - 1, n_out)
        resampled = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        logger.debug("Resampled (interp) %d→%d Hz (%d→%d samples)", src_rate, _GST_RATE, len(audio), len(resampled))
        return resampled, _GST_RATE


def _strip_for_tts(text: str) -> str:
    """Strip markdown formatting, emoji, and emotion tags from text before TTS."""
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = _EMOJI_RE.sub("", text)
    text = _EMOTION_TAG_RE.sub("", text)
    return text.strip()


def _drain_queue(q: asyncio.Queue) -> None:
    """Drain all items from a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break
