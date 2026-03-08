"""ConversationPlugin -- dual-pipeline STT/TTS/gateway conversation loop.

Three concurrent tasks connected by queues and an interrupt event:
  _audio_loop:          mic → VAD → state machine → STT → send to AI
  _sentence_accumulator: stream deltas → sentence splitting → sentence_queue
  _output_pipeline:     sentence_queue → TTS → interruptible playback
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Coroutine

import numpy as np

from ..audio import AudioCapture, WakeWordDetector
from ..gateway import DesktopRobotClient
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


# ── Plugin ────────────────────────────────────────────────────────────


class ConversationPlugin(Plugin):
    """Dual-pipeline conversation: continuous listen + concurrent TTS output."""

    name = "conversation"

    def __init__(self, app):
        super().__init__(app)
        self._client: DesktopRobotClient | None = None
        self._stt = None
        self._tts = None
        self._vad = None
        self._audio: AudioCapture | None = None
        self._wake_detector: WakeWordDetector | None = None
        self._wobbler: HeadWobbler | None = None

        # Queues connecting the four pipeline stages
        self._stream_text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._sentence_queue: asyncio.Queue[SentenceItem | None] = asyncio.Queue()
        # Audio queue: (SentenceItem, list_of_chunks | None) — TTS worker feeds this
        self._audio_queue: asyncio.Queue[tuple[SentenceItem, list | None] | None] = asyncio.Queue(maxsize=3)
        self._interrupt_event = asyncio.Event()

        self._current_run_id: str | None = None
        self._conversation_active = False
        self._state = ConvState.IDLE
        self._pending_tasks: set[asyncio.Task] = set()

    def setup(self) -> bool:
        return True

    async def start(self):
        config = self.app.config

        # Initialize STT
        logger.info("Loading speech recognition model...")
        self._stt = create_stt_backend(config)
        await asyncio.to_thread(self._stt.preload)
        logger.info("Speech recognition ready")

        # Initialize TTS
        self._tts = create_tts_backend(
            backend=config.tts_backend,
            voice=config.tts_voice,
            model=config.tts_model,
            config=config,
        )
        logger.info(f"TTS backend: {config.tts_backend}")

        # Initialize VAD
        self._vad = create_vad_backend(backend=config.vad_backend, config=config)
        await asyncio.to_thread(self._vad.preload)

        # Initialize audio capture
        self._audio = AudioCapture(config, self.app.reachy, vad=self._vad)

        if config.wake_word:
            self._wake_detector = WakeWordDetector(config.wake_word)

        # Initialize head wobbler if motion plugin is available
        motion_plugin = self._find_motion_plugin()
        if motion_plugin:
            self._wobbler = HeadWobbler(
                set_speech_offsets=motion_plugin.set_speech_offsets,
                sample_rate=config.sample_rate,
            )

        # Connect to OpenClaw
        if not config.standalone_mode:
            self._client = DesktopRobotClient(config)
            await self._client.connect()

            # Warm up gateway + TTS in parallel (callbacks NOT yet wired,
            # so the warmup response won't be spoken)
            warmup_tasks = []
            if config.gateway_warmup:
                warmup_tasks.append(self._client.warmup_session())
            if self._tts:
                warmup_tasks.append(self._warmup_tts())
            if warmup_tasks:
                await asyncio.gather(*warmup_tasks, return_exceptions=True)

            # NOW wire callbacks — only real user messages will be spoken
            self._setup_callbacks()
        else:
            logger.info("Running in standalone mode - no server connection")
            if self._tts:
                await self._warmup_tts()

        # Start continuous audio capture
        await self._audio.start_continuous()

        # Wake up the robot
        if self.app.reachy:
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

        logger.info("=" * 50)
        if config.wake_word:
            logger.info(f'Say "{config.wake_word}" to activate')
        else:
            logger.info("Speak anytime - always listening!")
        logger.info("=" * 50)

        # Run four concurrent pipeline tasks
        try:
            await asyncio.gather(
                self._audio_loop(),
                self._sentence_accumulator(),
                self._tts_worker(),
                self._output_pipeline(),
            )
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

    def _set_state(self, new_state: ConvState) -> None:
        if self._state != new_state:
            logger.debug(f"State: {self._state.value} → {new_state.value}")
            self._state = new_state

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

    async def _on_stream_start(self, run_id: str) -> None:
        logger.debug(f"Stream started: {run_id}")
        self._current_run_id = run_id
        self._set_state(ConvState.THINKING)
        _drain_queue(self._stream_text_queue)

    async def _on_stream_delta(self, text: str, run_id: str) -> None:
        if self._state == ConvState.THINKING:
            # First delta = TTFT measurement point
            if hasattr(self, "_t_send") and self._t_send:
                ttft = (time.perf_counter() - self._t_send) * 1000
                logger.info(f"TTFT: {ttft:.0f}ms (send → first delta)")
            self._set_state(ConvState.SPEAKING)
        await self._stream_text_queue.put(text)

    async def _on_stream_end(self, full_text: str, run_id: str) -> None:
        if hasattr(self, "_t_send") and self._t_send:
            total = (time.perf_counter() - self._t_send) * 1000
            logger.info(f"Response complete ({len(full_text)} chars, {total:.0f}ms)")
            self._t_send = None
        else:
            logger.info(f"Response complete ({len(full_text)} chars)")
        await self._stream_text_queue.put(None)

    async def _on_stream_abort(self, reason: str, run_id: str) -> None:
        logger.info(f"Stream aborted: {reason}")
        await self._stream_text_queue.put(None)

    async def _on_tool_start(self, tool_name: str, run_id: str) -> None:
        logger.info(f"Tool started: {tool_name}")

    async def _on_tool_end(self, tool_name: str, run_id: str) -> None:
        logger.info(f"Tool ended: {tool_name}")

    async def _on_task_spawned(self, label: str, task_run_id: str) -> None:
        logger.info(f"Background task started: {label}")

    async def _on_task_completed(self, summary: str, task_run_id: str) -> None:
        logger.info(f"Background task completed: {summary[:100]}")
        short = summary[:200] if len(summary) > 200 else summary
        # Route notifications through the normal output queue to avoid
        # blocking gateway listener callbacks on TTS playback.
        await self._sentence_queue.put(SentenceItem(text=short, is_last=True))

    # ── Pipeline task 1: Audio loop (mic → VAD → STT → send) ─────────

    async def _audio_loop(self) -> None:
        """Continuous mic read → VAD state machine → STT → send to AI."""
        speech_frames: list[np.ndarray] = []
        silence_count = 0
        barge_in_count = 0
        max_silence = int(self.app.config.silence_duration * self.app.config.sample_rate / 1024)
        max_frames = int(self.app.config.max_recording_duration * self.app.config.sample_rate / 1024)
        confirm_frames = self.app.config.barge_in_confirm_frames
        streaming_stt = self._stt.supports_streaming

        self._set_state(ConvState.IDLE)

        if self._client:
            await self._client.send_state_change("listening")

        if streaming_stt:
            logger.info("Listening... (streaming STT mode)")
        else:
            logger.info("Listening... (speak now)")

        while self._running:
            chunk = await self._audio.read_chunk(1024)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            if not isinstance(chunk, np.ndarray):
                chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

            has_speech = await asyncio.to_thread(self._audio._detect_speech, chunk)

            # ── SPEAKING state: detect barge-in ──
            if self._state == ConvState.SPEAKING:
                if not self.app.config.barge_in_enabled:
                    continue
                if has_speech:
                    barge_in_count += 1
                    if barge_in_count >= confirm_frames:
                        logger.info(f"Barge-in confirmed ({barge_in_count} frames)")
                        speech_frames = [chunk]
                        await self._fire_interrupt()
                        if streaming_stt:
                            self._stt.cancel_stream()
                            await asyncio.to_thread(
                                self._stt.start_stream, self.app.config.sample_rate
                            )
                            await asyncio.to_thread(self._stt.feed_chunk, chunk)
                        self._set_state(ConvState.LISTENING)
                        silence_count = 0
                        barge_in_count = 0
                else:
                    barge_in_count = 0
                continue

            # ── TRANSCRIBING / THINKING states: wait ──
            if self._state in (ConvState.TRANSCRIBING, ConvState.THINKING):
                continue

            # ── IDLE / LISTENING states: accumulate speech ──
            if has_speech:
                if self._state == ConvState.IDLE:
                    self._set_state(ConvState.LISTENING)
                    logger.info("Speech detected!")
                    if streaming_stt:
                        await asyncio.to_thread(
                            self._stt.start_stream, self.app.config.sample_rate
                        )
                speech_frames.append(chunk)
                silence_count = 0

                # Feed chunk to streaming STT
                if streaming_stt:
                    partial = await asyncio.to_thread(self._stt.feed_chunk, chunk)
                    if partial and partial.text:
                        logger.debug(f"Partial: \"{partial.text}\" (final={partial.is_final}, stable={partial.is_stable})")
                        # If ASR sent is_final, skip silence wait and process immediately
                        if partial.is_final:
                            logger.info("ASR is_final received — skipping silence wait")
                            self._set_state(ConvState.TRANSCRIBING)
                            text = await asyncio.to_thread(self._stt.finish_stream)
                            speech_frames = []
                            silence_count = 0
                            barge_in_count = 0
                            if self._vad:
                                self._vad.reset()
                            self._spawn_task(
                                self._process_and_send(text),
                                name="conversation.process_and_send",
                            )
                            continue

            elif self._state == ConvState.LISTENING:
                silence_count += 1
                speech_frames.append(chunk)

                # Keep feeding silence to streaming STT (it needs continuity)
                if streaming_stt:
                    await asyncio.to_thread(self._stt.feed_chunk, chunk)

                if silence_count >= max_silence or len(speech_frames) >= max_frames:
                    # End of utterance
                    self._set_state(ConvState.TRANSCRIBING)

                    if streaming_stt:
                        # Finish streaming — get final text
                        text = await asyncio.to_thread(self._stt.finish_stream)
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

    async def _transcribe_and_send(self, audio: np.ndarray) -> None:
        """Transcribe audio (batch) and send to AI."""
        try:
            text = await asyncio.to_thread(
                self._stt.transcribe, audio, self.app.config.sample_rate
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self._set_state(ConvState.IDLE)
            return

        await self._process_and_send(text)

    async def _process_and_send(self, text: str) -> None:
        """Process transcribed text (wake word check etc.) and send to AI."""
        if not text or not text.strip():
            logger.info("(no speech detected)")
            self._set_state(ConvState.IDLE)
            return

        logger.info(f'You said: "{text}"')

        # Check wake word
        if self._wake_detector and not self._conversation_active:
            if not self._wake_detector.detect(text):
                logger.info(f'Waiting for wake word "{self.app.config.wake_word}"...')
                self._set_state(ConvState.IDLE)
                return
            logger.info("Wake word detected!")
            if self.app.reachy:
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
        logger.info("Sending to AI...")
        self._set_state(ConvState.THINKING)
        self._t_send = time.perf_counter()

        if self.app.config.play_emotions:
            self.app.emotions.queue_emotion("thinking")

        if self._client:
            await self._client.send_state_change("thinking")

        try:
            await self._client.send_message_streaming(text)
        except Exception as e:
            logger.error(f"Error sending to AI: {e}")
            if self.app.config.play_emotions:
                self.app.emotions.queue_emotion("sad")
            self._set_state(ConvState.IDLE)

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

            if not item.text:
                continue

            clean = item.text.replace("**", "").replace("*", "").replace("`", "")

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

                if self._interrupt_event.is_set():
                    continue

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

            if interrupted or self._interrupt_event.is_set():
                _drain_queue(self._audio_queue)
                _drain_queue(self._sentence_queue)
                self._interrupt_event.clear()
                await self._finish_speaking()
                continue

            if item.is_last:
                await self._finish_speaking()

    async def _finish_speaking(self) -> None:
        """Clean up after speaking is done."""
        self.app.is_speaking = False
        if self._wobbler:
            self._wobbler.reset()
        self._set_state(ConvState.IDLE)
        if self._client:
            try:
                await self._client.send_state_change("listening")
            except Exception:
                pass
        logger.info("Ready for next turn")

    # ── Interrupt ─────────────────────────────────────────────────────

    async def _fire_interrupt(self) -> None:
        """Fire barge-in interrupt: stop playback, drain queues, notify server."""
        logger.info("Firing interrupt")
        self._interrupt_event.set()

        if self._client:
            await self._client.send_interrupt()

        _drain_queue(self._stream_text_queue)
        _drain_queue(self._sentence_queue)
        _drain_queue(self._audio_queue)

        self.app.is_speaking = False
        if self._wobbler:
            self._wobbler.reset()

    # ── TTS + interruptible playback ──────────────────────────────────

    async def _speak_interruptible(
        self, text: str, prefetched_chunks: list | None = None
    ) -> bool:
        """Synthesize and play one sentence. Returns True if interrupted."""
        if not text.strip():
            return False

        clean_text = text.replace("**", "").replace("*", "").replace("`", "")

        # Use streaming TTS if backend supports it
        if self._tts.supports_streaming:
            return await self._speak_streaming_tts(clean_text, prefetched_chunks)

        return await self._speak_batch_tts(clean_text)

    async def _speak_streaming_tts(
        self, text: str, prefetched_chunks: list | None = None
    ) -> bool:
        """Stream TTS: play audio chunks as they arrive from the backend."""
        try:
            if prefetched_chunks:
                logger.info(
                    f"TTS stream: playing prefetched audio ({len(text)} chars, "
                    f"{len(prefetched_chunks)} chunks)"
                )
            else:
                logger.info(f"TTS stream: generating speech ({len(text)} chars)...")

            # Build chunk source: prefetched list or live stream
            if prefetched_chunks:
                async def _chunk_source():
                    for item in prefetched_chunks:
                        yield item
                source = _chunk_source()
            else:
                source = self._tts.synthesize_streaming(text)

            if self.app.reachy and hasattr(self.app.reachy, "media"):
                reachy = self.app.reachy
                reachy.media.start_playing()
                if self._wobbler:
                    self._wobbler.start()

                interrupted = False
                try:
                    async for chunk, sr in source:
                        if self._interrupt_event.is_set():
                            interrupted = True
                            break
                        reachy.media.push_audio_sample(chunk)
                        if self._wobbler:
                            self._wobbler.feed(chunk)
                        await asyncio.sleep(len(chunk) / sr * 0.9)
                finally:
                    if self._wobbler:
                        self._wobbler.reset()
                    reachy.set_target_antenna_joint_positions([0.0, 0.0])
                    if not interrupted:
                        await asyncio.sleep(0.05)
                    reachy.media.stop_playing()
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

            if self.app.reachy and hasattr(self.app.reachy, "media"):
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
            prefix="clawd_stream_", suffix=".wav", delete=False
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
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    temp_wav_path,
                ],
                capture_output=True,
                check=True,
            )

            import wave

            with wave.open(temp_wav_path, "rb") as wf:
                audio_data = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                )
                audio_float = audio_data.astype(np.float32) / 32768.0

            reachy = self.app.reachy
            reachy.media.start_playing()

            if self._wobbler:
                self._wobbler.start()

            sample_rate = 16000
            chunk_size = 1600  # 100ms chunks
            chunk_duration = chunk_size / sample_rate
            interrupted = False

            for i in range(0, len(audio_float), chunk_size):
                if self._interrupt_event.is_set():
                    interrupted = True
                    break

                chunk = audio_float[i : i + chunk_size]
                reachy.media.push_audio_sample(chunk)

                if self._wobbler:
                    self._wobbler.feed(chunk)

                await asyncio.sleep(chunk_duration * 0.9)

            if self._wobbler:
                self._wobbler.reset()

            reachy.set_target_antenna_joint_positions([0.0, 0.0])
            if not interrupted:
                await asyncio.sleep(0.3)
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

            finished = asyncio.Event()

            def _play_blocking():
                sd.play(audio, samplerate=sample_rate, blocking=True)
                finished.set()

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _play_blocking)

            while not finished.is_set():
                if self._interrupt_event.is_set():
                    sd.stop()
                    return True
                await asyncio.sleep(0.02)  # 20ms poll

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
        try:
            proc = await asyncio.create_subprocess_exec(
                "afplay", audio_path,
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

    # ── Single-shot speak (for task_completed notifications) ──────────

    async def _speak_single(self, text: str) -> None:
        """Speak a single utterance without the pipeline (for notifications)."""
        if not text.strip():
            return

        clean_text = text.replace("**", "").replace("*", "").replace("`", "")
        temp_audio_path: str | None = None

        try:
            temp_audio_path = await self._tts.synthesize(clean_text)

            if self.app.reachy and hasattr(self.app.reachy, "media"):
                try:
                    await self._play_on_reachy_interruptible(temp_audio_path)
                except Exception as e:
                    logger.error(f"Reachy playback failed: {e}")
                    await self._play_local_interruptible(temp_audio_path)
            else:
                await self._play_local_interruptible(temp_audio_path)
        except Exception as e:
            logger.error(f"TTS failed: {e}")
        finally:
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except FileNotFoundError:
                    pass


# ── Helpers ───────────────────────────────────────────────────────────


def _drain_queue(q: asyncio.Queue) -> None:
    """Drain all items from a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break
