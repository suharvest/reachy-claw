"""ConversationPlugin -- STT/TTS/gateway conversation loop.

Extracted from interface.py. Handles the full conversation cycle:
  listen -> STT -> send to gateway -> receive stream -> TTS -> speak
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile

import numpy as np

from ..audio import AudioCapture, WakeWordDetector
from ..gateway import DesktopRobotClient
from ..motion.head_wobbler import HeadWobbler
from ..plugin import Plugin
from ..stt import create_stt_backend
from ..tts import create_tts_backend

logger = logging.getLogger(__name__)


class ConversationPlugin(Plugin):
    """Conversation loop: STT -> gateway -> TTS -> speak."""

    name = "conversation"

    def __init__(self, app):
        super().__init__(app)
        self._client: DesktopRobotClient | None = None
        self._stt = None
        self._tts = None
        self._audio: AudioCapture | None = None
        self._wake_detector: WakeWordDetector | None = None
        self._wobbler: HeadWobbler | None = None

        self._stream_text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._current_run_id: str | None = None
        self._conversation_active = False

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
        )
        logger.info(f"TTS backend: {config.tts_backend}")

        # Initialize audio capture
        self._audio = AudioCapture(config, self.app.reachy)

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
            self._setup_callbacks()
            await self._client.connect()
        else:
            logger.info("Running in standalone mode - no server connection")

        # Start audio capture
        await self._audio.start()

        # Wake up the robot
        if self.app.reachy:
            logger.info("Waking up Reachy...")
            await asyncio.to_thread(self.app.reachy.wake_up)
            await asyncio.sleep(0.5)
            # Startup antenna snap
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

        # Main conversation loop
        try:
            while self._running:
                await self._conversation_turn()
        except asyncio.CancelledError:
            logger.info("Conversation loop cancelled")

    async def stop(self):
        self._running = False

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

    # -- Callbacks from desktop-robot protocol --

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
        while not self._stream_text_queue.empty():
            try:
                self._stream_text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _on_stream_delta(self, text: str, run_id: str) -> None:
        logger.debug(f"Delta [{run_id[:8]}]: {text}")
        await self._stream_text_queue.put(text)

    async def _on_stream_end(self, full_text: str, run_id: str) -> None:
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
        if self._tts:
            short = summary[:200] if len(summary) > 200 else summary
            await self._speak(short)

    # -- Conversation turn --

    async def _conversation_turn(self) -> None:
        if self._client:
            await self._client.send_state_change("listening")
        logger.info("Listening... (speak now)")
        audio = await self._audio.capture_utterance()

        if audio is None:
            await asyncio.sleep(0.1)
            return

        # Transcribe
        logger.info("Processing speech...")
        try:
            text = await asyncio.to_thread(
                self._stt.transcribe, audio, self.app.config.sample_rate
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return

        if not text or not text.strip():
            logger.info("(no speech detected)")
            return

        logger.info(f'You said: "{text}"')

        # Check wake word
        if self._wake_detector and not self._conversation_active:
            if not self._wake_detector.detect(text):
                logger.info(
                    f'Waiting for wake word "{self.app.config.wake_word}"...'
                )
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
            return

        # Standalone mode
        if self.app.config.standalone_mode:
            response = f"I heard you say: {text}"
            await self._speak(response)
            return

        # Send to AI and stream response
        logger.info("Sending to AI...")

        # Queue thinking emotion
        if self.app.config.play_emotions:
            self.app.emotions.queue_emotion("thinking")

        try:
            await self._client.send_message_streaming(text)
            full_response = await self._speak_streaming()
        except Exception as e:
            logger.error(f"Error: {e}")
            if self.app.config.play_emotions:
                self.app.emotions.queue_emotion("sad")
            full_response = ""

        if full_response:
            logger.info(f'Response: "{full_response[:100]}..."')

        if self._client:
            await self._client.send_state_change("speaking_done")

        logger.info("Ready for next turn")

    # -- Streaming speech --

    async def _speak_streaming(self) -> str:
        buffer = ""
        full_text = ""
        self.app.is_speaking = True

        sentence_ends = {".", "!", "?", "\n", "\u3002", "\uff01", "\uff1f", "\uff1b"}

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self._stream_text_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    break

                if chunk is None:
                    if buffer.strip():
                        await self._speak(buffer.strip())
                        full_text += buffer
                    break

                buffer += chunk

                while True:
                    idx = -1
                    for ch in sentence_ends:
                        pos = buffer.find(ch)
                        if pos >= 0 and (idx < 0 or pos < idx):
                            idx = pos

                    if idx < 0 or idx < 5:
                        break

                    sentence = buffer[: idx + 1].strip()
                    buffer = buffer[idx + 1 :]

                    if sentence:
                        await self._speak(sentence)
                        full_text += sentence + " "

                        if self._was_barged_in():
                            logger.info("Barge-in detected between sentences")
                            if self._client:
                                await self._client.send_interrupt()
                            while not self._stream_text_queue.empty():
                                try:
                                    self._stream_text_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            return full_text.strip()
        finally:
            self.app.is_speaking = False
            if self._wobbler:
                self._wobbler.reset()

        return full_text.strip()

    def _was_barged_in(self) -> bool:
        if not self.app.config.barge_in_enabled:
            return False
        if not self._audio or not self._audio._running:
            return False

        try:
            import sounddevice as sd  # noqa: F401

            if (
                not hasattr(self._audio, "_input_stream")
                or self._audio._input_stream is None
            ):
                return False
            data, _ = self._audio._input_stream.read(512)
            energy = np.abs(data).mean()
            if energy > self.app.config.barge_in_energy_threshold:
                logger.debug(f"Barge-in energy: {energy:.4f}")
                return True
        except Exception:
            pass
        return False

    # -- TTS playback --

    async def _speak(self, text: str) -> None:
        if not text.strip():
            return

        clean_text = text.replace("**", "").replace("*", "").replace("`", "")
        temp_audio_path: str | None = None

        try:
            logger.info(f"TTS: generating speech ({len(clean_text)} chars)...")
            temp_audio_path = await self._tts.synthesize(clean_text)

            if self.app.reachy and hasattr(self.app.reachy, "media"):
                try:
                    await self._play_on_reachy(temp_audio_path)
                except Exception as e:
                    logger.error(f"Reachy playback failed: {e}")
                    self._play_local_fallback(temp_audio_path)
            else:
                self._play_local_fallback(temp_audio_path)

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            logger.info(f"[TTS] {text}")
        finally:
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except FileNotFoundError:
                    pass

    async def _play_on_reachy(self, audio_path: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            temp_wav_path = wf.name

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ar", "16000", "-ac", "1", temp_wav_path,
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

            # Start wobbler if available
            if self._wobbler:
                self._wobbler.start()

            sample_rate = 16000
            chunk_size = 1600  # 100ms chunks
            chunk_duration = chunk_size / sample_rate

            for i in range(0, len(audio_float), chunk_size):
                chunk = audio_float[i : i + chunk_size]
                reachy.media.push_audio_sample(chunk)

                # Feed audio to wobbler for speech animation
                if self._wobbler:
                    self._wobbler.feed(chunk)

                await asyncio.sleep(chunk_duration * 0.9)

            if self._wobbler:
                self._wobbler.reset()

            reachy.set_target_antenna_joint_positions([0.0, 0.0])
            await asyncio.sleep(0.3)
            reachy.media.stop_playing()
        finally:
            try:
                os.unlink(temp_wav_path)
            except FileNotFoundError:
                pass

    def _play_local_fallback(self, audio_path: str) -> None:
        try:
            subprocess.run(["afplay", audio_path], capture_output=True)
        except FileNotFoundError:
            logger.warning("No local audio player available")
