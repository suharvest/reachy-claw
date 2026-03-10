"""Audio-driven head movement for natural speech animation.

Analyzes audio output in real-time and generates subtle head movements
that make the robot appear more expressive while speaking.
"""

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# (roll, pitch, yaw) offsets in degrees
SpeechOffsets = Tuple[float, float, float]


class HeadWobbler:
    """Generate audio-driven head movements for expressive speech."""

    def __init__(
        self,
        set_speech_offsets: Callable[[SpeechOffsets], None],
        sample_rate: int = 16000,
        update_rate: float = 30.0,
    ):
        self.set_speech_offsets = set_speech_offsets
        self.sample_rate = sample_rate
        self.update_period = 1.0 / update_rate

        # Movement parameters (degrees)
        self.roll_scale = 4.0
        self.pitch_scale = 3.0
        self.yaw_scale = 2.0
        self.smoothing = 0.3

        # State
        self._audio_buffer: deque[NDArray[np.float32]] = deque(maxlen=10)
        self._buffer_lock = threading.Lock()
        self._current_amplitude = 0.0
        self._current_offsets: SpeechOffsets = (0.0, 0.0, 0.0)

        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_feed_time = 0.0
        self._is_speaking = False

        # Decay
        self._decay_rate = 3.0
        self._speech_timeout = 0.3

    def start(self) -> None:
        """Start the wobbler thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.debug("HeadWobbler started")

    def stop(self) -> None:
        """Stop the wobbler thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.set_speech_offsets((0.0, 0.0, 0.0))
        logger.debug("HeadWobbler stopped")

    def reset(self) -> None:
        """Reset state (call when speech ends or is interrupted)."""
        with self._buffer_lock:
            self._audio_buffer.clear()
        self._current_amplitude = 0.0
        self._is_speaking = False
        self.set_speech_offsets((0.0, 0.0, 0.0))

    def feed(self, audio_float32: NDArray[np.float32]) -> None:
        """Feed float32 PCM audio data to the wobbler."""
        with self._buffer_lock:
            self._audio_buffer.append(audio_float32)
        self._last_feed_time = time.monotonic()
        self._is_speaking = True

    def _compute_amplitude(self) -> float:
        with self._buffer_lock:
            if not self._audio_buffer:
                return 0.0
            audio = np.concatenate(list(self._audio_buffer))
        rms = np.sqrt(np.mean(audio**2))
        return min(1.0, rms * 3.0)

    def _compute_offsets(self, amplitude: float, t: float) -> SpeechOffsets:
        if amplitude < 0.01:
            return (0.0, 0.0, 0.0)

        roll = amplitude * self.roll_scale * np.sin(t * 3.0)
        pitch = amplitude * self.pitch_scale * np.sin(t * 5.0 + 0.5)
        yaw = amplitude * self.yaw_scale * np.sin(t * 2.0)

        return (roll, pitch, yaw)

    def _run_loop(self) -> None:
        start_time = time.monotonic()

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            t = loop_start - start_time

            silence_duration = loop_start - self._last_feed_time

            if silence_duration > self._speech_timeout:
                self._current_amplitude *= np.exp(
                    -self._decay_rate * self.update_period
                )
                self._is_speaking = False
            else:
                raw_amplitude = self._compute_amplitude()
                self._current_amplitude = (
                    self.smoothing * raw_amplitude
                    + (1 - self.smoothing) * self._current_amplitude
                )

            offsets = self._compute_offsets(self._current_amplitude, t)

            new_offsets = tuple(
                self.smoothing * new + (1 - self.smoothing) * old
                for new, old in zip(offsets, self._current_offsets)
            )
            self._current_offsets = new_offsets

            self.set_speech_offsets(new_offsets)

            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, self.update_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
