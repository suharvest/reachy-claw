"""Tests for the HeadWobbler (audio-reactive head movement)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from clawd_reachy_mini.motion.head_wobbler import HeadWobbler


class TestHeadWobblerLifecycle:
    def test_start_stop(self):
        offsets = []
        wobbler = HeadWobbler(set_speech_offsets=lambda o: offsets.append(o))
        wobbler.start()
        assert wobbler._thread is not None
        assert wobbler._thread.is_alive()

        wobbler.stop()
        assert wobbler._thread is None

    def test_double_start_no_crash(self):
        wobbler = HeadWobbler(set_speech_offsets=lambda o: None)
        wobbler.start()
        wobbler.start()  # Should not crash
        wobbler.stop()

    def test_stop_resets_to_zero(self):
        last_offsets = [None]
        wobbler = HeadWobbler(set_speech_offsets=lambda o: last_offsets.__setitem__(0, o))
        wobbler.start()
        wobbler.stop()
        # After stop, offsets should be (0, 0, 0)
        assert last_offsets[0] == (0.0, 0.0, 0.0)


class TestAudioFeed:
    def test_feed_generates_movement(self):
        offsets_log = []
        wobbler = HeadWobbler(
            set_speech_offsets=lambda o: offsets_log.append(o),
            update_rate=60.0,
        )
        wobbler.start()

        # Feed loud audio
        loud_audio = np.random.randn(1600).astype(np.float32) * 0.5
        for _ in range(5):
            wobbler.feed(loud_audio)
            time.sleep(0.02)

        time.sleep(0.15)
        wobbler.stop()

        # Should have produced non-zero offsets
        non_zero = [o for o in offsets_log if any(abs(v) > 0.01 for v in o)]
        assert len(non_zero) > 0, "Expected non-zero offsets from loud audio"

    def test_silence_decays_to_zero(self):
        offsets_log = []
        wobbler = HeadWobbler(
            set_speech_offsets=lambda o: offsets_log.append(o),
            update_rate=60.0,
        )
        wobbler.start()

        # Feed loud audio then stop
        loud = np.random.randn(1600).astype(np.float32) * 0.5
        wobbler.feed(loud)
        time.sleep(0.1)

        # Wait for decay
        time.sleep(0.5)
        wobbler.stop()

        # Last offsets should be near zero
        if offsets_log:
            last = offsets_log[-1]
            assert all(abs(v) < 0.5 for v in last), f"Expected decay to near-zero: {last}"

    def test_reset_clears_state(self):
        last_offsets = [None]
        wobbler = HeadWobbler(set_speech_offsets=lambda o: last_offsets.__setitem__(0, o))
        wobbler.start()

        loud = np.random.randn(1600).astype(np.float32) * 0.5
        wobbler.feed(loud)
        time.sleep(0.05)

        wobbler.reset()
        assert last_offsets[0] == (0.0, 0.0, 0.0)
        assert wobbler._current_amplitude == 0.0
        assert not wobbler._is_speaking

        wobbler.stop()


class TestComputeOffsets:
    def test_zero_amplitude_returns_zeros(self):
        wobbler = HeadWobbler(set_speech_offsets=lambda o: None)
        offsets = wobbler._compute_offsets(0.0, 1.0)
        assert offsets == (0.0, 0.0, 0.0)

    def test_nonzero_amplitude_returns_nonzero(self):
        wobbler = HeadWobbler(set_speech_offsets=lambda o: None)
        offsets = wobbler._compute_offsets(0.5, 1.0)
        assert any(v != 0.0 for v in offsets)

    def test_amplitude_scales_output(self):
        wobbler = HeadWobbler(set_speech_offsets=lambda o: None)
        t = 1.0
        low = wobbler._compute_offsets(0.1, t)
        high = wobbler._compute_offsets(0.9, t)
        # Higher amplitude should produce larger absolute values
        assert sum(abs(v) for v in high) > sum(abs(v) for v in low)
