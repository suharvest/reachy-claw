"""Shared test fixtures: mock Reachy Mini robot and ClawdApp."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, PropertyMock

import numpy as np
import pytest

from clawd_reachy_mini.app import ClawdApp
from clawd_reachy_mini.config import Config


def _make_mock_reachy():
    """Create a mock ReachyMini with all methods the codebase uses."""
    reachy = MagicMock(name="ReachyMini")

    # Head & antenna
    reachy.goto_target = MagicMock()
    reachy.set_target_head_pose = MagicMock()
    reachy.set_target_antenna_joint_positions = MagicMock()
    reachy.set_target = MagicMock()
    reachy.wake_up = MagicMock()
    reachy.goto_sleep = MagicMock()
    reachy.get_current_head_pose = MagicMock(return_value=np.eye(4))
    reachy.get_present_antenna_joint_positions = MagicMock(return_value=[0.0, 0.0])

    # Media manager
    media = MagicMock(name="MediaManager")
    media.start_playing = MagicMock()
    media.push_audio_sample = MagicMock()
    media.stop_playing = MagicMock()
    media.start_recording = MagicMock()
    media.get_audio_sample = MagicMock(return_value=None)
    media.stop_recording = MagicMock()
    media.get_input_audio_samplerate = MagicMock(return_value=16000)
    media.get_output_audio_samplerate = MagicMock(return_value=24000)
    reachy.media = media

    # Context manager
    reachy.__enter__ = MagicMock(return_value=reachy)
    reachy.__exit__ = MagicMock(return_value=False)

    return reachy


@pytest.fixture
def mock_reachy():
    """A mock ReachyMini instance."""
    return _make_mock_reachy()


@pytest.fixture
def config():
    """Default test config."""
    return Config(
        standalone_mode=True,
        idle_animations=False,
        play_emotions=True,
        enable_face_tracker=False,
        enable_motion=True,
        tts_backend="none",
        stt_backend="whisper",
    )


@pytest.fixture
def app(config, mock_reachy):
    """ClawdApp with a mock robot attached."""
    a = ClawdApp(config)
    a.reachy = mock_reachy
    return a
