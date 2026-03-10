"""OpenClaw skill for Reachy Mini robot integration."""

from reachy_claw.bridge import ReachyBridge
from reachy_claw.tools import (
    reachy_connect,
    reachy_disconnect,
    reachy_move_head,
    reachy_move_antennas,
    reachy_play_emotion,
    reachy_dance,
    reachy_capture_image,
    reachy_say,
    reachy_status,
)

__version__ = "0.1.0"

__all__ = [
    "ReachyBridge",
    "reachy_connect",
    "reachy_disconnect",
    "reachy_move_head",
    "reachy_move_antennas",
    "reachy_play_emotion",
    "reachy_dance",
    "reachy_capture_image",
    "reachy_say",
    "reachy_status",
]
