"""Dance routines for Reachy Mini.

Each dance is a sequence of steps with head pose + antenna positions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DanceStep:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    antenna_left: float = 0.0
    antenna_right: float = 0.0
    duration: float = 0.4


@dataclass
class DanceRoutine:
    name: str
    description: str
    steps: list[DanceStep]


DANCE_ROUTINES: dict[str, DanceRoutine] = {
    "nod": DanceRoutine(
        name="nod",
        description="Simple nodding dance",
        steps=[
            DanceStep(pitch=12, antenna_left=30, antenna_right=30, duration=0.3),
            DanceStep(pitch=-5, antenna_left=-10, antenna_right=-10, duration=0.3),
            DanceStep(pitch=12, antenna_left=30, antenna_right=30, duration=0.3),
            DanceStep(pitch=-5, antenna_left=-10, antenna_right=-10, duration=0.3),
            DanceStep(duration=0.4),
        ],
    ),
    "wiggle": DanceRoutine(
        name="wiggle",
        description="Side-to-side wiggle",
        steps=[
            DanceStep(yaw=20, roll=10, antenna_left=40, antenna_right=-20, duration=0.3),
            DanceStep(yaw=-20, roll=-10, antenna_left=-20, antenna_right=40, duration=0.3),
            DanceStep(yaw=20, roll=10, antenna_left=40, antenna_right=-20, duration=0.3),
            DanceStep(yaw=-20, roll=-10, antenna_left=-20, antenna_right=40, duration=0.3),
            DanceStep(duration=0.4),
        ],
    ),
    "celebrate": DanceRoutine(
        name="celebrate",
        description="Celebratory dance with big movements",
        steps=[
            DanceStep(pitch=15, antenna_left=50, antenna_right=50, duration=0.3),
            DanceStep(yaw=15, roll=10, antenna_left=30, antenna_right=-30, duration=0.3),
            DanceStep(pitch=15, antenna_left=50, antenna_right=50, duration=0.3),
            DanceStep(yaw=-15, roll=-10, antenna_left=-30, antenna_right=30, duration=0.3),
            DanceStep(pitch=15, antenna_left=50, antenna_right=50, duration=0.25),
            DanceStep(duration=0.5),
        ],
    ),
    "curious_look": DanceRoutine(
        name="curious_look",
        description="Look around curiously",
        steps=[
            DanceStep(yaw=25, pitch=-5, roll=5, antenna_left=20, antenna_right=20, duration=0.5),
            DanceStep(yaw=-25, pitch=-5, roll=-5, antenna_left=20, antenna_right=20, duration=0.6),
            DanceStep(pitch=-10, antenna_left=30, antenna_right=30, duration=0.5),
            DanceStep(duration=0.5),
        ],
    ),
    "lobster": DanceRoutine(
        name="lobster",
        description="Lobster claw antenna snap",
        steps=[
            DanceStep(antenna_left=40, antenna_right=40, duration=0.2),
            DanceStep(antenna_left=-40, antenna_right=-40, duration=0.2),
            DanceStep(antenna_left=40, antenna_right=40, duration=0.2),
            DanceStep(antenna_left=-40, antenna_right=-40, duration=0.2),
            DanceStep(duration=0.3),
        ],
    ),
}

AVAILABLE_DANCES = sorted(DANCE_ROUTINES.keys())
