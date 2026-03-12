"""Map emotions to Reachy Mini robot movements.

Translates emotion strings into physical robot expressions using
head poses, antenna movements, and recorded moves.

Antenna animations use parametric sine waves for organic, dynamic
movement (e.g. wagging when happy, drooping when sad).
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """Target head pose in degrees."""

    yaw: float = 0.0  # left/right rotation
    pitch: float = 0.0  # up/down tilt
    roll: float = 0.0  # side tilt
    duration: float = 0.8


@dataclass
class AntennaMotion:
    """Antenna target positions in degrees (static, one-shot)."""

    left: float = 0.0
    right: float = 0.0
    duration: float = 0.5


@dataclass
class AntennaAnimation:
    """Parametric antenna animation — continuous sine-wave oscillation.

    Each antenna: angle(t) = center + amplitude * sin(2π * freq * t + phase)
    phase_offset shifts the right antenna relative to left:
      0    = symmetric (both move together)
      π    = opposite (wagging like a dog tail)
      π/2  = wave-like
    """

    center: float = 0.0  # degrees — base position (positive = up)
    amplitude: float = 20.0  # degrees — swing range
    frequency: float = 2.0  # Hz
    phase_offset: float = math.pi  # radians between L/R
    duration: float = 2.0  # seconds
    description: str = ""


@dataclass
class RobotExpression:
    """A complete robot expression combining head and antenna movement."""

    head: Optional[HeadPose] = None
    antenna: Optional[AntennaMotion] = None
    antenna_anim: Optional[AntennaAnimation] = None  # dynamic animation (overrides antenna)
    description: str = ""


EMOTION_MAP: dict[str, list[RobotExpression]] = {
    "happy": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=5, roll=0, duration=0.6),
            antenna_anim=AntennaAnimation(
                center=25, amplitude=20, frequency=2.5,
                phase_offset=math.pi, duration=2.5,
            ),
            description="Happy wag — fast tail-like swing",
        ),
        RobotExpression(
            head=HeadPose(yaw=10, pitch=5, roll=5, duration=0.5),
            antenna_anim=AntennaAnimation(
                center=30, amplitude=15, frequency=2.0,
                phase_offset=math.pi, duration=2.0,
            ),
            description="Happy tilt with wag",
        ),
    ],
    "laugh": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=10, roll=0, duration=0.3),
            antenna_anim=AntennaAnimation(
                center=35, amplitude=25, frequency=3.5,
                phase_offset=math.pi, duration=2.0,
            ),
            description="Laughing — excited rapid wag",
        ),
    ],
    "excited": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=8, roll=0, duration=0.4),
            antenna_anim=AntennaAnimation(
                center=40, amplitude=25, frequency=3.0,
                phase_offset=math.pi, duration=3.0,
            ),
            description="Excited — high fast wag",
        ),
    ],
    "thinking": [
        RobotExpression(
            head=HeadPose(yaw=15, pitch=-5, roll=10, duration=1.0),
            antenna_anim=AntennaAnimation(
                center=5, amplitude=8, frequency=0.5,
                phase_offset=math.pi / 2, duration=3.0,
            ),
            description="Thinking — slow asymmetric sway",
        ),
    ],
    "confused": [
        RobotExpression(
            head=HeadPose(yaw=-10, pitch=0, roll=-15, duration=0.8),
            antenna_anim=AntennaAnimation(
                center=-5, amplitude=15, frequency=0.8,
                phase_offset=math.pi, duration=2.5,
            ),
            description="Confused — unsteady wobble",
        ),
    ],
    "curious": [
        RobotExpression(
            head=HeadPose(yaw=10, pitch=-8, roll=5, duration=0.7),
            antenna_anim=AntennaAnimation(
                center=20, amplitude=10, frequency=1.5,
                phase_offset=0, duration=2.0,
            ),
            description="Curious — perked up, gentle sync bounce",
        ),
    ],
    "sad": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=-15, roll=0, duration=1.2),
            antenna_anim=AntennaAnimation(
                center=-25, amplitude=5, frequency=0.3,
                phase_offset=0, duration=3.0,
            ),
            description="Sad — drooping, barely moving",
        ),
    ],
    "angry": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=-5, roll=0, duration=0.3),
            antenna_anim=AntennaAnimation(
                center=-10, amplitude=12, frequency=4.0,
                phase_offset=0, duration=1.5,
            ),
            description="Angry — tense fast vibration",
        ),
    ],
    "surprised": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=15, roll=0, duration=0.3),
            antenna_anim=AntennaAnimation(
                center=40, amplitude=20, frequency=3.0,
                phase_offset=math.pi, duration=2.0,
            ),
            description="Surprised — snap up then wag",
        ),
    ],
    "fear": [
        RobotExpression(
            head=HeadPose(yaw=-5, pitch=-10, roll=-5, duration=0.4),
            antenna_anim=AntennaAnimation(
                center=-15, amplitude=10, frequency=5.0,
                phase_offset=0, duration=2.0,
            ),
            description="Fear — trembling shiver",
        ),
    ],
    "neutral": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=0, roll=0, duration=0.8),
            antenna=AntennaMotion(left=0, right=0, duration=0.6),
            description="Neutral - return to center",
        ),
    ],
    "listening": [
        RobotExpression(
            head=HeadPose(yaw=5, pitch=-3, roll=3, duration=0.6),
            antenna_anim=AntennaAnimation(
                center=15, amplitude=5, frequency=0.8,
                phase_offset=math.pi / 3, duration=3.0,
            ),
            description="Listening — attentive gentle sway",
        ),
    ],
    "agreeing": [
        RobotExpression(
            head=HeadPose(yaw=0, pitch=8, roll=0, duration=0.4),
            antenna_anim=AntennaAnimation(
                center=15, amplitude=10, frequency=2.0,
                phase_offset=0, duration=1.5,
            ),
            description="Agreeing — nod with happy bounce",
        ),
    ],
    "disagreeing": [
        RobotExpression(
            head=HeadPose(yaw=15, pitch=0, roll=0, duration=0.3),
            antenna_anim=AntennaAnimation(
                center=-5, amplitude=12, frequency=2.5,
                phase_offset=math.pi, duration=1.5,
            ),
            description="Disagreeing — shake with agitated wag",
        ),
    ],
}


class EmotionMapper:
    """Maps emotion strings to robot movements via a queue."""

    def __init__(self, intensity: float = 0.7):
        self._intensity = max(0.0, min(1.0, intensity))
        self._move_queue: Queue[RobotExpression] = Queue(maxsize=20)
        self._last_emotion = "neutral"
        self._last_expression_time = 0.0

    @property
    def move_queue(self) -> Queue:
        return self._move_queue

    def map_emotion(self, emotion: str) -> Optional[RobotExpression]:
        """Convert an emotion string to a RobotExpression."""
        normalized = emotion.lower().strip()
        expressions = EMOTION_MAP.get(normalized)
        if not expressions:
            logger.debug(f"No mapping for emotion: {emotion}")
            return None

        template = random.choice(expressions)
        # Clone template to avoid mutating global EMOTION_MAP entries.
        expr = RobotExpression(
            head=HeadPose(
                yaw=template.head.yaw,
                pitch=template.head.pitch,
                roll=template.head.roll,
                duration=template.head.duration,
            )
            if template.head
            else None,
            antenna=AntennaMotion(
                left=template.antenna.left,
                right=template.antenna.right,
                duration=template.antenna.duration,
            )
            if template.antenna
            else None,
            antenna_anim=AntennaAnimation(
                center=template.antenna_anim.center,
                amplitude=template.antenna_anim.amplitude,
                frequency=template.antenna_anim.frequency,
                phase_offset=template.antenna_anim.phase_offset,
                duration=template.antenna_anim.duration,
                description=template.antenna_anim.description,
            )
            if template.antenna_anim
            else None,
            description=template.description,
        )

        # Apply intensity scaling
        if expr.head:
            expr.head.yaw *= self._intensity
            expr.head.pitch *= self._intensity
            expr.head.roll *= self._intensity
        if expr.antenna:
            expr.antenna.left *= self._intensity
            expr.antenna.right *= self._intensity
        if expr.antenna_anim:
            expr.antenna_anim.center *= self._intensity
            expr.antenna_anim.amplitude *= self._intensity

        return expr

    def queue_emotion(self, emotion: str) -> None:
        """Map an emotion and add it to the motion queue.

        Callers are responsible for rate-limiting (e.g. VisionClient
        cooldown/sustain, LLM tag dedup). The mapper just maps and queues.
        """
        expr = self.map_emotion(emotion)
        if expr:
            try:
                self._move_queue.put_nowait(expr)
                self._last_emotion = emotion
                self._last_expression_time = time.monotonic()
                logger.info(f"Queued expression: {expr.description}")
            except Exception:
                logger.warning("Motion queue full, dropping expression")

    def get_next_expression(self) -> Optional[RobotExpression]:
        """Get the next queued expression, or None."""
        try:
            return self._move_queue.get_nowait()
        except Empty:
            return None

    def get_idle_expression(self) -> RobotExpression:
        """Generate a subtle idle animation."""
        yaw = random.uniform(-5, 5)
        pitch = random.uniform(-3, 3)
        roll = random.uniform(-3, 3)
        return RobotExpression(
            head=HeadPose(yaw=yaw, pitch=pitch, roll=roll, duration=2.0),
            antenna_anim=AntennaAnimation(
                center=random.uniform(-5, 5),
                amplitude=random.uniform(3, 8),
                frequency=random.uniform(0.3, 0.7),
                phase_offset=random.uniform(0, math.pi),
                duration=2.5,
            ),
            description="Idle breathing",
        )
