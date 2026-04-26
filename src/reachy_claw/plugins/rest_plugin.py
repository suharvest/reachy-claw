"""RestPlugin — daily rest window orchestrator.

Polls app.config every 30s; when "now" enters the rest window, emits
`rest_start` (other plugins self-pause via on_rest_start) and runs registered
HousekeepingTasks. When the window ends, emits `rest_end`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, time as dtime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..plugin import Plugin

if TYPE_CHECKING:
    from .housekeeping_tasks import HousekeepingTask

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 30
HARD_OVERRUN_CAP_S = 600  # housekeeping may run up to 10 min past end_time


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(hour=int(h) % 24, minute=int(m))  # "24:00" → 00:00 next-day handled separately


def _is_in_window(now: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    """Whether `now` (timezone-aware) falls in [start, end) on the local clock.

    Special-cases:
      - end == "24:00" means end-of-day: the window is [start, 24:00) i.e.
        any time at or after start, on this calendar day, in the local tz.
      - Otherwise, if end <= start, the window wraps midnight.
    """
    start = _parse_hhmm(start_hhmm)
    cur = now.timetz().replace(tzinfo=None)

    if end_hhmm == "24:00":
        return cur >= start

    end = _parse_hhmm(end_hhmm)
    if start <= end:
        return start <= cur < end
    # wraparound: in window if cur >= start OR cur < end
    return cur >= start or cur < end


class RestPlugin(Plugin):
    name = "rest"

    def __init__(self, app) -> None:
        super().__init__(app)
        self._resting = False
        self._tasks: list[HousekeepingTask] = []
        self._zmq_ctx = None
        self._ctrl_pub = None

    def register_task(self, task: "HousekeepingTask") -> None:
        self._tasks.append(task)

    def _should_rest_now(self, now: datetime) -> bool:
        if not getattr(self.app.config, "rest_enabled", True):
            return False
        return _is_in_window(
            now,
            self.app.config.rest_window_start,
            self.app.config.rest_window_end,
        )

    def _now(self) -> datetime:
        try:
            tz = ZoneInfo(self.app.config.rest_timezone)
        except ZoneInfoNotFoundError:
            logger.warning(
                "Invalid rest_timezone %r; falling back to UTC",
                self.app.config.rest_timezone,
            )
            tz = ZoneInfo("UTC")
        return datetime.now(tz=tz)

    def _ensure_ctrl_pub(self):
        """Lazy-init the ZMQ PUB socket used to signal remote vision containers."""
        if self._ctrl_pub is not None:
            return
        try:
            import zmq
        except ImportError:
            logger.info("zmq not available; remote vision rest control disabled")
            return
        port = getattr(self.app.config, "rest_control_port", 18791)
        self._zmq_ctx = zmq.Context.instance()
        self._ctrl_pub = self._zmq_ctx.socket(zmq.PUB)
        self._ctrl_pub.bind(f"tcp://0.0.0.0:{port}")
        logger.info("Rest control PUB bound on tcp://0.0.0.0:%d", port)

    def _publish_ctrl(self, cmd: str) -> None:
        if self._ctrl_pub is None:
            return
        try:
            self._ctrl_pub.send_json({"cmd": cmd})
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to publish ctrl %s: %s", cmd, e)

    async def _enter_rest(self) -> None:
        if self._resting:
            return
        self._resting = True
        logger.info("Entering rest window")
        self._ensure_ctrl_pub()
        self._publish_ctrl("pause")
        self.app.events.emit("rest_start", {"started_at": int(time.time())})
        # Run housekeeping tasks sequentially in the background.
        asyncio.create_task(self._run_housekeeping())

    async def _exit_rest(self) -> None:
        if not self._resting:
            return
        self._resting = False
        logger.info("Exiting rest window")
        self._publish_ctrl("resume")
        self.app.events.emit("rest_end", {"ended_at": int(time.time())})

    async def _run_housekeeping(self) -> None:
        for task in self._tasks:
            self.app.events.emit("housekeeping_task_start", {"name": task.name})
            ok = True
            error = None
            try:
                await asyncio.wait_for(task.run(self.app), timeout=HARD_OVERRUN_CAP_S)
            except asyncio.TimeoutError:
                ok = False
                error = "timed out"
                logger.warning("Housekeeping task %r timed out", task.name)
            except Exception as e:  # noqa: BLE001
                ok = False
                error = str(e)
                logger.warning("Housekeeping task %r failed: %s", task.name, e)
            self.app.events.emit(
                "housekeeping_task_end", {"name": task.name, "ok": ok, "error": error}
            )

    async def start(self) -> None:
        self._running = True
        while self._running:
            should = self._should_rest_now(self._now())
            if should and not self._resting:
                await self._enter_rest()
            elif not should and self._resting:
                await self._exit_rest()
            elif self._resting:
                # Re-emit pause every tick. Defends against the ZMQ slow-joiner
                # problem: if a vision container restarts mid-rest, its newly
                # connected SUB will get the next heartbeat within POLL_INTERVAL_S
                # rather than miss the entire rest window.
                self._publish_ctrl("pause")
            try:
                await asyncio.sleep(POLL_INTERVAL_S)
            except asyncio.CancelledError:
                return
