"""Lightweight async-safe pub/sub event bus."""

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventBus:
    """Simple publish/subscribe event bus with async support."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._pending_tasks: set[asyncio.Task] = set()

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Cleanup completed task; log unhandled exceptions (except cancellation)."""
        self._pending_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception(
                "EventBus async callback raised unhandled exception",
                exc_info=exc,
            )

    def _track(self, task: asyncio.Task) -> None:
        self._pending_tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def subscribe(self, event: str, callback: Callable) -> None:
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable) -> None:
        try:
            self._subscribers[event].remove(callback)
        except ValueError:
            pass

    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event. Auto-awaits async callbacks via the running loop."""
        for cb in self._subscribers.get(event, []):
            try:
                if inspect.iscoroutinefunction(cb):
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        self._track(loop.create_task(cb(data)))
                    else:
                        loop.run_until_complete(cb(data))
                else:
                    cb(data)
            except Exception as e:
                logger.warning("EventBus callback error on %s: %s", event, e)

    def emit_sync(self, event: str, data: Any = None) -> None:
        """Thread-safe emit: schedule async callbacks on the running loop."""
        for cb in self._subscribers.get(event, []):
            try:
                if inspect.iscoroutinefunction(cb):
                    try:
                        loop = asyncio.get_running_loop()
                        self._track(loop.create_task(cb(data)))
                    except RuntimeError:
                        # Called from non-async thread — schedule on main loop
                        try:
                            loop = asyncio.get_event_loop()
                            asyncio.run_coroutine_threadsafe(cb(data), loop)
                        except Exception:
                            pass
                else:
                    cb(data)
            except Exception as e:
                logger.warning("EventBus emit_sync error on %s: %s", event, e)
