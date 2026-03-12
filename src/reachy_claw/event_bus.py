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
                        loop.create_task(cb(data))
                    else:
                        loop.run_until_complete(cb(data))
                else:
                    cb(data)
            except Exception as e:
                logger.warning("EventBus callback error on %s: %s", event, e)
