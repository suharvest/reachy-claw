"""Plugin base class for reachy-claw.

All feature modules (conversation, motion, face tracking, etc.)
extend this base class to plug into the ReachyClawApp lifecycle.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import ReachyClawApp

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class for plugins.

    Lifecycle:
      1. __init__(app) -- store reference to shared app context
      2. setup() -- check hardware / deps, return False to gracefully skip
      3. start() -- main async coroutine, gathered with other plugins
      4. stop() -- signal shutdown (called in reverse registration order)
    """

    name: str = "unnamed"

    def __init__(self, app: "ReachyClawApp") -> None:
        self.app = app
        self._running = False

    def setup(self) -> bool:
        """Check hardware availability and prerequisites.

        Returns True if the plugin can run, False to skip gracefully.
        Called synchronously before the async event loop starts.
        """
        return True

    @abstractmethod
    async def start(self) -> None:
        """Main async entry point. Runs until stop() is called."""
        ...

    async def stop(self) -> None:
        """Signal the plugin to shut down gracefully."""
        self._running = False
