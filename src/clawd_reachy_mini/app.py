"""Central application context and plugin orchestrator.

ClawdApp holds shared state (config, robot, head targets, emotions)
and manages the lifecycle of registered plugins.
"""

import asyncio
import logging
from typing import List

from .config import Config
from .motion.emotion_mapper import EmotionMapper
from .motion.head_target import HeadTargetBus
from .plugin import Plugin

logger = logging.getLogger(__name__)


class ClawdApp:
    """Central application that orchestrates all plugins."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reachy = None  # ReachyMini instance, set by connect_robot() or externally
        self._owns_reachy = False  # True if we created the connection ourselves
        self.head_targets = HeadTargetBus()
        self.emotions = EmotionMapper(
            intensity=config.motion_emotion_intensity,
        )

        # Shared flags for inter-plugin coordination
        self.is_speaking = False
        self.running = False

        self._plugins: List[Plugin] = []

    def connect_robot(self) -> None:
        """Connect to the Reachy Mini robot."""
        try:
            from reachy_mini import ReachyMini

            kwargs = {}
            if self.config.reachy_connection_mode != "auto":
                kwargs["connection_mode"] = self.config.reachy_connection_mode
            if self.config.reachy_media_backend != "default":
                kwargs["media_backend"] = self.config.reachy_media_backend

            self.reachy = ReachyMini(**kwargs)
            self.reachy.__enter__()
            self._owns_reachy = True
            logger.info("Connected to Reachy Mini")

        except ImportError:
            logger.warning("reachy-mini not installed, running without robot")
            self.reachy = None
        except Exception as e:
            logger.error(f"Failed to connect to Reachy Mini: {e}")
            self.reachy = None

    def register(self, plugin: Plugin) -> bool:
        """Register a plugin. Calls setup() and skips if it returns False."""
        try:
            if not plugin.setup():
                logger.info(f"Plugin '{plugin.name}' skipped (setup returned False)")
                return False
        except Exception as e:
            logger.warning(f"Plugin '{plugin.name}' setup failed: {e}")
            return False

        self._plugins.append(plugin)
        logger.info(f"Plugin '{plugin.name}' registered")
        return True

    async def run(self) -> None:
        """Start all registered plugins concurrently."""
        if not self._plugins:
            logger.warning("No plugins registered, nothing to run")
            return

        self.running = True
        plugin_names = [p.name for p in self._plugins]
        logger.info(f"Starting plugins: {', '.join(plugin_names)}")

        tasks = []
        for plugin in self._plugins:
            plugin._running = True
            tasks.append(asyncio.create_task(plugin.start(), name=plugin.name))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Plugins cancelled")
        except Exception as e:
            logger.error(f"Plugin error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Stop all plugins in reverse registration order."""
        logger.info("Shutting down plugins...")
        self.running = False
        for plugin in reversed(self._plugins):
            try:
                await plugin.stop()
                logger.debug(f"Plugin '{plugin.name}' stopped")
            except Exception as e:
                logger.warning(f"Error stopping plugin '{plugin.name}': {e}")

        if self.reachy:
            try:
                from reachy_mini.utils import create_head_pose

                self.reachy.goto_target(
                    head=create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
                    duration=0.5,
                )
                self.reachy.set_target_antenna_joint_positions([0.0, 0.0])
            except Exception:
                pass
            # Only close the connection if we created it ourselves
            if self._owns_reachy:
                try:
                    self.reachy.__exit__(None, None, None)
                except Exception:
                    pass
            self.reachy = None

        self._plugins.clear()
        logger.info("Shutdown complete")
