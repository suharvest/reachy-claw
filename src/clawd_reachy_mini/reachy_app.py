"""Reachy Mini application adapter.

Bridges our async ClawdApp into the ReachyMiniApp interface so it can
be discovered and managed by the Reachy Mini daemon, while also
supporting direct local execution.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from reachy_mini import ReachyMini
from reachy_mini.apps.app import ReachyMiniApp

from clawd_reachy_mini.app import ClawdApp
from clawd_reachy_mini.config import load_config

logger = logging.getLogger(__name__)


class ClawdReachyMiniApp(ReachyMiniApp):
    """Reachy Mini app wrapper around ClawdApp."""

    # No custom settings UI for now
    custom_app_url: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Entry point called by the Reachy Mini daemon."""
        config = load_config()
        app = ClawdApp(config)

        # Use the daemon-provided ReachyMini instead of connecting ourselves
        app.reachy = reachy_mini

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_async(app, stop_event))
        finally:
            loop.close()

    async def _run_async(self, app: ClawdApp, stop_event: threading.Event) -> None:
        """Run ClawdApp plugins until stop_event is set."""
        from clawd_reachy_mini.plugins.conversation_plugin import ConversationPlugin
        from clawd_reachy_mini.plugins.motion_plugin import MotionPlugin

        if app.config.enable_motion:
            app.register(MotionPlugin(app))

        if app.config.enable_face_tracker:
            try:
                from clawd_reachy_mini.plugins.face_tracker_plugin import (
                    FaceTrackerPlugin,
                )

                app.register(FaceTrackerPlugin(app))
            except ImportError:
                logger.info("Face tracker deps not available, skipping")

        app.register(ConversationPlugin(app))

        # Run plugins and watch for daemon stop signal
        run_task = asyncio.create_task(app.run())
        stop_task = asyncio.create_task(
            asyncio.to_thread(stop_event.wait)
        )

        done, pending = await asyncio.wait(
            [run_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await app.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = ClawdReachyMiniApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
