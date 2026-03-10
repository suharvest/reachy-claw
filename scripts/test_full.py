#!/usr/bin/env python3
"""Full pipeline test with Reachy mockup-sim, STT, and TTS.

This runs the full conversation loop:
  mic → STT → desktop-robot WS → streaming AI → TTS → Reachy speaker

With mockup-sim, the robot movements are instant (no physics) and
audio plays through your local speakers instead of the robot.

Usage:
    # Minimal: macOS say TTS (no API keys needed), local whisper STT
    uv run python scripts/test_full.py --tts macos-say --stt faster-whisper

    # With ElevenLabs TTS (needs ELEVENLABS_API_KEY)
    ELEVENLABS_API_KEY=sk-... uv run python scripts/test_full.py --tts elevenlabs

    # No TTS (text only), for testing just the pipeline
    uv run python scripts/test_full.py --tts none --stt faster-whisper

    # With specific audio device
    uv run python scripts/test_full.py --tts macos-say --audio-device "MacBook Pro Microphone"

Prerequisites:
    1. OpenClaw gateway running with desktop-robot enabled:
       cd /path/to/openclaw
       openclaw config set channels.desktopRobot.enabled true
       openclaw config set channels.desktopRobot.auth.allowAnonymous true
       openclaw gateway run

    2. Reachy Mini SDK installed (for mockup-sim):
       pip install -e /Users/harvest/project/reachy_mini/reachy_mini
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

sys.path.insert(0, "src")

from reachy_claw.config import Config, load_config
from reachy_claw.interface import ReachyInterface
from reachy_claw.main import parse_args, create_config, setup_logging


async def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    config = create_config(args)

    # Log what we're doing
    logging.info("=" * 60)
    logging.info("Full pipeline test")
    logging.info(f"  Server:  {config.desktop_robot_url}")
    logging.info(f"  STT:     {config.stt_backend} ({config.whisper_model})")
    logging.info(f"  TTS:     {config.tts_backend}")
    logging.info(f"  Barge-in: {'on' if config.barge_in_enabled else 'off'}")
    logging.info("=" * 60)

    interface = ReachyInterface(config)

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()

    def on_signal():
        logging.info("Shutting down...")
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal)

    try:
        run_task = asyncio.create_task(interface.run())
        stop_task = asyncio.create_task(shutdown.wait())

        done, pending = await asyncio.wait(
            [run_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
    finally:
        await interface.stop()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
