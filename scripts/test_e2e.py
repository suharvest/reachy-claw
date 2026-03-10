#!/usr/bin/env python3
"""End-to-end test: connect to desktop-robot WS, send a message, print streaming response.

Usage:
    # With default desktop-robot endpoint (ws://127.0.0.1:18790/desktop-robot)
    uv run python scripts/test_e2e.py

    # With custom endpoint
    uv run python scripts/test_e2e.py --host 127.0.0.1 --port 18790 --token my-secret

    # Interactive mode (type messages, see streaming responses)
    uv run python scripts/test_e2e.py --interactive

    # Interactive mode with TTS (type messages, hear AI speak)
    uv run python scripts/test_e2e.py --interactive --tts macos-say

    # Interactive with ElevenLabs TTS
    ELEVENLABS_API_KEY=sk-... uv run python scripts/test_e2e.py --interactive --tts elevenlabs
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys

# Add src to path for local dev
sys.path.insert(0, "src")

from reachy_claw.config import Config
from reachy_claw.gateway import DesktopRobotClient
from reachy_claw.tts import TTSBackend, create_tts_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def _play_audio(path: str) -> None:
    """Play an audio file using the best available local player."""
    if sys.platform == "darwin":
        cmd = ["afplay", path]
    elif sys.platform == "linux":
        cmd = ["aplay", path]
    else:
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()

    # Clean up temp file
    try:
        os.unlink(path)
    except OSError:
        pass


async def run_single_message(client: DesktopRobotClient, text: str) -> None:
    """Send a single message and print streaming response."""
    received_chunks: list[str] = []

    async def on_start(run_id: str) -> None:
        logger.info(f"--- stream start [{run_id[:8]}] ---")

    async def on_delta(text: str, run_id: str) -> None:
        received_chunks.append(text)
        print(text, end="", flush=True)

    async def on_end(full_text: str, run_id: str) -> None:
        print()  # newline
        logger.info(f"--- stream end [{run_id[:8]}] ({len(full_text)} chars) ---")

    async def on_abort(reason: str, run_id: str) -> None:
        print()
        logger.warning(f"--- stream aborted: {reason} ---")

    async def on_tool_start(tool: str, run_id: str) -> None:
        logger.info(f"  [tool] {tool} started")

    async def on_tool_end(tool: str, run_id: str) -> None:
        logger.info(f"  [tool] {tool} ended")

    async def on_task_spawned(label: str, task_run_id: str) -> None:
        logger.info(f"  [task] spawned: {label}")

    async def on_task_completed(summary: str, task_run_id: str) -> None:
        logger.info(f"  [task] completed: {summary[:100]}")

    client.callbacks.on_stream_start = on_start
    client.callbacks.on_stream_delta = on_delta
    client.callbacks.on_stream_end = on_end
    client.callbacks.on_stream_abort = on_abort
    client.callbacks.on_tool_start = on_tool_start
    client.callbacks.on_tool_end = on_tool_end
    client.callbacks.on_task_spawned = on_task_spawned
    client.callbacks.on_task_completed = on_task_completed

    logger.info(f'Sending: "{text}"')
    response = await client.send_message(text)
    logger.info(f"Full response: {response[:200]}...")


async def run_interactive(
    client: DesktopRobotClient, tts: TTSBackend | None = None
) -> None:
    """Interactive mode: type messages, see streaming responses, optionally hear TTS."""

    # Accumulate full response for TTS
    response_buffer: list[str] = []

    async def on_delta(text: str, run_id: str) -> None:
        response_buffer.append(text)
        print(text, end="", flush=True)

    async def on_end(full_text: str, run_id: str) -> None:
        print("\n")

    async def on_abort(reason: str, run_id: str) -> None:
        print(f"\n[aborted: {reason}]\n")

    async def on_tool_start(tool: str, run_id: str) -> None:
        print(f"\n  [tool: {tool}] ", end="", flush=True)

    async def on_tool_end(tool: str, run_id: str) -> None:
        pass

    client.callbacks.on_stream_delta = on_delta
    client.callbacks.on_stream_end = on_end
    client.callbacks.on_stream_abort = on_abort
    client.callbacks.on_tool_start = on_tool_start
    client.callbacks.on_tool_end = on_tool_end

    mode = "chat + TTS" if tts else "chat (text only)"
    print(f"Interactive mode ({mode}). Type a message and press Enter. Ctrl+C to quit.\n")

    while True:
        try:
            text = await asyncio.to_thread(input, "You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text.strip():
            continue

        if text.strip().lower() in ("quit", "exit"):
            break

        if text.strip() == "/interrupt":
            await client.send_interrupt()
            print("[interrupt sent]")
            continue

        response_buffer.clear()
        print("AI: ", end="", flush=True)
        await client.send_message(text)

        # Speak the response if TTS is enabled
        if tts:
            full_response = "".join(response_buffer).strip()
            if full_response:
                try:
                    audio_path = await tts.synthesize(full_response)
                    await _play_audio(audio_path)
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test desktop-robot WebSocket client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18790)
    parser.add_argument("--path", default="/desktop-robot")
    parser.add_argument("--token", default=None)
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--message", default="Hello! Tell me a short joke.", help="Message to send (non-interactive)")
    parser.add_argument(
        "--tts",
        choices=["elevenlabs", "macos-say", "piper", "none"],
        default=None,
        help="TTS backend for speaking AI responses (interactive mode)",
    )
    parser.add_argument("--tts-voice", default=None, help="TTS voice ID")
    parser.add_argument("--tts-model", default=None, help="TTS model path (piper)")
    args = parser.parse_args()

    config = Config(
        gateway_host=args.host,
        gateway_port=args.port,
        gateway_path=args.path,
        gateway_token=args.token,
    )

    # Create TTS backend if requested
    tts: TTSBackend | None = None
    if args.tts:
        tts = create_tts_backend(args.tts, voice=args.tts_voice, model=args.tts_model)

    client = DesktopRobotClient(config)

    try:
        await client.connect()
        logger.info(f"Connected! Session: {client.session_id}")

        if args.interactive:
            await run_interactive(client, tts=tts)
        else:
            await run_single_message(client, args.message)

    except ConnectionRefusedError:
        logger.error(
            f"Connection refused at {config.desktop_robot_url}\n"
            "Make sure the OpenClaw gateway is running with channels.desktopRobot.enabled=true"
        )
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        await client.disconnect()
        if tts:
            tts.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
