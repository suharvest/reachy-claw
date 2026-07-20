#!/usr/bin/env python3
"""Verify the Realtime V2 manual response barrier against a live Gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import wave

import websockets


async def _recv(ws, started: float) -> dict:
    message = await ws.recv()
    if isinstance(message, bytes):
        print(f"{time.monotonic() - started:7.3f} binary bytes={len(message)}")
        return {"type": "binary", "bytes": len(message)}
    event = json.loads(message)
    print(f"{time.monotonic() - started:7.3f} {event.get('type')}")
    return event


async def verify(url: str, wav_path: Path, barrier_s: float, language: str) -> None:
    started = time.monotonic()
    async with websockets.connect(
        url,
        subprotocols=["seeed.realtime.v2"],
        ping_interval=None,
        max_size=None,
    ) as ws:
        assert (await _recv(ws, started))["type"] == "session.created"
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": f"Reply briefly in {language}.",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm", "rate": 16000,
                            "channels": 1, "endianness": "little",
                        },
                        "transcription": {"language": language},
                        "turn_detection": {
                            "type": "server_vad", "backend": "silero",
                            "silence_duration_ms": 500,
                            "create_response": False,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm", "rate": 16000,
                            "channels": 1, "endianness": "little",
                        },
                        "language": language,
                    },
                },
            },
        }))
        assert (await _recv(ws, started))["type"] == "session.updated"

        with wave.open(str(wav_path), "rb") as wav:
            if (
                wav.getframerate() != 16000
                or wav.getnchannels() != 1
                or wav.getsampwidth() != 2
            ):
                raise ValueError("fixture must be 16 kHz mono PCM16 WAV")
            pcm = wav.readframes(wav.getnframes())
        chunk_bytes = 3200
        for pos in range(0, len(pcm), chunk_bytes):
            await ws.send(pcm[pos : pos + chunk_bytes])
            await asyncio.sleep(0.1)
        for _ in range(12):
            await ws.send(bytes(chunk_bytes))
            await asyncio.sleep(0.1)

        while True:
            event = await asyncio.wait_for(_recv(ws, started), timeout=15)
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                print(f"TRANSCRIPT={event.get('transcript')!r}")
                break

        barrier_events: list[str] = []
        barrier_until = time.monotonic() + barrier_s
        while time.monotonic() < barrier_until:
            try:
                event = await asyncio.wait_for(
                    _recv(ws, started), timeout=barrier_until - time.monotonic()
                )
                barrier_events.append(event["type"])
            except asyncio.TimeoutError:
                break
        forbidden = {"response.created", "binary"}
        if forbidden.intersection(barrier_events):
            raise AssertionError(f"response crossed manual barrier: {barrier_events}")
        print("MANUAL_BARRIER=passed")

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": (
                    "VISUAL_SENTINEL: Alice is present. "
                    f"Reply briefly in {language}."
                )
            },
        }))
        assert (await _recv(ws, started))["type"] == "session.updated"
        await ws.send(json.dumps({"type": "response.create", "response": {}}))

        saw_created = saw_audio = saw_done = False
        while not saw_done:
            event = await asyncio.wait_for(_recv(ws, started), timeout=30)
            saw_created |= event["type"] == "response.created"
            saw_audio |= event["type"] == "binary"
            saw_done |= event["type"] == "response.done"
        if not (saw_created and saw_audio and saw_done):
            raise AssertionError("incomplete response lifecycle after response.create")
        print("RESPONSE_AFTER_CREATE=passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--wav",
        type=Path,
        default=Path("tests/e2e/fixtures/wav-en/greeting.wav"),
    )
    parser.add_argument("--barrier-seconds", type=float, default=1.5)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    asyncio.run(verify(args.url, args.wav, args.barrier_seconds, args.language))


if __name__ == "__main__":
    main()
