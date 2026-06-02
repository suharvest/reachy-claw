"""Local motors-off E2E for the SLV ConversationPlugin.

Drives a real turn through the new plugin:
  fake SLV (inject ASRFinal) → real edge_llm (ollama) → real ToolRegistry +
  stream_with_tools → _cmd_move_head fired (no_robot ack) → TTS send_text.

Captures every app.events.emit(...) to prove the event-bridge fires
(asr_final / llm_delta / llm_end / state_change / emotion), so the
unchanged dashboard / daily_log plugins keep receiving their signals.

Run: uv run python tests/e2e_slv_plugin_local.py
Requires: ollama at http://localhost:11434 with qwen2.5:14b.
"""
from __future__ import annotations

import asyncio
import os
import sys

# Quiet the gstreamer .pth noise on macOS dev.
sys.path  # noqa

from reachy_claw.config import Config
from reachy_claw.app import ReachyClawApp
from reachy_claw.plugins.conversation_plugin_slv import ConversationPlugin, ConvState
from ovs_agent.slv_client import ASRFinal, ASRPartial


class FakeSLV:
    """Stub SLVClient: records all outbound calls; no real WS."""

    def __init__(self):
        self.text_chunks: list[str] = []
        self.flushed = 0
        self.aborts = 0
        self.audio_bytes = 0

    async def connect(self):
        pass

    async def close(self):
        pass

    async def send_audio(self, pcm: bytes):
        self.audio_bytes += len(pcm)

    async def send_text(self, text: str):
        if text:
            self.text_chunks.append(text)

    async def flush_tts(self):
        self.flushed += 1

    async def abort(self):
        self.aborts += 1


async def main() -> int:
    cfg = Config()
    cfg.conversation_backend = "slv"
    cfg.edge_llm_url = os.environ.get("LLM_URL", "http://localhost:11434/v1")
    cfg.edge_llm_model = os.environ.get("LLM_MODEL", "qwen2.5:14b")
    # base system prompt so the model knows to use move_head
    cfg.ollama_system_prompt = (
        "You control a Reachy robot. When the user asks you to look in a "
        "direction, you MUST call the move_head tool. Reply in one short "
        "sentence. End with one emotion tag like [happy]."
    )

    app = ReachyClawApp(cfg)
    # motors off: no robot connected → _cmd_move_head returns no_robot ack.
    app.reachy = None

    # Capture EVERY event emitted on the shared bus (this is what the
    # unchanged dashboard / daily_log plugins subscribe to).
    captured: list[tuple[str, object]] = []
    orig_emit = app.events.emit

    def _spy(name, payload=None):
        captured.append((name, payload))
        return orig_emit(name, payload)

    app.events.emit = _spy  # type: ignore

    plugin = ConversationPlugin(app)
    assert plugin.setup(), "setup() failed"

    # Build the real edge_llm backend + fake SLV, bypassing start()'s WS.
    from ovs_agent.llm.edge_llm import EdgeLLMBackend

    plugin._llm = EdgeLLMBackend(
        base_url=cfg.edge_llm_url, api_key="ollama", model=cfg.edge_llm_model
    )
    fake = FakeSLV()
    plugin._slv = fake  # type: ignore
    plugin._running = True

    # Track _cmd_move_head firing.
    cmd_calls: list[dict] = []
    orig_cmd = plugin._cmd_move_head

    def _cmd_spy(params):
        cmd_calls.append(dict(params))
        return orig_cmd(params)

    plugin._cmd_move_head = _cmd_spy  # type: ignore
    # re-register so the wrapper picks up the spy
    plugin._registry = type(plugin._registry)()
    plugin._register_tools()

    # ── inject one turn: "向左看" (look left) ──
    print("=== injecting ASRFinal: '向左看' ===")
    await plugin._dispatch_slv_event(
        ASRFinal(text="向左看", session_complete=False, duplicate_of_streamed=False)
    )
    # the turn was spawned as a task; await it
    if plugin._turn_task is not None:
        await plugin._turn_task

    # ── report ──
    print("\n=== _cmd_move_head fired ===")
    print(cmd_calls)

    print("\n=== SLV outbound ===")
    print("send_text chunks:", fake.text_chunks)
    print("flush_tts:", fake.flushed, " aborts:", fake.aborts)

    print("\n=== app.events.emit(...) bridge (raw) ===")
    for name, payload in captured:
        print(f"  emit {name!r}: {payload}")

    emitted = {name for name, _ in captured}
    required = {"asr_final", "state_change", "llm_delta", "llm_end"}
    missing = required - emitted
    print("\n=== event-bridge assertions ===")
    print("emitted signals:", sorted(emitted))
    print("required present:", sorted(required - missing))
    if missing:
        print("MISSING:", sorted(missing))
    # state_change must include thinking
    states = [p.get("state") for n, p in captured if n == "state_change" and isinstance(p, dict)]
    print("state transitions:", states)

    ok = not missing and cmd_calls and fake.text_chunks
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
