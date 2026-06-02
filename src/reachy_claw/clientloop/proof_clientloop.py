"""CLIENT-LOOP tool round-trip PROOF launcher (throwaway).

Builds ``ReachyClawClientLoopApp`` from config.yaml, runs the
ReachyToolsPlugin ``setup()`` so move_head / play_emotion land in
``app.tool_registry``, then drives ONE user turn through ovs_agent's
CLIENT-SIDE tool runner — ``ovs_agent.tools.runner.stream_with_tools``
— exactly as ``app_mode.py`` invokes it.

This isolates the thing under test (the client-loop LLM <-> tool loop)
from the SLV engine: the engine is a pass-through ASR/TTS transport and
is NOT required for the tool loop itself. ASR text is injected directly
(simulating the asr_final the engine would emit), and TTS callbacks are
stubbed to log instead of opening an audio socket.

Run:
  LLM_MODEL=qwen2.5:14b \
  uv run python -m reachy_claw.clientloop.proof_clientloop \
    --text "向左看"
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ovs_agent.config import load_config
from ovs_agent.session import Session
from ovs_agent.tools import ToolCallCtx, stream_with_tools

from reachy_claw.clientloop.app import ReachyClawClientLoopApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("proof_clientloop")

_CONFIG = Path(__file__).resolve().parent / "config.yaml"


async def run(text: str, config_path: Path) -> int:
    cfg = load_config(config_path)
    log.info(
        "loaded config: llm_backend=%s model=%s base_url=%s "
        "tools_enabled=%s max_iter=%s",
        cfg.llm_backend, cfg.llm_model, cfg.llm_base_url,
        cfg.tools_enabled, cfg.tools_max_iterations,
    )

    # 1. Build the app (registers ReachyToolsPlugin, builds edge_llm
    #    backend + default tool_registry + session).
    app = ReachyClawClientLoopApp(cfg)

    # 2. Run plugin setup() so the Reachy tools are registered. BaseApp
    #    runs this in its full start path; we invoke it directly for the
    #    isolated proof (sync per Plugin.setup contract).
    for plugin in app.plugins:
        ok = plugin.setup()
        log.info("plugin %s setup() -> %s", plugin.name, ok)

    registry = app.tool_registry
    advertised = registry.list_openai_tools()
    log.info(
        "tool_registry advertises %d tools: %s",
        len(advertised),
        [t["function"]["name"] for t in advertised],
    )

    # 3. Build the per-turn pieces app_mode builds: a Session with the
    #    user utterance, a ToolCallCtx, and the messages list.
    session: Session = app.session
    session.add_user(text)
    messages = session.messages(cfg.system_prompt)
    log.info(">>> USER UTTERANCE injected: %r", text)
    log.info(
        ">>> LLM request will carry %d messages + %d tools "
        "(temperature=0 via mode_override)",
        len(messages), len(advertised),
    )

    tool_ctx = ToolCallCtx(
        session=session,
        mode_manager=None,
        event_bus=app.events,
        config=cfg,
    )

    # 4. Callbacks — stubbed TTS (log instead of SLV socket). These are
    #    the SAME hook surface app_mode wires to slv.send_text.
    spoken: list[str] = []

    async def on_assistant_token(tok: str) -> None:
        spoken.append(tok)
        sys.stdout.write(tok)
        sys.stdout.flush()

    async def on_tool_started(tc: dict) -> None:
        log.info(
            ">>> TOOL_CALL STARTED name=%s id=%s args=%s",
            tc.get("function", {}).get("name"),
            tc.get("id"),
            tc.get("function", {}).get("arguments"),
        )

    async def on_tool_preamble(t: str) -> None:
        log.info(">>> TOOL PREAMBLE (would speak): %r", t)

    async def on_tool_completion_text(t: str) -> None:
        log.info(">>> TOOL COMPLETION_TEXT (would speak): %r", t)

    async def on_tool_completed(tc: dict, result: dict, dt_ms: float) -> None:
        log.info(
            ">>> TOOL_CALL COMPLETED name=%s result=%s (%.0fms)",
            tc.get("function", {}).get("name"), result, dt_ms,
        )

    # 5. Drive the CLIENT-LOOP runner — identical entrypoint to app_mode.
    llm_kwargs: dict = {
        "session": session,
        "temperature": 0.0,
    }
    log.info("=" * 70)
    log.info(">>> ENTERING CLIENT-LOOP RUNNER (stream_with_tools)")
    log.info("=" * 70)

    final_text = await stream_with_tools(
        app.llm,
        messages,
        session=session,
        registry=registry,
        allowed_tools=None,  # None = all registered (tools_default_allowlist: [])
        ctx=tool_ctx,
        max_iterations=cfg.tools_max_iterations,
        on_assistant_token=on_assistant_token,
        on_tool_started=on_tool_started,
        on_tool_preamble=on_tool_preamble,
        on_tool_completion_text=on_tool_completion_text,
        on_tool_completed=on_tool_completed,
        llm_kwargs=llm_kwargs,
        first_token_timeout_s=cfg.llm_first_token_timeout_s,
        idle_timeout_s=cfg.llm_stream_idle_timeout_s,
    )
    sys.stdout.write("\n")

    await app.llm.aclose()

    log.info("=" * 70)
    log.info(">>> FINAL ASSISTANT TEXT (TTS would speak): %r", final_text)
    log.info(">>> SESSION HISTORY (%d messages):", len(session.history))
    for i, m in enumerate(session.history):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            log.info(
                "    [%d] assistant tool_calls=%s content=%r",
                i,
                [
                    (tc["function"]["name"], tc["function"]["arguments"])
                    for tc in m["tool_calls"]
                ],
                m.get("content"),
            )
        elif role == "tool":
            log.info("    [%d] tool result content=%s", i, m.get("content"))
        else:
            log.info("    [%d] %s content=%r", i, role, m.get("content"))

    # Verdict: a tool_call must have been issued AND a tool result
    # appended (the client-loop runner's defining behaviour).
    saw_tool_call = any(
        m.get("role") == "assistant" and m.get("tool_calls")
        for m in session.history
    )
    saw_tool_result = any(m.get("role") == "tool" for m in session.history)
    proven = saw_tool_call and saw_tool_result
    log.info("=" * 70)
    log.info(
        "SUMMARY: tool_call_issued=%s tool_result_appended=%s "
        "final_text_nonempty=%s",
        saw_tool_call, saw_tool_result, bool(final_text),
    )
    log.info(
        "VERDICT: client-loop reachy slice %s",
        "PROVEN" if proven else "NOT-PROVEN",
    )
    return 0 if proven else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--text", default="向左看",
        help="user utterance to inject (simulated asr_final)",
    )
    ap.add_argument("--config", type=Path, default=_CONFIG)
    a = ap.parse_args()
    return asyncio.run(run(a.text, a.config))


if __name__ == "__main__":
    sys.exit(main())
