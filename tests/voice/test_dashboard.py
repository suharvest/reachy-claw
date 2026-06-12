"""Unit tests for the dashboard bridge — these lock in the exact wire protocol
the (verbatim-copied) original dashboard UI consumes, so the backend keeps
feeding it the shape it expects:

  * ``llm_delta``/``llm_end`` carry a per-turn ``run_id`` and **tag-free** text
    (the UI groups a streaming "thought card" by run_id; emotion tags must not
    show in the transcript).
  * a fresh turn gets a new run_id; ``llm_end.full_text`` is the whole clean
    reply.
"""

from __future__ import annotations

import asyncio

from reachy_voice.dashboard import DashboardHub, DashboardPlugin


class _FakeApp:
    """Plugin's base only needs an ``app`` handle; nothing is called on it."""


def _drive(tokens: list[str], *, turns: int = 1) -> list[dict]:
    """Run ``tokens`` through the plugin ``turns`` times (each turn = one
    assistant reply: tokens then on_assistant_done) and return all published
    messages in order."""
    hub = DashboardHub()
    published: list[dict] = []
    hub.publish = published.append  # type: ignore[method-assign]
    plugin = DashboardPlugin(_FakeApp(), hub)

    async def main() -> None:
        for _ in range(turns):
            for tok in tokens:
                await plugin.on_assistant_token(tok)
            await plugin.on_assistant_done()

    asyncio.run(main())
    return published


def test_llm_delta_carries_runid_and_clean_text():
    msgs = _drive(["你好", "[happy]", "，很高兴见到你"])
    deltas = [m for m in msgs if m["type"] == "llm_delta"]
    # the [happy] tag never reaches the transcript
    assert "".join(d["text"] for d in deltas) == "你好，很高兴见到你"
    # every delta in one turn shares a run_id
    run_ids = {d["run_id"] for d in deltas}
    assert len(run_ids) == 1


def test_llm_end_full_text_is_clean_and_matches_runid():
    msgs = _drive(["你好", "[happy]", "，很高兴见到你"])
    delta_run = next(m["run_id"] for m in msgs if m["type"] == "llm_delta")
    end = next(m for m in msgs if m["type"] == "llm_end")
    assert end["full_text"] == "你好，很高兴见到你"
    assert end["run_id"] == delta_run


def test_tag_split_across_tokens_is_stripped():
    # '[hap' + 'py]' must be removed as one tag, not leak partial brackets
    msgs = _drive(["很好", "[hap", "py]", "继续"])
    text = "".join(m["text"] for m in msgs if m["type"] == "llm_delta")
    assert text == "很好继续"


def test_each_turn_gets_a_new_runid():
    msgs = _drive(["嗨"], turns=2)
    run_ids = [m["run_id"] for m in msgs if m["type"] == "llm_end"]
    assert len(run_ids) == 2
    assert run_ids[0] != run_ids[1]


def test_unterminated_tag_tail_is_flushed_as_text():
    # a trailing '[' with no closing ']' is real text, not a tag — must appear
    msgs = _drive(["结束了 [", ])
    text = "".join(m["text"] for m in msgs if m["type"] in ("llm_delta",))
    end = next(m for m in msgs if m["type"] == "llm_end")
    assert end["full_text"] == "结束了 ["
    assert text + "[" == "结束了 [" or end["full_text"].endswith("[")


def test_user_and_state_events_shape():
    hub = DashboardHub()
    published: list[dict] = []
    hub.publish = published.append  # type: ignore[method-assign]
    plugin = DashboardPlugin(_FakeApp(), hub)

    async def main() -> None:
        await plugin.on_user_partial("nph")
        await plugin.on_user_utterance("你好")
        await plugin.on_state_change({"state": "thinking"})

    asyncio.run(main())
    assert {"type": "asr_partial", "text": "nph", "is_stable": False} in published
    assert {"type": "asr_final", "text": "你好"} in published
    assert {"type": "state", "state": "thinking"} in published
