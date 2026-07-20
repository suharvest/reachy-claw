"""Unit tests for provider-neutral ReachyMotionToolsPlugin motion tools.

Verifies the plugin registers the 4 tools onto a tool registry and that each
dispatches into the MotionController via the compositor-safe enqueue methods
(NOT direct SDK goto_target off-thread). Uses a fake MotionController so the
test runs offline with no robot / no SDK.
"""

from __future__ import annotations

import asyncio

from ovs_agent.tools.registry import ToolRegistry

from reachy_voice.plugins.motion_tools import ReachyMotionToolsPlugin


class _FakeMotion:
    """Stand-in for MotionController: records calls instead of driving motors."""

    def __init__(self) -> None:
        self.head_calls: list[tuple] = []
        self.antenna_calls: list[tuple] = []
        self.emotion_calls: list[str] = []
        self.dance_calls: list[str] = []

    def command_head(self, yaw, pitch, roll=0.0):
        self.head_calls.append((yaw, pitch, roll))
        return {"ok": True, "moved_to": {"yaw": yaw, "pitch": pitch, "roll": roll}}

    def command_antennas(self, left, right):
        self.antenna_calls.append((left, right))
        return {"ok": True, "antennas": {"left": left, "right": right}}

    def play_emotion(self, emotion):
        self.emotion_calls.append(emotion)
        return None  # MotionController.play_emotion returns None (enqueue only)

    def play_dance(self, name):
        self.dance_calls.append(name)
        return {"ok": True, "dance": name}


class _FakeApp:
    """Minimal app surface the plugin touches: a tool_registry + current_emotion."""

    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()
        self.current_emotion = "neutral"


def _make_plugin(motion=None):
    app = _FakeApp()
    motion = motion or _FakeMotion()
    plugin = ReachyMotionToolsPlugin(app, motion=motion)
    return app, motion, plugin


# ── registration ──────────────────────────────────────────────────────


def test_registry_has_four_tool_names():
    app, _motion, plugin = _make_plugin()
    assert plugin.setup() is True
    names = set(app.tool_registry.list_names())
    assert {"move_head", "move_antennas", "play_emotion", "dance"} <= names


def test_setup_returns_false_without_motion():
    app = _FakeApp()
    plugin = ReachyMotionToolsPlugin(app, motion=None)
    assert plugin.setup() is False
    assert app.tool_registry.list_names() == []


def test_tools_exposed_in_openai_schema():
    app, _motion, plugin = _make_plugin()
    plugin.setup()
    tools = app.tool_registry.list_openai_tools()
    by_name = {t["function"]["name"]: t for t in tools}
    assert set(by_name) >= {"move_head", "move_antennas", "play_emotion", "dance"}
    # typed params propagate from the handler signature
    head_props = by_name["move_head"]["function"]["parameters"]["properties"]
    assert head_props["yaw"]["type"] == "number"
    # preamble is stored on the registered Tool (verbal ack before motion).
    assert app.tool_registry._tools["move_head"].preamble_text == "好的。"
    # dance gets a longer dispatch timeout than the 10s default (long routines).
    assert app.tool_registry._tools["dance"].timeout_s == 20.0


# ── dispatch routes into the compositor-safe MotionController methods ──


def test_dispatch_move_head():
    app, motion, plugin = _make_plugin()
    plugin.setup()
    res = asyncio.run(
        app.tool_registry.dispatch("move_head", {"yaw": 30, "pitch": 0}, ctx=None)
    )
    assert res["ok"] is True
    assert motion.head_calls == [(30, 0, 0.0)]


def test_dispatch_move_antennas():
    app, motion, plugin = _make_plugin()
    plugin.setup()
    res = asyncio.run(
        app.tool_registry.dispatch(
            "move_antennas", {"left": 20, "right": 10}, ctx=None
        )
    )
    assert res["ok"] is True
    assert motion.antenna_calls == [(20, 10)]


def test_dispatch_play_emotion_sets_app_slot():
    app, motion, plugin = _make_plugin()
    plugin.setup()
    res = asyncio.run(
        app.tool_registry.dispatch("play_emotion", {"emotion": "happy"}, ctx=None)
    )
    assert res == {"ok": True, "emotion": "happy"}
    assert motion.emotion_calls == ["happy"]
    assert app.current_emotion == "happy"


def test_dispatch_play_emotion_uses_structured_callback():
    app = _FakeApp()
    motion = _FakeMotion()
    emotions: list[str] = []
    plugin = ReachyMotionToolsPlugin(
        app, motion=motion, on_emotion=emotions.append
    )
    plugin.setup()
    asyncio.run(
        app.tool_registry.dispatch("play_emotion", {"emotion": "curious"}, ctx=None)
    )
    assert emotions == ["curious"]
    assert motion.emotion_calls == []


def test_dispatch_dance():
    app, motion, plugin = _make_plugin()
    plugin.setup()
    res = asyncio.run(
        app.tool_registry.dispatch("dance", {"dance_name": "simple_nod"}, ctx=None)
    )
    assert res["ok"] is True
    assert motion.dance_calls == ["simple_nod"]


def test_stop_unregisters_tools():
    app, _motion, plugin = _make_plugin()
    plugin.setup()
    asyncio.run(plugin.stop())
    names = set(app.tool_registry.list_names())
    assert not ({"move_head", "move_antennas", "play_emotion", "dance"} & names)
