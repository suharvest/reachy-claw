"""Verifies the ovs config built by reachy_voice selects CLIENT-LOOP tool
calling: tools_enabled=True, an empty allowlist (= all registered tools), a
bounded iteration count, and server_loop left OFF (so the client-loop path is
auto-selected by ovs_agent).
"""

from __future__ import annotations

from reachy_voice.config import Config
from reachy_voice.conversation import build_ovs_config


def test_tools_enabled_true():
    ovs = build_ovs_config(Config())
    assert ovs.tools_enabled is True


def test_empty_allowlist_means_all_tools():
    ovs = build_ovs_config(Config())
    # Empty allowlist + tools_enabled=True → ovs exposes ALL registered tools.
    assert list(ovs.tools_default_allowlist) == []


def test_bounded_tool_iterations():
    ovs = build_ovs_config(Config())
    assert ovs.tools_max_iterations == 3


def test_client_loop_selected_not_server_loop():
    ovs = build_ovs_config(Config())
    # server_loop OFF → ovs_agent runs the client-loop (stream_with_tools) path.
    assert bool(getattr(ovs, "server_loop", False)) is False
    assert ovs.server_loop_enabled() is False
