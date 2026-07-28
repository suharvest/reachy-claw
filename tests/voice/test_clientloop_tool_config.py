"""Verifies Reachy selects canonical Realtime V2 server-loop semantics."""

from __future__ import annotations

from reachy_voice.config import Config
from reachy_voice.conversation import build_ovs_config


def test_tools_disabled_by_default_for_local_edge_llm():
    ovs = build_ovs_config(Config())
    assert ovs.tools_enabled is False


def test_tools_can_be_enabled_explicitly():
    ovs = build_ovs_config(Config(tools_enabled=True))
    assert ovs.tools_enabled is True


def test_empty_allowlist_means_all_tools():
    ovs = build_ovs_config(Config(tools_enabled=True))
    # Empty allowlist + tools_enabled=True exposes ALL registered tools.
    assert list(ovs.tools_default_allowlist) == []


def test_bounded_tool_iterations():
    ovs = build_ovs_config(Config())
    assert ovs.tools_max_iterations == 3


def test_realtime_v2_client_loop_selected_by_default():
    ovs = build_ovs_config(Config())
    assert ovs.realtime_protocol_version == 2
    assert ovs.server_loop_enabled() is False
    # Dynamic visual context must land before the response starts.
    assert ovs.slv_config["create_response"] is False


def test_client_loop_remains_explicit_fallback():
    ovs = build_ovs_config(Config(server_loop=False))
    assert ovs.server_loop_enabled() is False
