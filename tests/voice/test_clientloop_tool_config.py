"""Verifies Reachy selects canonical Realtime V2 server-loop semantics."""

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


def test_realtime_v2_manual_server_loop_selected():
    ovs = build_ovs_config(Config())
    assert ovs.realtime_protocol_version == 2
    assert ovs.server_loop_enabled() is True
    # Dynamic visual context must land before the response starts.
    assert ovs.slv_config["create_response"] is False


def test_client_loop_remains_explicit_fallback():
    ovs = build_ovs_config(Config(server_loop=False))
    assert ovs.server_loop_enabled() is False
