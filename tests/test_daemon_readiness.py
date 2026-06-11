"""Regression test for the daemon readiness probe after the Zenoh->WebSocket
transport migration (reachy-mini SDK 1.5.0).

Before the upgrade the readiness probe was hardcoded to Zenoh's port 7447, which
no longer exists once the SDK talks WebSocket on the FastAPI port. This pins the
probe to the same port the SDK client connects to (config.reachy_daemon_port).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reachy_claw.app import ReachyClawApp
from reachy_claw.config import Config


def _socket_factory(probed_ports, connect_result=0):
    """Build a fake socket.socket that records the ports it probes."""

    def _factory(*_args, **_kwargs):
        s = MagicMock(name="socket")
        s.__enter__ = MagicMock(return_value=s)
        s.__exit__ = MagicMock(return_value=False)

        def connect_ex(addr):
            probed_ports.append(addr[1])
            return connect_result

        s.connect_ex = connect_ex
        return s

    return _factory


def test_ensure_daemon_probes_configured_port_not_zenoh():
    """Readiness probe must hit the configured daemon port, never the dead
    Zenoh port 7447."""
    config = Config(reachy_daemon_port=38001)
    app = ReachyClawApp(config)

    probed: list[int] = []
    # connect_result=0 => port open => "already running" early return,
    # so no subprocess is spawned.
    with patch("socket.socket", side_effect=_socket_factory(probed, connect_result=0)):
        app._ensure_daemon("auto")

    assert 38001 in probed, f"expected probe on configured port 38001, got {probed}"
    assert 7447 not in probed, "readiness probe must not use the removed Zenoh port 7447"


def test_ensure_daemon_honors_custom_port():
    config = Config(reachy_daemon_port=12345)
    app = ReachyClawApp(config)

    probed: list[int] = []
    with patch("socket.socket", side_effect=_socket_factory(probed, connect_result=0)):
        app._ensure_daemon("auto")

    assert probed and probed[0] == 12345


def test_ensure_daemon_spawns_with_matching_fastapi_port():
    """When the daemon isn't already running, it must be spawned on the same
    port the client connects to via --fastapi-port. The daemon CLI defaults to
    8000, so omitting the flag would silently bind the wrong port."""
    config = Config(reachy_daemon_port=38001)
    app = ReachyClawApp(config)

    # First probe (already-running check) => not running; then ready.
    results = iter([1, 0])

    def factory(*_a, **_k):
        s = MagicMock(name="socket")
        s.__enter__ = MagicMock(return_value=s)
        s.__exit__ = MagicMock(return_value=False)
        s.connect_ex = lambda addr: next(results, 0)
        return s

    proc = MagicMock()
    proc.poll.return_value = None  # process still alive
    captured = {}

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        return proc

    with patch("socket.socket", side_effect=factory), \
         patch("shutil.which", return_value="/usr/bin/reachy-mini-daemon"), \
         patch("subprocess.Popen", side_effect=fake_popen), \
         patch("time.sleep"):
        app._ensure_daemon("auto")

    cmd = captured["cmd"]
    assert "--fastapi-port" in cmd, f"spawn command missing --fastapi-port: {cmd}"
    assert cmd[cmd.index("--fastapi-port") + 1] == "38001"
