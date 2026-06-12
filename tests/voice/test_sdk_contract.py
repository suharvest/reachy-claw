"""SDK contract tests for the reachy-mini robot SDK.

Unlike the rest of the suite — which runs against a MagicMock ReachyMini and
therefore silently auto-stubs any method name — these tests introspect the
*real installed* ``reachy_mini`` package. They pin the exact API surface the
codebase depends on so that an SDK upgrade (e.g. 1.4.x -> 1.8.x) that renames,
removes, or changes the signature of a symbol fails here loudly instead of only
blowing up on the physical robot.

Scope is deliberately the robot data closed-loop: motion commands
(sensor -> process -> motor), media capture/playback, and the pose helper.

If a symbol is intentionally renamed in a future SDK, update both this file and
the corresponding call site in the same change — that is the point of the gate.

GStreamer-backed imports (camera_gstreamer / audio_gstreamer) require the native
``gi`` bindings, which are absent on plain dev machines but present on the robot
/ Jetson image. Those checks skip gracefully when ``gi`` is unavailable.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

reachy_mini = pytest.importorskip("reachy_mini")


def _skip_without_gstreamer() -> None:
    """Skip GStreamer-backed import checks when the native ``Gst`` typelib is
    absent (plain dev box / CI). ``gi`` alone isn't enough — the camera/audio
    classes do ``gi.require_version('Gst', '1.0')``, which needs the GStreamer
    introspection typelib only present on the robot / Jetson image."""
    gi = pytest.importorskip("gi", reason="native gi bindings absent")
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401
    except (ValueError, ImportError):
        pytest.skip("GStreamer (Gst) typelib not available")


# ── Motion / data-loop methods on the ReachyMini handle ───────────────
# Every name here is called somewhere in src/reachy_claw on the live robot.
REACHY_METHODS = [
    # motor lifecycle
    "wake_up",
    "enable_motors",
    "disable_motors",
    # actuation (process -> motor)
    "goto_target",
    "set_target_head_pose",
    "set_target_body_yaw",
    "set_target_antenna_joint_positions",
    # sensing (sensor -> process)
    "get_current_head_pose",
    "get_present_antenna_joint_positions",
]

# ── Media manager methods (audio + camera closed-loop) ────────────────
MEDIA_METHODS = [
    "get_frame",            # camera -> vision
    "get_audio_sample",     # mic -> STT
    "push_audio_sample",    # TTS -> speaker
    "start_playing",
    "stop_playing",
    "start_recording",
    "stop_recording",
]


class TestReachyMiniSurface:
    @pytest.mark.parametrize("method", REACHY_METHODS)
    def test_reachy_has_method(self, method):
        from reachy_mini import ReachyMini

        assert callable(getattr(ReachyMini, method, None)), (
            f"reachy_mini.ReachyMini.{method}() is missing — the SDK upgrade "
            f"likely renamed or removed it; update the call site."
        )

    def test_reachy_is_context_manager(self):
        from reachy_mini import ReachyMini

        # app.py uses `with ReachyMini(...) as reachy:`
        assert hasattr(ReachyMini, "__enter__")
        assert hasattr(ReachyMini, "__exit__")


class TestMediaManagerSurface:
    @pytest.mark.parametrize("method", MEDIA_METHODS)
    def test_media_has_method(self, method):
        from reachy_mini.media.media_manager import MediaManager

        assert callable(getattr(MediaManager, method, None)), (
            f"reachy_mini.media.media_manager.MediaManager.{method}() is missing "
            f"— the SDK upgrade likely renamed or removed it."
        )


class TestCreateHeadPose:
    def test_importable(self):
        from reachy_mini.utils import create_head_pose  # noqa: F401

    def test_accepts_keyword_args(self):
        """Call sites use create_head_pose(roll=, pitch=, yaw=, degrees=True)."""
        from reachy_mini.utils import create_head_pose

        params = inspect.signature(create_head_pose).parameters
        for kw in ("roll", "pitch", "yaw", "degrees"):
            assert kw in params, f"create_head_pose lost the '{kw}' keyword argument"

    def test_returns_4x4_matrix(self):
        from reachy_mini.utils import create_head_pose

        pose = np.asarray(create_head_pose(roll=0, pitch=10, yaw=0, degrees=True))
        assert pose.shape == (4, 4), (
            f"create_head_pose must return a 4x4 homogeneous matrix, got {pose.shape}"
        )


class TestAppBaseClass:
    def test_reachy_mini_app_run_signature(self):
        """reachy_app.ReachyClawApp subclasses ReachyMiniApp and overrides
        run(self, reachy_mini, stop_event). Pin that contract."""
        from reachy_mini.apps.app import ReachyMiniApp

        params = list(inspect.signature(ReachyMiniApp.run).parameters)
        # self + the two positional args the daemon passes in
        assert params[0] == "self"
        assert len(params) == 3, (
            f"ReachyMiniApp.run signature changed: {params} — "
            f"reachy_app.ClawdReachyMiniApp.run must match the daemon's call."
        )


class TestImportPaths:
    def test_camera_constants(self):
        from reachy_mini.media.camera_constants import (  # noqa: F401
            CameraSpecs,
            ReachyMiniLiteCamSpecs,
        )

    def test_gstreamer_camera_import(self):
        _skip_without_gstreamer()
        from reachy_mini.media.camera_gstreamer import GStreamerCamera  # noqa: F401

    def test_gstreamer_audio_import(self):
        _skip_without_gstreamer()
        from reachy_mini.media.audio_gstreamer import GStreamerAudio  # noqa: F401
