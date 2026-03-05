"""Tests for HeadTarget and HeadTargetBus (tracking fusion)."""

from __future__ import annotations

import time

import pytest

from clawd_reachy_mini.motion.head_target import HeadTarget, HeadTargetBus


class TestHeadTarget:
    def test_defaults(self):
        t = HeadTarget()
        assert t.yaw == 0.0
        assert t.pitch == 0.0
        assert t.confidence == 0.0
        assert t.source == ""
        assert t.timestamp > 0

    def test_custom_values(self):
        t = HeadTarget(yaw=15.0, pitch=-10.0, confidence=0.9, source="face")
        assert t.yaw == 15.0
        assert t.pitch == -10.0
        assert t.confidence == 0.9
        assert t.source == "face"


class TestHeadTargetBus:
    def test_no_data_returns_neutral(self):
        bus = HeadTargetBus()
        target = bus.get_fused_target()
        assert target.source == "none"
        assert target.confidence == 0.0
        assert target.yaw == 0.0
        assert target.pitch == 0.0

    def test_face_published_returns_face(self):
        bus = HeadTargetBus()
        bus.publish(
            HeadTarget(yaw=20.0, pitch=-5.0, confidence=0.9, source="face")
        )
        target = bus.get_fused_target()
        assert target.source == "face"
        assert target.yaw == pytest.approx(20.0)
        assert target.pitch == pytest.approx(-5.0)
        assert target.confidence == 0.9

    def test_doa_published_returns_doa(self):
        bus = HeadTargetBus()
        bus.publish(
            HeadTarget(yaw=30.0, pitch=0.0, confidence=0.8, source="doa")
        )
        target = bus.get_fused_target()
        assert target.source == "doa"
        assert target.yaw == pytest.approx(30.0)
        # DOA doesn't provide pitch
        assert target.pitch == 0.0

    def test_face_takes_priority_over_doa(self):
        bus = HeadTargetBus()
        now = time.monotonic()

        bus.publish(
            HeadTarget(
                yaw=30.0, confidence=0.8, source="doa", timestamp=now
            )
        )
        bus.publish(
            HeadTarget(
                yaw=10.0, pitch=-3.0, confidence=0.9, source="face", timestamp=now
            )
        )

        target = bus.get_fused_target()
        assert target.source == "face"
        assert target.yaw == pytest.approx(10.0)

    def test_stale_face_falls_back_to_doa(self):
        bus = HeadTargetBus(face_timeout=0.5, doa_timeout=3.0)
        now = time.monotonic()

        # Fresh DOA
        bus.publish(
            HeadTarget(yaw=25.0, confidence=0.8, source="doa", timestamp=now)
        )
        # Stale face (1 second ago)
        bus.publish(
            HeadTarget(
                yaw=10.0, confidence=0.9, source="face", timestamp=now - 1.0
            )
        )

        target = bus.get_fused_target()
        assert target.source == "doa"
        assert target.yaw == pytest.approx(25.0)

    def test_stale_both_returns_neutral(self):
        bus = HeadTargetBus(face_timeout=0.1, doa_timeout=0.1)

        old = time.monotonic() - 1.0
        bus.publish(
            HeadTarget(yaw=10.0, confidence=0.9, source="face", timestamp=old)
        )
        bus.publish(
            HeadTarget(yaw=20.0, confidence=0.8, source="doa", timestamp=old)
        )

        target = bus.get_fused_target()
        assert target.source == "none"
        assert target.confidence == 0.0

    def test_zero_confidence_ignored(self):
        bus = HeadTargetBus()
        bus.publish(
            HeadTarget(yaw=10.0, confidence=0.0, source="face")
        )
        target = bus.get_fused_target()
        assert target.source == "none"

    def test_thread_safety(self):
        """Publish from multiple threads, read doesn't crash."""
        import threading

        bus = HeadTargetBus()
        errors = []

        def writer(source, n):
            try:
                for i in range(n):
                    bus.publish(
                        HeadTarget(
                            yaw=float(i), confidence=0.9, source=source
                        )
                    )
            except Exception as e:
                errors.append(e)

        def reader(n):
            try:
                for _ in range(n):
                    bus.get_fused_target()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("face", 100)),
            threading.Thread(target=writer, args=("doa", 100)),
            threading.Thread(target=reader, args=(200,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_overwrite_latest_face(self):
        bus = HeadTargetBus()
        bus.publish(HeadTarget(yaw=5.0, confidence=0.9, source="face"))
        bus.publish(HeadTarget(yaw=15.0, confidence=0.95, source="face"))

        target = bus.get_fused_target()
        assert target.yaw == pytest.approx(15.0)
        assert target.confidence == 0.95
