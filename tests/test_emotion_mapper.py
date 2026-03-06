"""Tests for the EmotionMapper and expression system."""

from __future__ import annotations

import time

import pytest

from clawd_reachy_mini.motion.emotion_mapper import (
    AntennaMotion,
    EmotionMapper,
    HeadPose,
    RobotExpression,
    EMOTION_MAP,
)


class TestEmotionMapBasic:
    def test_known_emotion_returns_expression(self):
        em = EmotionMapper()
        expr = em.map_emotion("happy")
        assert expr is not None
        assert expr.head is not None

    def test_unknown_emotion_returns_none(self):
        em = EmotionMapper()
        assert em.map_emotion("nonexistent_emotion_xyz") is None

    def test_all_mapped_emotions_return_expression(self):
        em = EmotionMapper(intensity=1.0)
        for emotion_name in EMOTION_MAP:
            expr = em.map_emotion(emotion_name)
            assert expr is not None, f"No expression for: {emotion_name}"

    def test_case_insensitive(self):
        em = EmotionMapper()
        assert em.map_emotion("Happy") is not None
        assert em.map_emotion("HAPPY") is not None
        assert em.map_emotion("  happy  ") is not None


class TestIntensityScaling:
    def test_full_intensity_preserves_values(self):
        em = EmotionMapper(intensity=1.0)
        # Neutral always has 0 values, use surprised which has clear non-zero
        expr = em.map_emotion("surprised")
        assert expr is not None
        # pitch should be 15 * 1.0 = 15
        assert expr.head.pitch == pytest.approx(15.0)
        assert expr.antenna.left == pytest.approx(50.0)

    def test_half_intensity_scales_values(self):
        em = EmotionMapper(intensity=0.5)
        expr = em.map_emotion("surprised")
        assert expr is not None
        assert expr.head.pitch == pytest.approx(7.5)
        assert expr.antenna.left == pytest.approx(25.0)

    def test_zero_intensity_zeroes_values(self):
        em = EmotionMapper(intensity=0.0)
        expr = em.map_emotion("surprised")
        assert expr is not None
        assert expr.head.pitch == 0.0
        assert expr.head.yaw == 0.0
        assert expr.antenna.left == 0.0

    def test_intensity_clamped_to_01(self):
        em_high = EmotionMapper(intensity=5.0)
        assert em_high._intensity == 1.0
        em_low = EmotionMapper(intensity=-1.0)
        assert em_low._intensity == 0.0

    def test_scaling_does_not_mutate_global_templates(self):
        em = EmotionMapper(intensity=0.5)
        first = em.map_emotion("surprised")
        second = em.map_emotion("surprised")
        third = em.map_emotion("surprised")
        assert first is not None and second is not None and third is not None
        assert first.head.pitch == pytest.approx(7.5)
        assert second.head.pitch == pytest.approx(7.5)
        assert third.head.pitch == pytest.approx(7.5)


class TestEmotionQueue:
    def test_queue_emotion_adds_to_queue(self):
        em = EmotionMapper()
        em.queue_emotion("happy")
        expr = em.get_next_expression()
        assert expr is not None
        assert "appy" in expr.description.lower() or "Happy" in expr.description

    def test_get_next_expression_empty_returns_none(self):
        em = EmotionMapper()
        assert em.get_next_expression() is None

    def test_queue_debounce_same_emotion(self):
        em = EmotionMapper()
        em.queue_emotion("happy")
        em.queue_emotion("happy")  # debounced
        em.queue_emotion("happy")  # debounced

        # Only 1 should be queued
        expr1 = em.get_next_expression()
        expr2 = em.get_next_expression()
        assert expr1 is not None
        assert expr2 is None

    def test_queue_allows_different_emotions(self):
        em = EmotionMapper()
        em.queue_emotion("happy")
        em.queue_emotion("sad")

        assert em.get_next_expression() is not None
        assert em.get_next_expression() is not None
        assert em.get_next_expression() is None

    def test_queue_allows_same_emotion_after_delay(self):
        em = EmotionMapper()
        em.queue_emotion("happy")
        # Fake the timestamp to be old
        em._last_expression_time -= 3.0
        em.queue_emotion("happy")

        assert em.get_next_expression() is not None
        assert em.get_next_expression() is not None

    def test_unknown_emotion_not_queued(self):
        em = EmotionMapper()
        em.queue_emotion("zzz_unknown_zzz")
        assert em.get_next_expression() is None


class TestIdleExpression:
    def test_idle_expression_has_head_and_antenna(self):
        em = EmotionMapper()
        expr = em.get_idle_expression()
        assert expr.head is not None
        assert expr.antenna is not None
        assert expr.description == "Idle breathing"

    def test_idle_expressions_vary(self):
        em = EmotionMapper()
        poses = set()
        for _ in range(20):
            expr = em.get_idle_expression()
            poses.add(round(expr.head.yaw, 2))
        # Should have multiple distinct values (randomized)
        assert len(poses) > 1


class TestDataclasses:
    def test_head_pose_defaults(self):
        hp = HeadPose()
        assert hp.yaw == 0.0
        assert hp.pitch == 0.0
        assert hp.roll == 0.0
        assert hp.duration == 0.8

    def test_antenna_motion_defaults(self):
        am = AntennaMotion()
        assert am.left == 0.0
        assert am.right == 0.0
        assert am.duration == 0.5

    def test_robot_expression_head_only(self):
        expr = RobotExpression(head=HeadPose(yaw=10), description="yaw only")
        assert expr.head is not None
        assert expr.antenna is None
        assert expr.description == "yaw only"
