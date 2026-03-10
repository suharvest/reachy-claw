"""Tests for the dance routines module."""

from reachy_claw.motion.dances import (
    AVAILABLE_DANCES,
    DANCE_ROUTINES,
    DanceRoutine,
    DanceStep,
)


class TestDanceRoutines:
    def test_all_routines_exist(self):
        expected = {"nod", "wiggle", "celebrate", "curious_look", "lobster"}
        assert set(DANCE_ROUTINES.keys()) == expected

    def test_available_dances_sorted(self):
        assert AVAILABLE_DANCES == sorted(DANCE_ROUTINES.keys())

    def test_all_routines_have_steps(self):
        for name, routine in DANCE_ROUTINES.items():
            assert len(routine.steps) > 0, f"Dance '{name}' has no steps"
            assert routine.name == name
            assert routine.description

    def test_all_routines_end_at_neutral(self):
        """Last step should return to roughly neutral position."""
        for name, routine in DANCE_ROUTINES.items():
            last = routine.steps[-1]
            assert abs(last.yaw) <= 1, f"Dance '{name}' ends with yaw={last.yaw}"
            assert abs(last.pitch) <= 1, f"Dance '{name}' ends with pitch={last.pitch}"
            assert abs(last.roll) <= 1, f"Dance '{name}' ends with roll={last.roll}"

    def test_step_durations_positive(self):
        for name, routine in DANCE_ROUTINES.items():
            for i, step in enumerate(routine.steps):
                assert step.duration > 0, f"Dance '{name}' step {i} has duration={step.duration}"

    def test_dance_step_defaults(self):
        step = DanceStep()
        assert step.yaw == 0.0
        assert step.pitch == 0.0
        assert step.roll == 0.0
        assert step.antenna_left == 0.0
        assert step.antenna_right == 0.0
        assert step.duration == 0.4

    def test_dance_routine_dataclass(self):
        routine = DanceRoutine(
            name="test",
            description="Test dance",
            steps=[DanceStep(yaw=10, duration=0.5)],
        )
        assert routine.name == "test"
        assert len(routine.steps) == 1
        assert routine.steps[0].yaw == 10
