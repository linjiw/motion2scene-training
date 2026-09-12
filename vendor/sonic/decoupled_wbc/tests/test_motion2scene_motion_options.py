import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_motion_options import (
    MotionOption,
    mechanical_work,
    qualification_predicates,
)


def test_supported_interface_rejects_unavailable_duration_and_off_grid_entry():
    for entry in (0.19, 0.21, 0.41, float("nan")):
        with pytest.raises(ValueError):
            MotionOption("crouch", "d040", entry)
    with pytest.raises(ValueError, match="return"):
        MotionOption("long", "d040", return_request_time_s=3.5)
    assert MotionOption("early", "d040", 0.2).action == 1
    assert MotionOption("walk", "neutral").action == 0


def test_passage_cannot_qualify_refused_adaptation():
    flags = {k: True for k in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")}
    interface = {
        "switches": [],
        "observations": [{"transition": {**flags, "attempted": True, "allowed": False}}],
    }
    passage = {"pass": True, "reset_count": 0, "fall_observed": False}
    predicates = qualification_predicates(MotionOption("low", "d040"), interface, passage)
    assert predicates["physical_passage"]
    assert not predicates["requested_entry_executed"]
    assert not predicates["no_transition_refusal"]
    assert not all(predicates.values())


def test_work_does_not_cancel_positive_and_negative_joints():
    result = mechanical_work([[2, -3], [2, -3]], np.ones((2, 2)), 0.1)
    assert result["positive_mechanical_work_j"] == pytest.approx(0.4)
    assert result["absolute_mechanical_work_j"] == pytest.approx(1)
    assert result["net_mechanical_work_j"] == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        mechanical_work([[1]], [[float("nan")]], 0.1)
