"""Continuation probes must preserve timing, frames and admission boundaries."""

import numpy as np
import pytest

from gear_sonic.research.scene_distillation.navigation_continuation import (
    continuation_commands,
    continuation_outcomes,
    validate_continuation,
)


def config(kind="goal_velocity_keypoints"):
    return dict(
        kind=kind,
        position_gain=0.8,
        velocity_gain=0.5,
        max_speed_mps=0.35,
        max_shift_m=0.2,
        activation_radius_m=0.6,
    )


def test_world_horizontal_feedback_rotates_into_current_body():
    q = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    c = np.zeros(114, np.float32)
    result = continuation_commands(
        c, [0, 0, 0.8], q, [0, 0, 0.8], [0.2, 0, 0.8], [0, -0.1, 0, 1], config()
    )
    np.testing.assert_allclose(result[2:5], [0, -0.11, 0], atol=1e-7)
    np.testing.assert_allclose(
        result[8:50].reshape(14, 3), np.tile([0, -0.2, 0], (14, 1)), atol=1e-7
    )
    np.testing.assert_array_equal(result[50:], c[50:])
    np.testing.assert_array_equal(c, np.zeros(114))


@pytest.mark.parametrize(
    "kind,valid,goal", [("nominal", 1, 0.2), ("goal_velocity", 0, 0.2), ("goal_velocity", 1, 2.0)]
)
def test_inactive_provider_preserves_nominal(kind, valid, goal):
    c = np.arange(114, dtype=np.float32)
    result = continuation_commands(
        c, [0, 0, 0.8], [1, 0, 0, 0], [0, 0, 0.8], [goal, 0, 0.8], [0, 0, 0, valid], config(kind)
    )
    np.testing.assert_array_equal(result, c)


def test_late_local_stabilization_is_not_timely_support():
    task = dict(
        goal_xyz=[0, 0, 0.8],
        goal_tolerance_m=0.25,
        terminal_speed_mps=0.1,
        hold_ticks=50,
        deadline_ticks=100,
    )
    root = np.tile([0, 0, 0.8], (125, 1))
    speed = np.ones(125)
    speed[75:] = 0
    trace = dict(root_xyz=root, speed=speed, undesired_force=np.zeros(125))
    outcome = continuation_outcomes(task, trace, 70)
    assert outcome["local_stabilized"]
    assert not outcome["original_deadline_success"]
    assert not outcome["timely_suffix_supported"]
    assert outcome["diagnostic_extension_ticks"] == 25
    assert outcome["timely_suffix_score"]["max_hold_ticks"] == 25


def test_no_hidden_deadline_or_unbounded_candidate_parameter():
    with pytest.raises(ValueError):
        validate_continuation({**config(), "deadline_gain": 1})
    with pytest.raises(ValueError):
        validate_continuation({**config(), "max_shift_m": 100})
