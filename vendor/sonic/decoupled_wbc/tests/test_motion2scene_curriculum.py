from dataclasses import replace

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_curriculum import (
    CostWeights,
    PhysicalBranch,
    absolute_mechanical_work,
    curriculum_probabilities,
    solution_preserving_screen,
    teacher_decision,
    verified_gap,
)


def branch(option, passed, elapsed=3.0, **kwargs):
    return PhysicalBranch(option, "prefix-sha", 7, passed, True, True, 150, elapsed, 0, **kwargs)


def test_solution_preserved_when_perturbation_removes_need_to_crouch():
    values = np.array([[[-0.03, 0.04], [0.02, 0.015]], [[-0.04, 0.03], [-0.02, 0.005]]])
    result = solution_preserving_screen(values, 1, 0)
    assert result["eligible"].tolist() == [True, False]
    assert result["negative_all_offsets_blocked"].tolist() == [False, True]
    with pytest.raises(ValueError):
        solution_preserving_screen(values * np.nan, 1, 0)


def test_teacher_requires_matched_physical_success_and_measured_cost():
    branches = [branch("walk", False, 1), branch("early", True, 4), branch("late", True, 3)]
    assert teacher_decision(branches).option_id == "late"
    assert verified_gap(branches, "walk") == 1
    assert verified_gap(branches, "early") == 0.25
    assert verified_gap(iter(branches), "early") == 0.25
    assert verified_gap(branches, "unexecuted") is None
    with pytest.raises(ValueError, match="state/history"):
        teacher_decision([branches[0], replace(branches[1], state_history_id="different")])
    with pytest.raises(ValueError, match="measured work"):
        teacher_decision(branches, CostWeights(work=1))


def test_no_target_is_not_a_stop_and_course_targets_require_continuation():
    assert teacher_decision([branch("walk", False)]) is None
    assert teacher_decision([branch("walk", True)], sequential=True) is None
    passed = branch("walk", True, continuation_verified=True)
    assert teacher_decision([passed], sequential=True) == passed
    assert verified_gap([passed, branch("late", False)], "late", sequential=True) is None


def test_uniform_exploration_survives_regret_focus():
    probabilities = curriculum_probabilities([0, 1, None], ["walk", "low", "low"])
    assert np.isclose(probabilities.sum(), 1)
    assert (probabilities >= 0.2 / 3).all()
    assert probabilities[1] > probabilities[0] > probabilities[2]
    assert np.allclose(curriculum_probabilities([None, None], ["a", "b"]), [0.5, 0.5])


def test_work_integrates_absolute_power_and_rejects_resets():
    assert absolute_mechanical_work([0, 0.5, 1], [[2], [-2], [2]], [[3], [3], [3]]) == 6
    with pytest.raises(ValueError):
        absolute_mechanical_work([0, 0.5, 0], [[2], [2], [2]], [[3], [3], [3]])
