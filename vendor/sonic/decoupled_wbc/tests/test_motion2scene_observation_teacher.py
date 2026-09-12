import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_observation_teacher import observation_consistent_teacher


def teacher(keys, passed, times=None, admitted=None):
    passed = np.array(passed, bool)
    return observation_consistent_teacher(
        keys,
        [15, 70],
        [None, 15, 70, 70],
        passed,
        np.where(passed, 4.0, np.nan) if times is None else times,
        np.ones_like(passed) if admitted is None else admitted,
    )


def test_wait_rejects_privileged_choice_between_identical_histories():
    rows = teacher([["same", "same"], ["same", "same"]], [[0, 0, 1, 0], [0, 0, 0, 1]])
    assert all(r["information_conflict"] for r in rows)
    assert all(r["teacher_action"] is None for r in rows)


def test_wait_succeeds_when_future_observation_disambiguates():
    rows = teacher([["same", "left"], ["same", "right"]], [[0, 0, 1, 0], [0, 0, 0, 1]])
    assert rows[0]["acceptable_success_actions"] == [0]
    assert rows[0]["selected_schedule_by_encounter"] == {0: 2, 1: 3}


def test_common_action_retained_even_when_scene_optima_disagree():
    rows = teacher(
        [["same", "same"], ["same", "same"]],
        [[0, 1, 1, 1], [0, 1, 1, 1]],
        [[np.nan, 6, 3, 5], [np.nan, 6, 5, 3]],
    )
    assert rows[0]["acceptable_success_actions"] == [0, 1]
    assert rows[0]["teacher_action"] == 0
    assert rows[1]["optimal_time_actions"] == [2, 3]
    assert rows[0]["mean_passage_time_s_by_action"][0] == 4


def test_unknown_branch_is_not_a_measured_information_conflict():
    admitted = np.ones((2, 4), bool)
    admitted[0, 2] = False
    rows = teacher(
        [["same", "same"], ["same", "same"]], [[0, 0, 1, 0], [0, 0, 0, 1]], admitted=admitted
    )
    assert all(not r["complete"] and not r["information_conflict"] for r in rows)


def test_information_cannot_forget_earlier_disambiguation():
    with pytest.raises(ValueError, match="refining"):
        teacher([["left", "same"], ["right", "same"]], [[0, 1, 1, 0], [0, 1, 0, 1]])


def test_all_failed_group_remains_failed_not_information_conflict():
    rows = teacher([["same", "same"]], [[0, 0, 0, 0]])
    assert all(not r["information_conflict"] and r["teacher_action"] is None for r in rows)
