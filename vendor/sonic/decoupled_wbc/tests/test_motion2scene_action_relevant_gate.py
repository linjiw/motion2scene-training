"""Visibility is neither necessary nor sufficient for a common finite action."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_action_relevant_gate as gate
from motion2scene_action_relevant_gate import action_compatible_gates


def example(reveal=False):
    keys = np.array([["same", "a" if reveal else "same"], ["same", "b" if reveal else "same"]])
    passed = np.array([[False, True, True, False], [False, True, False, True]])
    times = np.array([[np.nan, 3, 1, np.nan], [np.nan, 3, np.nan, 1]])
    return keys, [1, 2], [None, 1, 2, 2], passed, times, np.ones_like(passed), [[0, 2], [0, 3]]


def test_shared_early_action_is_usable_without_scene_distinction():
    result = action_compatible_gates(*example())
    early = [r for r in result if r["phase_tick"] == 1]
    assert all(r["eligible"] and r["acceptable_actions"] == [1] for r in early)
    assert all(r["common_teacher_action"] == 1 and r["selected_continuation"] == 1 for r in early)


def test_gate_cannot_reuse_scene_wise_wait_when_late_actions_conflict():
    result = action_compatible_gates(*example())
    early = [r for r in result if r["phase_tick"] == 1]
    assert all(
        r["requires_action_retargeting"] and not r["original_action_compatible"] for r in early
    )
    late = [r for r in result if r["phase_tick"] == 2]
    assert all(not r["eligible"] and r["information_conflict"] for r in late)


def test_later_reveal_makes_wait_causally_usable():
    result = action_compatible_gates(*example(reveal=True))
    early = [r for r in result if r["phase_tick"] == 1]
    assert all(r["acceptable_actions"] == [0, 1] for r in early)
    assert all(r["common_teacher_action"] == 0 and r["original_action_compatible"] for r in early)
    assert [r["selected_continuation"] for r in early] == [2, 3]


def test_missing_outcome_withholds_not_measured_failure():
    args = list(example())
    args[5][0, 1] = False
    early = [r for r in action_compatible_gates(*args) if r["phase_tick"] == 1]
    assert all(not r["complete"] and not r["eligible"] for r in early)
    assert all(not r["information_conflict"] for r in early)


def test_common_target_retains_set_valued_supervision():
    args = list(example())
    args[3][:, 2:] = True
    args[4][:, 2:] = 1
    late = [r for r in action_compatible_gates(*args) if r["phase_tick"] == 2]
    assert all(r["acceptable_actions"] == [2, 3] and r["original_action_compatible"] for r in late)


@pytest.mark.parametrize("action", [4, -1, True])
def test_illegal_original_targets_are_rejected(action):
    args = list(example())
    args[-1][0][0] = action
    with pytest.raises(ValueError, match="legal action"):
        action_compatible_gates(*args)


def test_unavailable_original_target_is_not_silently_compatible():
    args = list(example())
    args[-1][0][0] = None
    row = action_compatible_gates(*args)[0]
    assert row["eligible"] and row["requires_action_retargeting"]


def test_information_cannot_forget_an_earlier_distinction():
    args = list(example())
    args[0] = np.array([["a", "same"], ["b", "same"]])
    with pytest.raises(ValueError, match="refining"):
        action_compatible_gates(*args)


def compare_fixture(monkeypatch, *, raw_cue):
    keys, phases, entries, passed, times, admitted, actions = example()
    groups = [
        dict(
            scene_id=str(i),
            scene=dict(beam_collision_enabled=[True, True]),
            neutral_visibility=[],
            targets=[
                dict(
                    phase_tick=p,
                    features=[0.0],
                    feature_names=["unused"],
                    teacher_action=actions[i][k],
                )
                for k, p in enumerate(phases)
            ],
        )
        for i in range(2)
    ]
    monkeypatch.setattr(gate, "mask_scene_features", lambda *a: np.zeros((2, 2, 1)))
    monkeypatch.setattr(gate, "feature_keys", lambda *a: keys)
    monkeypatch.setattr(gate, "complete_schedules", lambda *a: (passed, times, admitted))
    monkeypatch.setattr(gate, "best_complete_teacher", lambda *a: dict(continuation_option_index=2))
    monkeypatch.setattr(gate, "deadline_eligibility", lambda *a: dict(current_eligible=raw_cue))
    return groups, phases, entries


def test_missing_distant_cue_does_not_preclude_common_action(monkeypatch):
    result = gate.compare(*compare_fixture(monkeypatch, raw_cue=False), reveal=1)
    assert result["counts"]["eligible"] == result["counts"]["newly_eligible"] == 2
    assert result["counts"]["requires_action_retargeting"] == 2
    assert result["counts"]["all_beam_cue_eligible"] == 0


def test_available_cues_do_not_resolve_task_information_conflict(monkeypatch):
    result = gate.compare(*compare_fixture(monkeypatch, raw_cue=True), reveal=1)
    assert result["counts"]["all_beam_cue_eligible"] == 4
    assert result["counts"]["cue_eligible_without_common_success"] == 2


def test_masked_recorded_receipt_is_not_currently_usable_information(monkeypatch):
    result = gate.compare(*compare_fixture(monkeypatch, raw_cue=True), reveal=None)
    assert result["counts"]["all_beam_cue_eligible"] == 0
    assert result["counts"]["newly_eligible"] == 2


def test_same_wait_action_does_not_authorize_reusing_its_scene_wise_continuation(monkeypatch):
    groups, phases, entries = compare_fixture(monkeypatch, raw_cue=True)
    passed = np.array([[False, True, True, True]] * 2)
    times = np.array([[np.nan, 3, 1, 1]] * 2)
    monkeypatch.setattr(
        gate, "complete_schedules", lambda *a: (passed, times, np.ones_like(passed))
    )
    monkeypatch.setattr(gate, "best_complete_teacher", lambda *a: dict(continuation_option_index=3))
    early = [r for r in gate.compare(groups, phases, entries, 1)["rows"] if r["phase_tick"] == 1]
    assert all(r["common_teacher_action"] == 0 and r["original_action_compatible"] for r in early)
    assert all(not r["requires_action_retargeting"] for r in early)
    assert all(not r["original_selected_continuation_matches_common_teacher"] for r in early)
