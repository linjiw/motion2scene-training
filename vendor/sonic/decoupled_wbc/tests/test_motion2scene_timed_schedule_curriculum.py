"""Physical gap semantics and encounter weighting; fixtures are not robot evidence."""

import copy

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_timed_schedule_policy import bank_fixture
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (
    DEFAULT_RULE,
    best_complete_teacher,
    coverage_key,
    deadline_eligibility,
    encounter_replay_weights,
    unique_capture_accounting,
    verified_decision_gap,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    TimedScheduledOutcome,
    schedule_layout,
    timed_schedule_teacher,
)


def target(tick=15):
    bank = bank_fixture()
    phases, legal, _ = schedule_layout(bank)
    branches = [
        TimedScheduledOutcome(
            str(i),
            i,
            8731,
            i == 6,
            3.0 if i == 6 else None,
            True,
            {15: "same", 50: "same", 70: "same"},
            1192,
        )
        for i in range(7)
    ]
    return {
        **timed_schedule_teacher(
            bank, branches, tick, "same", 8731, legal[list(phases).index(tick)]
        ),
        "recorded_history_sha256": "same",
        "features": [0.0] * 114,
    }


def student(row, passed=True, time=5.0):
    return dict(
        source_admitted=True,
        task_outcome_admitted=True,
        neutral_phase_available=True,
        matched_history=True,
        recorded_history_sha256="same",
        features=row["features"],
        legal_mask=row["legal_mask"],
        passed=passed,
        passage_time_s=time,
        actual_selected_option_id="neutral",
        predicted_failure_probability=0.99,
    )


def timing(first_delivery):
    return [
        dict(
            entry_tick=tick,
            beams=[
                {
                    cue: {
                        "first_delivery_elapsed_s": first_delivery,
                        "delivered_by_entry": first_delivery is not None
                        and first_delivery <= (tick - 1) / 50,
                    }
                    for cue in ("surface", "underside")
                }
            ],
        )
        for tick in (15, 50, 70)
    ]


def test_wait_teacher_uses_future_continuation_but_gap_uses_actual_student_episode():
    row = target()
    best = best_complete_teacher(row)
    assert best["immediate_action_index"] == 0 and best["continuation_option_index"] == 6
    actual = student(row, passed=False, time=None)
    result = verified_decision_gap(row, actual, {"current_eligible": True})
    assert result["gap"] == 1.0  # Student chose WAIT, but its actual continuation failed.
    actual.update(passed=True, passage_time_s=5.0)
    assert verified_decision_gap(row, actual, {"current_eligible": True})["gap"] == pytest.approx(
        0.4
    )
    actual.update(passage_time_s=2.9, predicted_failure_probability=1.0)
    assert verified_decision_gap(row, actual, {"current_eligible": True})["gap"] == 0


@pytest.mark.parametrize("fault", ["expected_count", "unadmitted", "incomplete_flag"])
def test_missing_future_branch_cannot_receive_positive_priority(fault):
    row = target()
    if fault == "expected_count":
        row["admitted_continuation_counts"][0] -= 1
    elif fault == "unadmitted":
        row["admitted"][0] = False
    else:
        row["complete_legal_action_table"] = False
    assert best_complete_teacher(row) is None
    assert (
        verified_decision_gap(row, student(row, False, None), {"current_eligible": True})["gap"]
        is None
    )


@pytest.mark.parametrize(
    "fault",
    [
        "source_admitted",
        "task_outcome_admitted",
        "neutral_phase_available",
        "matched_history",
        "hash",
        "features",
        "legality",
        "missing",
    ],
)
def test_unknown_or_unmatched_student_never_gets_regret_priority(fault):
    row = target()
    actual = student(row, False, None)
    if fault == "missing":
        actual = None
    elif fault == "hash":
        actual["recorded_history_sha256"] = "other_scene_seed_or_prefix"
    elif fault == "features":
        actual["features"] = [1.0] * 114
    elif fault == "legality":
        actual["legal_mask"] = [True] * 7
    else:
        actual[fault] = False
    assert verified_decision_gap(row, actual, {"current_eligible": True})["gap"] is None


def test_known_failure_survives_unavailable_later_measurement():
    row = target()
    actual = student(row, False, None)
    actual["measurement_admitted"] = (
        False  # Earlier phase and physical failure are verified separately.
    )
    assert verified_decision_gap(row, actual, {"current_eligible": True})["gap"] == 1


def test_current_wait_and_future_entry_cues_remain_distinct():
    receipt = deadline_eligibility(timing(0.5), [True], 15, 70, DEFAULT_RULE)
    assert not receipt["current_eligible"]
    assert receipt["teacher_continuation_entry"]["eligible"]
    row = target()
    assert verified_decision_gap(row, student(row, False, None), receipt)["gap"] is None
    early = deadline_eligibility(timing(0.26), [True], 15, 70, DEFAULT_RULE)
    assert early["current_eligible"]
    rule = copy.deepcopy(DEFAULT_RULE)
    rule["sensing_margin_s"] = 0.04
    assert not deadline_eligibility(timing(0.26), [True], 15, 70, rule)["current_eligible"]


def test_empty_scene_needs_no_obstacle_cue_and_unknown_course_beam_is_retained():
    assert deadline_eligibility(timing(None), [False], 15, None, DEFAULT_RULE)["current_eligible"]
    measurements = timing(0.0)
    for phase in measurements:
        phase["beams"].append(timing(None)[0]["beams"][0])
    assert not deadline_eligibility(measurements, [True, True], 15, 70, DEFAULT_RULE)[
        "current_eligible"
    ]
    unavailable = [{"entry_tick": 15, "available": False, "beams": None}]
    assert not deadline_eligibility(unavailable, [True], 15, 70, DEFAULT_RULE)["current_eligible"]


def test_coverage_uses_declared_future_option_length_and_phase_without_sensor_features():
    row = target()
    scene = {
        "beams": [{"length_m": 0.1}, {"length_m": 0.8}],
        "beam_collision_enabled": [True, True],
    }
    key = coverage_key(bank_fixture(), scene, row, DEFAULT_RULE)
    assert key == ("sustained", (0, 2), 15, 70, 255)
    assert row["features"] == [0.0] * 114


def replay_row(group, tick, gap, key="same"):
    return dict(
        encounter_id=group,
        phase_tick=tick,
        gap=gap,
        supervision_available=True,
        coverage_key=(key,),
    )


def test_three_wait_phases_do_not_triple_encounter_mass():
    rows = [replay_row("one_phase", 15, 1)] + [
        replay_row("three_phases", t, 1) for t in (15, 50, 70)
    ]
    report = encounter_replay_weights(rows)
    assert report["encounter_masses"] == pytest.approx({"one_phase": 0.5, "three_phases": 0.5})
    assert sum(report["weights"]) == pytest.approx(1)
    assert np.sum(report["components"]["gap"]) == pytest.approx(0.6)


def test_unknown_gaps_have_no_priority_and_fallback_is_labeled_separately():
    rows = [replay_row("known", 15, 1), replay_row("unknown", 15, None)]
    report = encounter_replay_weights(rows)
    assert report["components"]["gap"][1] == 0
    assert report["weights"] == pytest.approx([0.8, 0.2])
    rows[0]["gap"] = 0
    report = encounter_replay_weights(rows)
    assert report["components"]["gap"] == [0, 0]
    assert sum(report["components"]["fallback"]) == pytest.approx(0.6)
    assert report["weights"] == pytest.approx([0.5, 0.5])


def test_unavailable_supervision_is_preserved_without_a_positive_training_weight():
    rows = [replay_row("valid", 15, None), replay_row("unsolved", 15, None)]
    rows[1].update(supervision_available=False, coverage_key=None)
    report = encounter_replay_weights(rows)
    assert report["weights"] == pytest.approx([1, 0])
    assert report["excluded_rows"] == [1]
    with pytest.raises(ValueError, match="one replay row"):
        encounter_replay_weights(rows + [rows[0]])


def test_source_identity_deduplicates_shared_capture_but_not_identical_independent_bytes(tmp_path):
    one = tmp_path / "one.pkl"
    one.write_text("same")
    alias = tmp_path / "alias.pkl"
    alias.symlink_to(one)
    two = tmp_path / "two.pkl"
    two.write_text("same")
    rows = [
        dict(path=str(path), sha256="same_hash", physics_steps=1192, role=role)
        for path, role in ((one, "teacher"), (alias, "student"), (two, "student"))
    ]
    report = unique_capture_accounting(rows)
    assert report["unique_recorded_captures"] == 2
    assert report["unique_recorded_physics_steps"] == 2384
    rows[1]["physics_steps"] = 1188
    with pytest.raises(ValueError, match="conflicting"):
        unique_capture_accounting(rows)


def test_explicit_unavailable_phase_retains_no_invented_teacher():
    row = dict(
        phase_tick=70,
        available=False,
        complete_legal_action_table=False,
        features=None,
        recorded_history_sha256=None,
        pass_labels=None,
        legal_mask=None,
    )
    assert best_complete_teacher(row) is None
    result = verified_decision_gap(row, None, {"current_eligible": False})
    assert result["gap"] is None and "no_complete_feasible_teacher" in result["reasons"]


def test_zero_gap_keeps_coverage_mass_for_underrepresented_bins():
    # Three encounters: two share one coverage bin and one occupies a second.
    # Each bin has equal coverage mass, so encounter coverage is (1/4,1/4,1/2).
    rows = [
        dict(
            encounter_id=name,
            phase_tick=15,
            gap=gap,
            supervision_available=True,
            coverage_key=coverage,
        )
        for name, gap, coverage in (
            ("a", None, "shared"),
            ("b", 0.0, "shared"),
            ("c", None, "rare"),
        )
    ]
    result = encounter_replay_weights(rows)
    expected = 0.8 * np.array([1 / 3, 1 / 3, 1 / 3]) + 0.2 * np.array([1 / 4, 1 / 4, 1 / 2])
    np.testing.assert_allclose(result["weights"], expected, rtol=0, atol=1e-15)
    assert result["no_positive_gap_fallback"] is True
    assert result["distribution"]["component_masses"] == pytest.approx(
        dict(uniform=0.2, coverage=0.2, gap=0.0, fallback=0.6)
    )
    assert result["weights"][2] > result["weights"][0]
