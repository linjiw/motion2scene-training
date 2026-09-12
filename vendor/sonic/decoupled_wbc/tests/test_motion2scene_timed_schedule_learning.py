"""Finite lookahead must preserve future alternatives and phase legality."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_policy import (
    expected_feature_names,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    TimedScheduledOutcome,
    fit_timed_schedule_policy,
    schedule_layout,
    timed_schedule_teacher,
)


def bank():
    return SimpleNamespace(
        online_verified=True,
        option_ids=("neutral", "early", "middle", "late"),
        request=dict(
            max_entries_per_episode=1,
            options=[
                dict(option_id=name, entry_tick=tick)
                for name, tick in [("early", 15), ("middle", 50), ("late", 70)]
            ],
        ),
    )


def branch(index, passed=False, time=None, prefix="matched"):
    return TimedScheduledOutcome(
        str(index), index, 8731, passed, time, True, {15: prefix, 50: prefix, 70: prefix}, 1192
    )


def test_wait_uses_future_verified_schedule_and_missing_future_is_not_failure():
    b = bank()
    outcomes = [branch(0), branch(1), branch(2, True, 2.9), branch(3, True, 3.2)]
    result = timed_schedule_teacher(
        b, outcomes, 15, "matched", 8731, np.array([1, 1, 0, 0], dtype=bool)
    )
    assert result["teacher_action"] == 0
    assert result["pass_labels"][0]
    assert result["continuation_option_indices"][0] == 2
    assert result["waiting_has_future_adaptation"]
    assert result["expected_continuation_counts"][0] == 3
    missing = timed_schedule_teacher(
        b, outcomes[:-1], 15, "matched", 8731, np.array([1, 1, 0, 0], dtype=bool)
    )
    assert not missing["admitted"][0]
    assert missing["teacher_action"] is None
    mismatched = timed_schedule_teacher(
        b,
        [*outcomes[:-1], branch(3, True, 3.2, "different")],
        15,
        "matched",
        8731,
        np.array([1, 1, 0, 0], dtype=bool),
    )
    assert mismatched["teacher_action"] is None


def test_late_decision_excludes_already_entered_schedule_and_illegal_entry():
    b = bank()
    outcomes = [
        branch(0, True, 3.5),
        branch(1, True, 2.0),
        branch(2, True, 3.3),
        branch(3, True, 3.2),
    ]
    result = timed_schedule_teacher(
        b, outcomes, 50, "matched", 8731, np.array([1, 0, 1, 0], dtype=bool)
    )
    assert result["teacher_action"] == 0
    assert result["continuation_option_indices"][0] == 3
    with pytest.raises(ValueError, match="neutral decision"):
        timed_schedule_teacher(b, outcomes, 50, "matched", 8731, np.array([1, 1, 1, 0], dtype=bool))


def data():
    b = bank()
    phases, qualified, _ = schedule_layout(b)
    names = (
        expected_feature_names()[:100]
        + tuple(f"active_option_{i}" for i in range(4))
        + tuple(f"option_{i}_legal" for i in range(4))
    )
    x = np.zeros((6, len(names)))
    ticks = np.repeat(phases, 2)
    legal = np.repeat(qualified, 2, axis=0)
    passed = np.zeros((6, 4), dtype=bool)
    times = np.full((6, 4), np.nan)
    for i, tick in enumerate(ticks):
        x[i, 0] = i % 2
        x[i, names.index("phase_s")] = tick / 50
        x[i, names.index("active_option_0")] = 1
        x[i, -4:] = legal[i]
        chosen = 0 if i % 2 == 0 else np.flatnonzero(legal[i])[1]
        passed[i, chosen] = True
        times[i, chosen] = 3.0
    return b, names, x, ticks, passed, times, legal


def test_phase_fitting_never_creates_unqualified_columns():
    b, names, x, ticks, passed, times, legal = data()
    model, report = fit_timed_schedule_policy(
        b, x, names, ticks, passed, times, np.ones_like(legal), legal
    )
    np.testing.assert_array_equal(model["trained_mask"], model["qualified_mask"])
    assert model["weights"].shape == (3, 108, 4)
    assert report["supervised_decisions"] == 6
    assert all(all(v == 0 for v in r["measured_selected_regret"]) for r in report["phase_fits"])
    for phase in range(3):
        assert (model["weights"][phase][:, ~model["qualified_mask"][phase]] == 0).all()


def test_unqualified_phase_or_unmeasured_phase_cannot_create_a_head():
    b, names, x, ticks, passed, times, legal = data()
    with pytest.raises(ValueError, match="unregistered training phase"):
        fit_timed_schedule_policy(b, x, names, ticks + 1, passed, times, np.ones_like(legal), legal)
    admitted = np.ones_like(legal)
    admitted[-2:] = False
    with pytest.raises(ValueError, match="each phase requires"):
        fit_timed_schedule_policy(b, x, names, ticks, passed, times, admitted, legal)
