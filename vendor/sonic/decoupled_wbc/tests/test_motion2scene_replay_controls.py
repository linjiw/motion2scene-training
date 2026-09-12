from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_replay_controls import (
    current_supervised_error,
    replay_control_weights,
)


def records():
    return [
        dict(encounter_id=g, phase_tick=k, supervision_available=True, coverage_key=(g, k), gap=gap)
        for g, k, gap in [("a", 15, 1.0), ("a", 50, 0.0), ("b", 15, 0.0)]
    ]


def test_uniform_balances_encounters_instead_of_decision_count():
    assert replay_control_weights(records(), "uniform")["weights"] == [0.25, 0.25, 0.5]


def test_current_error_moves_mass_after_old_failure_is_learned():
    rows = records()
    historical = replay_control_weights(rows, "historical_gated")["weights"]
    refreshed = replay_control_weights(rows, "current_error", supervised_error=[0, 0, 1])["weights"]
    assert sum(historical[:2]) > historical[2]
    assert sum(refreshed[:2]) < refreshed[2]
    assert sum(refreshed) == pytest.approx(1)


def test_zero_error_recovers_exact_uniform_coverage_control():
    rows = records()
    assert replay_control_weights(rows, "current_error", supervised_error=[0, 0, 0])[
        "weights"
    ] == pytest.approx(replay_control_weights(rows, "uniform_coverage")["weights"])


def test_hybrid_does_not_turn_missing_physics_into_measured_failure():
    rows = records()
    rows[0]["gap"] = None
    assert replay_control_weights(rows, "historical_current_error", supervised_error=[1, 1, 1])[
        "weights"
    ] == pytest.approx(replay_control_weights(rows, "uniform_coverage")["weights"])
    with pytest.raises(ValueError, match="separately measured"):
        replay_control_weights(rows, "historical_ungated")


def test_prediction_error_excludes_unknown_and_all_failed_tables():
    passed = np.array([[1, 0], [1, 0], [0, 0]], bool)
    admitted = np.array([[1, 1], [1, 0], [1, 1]], bool)
    error = current_supervised_error(
        np.zeros((3, 2)),
        passed,
        [[3, np.nan], [3, np.nan], [np.nan, np.nan]],
        admitted,
        np.ones((3, 2), bool),
    )
    assert error[0] == 0.5
    assert np.isnan(error[1:]).all()
