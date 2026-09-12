"""Development tuning uses legal measured continuations, not prediction confidence."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_tune_schedule_learner import score  # noqa: E402


def model_and_row():
    model = dict(
        phase_ticks=np.array([15]),
        trained_mask=np.array([[True, True]]),
        mean=np.zeros((1, 1)),
        std=np.ones((1, 1)),
        weights=np.zeros((1, 1, 2)),
        bias=np.array([[0.0, 10.0]]),
    )
    row = dict(
        features=[0.0],
        phase_tick=15,
        pass_labels=[True, False],
        passage_time_s=[4.0, None],
        admitted=[True, True],
        legal_mask=[True, True],
    )
    return model, row


def test_confident_failed_continuation_is_scored_as_measured_failure():
    model, row = model_and_row()
    result = score(model, [row])
    assert result["mean_teacher_regret"] == 1.0
    assert result["failure_choice_fraction"] == 1.0


def test_illegal_high_value_is_never_a_selected_continuation():
    model, row = model_and_row()
    row["pass_labels"] = [True, True]
    row["passage_time_s"] = [4.0, 5.0]
    row["legal_mask"] = [True, False]
    # A single legal continuation has no consequential comparison to tune against.
    with pytest.raises(ValueError, match="no consequential"):
        score(model, [row])


def test_cost_proxy_and_untrained_legal_option_are_distinct():
    model, row = model_and_row()
    row["pass_labels"] = [True, True]
    row["passage_time_s"] = [4.0, 5.0]
    result = score(model, [row])
    assert result["failure_choice_fraction"] == 0
    assert result["mean_teacher_regret"] > 0
    model["trained_mask"][0, 1] = False
    with pytest.raises(ValueError, match="no training evidence"):
        score(model, [row])


def test_nonfinite_heldout_features_do_not_become_an_argmax_success():
    model, row = model_and_row()
    row["features"] = [float("nan")]
    with pytest.raises(ValueError, match="finite held-out"):
        score(model, [row])
