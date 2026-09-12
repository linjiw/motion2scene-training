import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_policy import (
    choose_option,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_value_imitation import (
    fit_value_policy,
)


def test_identical_observations_choose_shared_successful_option():
    passed = np.array([[True, False, True], [False, True, True]])
    model, report = fit_value_policy(
        [[0, 1], [0, 1]],
        ["seen", "unknown"],
        ["walk", "shallow", "deep"],
        passed,
        [[2.4, np.nan, 2.5], [np.nan, 2.4, 2.5]],
        np.ones_like(passed),
        np.ones_like(passed),
    )
    assert model["bias"].argmax() == 2
    assert [row["action"] for row in report["fitted_decisions"]] == [2, 2]
    assert np.allclose(-model["bias"], [0.5, 0.5, 0.04])


def test_incomplete_rows_do_not_change_fit_or_normalization():
    common = (["distance"], ["walk", "adapt"])
    passed = np.array([[True, True], [False, True]])
    reference, _ = fit_value_policy(
        [[0], [1]],
        *common,
        passed,
        [[2, 3], [np.nan, 3]],
        np.ones_like(passed),
        np.ones_like(passed),
    )
    model, report = fit_value_policy(
        [[0], [1], [1000]],
        *common,
        np.vstack([passed, [False, True]]),
        [[2, 3], [np.nan, 3], [np.nan, 100]],
        np.array([[True, True], [True, True], [True, False]]),
        np.ones((3, 2), dtype=bool),
    )
    for key in ("mean", "std", "weights", "bias"):
        assert np.array_equal(model[key], reference[key])
    assert report["excluded_decision_indices"] == [2]


def test_illegal_labels_are_excluded_and_runtime_masks_highest_value():
    legal = np.array([[True, True, False], [False, True, True]])
    passed = np.array([[True, True, False], [False, True, True]])
    model, report = fit_value_policy(
        [[0], [1]],
        ["distance"],
        ["walk", "shallow", "deep"],
        passed,
        [[2, 3, np.nan], [np.nan, 3, 2]],
        legal.copy(),
        legal,
    )
    changed, _ = fit_value_policy(
        [[0], [1]],
        ["distance"],
        ["walk", "shallow", "deep"],
        np.ones_like(passed),
        [[2, 3, 999], [999, 3, 2]],
        legal.copy(),
        legal,
    )
    assert np.array_equal(changed["weights"], model["weights"])
    assert np.array_equal(changed["bias"], model["bias"])
    assert [r["target_count"] for r in report["option_fits"]] == [1, 2, 1]
    model["bias"][2] = 1000  # An illegal option must still be unavailable online.
    action, _ = choose_option("learned", ["distance"], np.array([0]), legal[0], 0, 0.3, 2, model)
    assert action == 0


def test_never_observed_option_is_not_assigned_a_synthetic_failure():
    with pytest.raises(ValueError, match="no complete legal value target"):
        fit_value_policy(
            [[0]],
            ["distance"],
            ["walk", "adapt", "unseen"],
            np.array([[True, True, False]]),
            [[2, 3, np.nan]],
            np.array([[True, True, False]]),
            np.array([[True, True, False]]),
        )
