import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (
    fit_multi_option_policy,
    physical_regret,
)


def test_missing_or_infeasible_rows_are_not_negative_targets():
    passed = np.array([[False, True, True], [False, False, False], [True, True, False]])
    admitted = np.array([[True, True, True], [True, True, True], [True, False, True]])
    regret, supervised = physical_regret(
        passed,
        [[np.nan, 2, 2.5], [np.nan, np.nan, np.nan], [2, 2, np.nan]],
        admitted,
        np.ones_like(passed),
    )
    assert np.allclose(regret[0], [1, 0, 0.2])
    assert supervised.tolist() == [True, False, False]


def test_shared_success_beats_sensor_aliasing_across_three_options():
    # Option2 is the only passing option common to both identical observations.
    passed = np.array([[True, False, True], [False, True, True]])
    model, report = fit_multi_option_policy(
        [[0, 1], [0, 1]],
        ["seen", "unknown"],
        ["walk", "shallow", "deep"],
        passed,
        [[2.4, np.nan, 2.5], [np.nan, 2.4, 2.5]],
        np.ones_like(passed),
        np.ones_like(passed),
    )
    assert model["bias"].argmax() == 2
    assert report["history"][-1]["mean_expected_physical_regret"] < 0.1


def test_verified_replay_preserves_common_success_under_unequal_sampling():
    passed = np.array([[True, False, True], [False, True, True]])
    args = (
        [[0], [0]],
        ["unknown"],
        ["walk", "shallow", "deep"],
        passed,
        [[2.4, np.nan, 2.5], [np.nan, 2.4, 2.5]],
        np.ones_like(passed),
        np.ones_like(passed),
    )
    model, report = fit_multi_option_policy(*args, sample_weights=[0.2, 0.8])
    assert model["bias"].argmax() == 2
    assert report["replay_weights"] == [0.2, 0.8]
    for invalid in ([0, 1], [np.nan, 1], [1]):
        with pytest.raises(ValueError, match="replay weights"):
            fit_multi_option_policy(*args, sample_weights=invalid)
