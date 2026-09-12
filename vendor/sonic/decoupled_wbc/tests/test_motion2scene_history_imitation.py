import hashlib

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_closed_loop_policy import (
    choose_history_skill,
    load_history_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_history_imitation import (
    fit_history_policy,
    physical_targets,
)


def test_physical_weights_do_not_penalize_adaptation_arbitrarily():
    passed = np.array([[False, True], [True, True], [False, False], [True, True]])
    times = [[np.nan, 3], [2, 2.5], [np.nan, np.nan], [2, 2]]
    labels, weights = physical_targets(passed, times, np.ones_like(passed))
    assert labels.tolist() == [1, 0, 0, 0]
    assert np.allclose(weights, [1, 0.2, 0, 0])


def test_aliasing_learns_shared_successful_option_and_round_trips_runtime(tmp_path):
    # Identical sensor inputs: an inexpensive extra adaptation is outweighed by
    # the physically verified failure of walking in an indistinguishable scene.
    passed = np.array([[True, True], [False, True]])
    model, report = fit_history_policy(
        [[0, 1], [0, 1]],
        ["observed", "unknown"],
        passed,
        [[2.4, 2.5], [np.nan, 2.5]],
        np.ones_like(passed),
    )
    path = tmp_path / "policy.npz"
    np.savez_compressed(path, **model)
    loaded = load_history_policy(path, hashlib.sha256(path.read_bytes()).hexdigest())
    action, _ = choose_history_skill(
        "learned", ["observed", "unknown"], [0, 1], [True, True], 0, 0.3, loaded
    )
    assert action == 1
    assert report["consequential_supervised_rows"] == 2
