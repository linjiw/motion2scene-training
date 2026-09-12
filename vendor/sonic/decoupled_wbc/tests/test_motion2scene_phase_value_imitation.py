import copy
import hashlib

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_policy import (
    choose_option,
    load_multi_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_phase_value_imitation import (
    OPTION_IDS,
    PHASES,
    QUALIFIED,
    choose_phase_option,
    fit_phase_value_policy,
    load_phase_policy,
)


def dataset():
    phases = np.repeat(PHASES, 2)
    features = np.c_[np.tile([-1.0, 1.0], 3), phases]
    legal = np.repeat(QUALIFIED, 2, axis=0)
    passed = legal.copy()
    times = np.where(legal, 5.0, np.nan)
    expected = [0, 1, 1, 0, 0, 1]
    for row, action in enumerate(expected):
        times[row, action] = 2.0
        times[row, 1 - action] = 3.0
    return {
        "features": features,
        "feature_names": ["sensor", "phase_s"],
        "option_ids": OPTION_IDS,
        "phases_s": phases,
        "passed": passed,
        "passage_time_s": times,
        "admitted": legal.copy(),
        "legality": legal,
    }, expected


def test_phase_heads_represent_opposing_sensor_decisions_without_changing_features():
    data, expected = dataset()
    model, report = fit_phase_value_policy(**data)
    actions = [
        choose_phase_option(model, data["feature_names"], x, legal, phase)[0]
        for x, legal, phase in zip(
            data["features"], data["legality"], data["phases_s"], strict=True
        )
    ]
    assert actions == expected
    assert np.array_equal(model["trained_mask"], QUALIFIED)
    assert report["l2"] == 1e-6
    assert [row["complete_decision_indices"] for row in report["phase_fits"]] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert [len(row["option_fits"]) for row in report["phase_fits"]] == [3, 5, 3]


def test_other_phases_and_incomplete_rows_cannot_change_a_head():
    data, _ = dataset()
    reference, _ = fit_phase_value_policy(**data)
    changed = copy.deepcopy(data)
    changed["features"][2:4, 0] += 100
    changed["passage_time_s"][2:4, 0] = 9
    model, _ = fit_phase_value_policy(**changed)
    for key in ("mean", "std", "weights", "bias"):
        np.testing.assert_array_equal(model[key][[0, 2]], reference[key][[0, 2]])
    changed = copy.deepcopy(data)
    for key in ("features", "passed", "passage_time_s", "admitted", "legality"):
        changed[key] = np.concatenate([changed[key], changed[key][:1]], axis=0)
    changed["phases_s"] = np.r_[changed["phases_s"], 0.2]
    changed["features"][-1, 0] = 1e6
    changed["admitted"][-1, 4] = False
    model, report = fit_phase_value_policy(**changed)
    for key in ("mean", "std", "weights", "bias"):
        np.testing.assert_array_equal(model[key], reference[key])
    assert report["excluded_decision_indices"] == [6]


def test_illegal_labels_never_become_synthetic_failure_targets():
    data, _ = dataset()
    reference, _ = fit_phase_value_policy(**data)
    data["passed"][~data["legality"]] = True
    data["passage_time_s"][~data["legality"]] = -999
    changed, _ = fit_phase_value_policy(**data)
    for key in ("mean", "std", "weights", "bias"):
        np.testing.assert_array_equal(changed[key], reference[key])
    illegal_request = QUALIFIED[0].copy()
    illegal_request[2] = True
    with pytest.raises(ValueError, match="lacks qualification"):
        choose_phase_option(
            changed, data["feature_names"], data["features"][0], illegal_request, 0.2
        )


def test_missing_trained_action_and_unknown_phase_error_instead_of_extrapolation():
    data, _ = dataset()
    data["legality"][2:4, 3] = False
    data["admitted"][2:4, 3] = False
    model, _ = fit_phase_value_policy(**data)
    assert not model["trained_mask"][1, 3]
    with pytest.raises(ValueError, match="no trained physical value"):
        choose_phase_option(model, data["feature_names"], data["features"][2], QUALIFIED[1], 0.3)
    with pytest.raises(ValueError, match="unsupported decision phase"):
        choose_phase_option(model, data["feature_names"], data["features"][2], QUALIFIED[1], 0.32)
    data["phases_s"][2] = 0.32
    with pytest.raises(ValueError, match="unsupported decision phase"):
        fit_phase_value_policy(**data)


def test_replay_weights_change_only_measured_value_objective():
    data, _ = dataset()
    data["features"][:, 0] = 0
    weights = np.tile([1.0, 9.0], 3)
    model, report = fit_phase_value_policy(**data, sample_weights=weights)
    assert [row["action"] for row in report["fitted_decisions"]] == [1, 1, 0, 0, 1, 1]
    np.testing.assert_allclose(model["mean"][:, 0], 0)


def test_feature_phase_and_model_hash_are_bound(tmp_path):
    data, _ = dataset()
    model, _ = fit_phase_value_policy(**data)
    path = tmp_path / "phase.npz"
    np.savez_compressed(path, **model)
    sha = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_phase_policy(path, sha, OPTION_IDS)
    assert loaded["weights"].shape == (3, 2, 5)
    dispatched = load_multi_policy(path, sha, list(OPTION_IDS))
    np.testing.assert_array_equal(dispatched["weights"], loaded["weights"])
    with pytest.raises(ValueError, match="hash mismatch"):
        load_phase_policy(path, "sha256:bad", OPTION_IDS)
    mismatch = data["features"][0].copy()
    mismatch[1] = 0.3
    with pytest.raises(ValueError, match="disagrees"):
        choose_phase_option(model, data["feature_names"], mismatch, QUALIFIED[0], 0.2)


def test_runtime_dispatch_holds_and_recovers_without_learning_termination():
    data, _ = dataset()
    model, _ = fit_phase_value_policy(**data)
    names, features = data["feature_names"], data["features"][0]
    action, logits = choose_option(
        "learned", names, features, [True, False, False, False, False], 0, 0.18, 4, model
    )
    assert action == 0 and logits is None
    action, logits = choose_option(
        "learned", names, features, [True, False, False, False, True], 4, 3.3, 4, model
    )
    assert action == 0 and logits is None
    action, logits = choose_option(
        "learned", names, features, [False, False, False, False, True], 4, 3.3, 4, model
    )
    assert action == 4 and logits is None  # A refused return is not a stop.
    with pytest.raises(ValueError, match="unsupported decision phase"):
        choose_option("learned", names, features, QUALIFIED[0], 0, 0.22, 4, model)
