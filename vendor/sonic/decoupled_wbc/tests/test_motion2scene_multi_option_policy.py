import hashlib

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_policy import (
    choose_option,
    legal_option_mask,
    load_multi_policy,
)


def test_only_qualified_entry_times_are_offered():
    ticks = [[0.3], [0.2, 0.3, 0.4], [0.3]]
    assert legal_option_mask(0, 0.2, [0, 0, 0], [0, 0, 0], ticks).tolist() == [True, True, False]
    assert legal_option_mask(0, 0.3, [0, 0.051, 0], [0, 0, 0], ticks).tolist() == [
        True,
        False,
        True,
    ]


def test_no_unqualified_adaptation_to_adaptation_switch():
    ticks = [[0.3]] * 4
    assert legal_option_mask(2, 0.3, [0] * 4, [0] * 4, ticks).tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert legal_option_mask(2, 3.3, [0] * 4, [0] * 4, ticks).tolist() == [True, False, True, False]
    assert legal_option_mask(2, 3.6, [0] * 4, [0] * 4, ticks).tolist() == [
        False,
        False,
        True,
        False,
    ]


def test_option_readout_masks_illegal_largest_logit_and_terminates():
    names = ("corridor_upper_hit",)
    requested, _ = choose_option(
        "always_adapt", names, np.array([1]), [True, True, False], 0, 0.3, 2
    )
    assert requested == 0
    requested, _ = choose_option(
        "always_adapt", names, np.array([1]), [True, False, True], 2, 3.3, 2
    )
    assert requested == 0


def test_multi_policy_uses_canonical_artifact_hash_and_registry_order(tmp_path):
    path = tmp_path / "policy.npz"
    np.savez_compressed(
        path,
        schema_version="motion2scene_history_multioption_v1",
        feature_names=np.array(["phase_s"]),
        option_ids=np.array(["neutral", "d085"]),
        classes=np.arange(2),
        mean=np.zeros(1),
        std=np.ones(1),
        weights=np.zeros((1, 2)),
        bias=np.zeros(2),
    )
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_multi_policy(path, digest, ["neutral", "d085"])["weights"].shape == (1, 2)
    with pytest.raises(ValueError, match="registry"):
        load_multi_policy(path, digest, ["neutral", "d040"])
