from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_information_ablation import mask_scene_features, maximum_causal_passages


def test_mask_only_changes_named_scene_summary_before_reveal():
    names = [f"corridor_{i}" for i in range(28)] + ["state", "phase"]
    x = np.arange(2 * 3 * 30).reshape(2, 3, 30)
    masked = mask_scene_features(x, names, [15, 50, 70], 50)
    assert np.all(masked[:, 0, :28] == 0)
    np.testing.assert_array_equal(masked[:, 1:], x[:, 1:])
    np.testing.assert_array_equal(masked[:, :, 28:], x[:, :, 28:])


def test_finite_capability_cannot_use_unobserved_scene_identity():
    phases, entries = [15, 50], [None, 50, 50]
    passed = np.array([[False, True, False], [False, False, True]])
    times = np.where(passed, 3.0, np.nan)
    admitted = np.ones_like(passed)
    blind = np.array([["same", "same"], ["same", "same"]])
    visible = np.array([["same", "a"], ["same", "b"]])
    assert maximum_causal_passages(blind, phases, entries, passed, times, admitted) == 1
    assert maximum_causal_passages(visible, phases, entries, passed, times, admitted) == 2
    admitted[0, 2] = False
    with pytest.raises(ValueError, match="complete measured"):
        maximum_causal_passages(visible, phases, entries, passed, times, admitted)


def test_dynamic_program_matches_exhaustive_shared_policy_search():
    phases, entries = [15, 50, 70], [None, 15, 15, 50, 70]
    keys = np.array([["root", "a", "a1"], ["root", "a", "a2"], ["root", "b", "b1"]])
    nodes = [(k, key) for k in range(3) for key in sorted(set(keys[:, k]))]
    choices = [[0] + [a for a, tick in enumerate(entries) if tick == phases[k]] for k, _ in nodes]
    rng = np.random.default_rng(20260909)
    for _ in range(20):
        passed = rng.integers(0, 2, size=(3, 5)).astype(bool)
        times = np.where(passed, 3.0, np.nan)
        maximum = 0
        for actions in product(*choices):
            policy = dict(zip(nodes, actions, strict=True))
            count = 0
            for i in range(3):
                selected = next(
                    (policy[k, keys[i, k]] for k in range(3) if policy[k, keys[i, k]]), 0
                )
                count += int(passed[i, selected])
            maximum = max(maximum, count)
        assert (
            maximum_causal_passages(keys, phases, entries, passed, times, np.ones_like(passed))
            == maximum
        )
