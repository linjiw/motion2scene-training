"""Scientific checks for the post-hoc full-observation alias diagnostic."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from audit_motion2scene_submission import empirical_aliases  # noqa: E402


def test_conflicting_walk_labels_have_entropy_floor_but_common_successful_action():
    x = np.zeros((2, 214))
    y = np.array([[1.0, 1.0], [0.0, 1.0]])
    result = empirical_aliases(x, y, ["both_pass", "d040_only"])
    assert result["bce_infimum_for_identical_features"] == pytest.approx(np.log(2) / 2)
    assert result["head_error_minimum"] == 1
    # Outcome-label ambiguity is not an unavoidable passage error: action 1 succeeds in both.
    assert y[:, 1].all()


def test_proprioception_can_disambiguate_identical_rays():
    x = np.zeros((2, 214))
    x[1, 170] = 0.5
    result = empirical_aliases(x, np.array([[1.0, 1.0], [0.0, 1.0]]), ["a", "b"])
    assert result["conflicting_classes"] == []
    assert result["bce_infimum_for_identical_features"] == 0


def test_near_identical_features_do_not_justify_exact_entropy_bound():
    x = np.zeros((2, 214))
    x[1, 0] = 1e-8
    y = np.array([[1.0, 1.0], [0.0, 1.0]])
    assert empirical_aliases(x, y, ["a", "b"])["conflicting_classes"] == []
    result = empirical_aliases(x, y, ["a", "b"], precision=6)
    assert len(result["conflicting_classes"]) == 1
    assert result["bce_infimum_for_identical_features"] is None
    assert result["head_error_minimum"] is None


def test_entropy_bound_weights_label_multiplicity_and_both_heads():
    x = np.zeros((4, 214))
    x[-1, 0] = 1
    y = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    result = empirical_aliases(x, y, ["a", "b", "c", "d"])
    entropy = -(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3)
    assert result["bce_infimum_for_identical_features"] == pytest.approx(3 * entropy / 8)
    assert result["unique_observations"] == 2
