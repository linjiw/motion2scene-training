"""Test explicit authored operator semantics, independent of any learned candidate."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "scripts/research/motion2scene_project_splice_prior.py"
)
SPEC = importlib.util.spec_from_file_location("prior_splice", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_sources():
    neutral = np.zeros((180, 36))
    neutral[:, 3] = 1
    raw = neutral.copy()
    raw[:, 0] = 0.2
    raw[:, 2] = 0.8
    raw[:, 7:] = 0.3
    limits = np.tile([-1.0, 1.0], (29, 1))
    return neutral, raw, limits


def test_projection_only_changes_outside_hinges_and_exact_authored_regions():
    neutral, raw, limits = fixture_sources()
    raw[80, 7] = 1.3
    raw[81, 8] = -1.4
    result, projected, weight = MODULE.project_and_splice(neutral, raw, limits)
    delta = projected - raw
    assert np.count_nonzero(delta) == 2
    assert projected[80, 7] == 1
    assert projected[81, 8] == -1
    assert np.array_equal(result[:31], neutral[:31])
    assert np.array_equal(result[150:], neutral[150:])
    assert np.array_equal(result[42:136], projected[42:136])
    assert weight[30] == 0 and weight[42] == 1
    assert weight[135] == 1 and weight[150] == 0
    assert np.all((weight >= 0) & (weight <= 1))
    assert np.isfinite(result).all()


def test_quaternion_shortest_arc_antipodal_and_midpoint():
    neutral, raw, limits = fixture_sources()
    raw[:, 3:7] *= -1
    result, _, _ = MODULE.project_and_splice(neutral, raw, limits)
    assert np.allclose(np.linalg.norm(result[:, 3:7], axis=1), 1)
    assert np.allclose(abs(result[:, 3]), 1)
    a = np.array([[1.0, 0, 0, 0]])
    b = np.array([[0.0, 0, 0, 1]])
    halfway = MODULE.slerp_wxyz(a, b, np.array([0.5]))
    assert np.allclose(halfway, [[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
    assert MODULE.quintic(np.array([0.0, 0.5, 1.0])).tolist() == [0.0, 0.5, 1.0]


def test_reject_nonfinite_and_reversed_native_intervals():
    neutral, raw, limits = fixture_sources()
    limits[0] = [1, -1]
    with pytest.raises(ValueError, match="Reversed"):
        MODULE.project_and_splice(neutral, raw, limits)
    limits[0] = [-1, 1]
    raw[0, 7] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        MODULE.project_and_splice(neutral, raw, limits)


def test_sole_sphere_uses_native_rotation_and_radius_without_floor_correction():
    angle = np.pi / 2
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    native = {
        "global_translation": np.array([[[0.0, 0.0, 0.1]]]),
        "global_rotation_mat": rotation[None, None],
        "body_names": ["left_ankle_roll_link"],
    }
    sphere = [
        {"body": "left_ankle_roll_link", "local_center_m": [0.12, 0, -0.03], "radius_m": 0.005}
    ]
    assert np.allclose(MODULE.sole_heights(native, sphere), [[-0.025]])
