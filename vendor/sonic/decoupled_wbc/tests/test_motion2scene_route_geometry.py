import numpy as np
import pytest

from scripts.research.lflh_next.navigation.route_geometry import route_candidates


def test_negative_route_is_covered_and_recipes_are_invariant():
    a, r, anchors = route_candidates([[0, 0], [-8, 0]])
    assert a.shape == (225, 5)
    np.testing.assert_allclose(anchors[:, 0], [0, -2, -4, -6, -8])
    b, s, _ = route_candidates([[10, 3], [2, 3]])
    np.testing.assert_allclose(b[:, :2], a[:, :2] + [10, 3])
    np.testing.assert_array_equal(r, s)


def test_turning_route_and_rotation_equivariance():
    root = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    a, r, _ = route_candidates(root)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    b, s, _ = route_candidates(root @ rot.T)
    np.testing.assert_allclose(b[:, :2], a[:, :2] @ rot.T, atol=1e-12)
    np.testing.assert_allclose(np.cos(b[:, 4] - a[:, 4]), 0, atol=1e-12)
    np.testing.assert_allclose(np.sin(b[:, 4] - a[:, 4]), 1, atol=1e-12)
    np.testing.assert_array_equal(r, s)


def test_stationary_and_repeated_frames_remain_finite():
    a, r, _ = route_candidates([[0, 0], [0, 0]], np.pi / 2)
    assert np.isfinite(a).all()
    assert len(np.unique(a, axis=0)) == 45
    b, _, _ = route_candidates([[0, 0], [0, 0], [1, 0], [1, 0]])
    c, _, _ = route_candidates([[0, 0], [1, 0]])
    np.testing.assert_allclose(b, c)
    with pytest.raises(ValueError):
        route_candidates([[float("nan"), 0]])
