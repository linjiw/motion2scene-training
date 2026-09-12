from __future__ import annotations

import numpy as np
import pytest

from motion2scene.motion.route_semantics import classify_route


def arc(sign: float = 1.0) -> np.ndarray:
    angle = np.linspace(0.0, np.pi / 3.0, 120)
    radius = 3.0
    return np.column_stack([radius * np.sin(angle), sign * radius * (1.0 - np.cos(angle))])


def test_straight_route_is_valid() -> None:
    path = np.column_stack([np.linspace(0.0, 3.0, 120), np.zeros(120)])
    result = classify_route(path, "straight")
    assert result.validity_class == "valid_straight"
    assert result.net_to_path_ratio == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("sign", "label", "expected"),
    [(1.0, "gentle_left", "valid_gentle_left"), (-1.0, "gentle_right", "valid_gentle_right")],
)
def test_gentle_arcs_preserve_direction(sign: float, label: str, expected: str) -> None:
    result = classify_route(arc(sign), label)
    assert result.validity_class == expected
    assert np.sign(result.signed_heading_change_rad) == sign


def test_loop_is_rejected_even_if_it_has_net_progress() -> None:
    path = np.array(
        [[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    result = classify_route(path, "straight")
    assert result.self_intersection
    assert result.validity_class == "looping_reversal"


def test_short_motion_is_insufficient_progress() -> None:
    path = np.column_stack([np.linspace(0.0, 0.5, 20), np.zeros(20)])
    assert classify_route(path, "straight").validity_class == "insufficient_progress"


def test_bad_shape_and_unknown_route_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        classify_route(np.zeros((2, 2)), "straight")
    with pytest.raises(ValueError, match="unknown"):
        classify_route(np.zeros((3, 2)), "circle")
    with pytest.raises(ValueError, match="fps"):
        classify_route(np.zeros((3, 2)), "straight", fps=0.0)


def test_stride_scale_sway_does_not_become_route_curvature() -> None:
    progress = np.linspace(0.0, 4.0, 120)
    sway = 0.03 * np.sin(np.linspace(0.0, 16.0 * np.pi, 120))
    route = classify_route(np.column_stack([progress, sway]), "straight", fps=30.0)
    assert route.validity_class == "valid_straight"
