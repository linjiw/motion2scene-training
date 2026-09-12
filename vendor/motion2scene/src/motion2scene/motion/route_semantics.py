"""Versioned route-semantic measurements for generated and achieved motions.

The classifier is intentionally small and deterministic. Thresholds are physical design
choices frozen before the shared-seed v2 corpus, not values fitted to that corpus. Applying it
to the existing v1 pilot is descriptive secondary analysis only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

ROUTE_PREDICATE_VERSION = "route_semantics_v2_preregistered"


@dataclass(frozen=True)
class RoutePolicy:
    """Frozen route thresholds in metres and radians."""

    min_net_displacement_m: float = 1.0
    min_net_to_path_ratio: float = 0.65
    min_monotonic_progress_fraction: float = 0.90
    reversal_excursion_m: float = 0.15
    centerline_smoothing_s: float = 0.50
    centerline_station_spacing_m: float = 0.50
    straight_max_heading_rad: float = math.radians(20.0)
    straight_max_total_curvature_rad: float = math.radians(55.0)
    gentle_min_heading_rad: float = math.radians(15.0)
    gentle_max_heading_rad: float = math.radians(105.0)
    gentle_max_total_curvature_rad: float = math.radians(125.0)


DEFAULT_ROUTE_POLICY = RoutePolicy()


@dataclass(frozen=True)
class RouteResult:
    route_predicate_version: str
    expected_route: str
    validity_class: str
    net_displacement_m: float
    path_length_m: float
    net_to_path_ratio: float
    signed_heading_change_rad: float
    total_absolute_curvature_rad: float
    monotonic_progress_fraction: float
    maximum_lateral_departure_m: float
    self_intersection: bool
    reversal: bool

    def to_dict(self) -> dict[str, str | float | bool]:
        return asdict(self)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Return whether two closed planar segments cross away from roundoff."""

    ab = b - a
    cd = d - c
    denominator = _cross(ab, cd)
    if abs(denominator) <= 1e-12:
        return False
    ac = c - a
    t = _cross(ac, cd) / denominator
    u = _cross(ac, ab) / denominator
    epsilon = 1e-9
    return epsilon < t < 1.0 - epsilon and epsilon < u < 1.0 - epsilon


def _has_self_intersection(path_xy: np.ndarray) -> bool:
    for first in range(len(path_xy) - 1):
        for second in range(first + 2, len(path_xy) - 1):
            if _segments_intersect(
                path_xy[first],
                path_xy[first + 1],
                path_xy[second],
                path_xy[second + 1],
            ):
                return True
    return False


def _path_heading_metrics(path_xy: np.ndarray) -> tuple[float, float]:
    delta = np.diff(path_xy, axis=0)
    delta = delta[np.linalg.norm(delta, axis=1) > 1e-6]
    if len(delta) < 2:
        return 0.0, 0.0
    headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    turns = np.diff(headings)
    return float(headings[-1] - headings[0]), float(np.abs(turns).sum())


def _route_centerline(path_xy: np.ndarray, fps: float, policy: RoutePolicy) -> np.ndarray:
    """Suppress gait sway, then sample headings at a physical spatial scale."""

    window = max(3, round(policy.centerline_smoothing_s * fps))
    if window % 2 == 0:
        window += 1
    window = min(window, len(path_xy) - (1 - len(path_xy) % 2))
    pad = window // 2
    padded = np.pad(path_xy, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
    )

    segment_length = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_length)])
    if arclength[-1] <= policy.centerline_station_spacing_m:
        return smoothed
    stations = np.arange(
        0.0,
        arclength[-1],
        policy.centerline_station_spacing_m,
        dtype=np.float64,
    )
    stations = np.concatenate([stations, [arclength[-1]]])
    return np.column_stack([np.interp(stations, arclength, smoothed[:, axis]) for axis in range(2)])


def _progress_metrics(path_xy: np.ndarray, policy: RoutePolicy) -> tuple[float, float, bool]:
    chord = path_xy[-1] - path_xy[0]
    distance = float(np.linalg.norm(chord))
    if distance <= 1e-12:
        return 0.0, 0.0, True
    axis = chord / distance
    projected = (path_xy - path_xy[0]) @ axis
    increments = np.diff(projected)
    absolute_progress = float(np.abs(increments).sum())
    monotonic_fraction = (
        float(np.clip(increments, 0.0, None).sum() / absolute_progress)
        if absolute_progress > 1e-12
        else 0.0
    )
    running_peak = np.maximum.accumulate(projected)
    reversal = bool(np.max(running_peak - projected) > policy.reversal_excursion_m)
    normal = np.array([-axis[1], axis[0]])
    lateral = np.abs((path_xy - path_xy[0]) @ normal)
    return monotonic_fraction, float(lateral.max()), reversal


def classify_route(
    path_xy: np.ndarray,
    expected_route: str,
    *,
    fps: float = 30.0,
    policy: RoutePolicy = DEFAULT_ROUTE_POLICY,
) -> RouteResult:
    """Measure and classify a route as straight, gentle turn, or a named failure."""

    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        raise ValueError("path_xy must have shape [frames>=3, 2]")
    if not np.isfinite(path).all():
        raise ValueError("path_xy contains non-finite values")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    if expected_route not in {"straight", "gentle_left", "gentle_right"}:
        raise ValueError(f"unknown expected route: {expected_route}")

    raw_self_intersection = _has_self_intersection(path)
    path = _route_centerline(path, fps, policy)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_length = float(steps.sum())
    displacement = float(np.linalg.norm(path[-1] - path[0]))
    ratio = displacement / path_length if path_length > 1e-12 else 0.0
    signed_heading, total_curvature = _path_heading_metrics(path)
    monotonic_fraction, lateral_departure, reversal = _progress_metrics(path, policy)
    self_intersection = raw_self_intersection or _has_self_intersection(path)

    if self_intersection or reversal or monotonic_fraction < policy.min_monotonic_progress_fraction:
        validity = "looping_reversal"
    elif displacement < policy.min_net_displacement_m or ratio < policy.min_net_to_path_ratio:
        validity = "insufficient_progress"
    elif expected_route == "straight" and (
        abs(signed_heading) <= policy.straight_max_heading_rad
        and total_curvature <= policy.straight_max_total_curvature_rad
    ):
        validity = "valid_straight"
    elif expected_route == "gentle_left" and (
        policy.gentle_min_heading_rad <= signed_heading <= policy.gentle_max_heading_rad
        and total_curvature <= policy.gentle_max_total_curvature_rad
    ):
        validity = "valid_gentle_left"
    elif expected_route == "gentle_right" and (
        -policy.gentle_max_heading_rad <= signed_heading <= -policy.gentle_min_heading_rad
        and total_curvature <= policy.gentle_max_total_curvature_rad
    ):
        validity = "valid_gentle_right"
    else:
        validity = "over_turn"

    return RouteResult(
        route_predicate_version=ROUTE_PREDICATE_VERSION,
        expected_route=expected_route,
        validity_class=validity,
        net_displacement_m=displacement,
        path_length_m=path_length,
        net_to_path_ratio=ratio,
        signed_heading_change_rad=signed_heading,
        total_absolute_curvature_rad=total_curvature,
        monotonic_progress_fraction=monotonic_fraction,
        maximum_lateral_departure_m=lateral_departure,
        self_intersection=self_intersection,
        reversal=reversal,
    )
