"""Reference-to-achieved route-retention measurements.

Absolute route validity and controller retention are different measurements.  This module
compares a tracked root path with its own reference path without assigning a pass/fail label.
Only the initial translation is removed: rotation, scale, and endpoint error remain visible.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from motion2scene.motion.route_semantics import (
    DEFAULT_ROUTE_POLICY,
    RoutePolicy,
    RouteResult,
    classify_route,
)

ROUTE_RETENTION_VERSION = "reference_achieved_route_retention_v1_descriptive"
ROUTE_RETENTION_GATE_VERSION = "relative_route_retention_v2_heldout"


@dataclass(frozen=True)
class RouteRetentionPolicy:
    """Frozen physical tolerances for held-out reference-relative validation."""

    min_path_length_ratio: float = 0.90
    max_path_length_ratio: float = 1.10
    min_net_displacement_ratio: float = 0.90
    max_net_displacement_ratio: float = 1.10
    max_endpoint_error_m: float = 0.35
    max_abs_signed_heading_error_rad: float = math.radians(20.0)
    max_cross_track_rmse_m: float = 0.10
    max_cross_track_abs_error_m: float = 0.20
    min_monotonic_progress_fraction: float = 0.90


DEFAULT_ROUTE_RETENTION_POLICY = RouteRetentionPolicy()


@dataclass(frozen=True)
class RouteRetentionResult:
    """Descriptive route differences after initial-translation alignment."""

    route_retention_version: str
    alignment: str
    station_count: int
    reference_route: RouteResult
    achieved_route: RouteResult
    initial_offset_m: float
    path_length_ratio: float
    net_displacement_ratio: float
    signed_heading_error_rad: float
    total_absolute_curvature_excess_rad: float
    endpoint_error_m: float
    endpoint_along_chord_error_m: float
    endpoint_cross_chord_error_m: float
    route_shape_rmse_m: float
    route_shape_max_error_m: float
    along_track_rmse_m: float
    along_track_max_abs_error_m: float
    cross_track_rmse_m: float
    cross_track_max_abs_error_m: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["reference_route"] = self.reference_route.to_dict()
        payload["achieved_route"] = self.achieved_route.to_dict()
        return payload


@dataclass(frozen=True)
class RouteRetentionDecision:
    """Fail-closed decision under the held-out relative-route protocol."""

    route_retention_gate_version: str
    retained: bool
    checks: dict[str, bool]
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_path(path_xy: np.ndarray, name: str) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        raise ValueError(f"{name} must have shape [frames>=3, 2]")
    if not np.isfinite(path).all():
        raise ValueError(f"{name} contains non-finite values")
    return path


def _resample_normalized_arclength(path_xy: np.ndarray, stations: np.ndarray) -> np.ndarray:
    segment_length = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_length)])
    if arclength[-1] <= 1e-12:
        raise ValueError("route path has zero arclength")
    keep = np.concatenate([[True], np.diff(arclength) > 1e-12])
    arclength = arclength[keep] / arclength[-1]
    path = path_xy[keep]
    return np.column_stack(
        [np.interp(stations, arclength, path[:, axis]) for axis in range(2)]
    )


def _local_reference_frame(reference_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.gradient(reference_path, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    chord = reference_path[-1] - reference_path[0]
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm <= 1e-12:
        raise ValueError("reference route has zero net displacement")
    chord = chord / chord_norm
    valid = norm > 1e-12
    tangent[valid] /= norm[valid, None]
    tangent[~valid] = chord
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal


def _safe_ratio(numerator: float, denominator: float, name: str) -> float:
    if denominator <= 1e-12:
        raise ValueError(f"reference {name} is zero")
    return numerator / denominator


def compare_routes(
    reference_path_xy: np.ndarray,
    achieved_path_xy: np.ndarray,
    *,
    expected_route: str,
    reference_fps: float,
    achieved_fps: float,
    policy: RoutePolicy = DEFAULT_ROUTE_POLICY,
    station_count: int = 101,
) -> RouteRetentionResult:
    """Compare achieved and reference root routes without defining an acceptance threshold.

    Paths are translated independently to begin at the origin, then sampled at normalized
    arclength.  No rotation, scale, endpoint, or nonlinear shape alignment is applied.
    """

    reference = _validate_path(reference_path_xy, "reference_path_xy")
    achieved = _validate_path(achieved_path_xy, "achieved_path_xy")
    if station_count < 3:
        raise ValueError("station_count must be at least 3")
    if not math.isfinite(reference_fps) or reference_fps <= 0.0:
        raise ValueError("reference_fps must be positive and finite")
    if not math.isfinite(achieved_fps) or achieved_fps <= 0.0:
        raise ValueError("achieved_fps must be positive and finite")

    reference_route = classify_route(
        reference,
        expected_route,
        fps=reference_fps,
        policy=policy,
    )
    achieved_route = classify_route(
        achieved,
        expected_route,
        fps=achieved_fps,
        policy=policy,
    )

    initial_offset = achieved[0] - reference[0]
    reference = reference - reference[0]
    achieved = achieved - achieved[0]
    stations = np.linspace(0.0, 1.0, station_count)
    reference_sampled = _resample_normalized_arclength(reference, stations)
    achieved_sampled = _resample_normalized_arclength(achieved, stations)
    route_error = achieved_sampled - reference_sampled
    tangent, normal = _local_reference_frame(reference_sampled)
    along_error = np.sum(route_error * tangent, axis=1)
    cross_error = np.sum(route_error * normal, axis=1)

    reference_chord = reference[-1]
    chord_tangent = reference_chord / np.linalg.norm(reference_chord)
    chord_normal = np.array([-chord_tangent[1], chord_tangent[0]])
    endpoint_error = achieved[-1] - reference[-1]

    route_error_norm = np.linalg.norm(route_error, axis=1)
    return RouteRetentionResult(
        route_retention_version=ROUTE_RETENTION_VERSION,
        alignment="independent_initial_translation_only_then_normalized_arclength",
        station_count=station_count,
        reference_route=reference_route,
        achieved_route=achieved_route,
        initial_offset_m=float(np.linalg.norm(initial_offset)),
        path_length_ratio=_safe_ratio(
            achieved_route.path_length_m,
            reference_route.path_length_m,
            "path length",
        ),
        net_displacement_ratio=_safe_ratio(
            achieved_route.net_displacement_m,
            reference_route.net_displacement_m,
            "net displacement",
        ),
        signed_heading_error_rad=(
            achieved_route.signed_heading_change_rad
            - reference_route.signed_heading_change_rad
        ),
        total_absolute_curvature_excess_rad=(
            achieved_route.total_absolute_curvature_rad
            - reference_route.total_absolute_curvature_rad
        ),
        endpoint_error_m=float(np.linalg.norm(endpoint_error)),
        endpoint_along_chord_error_m=float(endpoint_error @ chord_tangent),
        endpoint_cross_chord_error_m=float(endpoint_error @ chord_normal),
        route_shape_rmse_m=float(np.sqrt(np.mean(np.square(route_error_norm)))),
        route_shape_max_error_m=float(route_error_norm.max()),
        along_track_rmse_m=float(np.sqrt(np.mean(np.square(along_error)))),
        along_track_max_abs_error_m=float(np.abs(along_error).max()),
        cross_track_rmse_m=float(np.sqrt(np.mean(np.square(cross_error)))),
        cross_track_max_abs_error_m=float(np.abs(cross_error).max()),
    )


def assess_route_retention(
    result: RouteRetentionResult,
    *,
    policy: RouteRetentionPolicy = DEFAULT_ROUTE_RETENTION_POLICY,
) -> RouteRetentionDecision:
    """Apply the held-out relative gate without using cumulative curvature.

    The reference must still pass its separately frozen absolute route predicate.  The achieved
    path is then judged relative to that reference, so controller-scale ripple is not mistaken
    for a different commanded route.
    """

    checks = {
        "reference_route_valid": result.reference_route.validity_class.startswith("valid_"),
        "achieved_no_self_intersection": not result.achieved_route.self_intersection,
        "achieved_no_reversal": not result.achieved_route.reversal,
        "achieved_monotonic_progress": (
            result.achieved_route.monotonic_progress_fraction
            >= policy.min_monotonic_progress_fraction
        ),
        "path_length_ratio_lower": result.path_length_ratio >= policy.min_path_length_ratio,
        "path_length_ratio_upper": result.path_length_ratio <= policy.max_path_length_ratio,
        "net_displacement_ratio_lower": (
            result.net_displacement_ratio >= policy.min_net_displacement_ratio
        ),
        "net_displacement_ratio_upper": (
            result.net_displacement_ratio <= policy.max_net_displacement_ratio
        ),
        "endpoint_error": result.endpoint_error_m <= policy.max_endpoint_error_m,
        "signed_heading_error": (
            abs(result.signed_heading_error_rad) <= policy.max_abs_signed_heading_error_rad
        ),
        "cross_track_rmse": result.cross_track_rmse_m <= policy.max_cross_track_rmse_m,
        "cross_track_max": (
            result.cross_track_max_abs_error_m <= policy.max_cross_track_abs_error_m
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return RouteRetentionDecision(
        route_retention_gate_version=ROUTE_RETENTION_GATE_VERSION,
        retained=not failures,
        checks=checks,
        failure_reasons=failures,
    )
