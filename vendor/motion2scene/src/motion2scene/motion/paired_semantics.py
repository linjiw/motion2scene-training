"""Paired S0--S4 evidence for height- and width-reduction events.

The target is always compared with the same-seed, same-route walk. This module consumes
path-aligned scalar envelopes; exact G1 body-envelope extraction remains a separate upstream
step. It therefore cannot silently substitute root height for whole-body height or world-y
extent for route-normal width.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from motion2scene.dataset.schema import SemanticStatus

SEMANTIC_PREDICATE_VERSION = "paired_functional_reduction_v2"


@dataclass(frozen=True)
class PairedSemanticPolicy:
    minimum_effect: float
    activation_fraction: float = 0.50
    boundary_guard_progress: float = 0.05
    minimum_event_duration_progress: float = 0.05
    maximum_event_duration_progress: float = 0.70
    minimum_controller_retention_ratio: float = 0.60
    minimum_event_iou: float = 0.30


@dataclass(frozen=True)
class PairedSemanticResult:
    semantic_predicate_version: str
    semantic_status: SemanticStatus
    peak_reduction: float
    event_interval: tuple[float, float] | None
    event_duration_progress: float | None
    route_valid: bool
    reason: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["semantic_status"] = self.semantic_status.value
        return payload


def _validated_series(
    route_progress: np.ndarray,
    target_envelope: np.ndarray,
    matched_walk_envelope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    progress = np.asarray(route_progress, dtype=np.float64)
    target = np.asarray(target_envelope, dtype=np.float64)
    walk = np.asarray(matched_walk_envelope, dtype=np.float64)
    if progress.ndim != 1 or target.shape != progress.shape or walk.shape != progress.shape:
        raise ValueError("progress, target, and matched-walk envelopes must be equal 1-D arrays")
    if len(progress) < 3 or not np.isfinite(np.column_stack([progress, target, walk])).all():
        raise ValueError("paired semantic profiles require at least 3 finite samples")
    if np.any(np.diff(progress) < 0.0) or progress[0] < 0.0 or progress[-1] > 1.0:
        raise ValueError("route progress must be monotonic within [0, 1]")
    return progress, walk - target


def assess_paired_reduction(
    route_progress: np.ndarray,
    target_envelope: np.ndarray,
    matched_walk_envelope: np.ndarray,
    *,
    route_valid: bool,
    policy: PairedSemanticPolicy,
) -> PairedSemanticResult:
    """Assign S0--S3 to a reference reduction event.

    Positive ``matched_walk - target`` means the target is lower or narrower. S4 requires an
    achieved-state profile and is assigned by :func:`assess_controller_retention`.
    """

    progress, reduction = _validated_series(route_progress, target_envelope, matched_walk_envelope)
    peak = float(reduction.max())
    if peak < policy.minimum_effect:
        return PairedSemanticResult(
            semantic_predicate_version=SEMANTIC_PREDICATE_VERSION,
            semantic_status=SemanticStatus.ABSENT,
            peak_reduction=peak,
            event_interval=None,
            event_duration_progress=None,
            route_valid=route_valid,
            reason="paired functional effect is below the frozen minimum",
        )

    active = reduction >= policy.minimum_effect * policy.activation_fraction
    indices = np.flatnonzero(active)
    onset = float(progress[indices[0]])
    recovery = float(progress[indices[-1]])
    duration = recovery - onset
    localized = (
        onset >= policy.boundary_guard_progress
        and recovery <= 1.0 - policy.boundary_guard_progress
        and policy.minimum_event_duration_progress
        <= duration
        <= policy.maximum_event_duration_progress
        and not active[0]
        and not active[-1]
    )
    status = SemanticStatus.ELICITED
    reason = "functional change was elicited but is not a finite recovered event"
    if localized:
        status = SemanticStatus.LOCALIZED
        reason = "paired functional change has an onset and recovery"
    if localized and route_valid:
        status = SemanticStatus.ROUTE_ALIGNED
        reason = "localized paired event occurs on a valid route"

    return PairedSemanticResult(
        semantic_predicate_version=SEMANTIC_PREDICATE_VERSION,
        semantic_status=status,
        peak_reduction=peak,
        event_interval=(onset, recovery),
        event_duration_progress=duration,
        route_valid=route_valid,
        reason=reason,
    )


def _interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0.0 else 0.0


def assess_controller_retention(
    reference: PairedSemanticResult,
    achieved: PairedSemanticResult,
    *,
    policy: PairedSemanticPolicy,
) -> PairedSemanticResult:
    """Promote S3 to S4 only when execution retains magnitude and event location."""

    if reference.semantic_status != SemanticStatus.ROUTE_ALIGNED:
        raise ValueError("controller retention requires an S3 route-aligned reference")
    if reference.event_interval is None:
        raise ValueError("route-aligned reference is missing its event interval")
    if achieved.semantic_status != SemanticStatus.ROUTE_ALIGNED or achieved.event_interval is None:
        return achieved
    retained = achieved.peak_reduction / max(reference.peak_reduction, 1e-12)
    overlap = _interval_iou(reference.event_interval, achieved.event_interval)
    if retained < policy.minimum_controller_retention_ratio or overlap < policy.minimum_event_iou:
        return PairedSemanticResult(
            semantic_predicate_version=SEMANTIC_PREDICATE_VERSION,
            semantic_status=SemanticStatus.ROUTE_ALIGNED,
            peak_reduction=achieved.peak_reduction,
            event_interval=achieved.event_interval,
            event_duration_progress=achieved.event_duration_progress,
            route_valid=achieved.route_valid,
            reason=(
                f"execution retained {retained:.1%} magnitude and {overlap:.1%} event IoU; "
                "S4 thresholds were not both met"
            ),
        )
    return PairedSemanticResult(
        semantic_predicate_version=SEMANTIC_PREDICATE_VERSION,
        semantic_status=SemanticStatus.CONTROLLER_RETAINED,
        peak_reduction=achieved.peak_reduction,
        event_interval=achieved.event_interval,
        event_duration_progress=achieved.event_duration_progress,
        route_valid=achieved.route_valid,
        reason=f"execution retained {retained:.1%} magnitude with {overlap:.1%} event IoU",
    )
