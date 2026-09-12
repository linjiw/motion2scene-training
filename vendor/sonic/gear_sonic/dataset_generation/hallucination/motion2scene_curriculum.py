"""Solution-preserving construction and physically verified curriculum replay.

Geometry admits proposals only. Teacher targets and curriculum regret require
matched physical branches, including a verified continuation for course targets.
No scene metadata or outcome labels are part of the student's input contract.
"""

from dataclasses import dataclass

import numpy as np


def solution_preserving_screen(clearance_m, positive, negative, nominal_index=0, margin_m=0.01):
    """Robust positive clearance plus nominal contrast on a *finite* offset set.

    ``clearance_m`` has axes (proposal, offset, executed option). The caller must
    include the nominal offset and bind all forecasts to their execution history.
    This deliberately differs from requiring contrast at every perturbation.
    """
    c = np.asarray(clearance_m, dtype=float)
    if c.ndim != 3 or 0 in c.shape or not np.isfinite(c).all():
        raise ValueError("finite, nonempty proposal/offset/option clearances required")
    if (
        positive == negative
        or not 0 <= positive < c.shape[2]
        or not 0 <= negative < c.shape[2]
        or not 0 <= nominal_index < c.shape[1]
        or not np.isfinite(margin_m)
        or margin_m < 0
    ):
        raise ValueError("invalid option, nominal offset, or margin")
    positive_min = c[:, :, positive].min(axis=1)
    negative_nominal = c[:, nominal_index, negative]
    slack = np.minimum(positive_min - margin_m, -negative_nominal - margin_m)
    return {
        "eligible": slack >= 0,
        "slack_m": slack,
        "positive_min_m": positive_min,
        "negative_nominal_m": negative_nominal,
        "negative_all_offsets_blocked": (c[:, :, negative] <= -margin_m).all(axis=1),
    }


@dataclass(frozen=True)
class PhysicalBranch:
    """One recorded legal execution from a common state/history and physics seed.

    Costs use seconds, joules of absolute mechanical work, and actual switch
    counts. Missing work stays unknown; it is never imputed as zero energy.
    ``continuation_verified`` is mandatory for sequential teacher targets.
    """

    option_id: str
    state_history_id: str
    physics_seed: int
    passed: bool
    admitted: bool
    legal: bool
    rollout_steps: int
    elapsed_s: float
    switches: int
    mechanical_work_j: float | None = None
    continuation_verified: bool = False

    def __post_init__(self):
        if not self.option_id or not self.state_history_id:
            raise ValueError("option and matched-history identifiers required")
        if (
            not isinstance(self.passed, bool)
            or not isinstance(self.admitted, bool)
            or not isinstance(self.legal, bool)
            or not isinstance(self.continuation_verified, bool)
            or not isinstance(self.rollout_steps, int)
            or self.rollout_steps <= 0
            or not isinstance(self.switches, int)
            or self.switches < 0
            or not np.isfinite(self.elapsed_s)
            or self.elapsed_s < 0
        ):
            raise ValueError("invalid physical branch measurement")
        if self.mechanical_work_j is not None and (
            not np.isfinite(self.mechanical_work_j) or self.mechanical_work_j < 0
        ):
            raise ValueError("invalid measured mechanical work")


@dataclass(frozen=True)
class CostWeights:
    """Explicit measured-cost weights; no implicit adaptation penalty."""

    time: float = 1.0
    work: float = 0.0
    switches: float = 0.0

    def __post_init__(self):
        values = np.array([self.time, self.work, self.switches])
        if not np.isfinite(values).all() or (values < 0).any() or not values.any():
            raise ValueError("nonnegative, nonzero measured-cost weights required")

    def cost(self, branch):
        if self.work and branch.mechanical_work_j is None:
            raise ValueError("mechanical-work objective requires measured work on every candidate")
        return (
            self.time * branch.elapsed_s
            + (self.work * branch.mechanical_work_j if self.work else 0)
            + self.switches * branch.switches
        )


def teacher_decision(branches, weights=CostWeights(), *, sequential=False):
    """Passage first, then measured cost among verified matched branches.

    A missing branch is not a failure. No passing branch returns no target, which
    must not be translated into a stop unless a verified stopping option exists.
    """
    branches = tuple(branches)
    if not branches:
        raise ValueError("physical branches required")
    if len({b.option_id for b in branches}) != len(branches):
        raise ValueError("duplicate option branches")
    if len({(b.state_history_id, b.physics_seed) for b in branches}) != 1:
        raise ValueError("teacher branches must share recorded state/history and physics seed")
    candidates = [
        b
        for b in branches
        if b.admitted and b.legal and b.passed and (not sequential or b.continuation_verified)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda b: (weights.cost(b), b.option_id))


def verified_gap(branches, student_option, weights=CostWeights(), *, sequential=False):
    """Bounded physical regret; unknown student execution is not high regret.

    A verified failed student branch has gap 1. A more costly successful branch
    has relative cost regret below 1. This ranking keeps passage dominant.
    """
    branches = tuple(branches)
    teacher = teacher_decision(branches, weights, sequential=sequential)
    student = next((b for b in branches if b.option_id == student_option), None)
    if (
        teacher is None
        or student is None
        or not student.admitted
        or not student.legal
        or (sequential and not student.continuation_verified)
    ):
        return None
    if not student.passed:
        return 1.0
    tc, sc = weights.cost(teacher), weights.cost(student)
    return float(max(0.0, (sc - tc) / max(sc, tc, 1e-12)))


def curriculum_probabilities(gaps, coverage_keys, uniform_fraction=0.2, coverage_fraction=0.2):
    """Mixture of uniform exploration, option/scene coverage, and verified regret.

    Unverified or unsolvable examples have gap ``None`` and receive no regret
    priority, but remain available to uniform/coverage sampling and diagnostics.
    ``coverage_keys`` should encode teacher option and predeclared scene bins.
    """
    if (
        len(gaps) == 0
        or len(gaps) != len(coverage_keys)
        or not np.isfinite([uniform_fraction, coverage_fraction]).all()
        or uniform_fraction <= 0
        or coverage_fraction < 0
        or uniform_fraction + coverage_fraction > 1
    ):
        raise ValueError("nonempty aligned examples and a positive uniform mixture required")
    values = np.array([0.0 if g is None else g for g in gaps], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("verified gaps must lie in [0,1], or be unknown")
    counts = {}
    for key in coverage_keys:
        counts[key] = counts.get(key, 0) + 1
    uniform = np.full(len(gaps), 1 / len(gaps))
    coverage = np.array([1 / counts[key] for key in coverage_keys])
    coverage /= coverage.sum()
    regret = values / values.sum() if values.sum() else uniform
    return (
        uniform_fraction * uniform
        + coverage_fraction * coverage
        + (1 - uniform_fraction - coverage_fraction) * regret
    )


def absolute_mechanical_work(time_s, joint_torque_nm, joint_velocity_rad_s):
    """Trapezoidal integral of sum |tau*qdot|; not electrical/battery energy."""
    t = np.asarray(time_s, dtype=float)
    torque, velocity = np.asarray(joint_torque_nm), np.asarray(joint_velocity_rad_s)
    if (
        t.ndim != 1
        or len(t) < 2
        or torque.shape != velocity.shape
        or torque.ndim != 2
        or torque.shape[0] != len(t)
        or not np.isfinite(t).all()
        or not np.isfinite(torque).all()
        or not np.isfinite(velocity).all()
        or (np.diff(t) <= 0).any()
    ):
        raise ValueError(
            "synchronized finite torque/velocity with strictly increasing time required"
        )
    power = np.abs(torque * velocity).sum(axis=1)
    return float(np.sum((power[1:] + power[:-1]) * np.diff(t) / 2))
