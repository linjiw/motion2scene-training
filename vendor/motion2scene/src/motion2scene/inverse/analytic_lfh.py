"""Analytic Learning-from-Hallucination baseline for overhead constraints.

The baseline does not generate a room. It identifies the one-dimensional set of beam underside
heights for which a lower target motion clears the beam and a taller, weaker alternative strikes
it. Scene rendering may later add nuisance geometry around samples from this interval.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OverheadCriticalInterval:
    """A counterfactually critical beam-height interval for one matched motion pair."""

    target_motion_id: str
    weaker_motion_id: str
    target_reach_m: float
    weaker_reach_m: float
    safety_margin_m: float
    strike_margin_m: float
    lower_m: float
    upper_m: float

    @property
    def raw_gap_m(self) -> float:
        """Unmargined reach difference between weaker and target motions."""
        return self.weaker_reach_m - self.target_reach_m

    @property
    def width_m(self) -> float:
        """Usable interval width; negative values expose a collapsed interval."""
        return self.upper_m - self.lower_m

    @property
    def nonempty(self) -> bool:
        return self.width_m > 0.0

    def clearance_m(self, beam_underside_m: float) -> float:
        """Signed target clearance: positive means the target is below the beam."""
        return float(beam_underside_m) - self.target_reach_m

    def weaker_deficit_m(self, beam_underside_m: float) -> float:
        """Signed weaker-motion collision depth: positive means the weaker motion is too tall."""
        return self.weaker_reach_m - float(beam_underside_m)

    def contains(self, beam_underside_m: float) -> bool:
        value = float(beam_underside_m)
        return self.nonempty and self.lower_m <= value <= self.upper_m

    def sample_stratified(self, count: int, *, seed: int) -> np.ndarray:
        """Draw one uniform sample from every equal-probability interval stratum.

        Stratification gives every accepted pair balanced coverage from the target-feasibility
        boundary to the weaker-motion collision boundary. A later renderer is responsible for
        nuisance variables such as thickness, width, material, and background clutter.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count must be a positive integer")
        if not self.nonempty:
            raise ValueError("cannot sample an empty critical interval")
        rng = np.random.default_rng(seed)
        unit = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        rng.shuffle(unit)
        return self.lower_m + self.width_m * unit

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            **asdict(self),
            "raw_gap_m": self.raw_gap_m,
            "width_m": self.width_m,
            "nonempty": self.nonempty,
        }


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def solve_overhead_interval(
    *,
    target_motion_id: str,
    weaker_motion_id: str,
    target_reach_m: float,
    weaker_reach_m: float,
    safety_margin_m: float = 0.0,
    strike_margin_m: float = 0.0,
) -> OverheadCriticalInterval:
    """Solve ``target + safety < beam < weaker - strike`` without hiding failure.

    The returned record is valid even when its interval is empty. This preserves the attempted
    denominator and lets an experiment report a failed matched pair instead of silently filtering
    it. Callers must inspect :attr:`OverheadCriticalInterval.nonempty` before sampling.
    """
    if not target_motion_id.strip() or not weaker_motion_id.strip():
        raise ValueError("motion identifiers must be nonempty")
    target = _finite_nonnegative(target_reach_m, "target_reach_m")
    weaker = _finite_nonnegative(weaker_reach_m, "weaker_reach_m")
    safety = _finite_nonnegative(safety_margin_m, "safety_margin_m")
    strike = _finite_nonnegative(strike_margin_m, "strike_margin_m")
    return OverheadCriticalInterval(
        target_motion_id=target_motion_id,
        weaker_motion_id=weaker_motion_id,
        target_reach_m=target,
        weaker_reach_m=weaker,
        safety_margin_m=safety,
        strike_margin_m=strike,
        lower_m=target + safety,
        upper_m=weaker - strike,
    )
