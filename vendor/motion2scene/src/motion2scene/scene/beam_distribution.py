"""Factorized critical and nuisance distributions for overhead beams."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from motion2scene.inverse import OverheadCriticalInterval


def _validate_range(bounds: tuple[float, float], name: str, *, positive: bool = True) -> None:
    lower, upper = (float(value) for value in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError(f"{name} must contain finite ordered bounds")
    if positive and lower <= 0.0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class BeamNuisanceDistribution:
    """Independent rendering variables that do not decide which motion clears."""

    thickness_m: tuple[float, float] = (0.04, 0.16)
    along_route_m: tuple[float, float] = (0.20, 0.50)
    across_route_m: tuple[float, float] = (1.00, 2.00)
    yaw_jitter_rad: tuple[float, float] = (-0.08, 0.08)
    materials: tuple[str, ...] = ("painted_wood", "steel", "concrete")

    def __post_init__(self) -> None:
        _validate_range(self.thickness_m, "thickness_m")
        _validate_range(self.along_route_m, "along_route_m")
        _validate_range(self.across_route_m, "across_route_m")
        _validate_range(self.yaw_jitter_rad, "yaw_jitter_rad", positive=False)
        if not self.materials or any(not value.strip() for value in self.materials):
            raise ValueError("materials must contain nonempty labels")


@dataclass(frozen=True)
class OverheadBeamSample:
    """One renderer-ready primitive sample with criticality diagnostics."""

    sample_index: int
    beam_underside_m: float
    thickness_m: float
    along_route_m: float
    across_route_m: float
    yaw_jitter_rad: float
    material: str
    target_clearance_m: float
    weaker_deficit_m: float


def sample_overhead_beams(
    interval: OverheadCriticalInterval,
    nuisance: BeamNuisanceDistribution,
    count: int,
    *,
    seed: int,
) -> tuple[OverheadBeamSample, ...]:
    """Sample a critical height and independent nuisance parameters for each scene.

    The critical-height stream depends only on ``interval``, ``count``, and ``seed``. Changing a
    nuisance range therefore cannot silently move a scene across the decision boundary.
    """
    heights = interval.sample_stratified(count, seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence((seed, 0x4D3253)))
    thickness = rng.uniform(*nuisance.thickness_m, size=count)
    along = rng.uniform(*nuisance.along_route_m, size=count)
    across = rng.uniform(*nuisance.across_route_m, size=count)
    yaw = rng.uniform(*nuisance.yaw_jitter_rad, size=count)
    material_indices = rng.integers(0, len(nuisance.materials), size=count)
    return tuple(
        OverheadBeamSample(
            sample_index=index,
            beam_underside_m=float(height),
            thickness_m=float(thickness[index]),
            along_route_m=float(along[index]),
            across_route_m=float(across[index]),
            yaw_jitter_rad=float(yaw[index]),
            material=nuisance.materials[int(material_indices[index])],
            target_clearance_m=interval.clearance_m(float(height)),
            weaker_deficit_m=interval.weaker_deficit_m(float(height)),
        )
        for index, height in enumerate(heights)
    )
