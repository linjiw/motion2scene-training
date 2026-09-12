"""Procedural scene-parameter distributions."""

from .beam_distribution import (
    BeamNuisanceDistribution,
    OverheadBeamSample,
    sample_overhead_beams,
)

__all__ = ["BeamNuisanceDistribution", "OverheadBeamSample", "sample_overhead_beams"]
