"""Explicit command curricula and sampling over recorded reference phase."""

import math

import numpy as np
import torch

from gear_sonic.research.scene_distillation.commands import COMMAND_DIM, PROFILES


def curriculum_weights(stages, step):
    """Select a stage by absolute, zero-based optimizer update; retain last stage."""
    previous = 0
    for stage in stages:
        end = stage["until_step"]
        weights = stage["weights"]
        if not isinstance(end, int) or end <= previous:
            raise ValueError("Curriculum boundaries must be increasing positive integers")
        if not weights or set(weights) - set(PROFILES):
            raise ValueError("Unknown or empty command profile mixture")
        if (
            any(not math.isfinite(w) or w < 0 for w in weights.values())
            or sum(weights.values()) <= 0
        ):
            raise ValueError("Command weights must be finite, nonnegative and nonzero")
        previous = end
    if not stages or step < 0:
        raise ValueError("A curriculum and nonnegative update are required")
    return next((s["weights"] for s in stages if step < s["until_step"]), stages[-1]["weights"])


def curriculum_mask(available, weights, generator=None):
    if available.ndim != 2 or available.shape[1] != COMMAND_DIM or available.dtype != torch.bool:
        raise ValueError("Expected B,79 availability")
    curriculum_weights([{"until_step": 1, "weights": weights}], 0)
    names = list(weights)
    probabilities = torch.tensor([weights[k] for k in names], device=available.device)
    chosen = torch.multinomial(probabilities.float(), len(available), True, generator=generator)
    masks = torch.zeros(len(names), COMMAND_DIM, dtype=torch.bool, device=available.device)
    for i, name in enumerate(names):
        masks[i, list(PROFILES[name])] = True
    return masks[chosen] & available


class PhaseBalancedSampler:
    """Balance episodes, then their occupied reference quarters, then recorded rows.

    Missing quarters stay missing. Phase must be recorded before query filtering;
    estimating it from the retained row index would turn prefixes into full clips.
    """

    def __init__(self, episodes):
        self.bins = []
        for episode in episodes:
            phase = episode.get("reference_phase")
            if phase is None or phase.shape != (len(episode["proprio"]),):
                raise ValueError("Phase sampling requires per-row original reference_phase")
            phase = phase.cpu().numpy()
            if not np.isfinite(phase).all() or np.any((phase < 0) | (phase > 1)):
                raise ValueError("Reference phase must be finite and within [0,1]")
            quarters = np.minimum((phase * 4).astype(int), 3)
            self.bins.append([np.flatnonzero(quarters == q) for q in range(4)])

    def sample(self, rng):
        episode = int(rng.integers(len(self.bins)))
        occupied = [b for b in self.bins[episode] if len(b)]
        rows = occupied[int(rng.integers(len(occupied)))]
        return episode, int(rows[int(rng.integers(len(rows)))])

    def coverage(self):
        return [[len(b) for b in bins] for bins in self.bins]
