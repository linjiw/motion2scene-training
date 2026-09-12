"""Curriculum boundaries, command availability and censored temporal coverage."""

import numpy as np
import pytest
import torch

from gear_sonic.research.scene_distillation.curriculum import (
    PhaseBalancedSampler,
    curriculum_mask,
    curriculum_weights,
)


def test_curriculum_absolute_boundaries_and_availability():
    stages = [
        {"until_step": 10, "weights": {"full": 1}},
        {"until_step": 20, "weights": {"navigation": 1}},
    ]
    assert curriculum_weights(stages, 9) == {"full": 1}
    assert curriculum_weights(stages, 10) == {"navigation": 1}
    assert curriculum_weights(stages, 200) == {"navigation": 1}
    available = torch.ones(8, 79, dtype=torch.bool)
    available[:, 6] = False
    mask = curriculum_mask(available, {"navigation": 1})
    assert mask.sum() == 24
    assert mask[:, [2, 3, 5]].all()
    assert not mask[:, 6].any()
    for weights in ({"typo": 1}, {"full": float("nan")}, {"full": -1}, {"full": 0}):
        with pytest.raises(ValueError):
            curriculum_weights([{"until_step": 10, "weights": weights}], 0)
    with pytest.raises(ValueError):
        curriculum_weights(stages[::-1], 0)


def test_phase_sampler_balances_occupied_quarters_without_inventing_late_rows():
    phases = torch.tensor([0.01] * 90 + [0.3] * 10)
    sampler = PhaseBalancedSampler([{"proprio": torch.zeros(100, 930), "reference_phase": phases}])
    assert sampler.coverage() == [[90, 10, 0, 0]]
    rng = np.random.default_rng(12)
    rows = [sampler.sample(rng)[1] for _ in range(4000)]
    assert 1800 < sum(row >= 90 for row in rows) < 2200
    assert max(rows) < 100
    with pytest.raises(ValueError):
        PhaseBalancedSampler([{"proprio": torch.zeros(100, 930)}])
    with pytest.raises(ValueError):
        PhaseBalancedSampler(
            [{"proprio": torch.zeros(1, 930), "reference_phase": torch.tensor([1.1])}]
        )
