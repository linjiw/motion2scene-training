from dataclasses import replace

import numpy as np
import pytest

from motion2scene.inverse import solve_overhead_interval
from motion2scene.scene import BeamNuisanceDistribution, sample_overhead_beams


def interval():
    return solve_overhead_interval(
        target_motion_id="crouch",
        weaker_motion_id="walk",
        target_reach_m=1.20,
        weaker_reach_m=1.30,
        safety_margin_m=0.01,
        strike_margin_m=0.01,
    )


def test_samples_keep_criticality_and_nuisance_in_support():
    nuisance = BeamNuisanceDistribution()
    samples = sample_overhead_beams(interval(), nuisance, 12, seed=5)

    assert len(samples) == 12
    assert all(sample.target_clearance_m >= 0.01 for sample in samples)
    assert all(sample.weaker_deficit_m >= 0.01 for sample in samples)
    assert all(nuisance.thickness_m[0] <= sample.thickness_m <= nuisance.thickness_m[1] for sample in samples)
    assert all(sample.material in nuisance.materials for sample in samples)


def test_nuisance_changes_do_not_change_critical_height_stream():
    first = sample_overhead_beams(interval(), BeamNuisanceDistribution(), 8, seed=17)
    changed = replace(BeamNuisanceDistribution(), thickness_m=(0.20, 0.30))
    second = sample_overhead_beams(interval(), changed, 8, seed=17)

    assert np.array_equal(
        [sample.beam_underside_m for sample in first],
        [sample.beam_underside_m for sample in second],
    )
    assert not np.array_equal(
        [sample.thickness_m for sample in first],
        [sample.thickness_m for sample in second],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thickness_m", (0.0, 0.1)),
        ("along_route_m", (0.5, 0.2)),
        ("across_route_m", (1.0, np.inf)),
        ("yaw_jitter_rad", (0.1, -0.1)),
        ("materials", ()),
    ],
)
def test_rejects_invalid_nuisance_distributions(field, value):
    with pytest.raises(ValueError):
        replace(BeamNuisanceDistribution(), **{field: value})
