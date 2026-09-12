from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_sensor_perturbations import (
    PerturbedObservationStream,
    SensorPerturbation,
    corrupt_rays,
)

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    FloorCeilingHistory,
    SensorRay,
)


def rays():
    return (
        SensorRay((0, 0, 1), (1, 0, 0), 4, 1, (-1, 0, 0)),
        SensorRay((0, 0, 1), (0, 0, 1), 4, 0.3, (0, 0, -1)),
        SensorRay((0, 0, 1), (0, 0, -1), 4, 1, (0, 0, 1)),
        SensorRay((0, 0, 1), (0, 1, 0), 4),
    )


def test_zero_setting_preserves_measurements_and_history_exactly():
    original = rays()
    corrupted, metadata = corrupt_rays(original, SensorPerturbation(), 0)
    assert corrupted == original
    assert metadata["measurement_valid_mask"] == [True] * 4
    stream = PerturbedObservationStream(SensorPerturbation())
    baseline = FloorCeilingHistory()
    cache, packet = stream.push(original, 0.02)
    baseline.push(original, 0)
    assert cache["measurements"] == [asdict(r) for r in original]
    assert packet["capture_elapsed_s"] == 0
    a = baseline.snapshot((0, 0, 1), (1, 0, 0, 0), 0)
    b = stream.history.snapshot((0, 0, 1), (1, 0, 0, 0), 0)
    for key in a:
        np.testing.assert_equal(a[key], b[key])


def test_missing_channels_are_unknown_not_clear_no_hit_rays():
    stream = PerturbedObservationStream(SensorPerturbation(dropout_probability=1))
    cache, packet = stream.push(rays(), 0.02)
    observation = stream.history.snapshot((0, 0, 1), (1, 0, 0, 0), 0)
    assert cache["measurements"] == packet["measurements"] == []
    assert observation["unknown"].all()
    assert not observation["free_sampled"].any()
    valid_no_hit = FloorCeilingHistory()
    valid_no_hit.push([rays()[-1]], 0)
    assert valid_no_hit.snapshot((0, 0, 1), (1, 0, 0, 0), 0)["free_sampled"].any()


def test_dropout_and_noise_are_paired_across_strengths_without_hit_dependent_rng():
    original = rays() * 25
    low, first = corrupt_rays(original, SensorPerturbation(0.1, 0.01, seed=901), 8)
    high, second = corrupt_rays(original, SensorPerturbation(0.3, 0.02, seed=901), 8)
    assert set(first["dropout_channel_indices"]) <= set(second["dropout_channel_indices"])
    by_index = dict(zip(first["retained_channel_indices"], low, strict=True))
    for index, ray in zip(second["retained_channel_indices"], high, strict=True):
        if ray.hit_distance_m is not None:
            base = original[index].hit_distance_m
            assert ray.hit_distance_m - base == pytest.approx(
                2 * (by_index[index].hit_distance_m - base)
            )
        else:
            assert by_index[index].hit_distance_m is None


def test_latency_never_admits_a_future_packet_and_preserves_capture_pose():
    stream = PerturbedObservationStream(SensorPerturbation(latency_s=0.04))
    assert stream.push(rays(), 0.02)[1] is None
    assert stream.push(rays(), 0.04)[1] is None
    cache, delivered = stream.push(rays()[::-1], 0.06)
    assert cache["capture_elapsed_s"] == 0.04
    assert cache["delivered_capture_elapsed_s"] == 0
    assert cache["observation_age_s"] == 0.04
    assert delivered["measurements"] == [asdict(r) for r in rays()]


def test_out_of_range_noise_is_missing_instead_of_a_clipped_surface():
    original = [SensorRay((0, 0, 0), (1, 0, 0), 1, 0.5)] * 100
    perturbed, metadata = corrupt_rays(
        original, SensorPerturbation(range_noise_std_m=100, seed=91), 0
    )
    assert metadata["out_of_range_channel_indices"]
    assert len(perturbed) + len(metadata["out_of_range_channel_indices"]) == 100
    assert all(0 <= r.hit_distance_m <= 1 for r in perturbed)


@pytest.mark.parametrize(
    "settings",
    [
        dict(dropout_probability=-1),
        dict(range_noise_std_m=np.nan),
        dict(latency_s=-0.1),
        dict(seed=True),
    ],
)
def test_invalid_sensor_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        SensorPerturbation(**settings)
