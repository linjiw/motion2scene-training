"""Two-beam contact evidence must retain mapped columns and worst substeps."""

import copy

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_course_measurements import (
    synchronize_pair_beam_forces,
)

BEAMS = ["/World/ground/terrain/CounterfactualBeam", "/World/ground/terrain/CounterfactualBeam_01"]


def fixture():
    names = ["head", "torso"]
    paths = ["/World/envs/env_0/Robot/head", "/World/ground/terrain/Structure/Floor", *BEAMS]
    pair = dict(
        force_w=np.zeros((8, 2, 4, 3)),
        physics_steps=np.arange(1, 9),
        physics_dt_s=0.005,
        body_names=np.array(names),
    )
    mapping = dict(
        pair_subject_body_names=names,
        sensors={
            name: dict(
                sensor_body_names=[name],
                native_body_paths=[f"/World/envs/env_0/Robot/{name}"],
                filter_paths=paths.copy(),
                native_filter_count=4,
            )
            for name in names
        },
    )
    return pair, mapping


def test_second_beam_contact_survives_vector_cancellation_and_output_order():
    pair, mapping = fixture()
    pair["force_w"][0, 0, 3] = [12, 0, 0]
    pair["force_w"][1, 0, 3] = [-12, 0, 0]
    values, receipt = synchronize_pair_beam_forces(pair, mapping, BEAMS, np.array([4, 8]))
    assert values.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(values[0, 1, 0], [12, 0, 0])
    assert receipt["maximum_force_n_by_beam"] == [0, 12]
    reverse, _ = synchronize_pair_beam_forces(pair, mapping, BEAMS[::-1], np.array([4, 8]))
    np.testing.assert_array_equal(reverse[:, 0], values[:, 1])


def test_native_filter_permutation_is_resolved_from_each_body_mapping():
    pair, mapping = fixture()
    pair["force_w"][2, 1, 3] = [0, 7, 0]
    expected, _ = synchronize_pair_beam_forces(pair, mapping, BEAMS, np.array([4, 8]))
    permutation = [3, 0, 2, 1]
    pair["force_w"][:, 1] = pair["force_w"][:, 1, permutation]
    sensor = mapping["sensors"]["torso"]
    sensor["filter_paths"] = [sensor["filter_paths"][i] for i in permutation]
    actual, _ = synchronize_pair_beam_forces(pair, mapping, BEAMS, np.array([4, 8]))
    np.testing.assert_array_equal(actual, expected)


def test_dropped_beam_and_partial_blocks_fail():
    pair, mapping = fixture()
    with pytest.raises(ValueError, match="every declared beam"):
        synchronize_pair_beam_forces(pair, mapping, BEAMS[:1], np.array([4, 8]))
    incomplete = copy.deepcopy(pair)
    incomplete["force_w"] = incomplete["force_w"][:-1]
    with pytest.raises(ValueError, match="four-substep"):
        synchronize_pair_beam_forces(incomplete, mapping, BEAMS, np.array([4]))
    with pytest.raises(ValueError, match="four-substep"):
        synchronize_pair_beam_forces(pair, mapping, BEAMS, np.array([3, 8]))
