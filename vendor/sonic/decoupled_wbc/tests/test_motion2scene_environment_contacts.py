import copy

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (
    EXTERNAL_PATHS,
    audit_environment_contacts,
)


def fixture():
    names = ["left_ankle_roll_link", "left_hip_roll_link", "left_wrist_yaw_link"]
    paths = [f"/World/envs/env_0/Robot/{name}" for name in names]
    # Intentionally use a different order for the aggregate native net sensor.
    net_names = names[::-1]
    values = np.zeros((2, 3, 9, 3))
    values[:, 0, 3, 2] = 100.0  # Allowed foot-floor support.
    values[:, 1, 2, 0] = 50.0  # Baseline hip-wrist self-contact.
    values[:, 2, 1, 0] = -50.0
    pairs = dict(
        force_w=values,
        physics_steps=np.arange(1, 3),
        physics_dt_s=0.005,
        body_names=np.asarray(names),
    )
    net = dict(
        net_force_w=values.sum(axis=2)[:, ::-1].copy(),
        physics_steps=np.arange(1, 3),
        physics_dt_s=0.005,
        body_names=np.asarray(net_names),
    )
    mapping = dict(
        pair_subject_body_names=names,
        robot_articulation_body_names=names,
        net_body_names=net_names,
        net_native_body_paths=paths[::-1],
        sensors={
            name: dict(
                sensor_body_names=[name],
                native_body_paths=[path],
                filter_paths=paths + EXTERNAL_PATHS,
                native_filter_count=9,
            )
            for name, path in zip(names, paths, strict=True)
        },
    )
    return pairs, net, mapping


def test_self_contacts_reported_and_native_order_resolved():
    pairs, net, mapping = fixture()
    result = audit_environment_contacts(
        pairs, net, mapping, 2, neutral_self_pairs=[["left_hip_roll_link", "left_wrist_yaw_link"]]
    )
    assert result["complete_synchronized_streams"]
    assert result["no_undesired_measured_contact"]
    assert result["maximum_pair_sum_net_residual_n"] == 0
    assert result["self_contacts"][0]["maximum_force_n"] == 50
    assert result["unexpected_self_contact_pairs"] == []


def test_beam_failure_is_valid_measurement_and_nonfoot_floor_is_rejected():
    for column in (3, 4, 8):
        pairs, net, mapping = fixture()
        pairs["force_w"][0, 1, column, 2] = 2.0
        net["net_force_w"] = pairs["force_w"].sum(axis=2)[:, ::-1].copy()
        result = audit_environment_contacts(pairs, net, mapping, 2)
        assert result["complete_synchronized_streams"]
        assert not result["no_undesired_measured_contact"]
        assert result["maximum_undesired_environment_force_n"] == 2
        assert result["unexpected_self_contact_pairs"]


def test_missing_clock_mapping_and_unexplained_net_never_admitted():
    for mutation in ("clock", "mapping", "net"):
        pairs, net, mapping = copy.deepcopy(fixture())
        if mutation == "clock":
            pairs["physics_steps"][1] = 1
        elif mutation == "mapping":
            mapping["sensors"]["left_hip_roll_link"]["filter_paths"] = []
        else:
            net["net_force_w"][0, 0, 0] += 3
        result = audit_environment_contacts(pairs, net, mapping, 2)
        assert not result["complete_synchronized_streams"]
        assert not result["no_undesired_measured_contact"]


def test_second_beam_is_valid_measured_failure_but_omission_is_incomplete():
    from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (
        BEAM_PATH,
    )

    pairs, net, mapping = fixture()
    two_paths = [BEAM_PATH, BEAM_PATH + "_01"]
    values = np.pad(pairs["force_w"], ((0, 0), (0, 0), (0, 1), (0, 0)))
    values[1, 1, -1, 1] = 3.0
    pairs["force_w"] = values
    net["net_force_w"] = values.sum(axis=2)[:, ::-1].copy()
    for sensor in mapping["sensors"].values():
        sensor["filter_paths"] = sensor["filter_paths"] + [two_paths[1]]
        sensor["native_filter_count"] = 10
    result = audit_environment_contacts(pairs, net, mapping, 2, beam_paths=two_paths)
    assert result["complete_synchronized_streams"]
    assert not result["no_undesired_measured_contact"]
    assert result["maximum_undesired_environment_force_n"] == 3.0
    # An omitted actual counterpart leaves normal forces unexplained, not a valid label.
    pairs["force_w"] = pairs["force_w"][:, :, :-1].copy()
    for sensor in mapping["sensors"].values():
        sensor["filter_paths"] = sensor["filter_paths"][:-1]
        sensor["native_filter_count"] = 9
    omitted = audit_environment_contacts(pairs, net, mapping, 2)
    assert not omitted["complete_synchronized_streams"]
    assert omitted["maximum_pair_sum_net_residual_n"] == 3.0


def test_beam_paths_require_explicit_consecutive_authored_names():
    import pytest

    from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (
        BEAM_PATH,
        environment_counterpart_paths,
        validated_beam_paths,
    )

    assert environment_counterpart_paths() == EXTERNAL_PATHS
    assert validated_beam_paths([BEAM_PATH, BEAM_PATH + "_01"]) == [BEAM_PATH, BEAM_PATH + "_01"]
    for invalid in ([], [BEAM_PATH + "_01"], [BEAM_PATH, BEAM_PATH + "_02"], BEAM_PATH):
        with pytest.raises(ValueError):
            validated_beam_paths(invalid)
