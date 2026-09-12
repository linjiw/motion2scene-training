import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_measurements import (
    audit_whole_body_contacts,
)


def capture():
    net = np.zeros((8, 3, 3))
    net[:, 1:, 2] = [100, 200]
    return {
        "net_force_w": net,
        "left_floor_force_w": net[:, 1:2].copy(),
        "right_floor_force_w": net[:, 2:3].copy(),
        "body_names": ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"],
        "physics_steps": np.arange(1, 9),
        "control_steps": np.array([4, 8]),
    }


def test_support_is_allowed_but_nonfoot_and_other_foot_contact_are_retained():
    values = capture()
    audit = audit_whole_body_contacts(values, 8)
    assert audit["complete_synchronized_streams"]
    assert audit["no_undesired_measured_contact"]
    values["net_force_w"][3, 0, 2] = 10
    assert not audit_whole_body_contacts(values, 8)["no_undesired_measured_contact"]
    values = capture()
    values["net_force_w"][3, 1, 0] = 5
    audit = audit_whole_body_contacts(values, 8)
    assert audit["maximum_foot_force_unexplained_by_floor_n"] == 5
    assert not audit["no_undesired_measured_contact"]


def test_missing_physics_or_wrong_control_clock_cannot_be_admitted():
    values = capture()
    assert not audit_whole_body_contacts(values, 12)["complete_synchronized_streams"]
    values["control_steps"] = np.array([3, 8])
    assert not audit_whole_body_contacts(values, 8)["complete_synchronized_streams"]
