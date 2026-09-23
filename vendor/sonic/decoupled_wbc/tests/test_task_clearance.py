import numpy as np
import pytest

from gear_sonic.research.scene_distillation.tasks import (
    MIN_START_GOAL_CLEARANCE_M,
    horizontal_clearance_m,
    obstacle_footprint,
    require_start_goal_clearance,
    shift_to_clearance,
    start_goal_clearance,
)


def box(center, yaw=0.0, size=(1.0, 0.2, 1.5), shape="box"):
    return dict(
        shape=shape,
        center_xyz=list(center),
        quaternion_wxyz=[float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
        full_dimensions_xyz=list(size),
    )


# The legacy nav-8192 00399-corridor walls (workspace/nav-8192/tasks/00399/corridor.json).
LEGACY_00399 = dict(
    start=[0.0, 0.0, 0.7367977499961853],
    goal=[-0.34931617975234985, 0.495400071144104, 0.7909403443336487],
    quaternion=[0.46028962003185403, 0.0, 0.0, 0.8877688132002224],
    centers=[
        [0.8122446358271864, 1.0507138863301606, 0.75],
        [-0.49537382722580703, 0.1286867961191802, 0.75],
    ],
)


def legacy_walls():
    return [
        dict(
            shape="box",
            center_xyz=c,
            quaternion_wxyz=LEGACY_00399["quaternion"],
            full_dimensions_xyz=[1.0, 0.2, 1.5],
        )
        for c in LEGACY_00399["centers"]
    ]


def test_box_footprint_distance_faces_corners_and_inside():
    wall = box([0.0, 0.0, 0.75])
    assert horizontal_clearance_m(wall, [0.0, 0.3]) == pytest.approx(0.2)
    assert horizontal_clearance_m(wall, [0.7, 0.0]) == pytest.approx(0.2)
    assert horizontal_clearance_m(wall, [0.8, 0.4]) == pytest.approx(np.hypot(0.3, 0.3))
    assert horizontal_clearance_m(wall, [0.1, 0.05]) == 0.0
    assert horizontal_clearance_m(wall, [0.5, 0.1]) == 0.0  # on the boundary
    # Height never matters: a pelvis at any z is compared with the projected footprint.
    assert horizontal_clearance_m(wall, [0.0, 0.3, 5.0]) == pytest.approx(0.2)


def test_rotated_box_beam_sphere_and_cylinder_footprints():
    turned = box([1.0, 2.0, 0.75], yaw=np.pi / 2)
    assert horizontal_clearance_m(turned, [1.3, 2.0]) == pytest.approx(0.2)
    assert horizontal_clearance_m(turned, [1.0, 2.7]) == pytest.approx(0.2)
    beam = box([0.0, 0.0, 1.1], size=(0.2, 1.4, 0.15), shape="beam")
    assert horizontal_clearance_m(beam, [0.5, 0.0]) == pytest.approx(0.4)
    sphere = dict(
        shape="sphere",
        center_xyz=[1.0, 0.0, 0.5],
        quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        full_dimensions_xyz=[0.4, 0.4, 0.4],
    )
    assert obstacle_footprint(sphere)[0] == "circle"
    assert horizontal_clearance_m(sphere, [0.0, 0.0]) == pytest.approx(0.8)
    upright = dict(sphere, shape="cylinder", full_dimensions_xyz=[0.4, 0.4, 1.0])
    assert horizontal_clearance_m(upright, [0.0, 0.0]) == pytest.approx(0.8)
    # Lying along x: footprint is a 1.0 x 0.4 rectangle (sampled rim, <0.1% radius error).
    lying = dict(upright, quaternion_wxyz=[np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    assert obstacle_footprint(lying)[0] == "polygon"
    assert horizontal_clearance_m(lying, [1.0, 0.5]) == pytest.approx(0.3, abs=1e-3)
    assert horizontal_clearance_m(lying, [-0.2, 0.0]) == pytest.approx(0.7, abs=1e-3)
    with pytest.raises(ValueError):
        obstacle_footprint(dict(sphere, shape="mesh"))
    with pytest.raises(ValueError):
        obstacle_footprint(box([0, 0, 0], size=(1.0, 0.0, 1.0)))


def test_legacy_00399_corridor_wall_violates_the_margin_at_start_and_goal():
    assert MIN_START_GOAL_CLEARANCE_M == 0.35
    rows = start_goal_clearance(legacy_walls(), LEGACY_00399["start"], LEGACY_00399["goal"])
    assert rows[0]["start_m"] > 1.0 and rows[0]["goal_m"] > 1.0
    assert rows[1]["start_m"] == pytest.approx(0.2307, abs=1e-3)
    assert rows[1]["goal_m"] == pytest.approx(0.2307, abs=1e-3)
    with pytest.raises(ValueError, match="obstacle 1 .box. start 0.231 m; obstacle 1 .box. goal"):
        require_start_goal_clearance(legacy_walls(), LEGACY_00399["start"], LEGACY_00399["goal"])
    ok = require_start_goal_clearance(
        legacy_walls(), LEGACY_00399["start"], LEGACY_00399["goal"], min_clearance_m=0.2
    )
    assert len(ok) == 2
    assert require_start_goal_clearance([], [0, 0, 0], [0, 0, 0]) == []
    with pytest.raises(ValueError):
        require_start_goal_clearance([], [0, 0, 0], [1, 0, 0], min_clearance_m=float("nan"))


def test_shift_to_clearance_is_minimal_whole_step_and_keeps_shape():
    wall = legacy_walls()[1]
    start, goal = LEGACY_00399["start"], LEGACY_00399["goal"]
    # Outward = away from the corridor midpoint (the mean of the two wall centers).
    direction = np.subtract(wall["center_xyz"][:2], np.mean(LEGACY_00399["centers"], 0)[:2])
    moved, shift = shift_to_clearance(wall, direction, start, goal)
    assert shift == pytest.approx(0.12)
    after = start_goal_clearance([moved], start, goal)[0]
    assert min(after["start_m"], after["goal_m"]) >= 0.35
    unit = direction / np.linalg.norm(direction)
    short = dict(wall, center_xyz=[*(np.add(wall["center_xyz"][:2], 0.11 * unit)), 0.75])
    before = start_goal_clearance([short], start, goal)[0]
    assert min(before["start_m"], before["goal_m"]) < 0.35
    assert shift_to_clearance(wall, direction, start, goal, max_shift_m=0.12)[0] == moved
    with pytest.raises(ValueError):
        shift_to_clearance(wall, direction, start, goal, max_shift_m=0.11)
    for key in ("shape", "quaternion_wxyz", "full_dimensions_xyz"):
        assert moved[key] == wall[key]
    assert moved["center_xyz"][2] == wall["center_xyz"][2]
    assert wall["center_xyz"] == LEGACY_00399["centers"][1]  # input untouched
    untouched, zero = shift_to_clearance(legacy_walls()[0], direction, start, goal)
    assert zero == 0 and untouched == legacy_walls()[0]
    with pytest.raises(ValueError):
        shift_to_clearance(wall, -direction, start, goal, max_shift_m=0.05)
    with pytest.raises(ValueError):
        shift_to_clearance(wall, [0.0, 0.0], start, goal)
