"""Coverage censoring, navigation label separation and contact qualification regressions."""

import numpy as np
import pytest

from gear_sonic.research.scene_distillation.collect import supported_prefix
from gear_sonic.research.scene_distillation.scene_qualification import score_scene
from gear_sonic.research.scene_distillation.tasks import (
    trajectory_labels,
    validate_task,
)


def test_prefix_censors_at_first_bad_state_and_native_end():
    assert supported_prefix([True, True, False, True], 4, 2).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert supported_prefix([True] * 5, 3, 2).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert not supported_prefix([True] * 5, 3, 4).any()


def test_future_labels_transform_and_mask_end_of_reference():
    labels = trajectory_labels(
        [0, 1, 2],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        1,
        [1, 0, 0],
        [0, 0, 0, 1],
        horizons=(0.5, 1, 2),
    )
    np.testing.assert_allclose(
        labels["label_future_pelvis_body"], [[-0.5, 0, 0], [-1, 0, 0], [0, 0, 0]]
    )
    assert labels["label_future_mask"].tolist() == [True, True, False]
    assert all(k.startswith("label_") for k in labels)
    with pytest.raises(ValueError):
        trajectory_labels([0, 0], np.zeros((2, 3)), 0, [0, 0, 0], [1, 0, 0, 0])


def fixture_trace():
    task = {
        "obstacles": [{}],
        "goal_xyz": [0, 0, 1],
        "hold_ticks": 50,
        "goal_tolerance_m": 0.25,
        "terminal_speed_mps": 0.1,
    }
    tracked = np.zeros((60, 1, 3))
    tracked[:, :, 2] = 1
    forces = np.zeros((240, 2, 2, 3))
    forces[:, 0, 0, 2] = 100
    names = ["left_ankle_roll_link", "pelvis"]
    return task, tracked, forces, names


def test_pair_contacts_allow_foot_floor_but_reject_body_floor_and_obstacles():
    task, tracked, forces, names = fixture_trace()
    result = score_scene(
        task, tracked, tracked, forces, names, terminated=False, valid_steps=60
    )
    assert result["state"] == "complete"
    forces[101, 1, 0, 0] = 2
    assert not score_scene(
        task, tracked, tracked, forces, names, terminated=False, valid_steps=60
    )["checks"]["environment_contacts"]
    forces[101, 1, 0, 0] = 0
    forces[101, 0, 1, 0] = 2
    assert not score_scene(
        task, tracked, tracked, forces, names, terminated=False, valid_steps=60
    )["checks"]["environment_contacts"]


def test_arrival_without_stable_hold_and_missing_substeps_fail():
    task, tracked, forces, names = fixture_trace()
    tracked[:-1, 0, 0] = 0.4
    result = score_scene(
        task, tracked, tracked, forces, names, terminated=False, valid_steps=60
    )
    assert result["checks"]["goal_reached"] and not result["checks"]["terminal_hold"]
    with pytest.raises(ValueError, match="Incomplete"):
        score_scene(
            task, tracked, tracked, forces[:-1], names, terminated=False, valid_steps=60
        )


def test_task_rejects_unknown_public_profile():
    with pytest.raises(ValueError, match="profile"):
        validate_task(
            {
                "schema": "bfm_known_map_navigation_task_v1",
                "split": "train",
                "observation_profile": "privileged_future_route",
            }
        )


def test_native_wrapper_reference_and_measured_roles():
    # Load this small method without starting Isaac Kit, whose imports need a GPU runtime.
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    tree = ast.parse(
        (
            Path(__file__).resolve().parents[2]
            / "gear_sonic/envs/wrapper/manager_env_wrapper.py"
        ).read_text()
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ManagerEnvWrapper"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_env_data"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "wrapper_role_test", "exec"
        ),
        namespace,
    )
    reference, measured = object(), object()
    env = SimpleNamespace(
        motion_command=SimpleNamespace(body_pos_w=reference, robot_body_pos_w=measured)
    )
    assert namespace["get_env_data"](env, "ref_body_pos_extend") is reference
    assert namespace["get_env_data"](env, "rigid_body_pos_extend") is measured


def test_prefix_manifest_requires_its_own_explicit_training_flag(tmp_path):
    import json

    from gear_sonic.research.scene_distillation.train import load_episodes

    p = tmp_path / "mixed.json"
    p.write_text(
        json.dumps(
            {
                "schema": "bfm_executed_foundation_v1",
                "teacher_sha256": "a",
                "source": "mixed",
                "episodes": [{"source": "executed_native_teacher_prefix"}],
            }
        )
    )
    with pytest.raises(ValueError, match="prefixes require an explicit"):
        load_episodes(p, "a", set(), "foundation", allow_exploratory_queries=True)


def test_public_request_identity_excludes_reference_but_binds_goal_and_scene():
    from gear_sonic.research.scene_distillation.tasks import navigation_request_sha256

    keys = (
        "observation_profile",
        "world_frame",
        "anchor_body",
        "start_xyz",
        "start_wxyz",
        "goal_xyz",
        "goal_tolerance_m",
        "terminal_speed_mps",
        "hold_ticks",
        "deadline_ticks",
        "obstacles",
        "scene_usd_sha256",
    )
    task = {k: 0 for k in keys}
    reference_changed = dict(task, motion_id="different", reference={"sha256": "new"})
    assert navigation_request_sha256(task) == navigation_request_sha256(
        reference_changed
    )
    assert navigation_request_sha256(task) != navigation_request_sha256(
        dict(task, goal_xyz=[1, 0, 0])
    )
    assert navigation_request_sha256(task) != navigation_request_sha256(
        dict(task, scene_usd_sha256="new")
    )


def test_whole_qualified_teacher_keeps_late_recovery_but_never_post_end_frames():
    from gear_sonic.research.scene_distillation.collect import teacher_support_mask

    local = [True, False, True, True, True]
    assert teacher_support_mask(local, 4, 2, True).tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert not teacher_support_mask(local, 4, 2, False).any()
