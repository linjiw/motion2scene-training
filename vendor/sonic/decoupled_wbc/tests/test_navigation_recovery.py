"""Recovery evidence cannot promote failed, prefix or deadline-relaxed labels."""

import json

import numpy as np
import pytest

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.direct_context import score_navigation_task
from gear_sonic.research.scene_distillation.navigation_recovery import suffix_support
from gear_sonic.research.scene_distillation.navigation_recovery_data import load_motor_recoveries


def bind(path):
    return dict(path=str(path), sha256=sha(path))


def dump(path, value):
    path.write_text(json.dumps(value))
    return bind(path)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gear_sonic.research.scene_distillation.navigation_recovery_data.validate_task", lambda t: t
    )
    n, switch = 55, 5
    task = dict(
        motion_id="m",
        split="train",
        goal_xyz=[0, 0, 0.8],
        goal_tolerance_m=0.25,
        terminal_speed_mps=0.1,
        hold_ticks=50,
        deadline_ticks=100,
    )
    task_ref = dump(tmp_path / "task.json", task)
    trace = dict(root_xyz=np.zeros((n, 3)), speed=np.zeros(n), undesired_force=np.zeros(n))
    trace["root_xyz"][:, 2] = 0.8
    mask, suffix = suffix_support(task, trace, switch, False)
    np.savez(tmp_path / "trace.npz", **trace)
    widths = dict(
        proprio=(930,),
        controls=(114,),
        actions=(29,),
        motor_actions=(29,),
        teacher_actions=(29,),
        teacher_tokens=(64,),
        navigation_context=(10,),
        obstacles_body=(5, 15),
    )
    arrays = {k: np.zeros((n, *shape), np.float32) for k, shape in widths.items()}
    arrays.update(
        control_mask=np.ones((n, 114), bool),
        obstacle_mask=np.zeros((n, 5), bool),
        query_mask=mask,
        learner_query_mask=np.arange(n) == switch,
        measured_root_xyz=np.zeros((n, 3)),
        measured_root_wxyz=np.tile([1.0, 0, 0, 0], (n, 1)),
        observation_time_s=np.arange(n) * 0.02,
        localization=np.zeros((n, 4), np.float32),
    )
    arrays["localization"][5:, 3] = 1
    arrays["control_mask"][:, 7] = False
    arrays["measured_root_xyz"][:, 2] = 0.8
    np.savez(tmp_path / "shard.npz", **arrays)
    (tmp_path / "behavior.pt").write_bytes(b"behavior")
    score = dict(
        state="complete",
        teacher_mode=False,
        teacher_sha256="teacher",
        student_sha256=sha(tmp_path / "behavior.pt"),
        task_sha256=task_ref["sha256"],
        control_dt=0.02,
        contact_dt=0.005,
        control_steps=n,
        fell=False,
    )
    score.update(
        score_navigation_task(task, trace["root_xyz"], trace["speed"], trace["undesired_force"])
    )
    score["actor_profile"] = "nav_goal_map_v1"
    receipt = dict(
        source="navigation_prefix_motor_recovery",
        schema="executed_motor_navigation_recovery_v1",
        motor_sha256="motor",
        teacher_sha256="teacher",
        task=task_ref,
        reference_phase_switch=False,
        score=dump(tmp_path / "score.json", score),
        shard=bind(tmp_path / "shard.npz"),
        behavior_checkpoint=bind(tmp_path / "behavior.pt"),
        takeover_tick=switch,
        suffix_score=suffix,
        supported=True,
        rows=n,
        supported_rows=50,
        learner_query_rows=1,
    )
    parent = dump(tmp_path / "receipt.json", receipt)
    parent["trace"] = bind(tmp_path / "trace.npz")
    manifest = dict(schema="motor_navigation_recovery_manifest_v1", parents=[parent])
    manifest_ref = dump(tmp_path / "manifest.json", manifest)
    catalog = dump(tmp_path / "catalog.json", [dict(id="m", split="train", group="ancestry")])
    tasks = dump(tmp_path / "tasks.json", dict(tasks=[dict(task_sha256=task_ref["sha256"])]))
    config = dict(
        dataset_manifest=manifest_ref["path"],
        dataset_manifest_sha256=manifest_ref["sha256"],
        ancestry_catalog=catalog["path"],
        ancestry_catalog_sha256=catalog["sha256"],
        task_manifest=tasks["path"],
        task_manifest_sha256=tasks["sha256"],
        teacher_sha256="teacher",
        motor_sha256="motor",
    )
    return config, arrays, receipt, manifest


def rewrite(tmp_path, config, arrays, receipt, manifest):
    np.savez(tmp_path / "shard.npz", **arrays)
    receipt["shard"] = bind(tmp_path / "shard.npz")
    parent = dump(tmp_path / "receipt.json", receipt)
    parent["trace"] = bind(tmp_path / "trace.npz")
    manifest["parents"] = [parent]
    config["dataset_manifest_sha256"] = dump(tmp_path / "manifest.json", manifest)["sha256"]


def test_suffix_hold_does_not_inherit_prefix_success():
    task = dict(goal_xyz=[0, 0, 0], goal_tolerance_m=0.25, terminal_speed_mps=0.1, hold_ticks=50)
    trace = dict(root_xyz=np.zeros((60, 3)), speed=np.zeros(60), undesired_force=np.zeros(60))
    mask, score = suffix_support(task, trace, 50, False)
    assert not mask.any()
    assert score["max_hold_ticks"] == 10


def test_qualified_suffix_retains_one_learner_query(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    (episode,) = load_motor_recoveries(config)
    assert episode["role"] == "recovery"
    assert episode["arrays"]["query_mask"].sum() == 50
    assert episode["arrays"]["learner_query_mask"].sum() == 1
    assert episode["arrays"]["decision_index"][5] == 5


@pytest.mark.parametrize(
    "attack", ["prefix", "learner_count", "unexecuted_action", "relaxed_deadline"]
)
def test_forged_recovery_admission_rejected(tmp_path, monkeypatch, attack):
    config, arrays, receipt, manifest = fixture(tmp_path, monkeypatch)
    if attack == "prefix":
        arrays["query_mask"][0] = True
        receipt["supported_rows"] += 1
    elif attack == "learner_count":
        arrays["learner_query_mask"][6] = True
        receipt["learner_query_rows"] += 1
    elif attack == "unexecuted_action":
        arrays["actions"][5, 0] = 1
    else:
        task = json.loads((tmp_path / "task.json").read_text())
        task["deadline_ticks"] = 200
        receipt["task"] = dump(tmp_path / "task.json", task)
    rewrite(tmp_path, config, arrays, receipt, manifest)
    with pytest.raises(ValueError):
        load_motor_recoveries(config)
