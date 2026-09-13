"""Information boundaries, motor retention and task-positive evidence admission."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.navigation_data import load_successful_tasks
from gear_sonic.research.scene_distillation.navigation_motor import (
    NavigationInput,
    NavigationMotorStudent,
)
from gear_sonic.research.scene_distillation.reference_layout import pack_reference


class Motor(nn.Module):
    command_dim = 114

    def __init__(self):
        super().__init__()
        self.forecaster = nn.Linear(114, 64)
        self.register_buffer("input_mean", torch.zeros(1044))
        self.register_buffer("input_std", torch.ones(1044))

    def prior_step(self, proprio, controls, mask):
        return dict(tokens=self.forecaster(torch.where(mask, controls, 0)))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 29)

    def forward(self, tokens, proprio):
        return self.linear(tokens)


def actor():
    return NavigationInput(
        torch.randn(2, 930),
        torch.ones(2, 10),
        torch.randn(2, 5, 15),
        torch.zeros(2, 5, dtype=torch.bool),
    )


def test_actor_boundary_padding_and_complete_map():
    student = NavigationMotorStudent(Motor(), Decoder(), width=32)
    a = actor()
    expected = student.navigation_step(a)["actions"]
    hidden = NavigationInput(
        a.proprio,
        a.navigation_context,
        torch.full_like(a.obstacles_body, float("nan")),
        a.obstacle_mask,
    )
    torch.testing.assert_close(student.navigation_step(hidden)["actions"], expected, atol=0, rtol=0)
    incomplete = a.navigation_context.clone()
    incomplete[:, 9] = 0
    with pytest.raises(ValueError, match="complete known map"):
        student.navigation_step(
            NavigationInput(a.proprio, incomplete, a.obstacles_body, a.obstacle_mask)
        )
    with pytest.raises(TypeError, match="public actor view"):
        student.navigation_step(dict(proprio=a.proprio, future_reference=torch.ones(2, 640)))


def test_navigation_update_preserves_full_motor_and_decoder():
    torch.manual_seed(5)
    student = NavigationMotorStudent(Motor(), Decoder(), width=32)
    a = actor()
    commands = torch.randn(2, 114)
    mask = torch.ones(2, 114, dtype=torch.bool)
    before = student.full_step(a.proprio, commands, mask)["actions"].detach().clone()
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=0.01)
    loss = student.navigation_step(a)["actions"].square().mean()
    loss.backward()
    assert all(p.grad is None for p in student.motor.parameters())
    assert all(p.grad is None for p in student.decoder.parameters())
    assert student.command_predictor[-1].weight.grad.abs().max() > 0
    optimizer.step()
    torch.testing.assert_close(
        student.full_step(a.proprio, commands, mask)["actions"], before, atol=0, rtol=0
    )


def test_runtime_uses_measured_pose_without_reference_commands(monkeypatch):
    from gear_sonic.research.scene_distillation.navigation_motor_runtime import (
        NavigationMotorCallback,
    )

    # The environment deliberately has no target joint positions, velocity,
    # tokenizer observation, reference clock or future-reference arrays.
    env = SimpleNamespace(
        device="cpu",
        motion_command=SimpleNamespace(
            robot_anchor_pos_w=torch.zeros(1, 3),
            robot_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ),
        env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3))),
    )
    monkeypatch.setattr(
        "gear_sonic.research.scene_distillation.navigation_motor_runtime.task_context",
        lambda task, pos, quat: dict(
            navigation_context=np.ones(10, np.float32),
            obstacles_body=np.zeros((5, 15), np.float32),
            obstacle_mask=np.zeros(5, bool),
        ),
    )
    callback = object.__new__(NavigationMotorCallback)
    student = NavigationMotorStudent(Motor(), Decoder(), width=32)
    action = callback._student_action(
        student,
        env,
        SimpleNamespace(running_mean_std=None),
        {"actor_obs": torch.zeros(1, 930)},
        {},
        None,
    )
    assert action.shape == (1, 29)


def task_fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=sha(path))

    n = 3
    arrays = dict(
        proprio=np.zeros((n, 930), np.float32),
        controls=np.zeros((n, 79), np.float32),
        control_mask=np.ones((n, 79), bool),
        teacher_tokens=np.zeros((n, 64), np.float32),
        teacher_actions=np.zeros((n, 29), np.float32),
        future_reference=pack_reference(torch.zeros(n, 10, 64)).numpy(),
        navigation_context=np.ones((n, 10), np.float32),
        obstacles_body=np.zeros((n, 5, 15), np.float32),
        obstacle_mask=np.zeros((n, 5), bool),
        query_mask=np.array([True, False, True]),
    )
    shard = tmp_path / "episode.npz"
    np.savez(shard, **arrays)
    task = write("task.json", dict(motion_id="00001", split="train"))
    score = write(
        "score.json",
        dict(
            state="complete",
            navigation_success=True,
            teacher_mode=True,
            teacher_sha256="teacher",
            task_sha256=task["sha256"],
            control_dt=0.02,
            contact_dt=0.005,
            control_steps=n,
        ),
    )
    episode = dict(
        path=str(shard),
        sha256=sha(shard),
        motion_id="00001",
        split="train",
        eligible=True,
        task_path=task["path"],
        task_sha256=task["sha256"],
        task_success_evidence=score,
        rows=n,
        supported_query_rows=2,
    )
    parent = write(
        "parent.json", dict(scene_task_success=True, teacher_sha256="teacher", episodes=[episode])
    )
    manifest = write(
        "manifest.json",
        dict(
            schema="bfm_executed_foundation_v1",
            context_schema="complete_known_map_goal_v1",
            teacher_sha256="teacher",
            parents=[parent],
            episodes=[episode],
        ),
    )
    ancestry = write("ancestry.json", [dict(id="00001", split="train", group="same-family")])
    return dict(
        dataset_manifest=manifest["path"],
        dataset_manifest_sha256=manifest["sha256"],
        ancestry_catalog=ancestry["path"],
        ancestry_catalog_sha256=ancestry["sha256"],
        teacher_sha256="teacher",
    )


def test_task_view_retains_gaps_and_lineage(tmp_path, monkeypatch):
    config = task_fixture(tmp_path)
    monkeypatch.setattr(
        "gear_sonic.research.scene_distillation.navigation_data.validate_task", lambda x: x
    )
    episodes = load_successful_tasks(config)
    assert episodes[0]["ancestry"] == "same-family"
    assert episodes[0]["arrays"]["decision_index"].tolist() == [0, 1, 2]
    assert episodes[0]["arrays"]["query_mask"].tolist() == [True, False, True]
    assert episodes[0]["arrays"]["controls"].shape == (3, 114)


@pytest.mark.parametrize(
    "failure", ["missing_task_success", "changed_score", "development_ancestry"]
)
def test_task_view_rejects_unproven_labels(tmp_path, failure):
    config = task_fixture(tmp_path)
    if failure == "changed_score":
        (tmp_path / "score.json").write_text("{}")
    elif failure == "development_ancestry":
        (tmp_path / "ancestry.json").write_text(
            json.dumps([dict(id="00001", split="development", group="g")])
        )
        config["ancestry_catalog_sha256"] = sha(tmp_path / "ancestry.json")
    else:
        parent = json.loads((tmp_path / "parent.json").read_text())
        parent["scene_task_success"] = False
        (tmp_path / "parent.json").write_text(json.dumps(parent))
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["parents"][0]["sha256"] = sha(tmp_path / "parent.json")
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        config["dataset_manifest_sha256"] = sha(tmp_path / "manifest.json")
    with pytest.raises(ValueError):
        load_successful_tasks(config)


def test_causal_localization_rotation_warmup_goal_reset_and_timestamps():
    from gear_sonic.research.scene_distillation.navigation_localization import CausalLocalization

    estimator = CausalLocalization(lag=2)
    goal = [5, 0, 0]
    yaw90 = [2**-0.5, 0, 0, 2**-0.5]
    assert np.array_equal(estimator.update([0, 0, 0], yaw90, 0.0, goal), np.zeros(4))
    assert estimator.update([0.1, 0, 0], yaw90, 0.1, goal)[3] == 0
    np.testing.assert_allclose(
        estimator.update([0.2, 0, 0], yaw90, 0.2, goal), [0, -1, 0, 1], atol=1e-6
    )
    with pytest.raises(ValueError, match="timestamps"):
        estimator.update([0.2, 0, 0], yaw90, 0.2, goal)
    assert estimator.update([0.3, 0, 0], yaw90, 0.3, [6, 0, 0])[3] == 0


def test_structured_loss_uses_motor_target_and_excludes_missing_arrival():
    from gear_sonic.research.scene_distillation.navigation_motor import navigation_loss

    student = NavigationMotorStudent(Motor(), Decoder(), width=32)
    a = actor()
    batch = {k: getattr(a, k) for k in a.__dataclass_fields__}
    batch.update(
        controls=torch.randn(2, 114),
        teacher_tokens=torch.randn(2, 64),
        teacher_actions=torch.randn(2, 29),
    )
    first = navigation_loss(student, batch, objective="structured_motor")
    first["loss"].backward()
    assert student.command_predictor[-1].weight.grad.abs().max() > 0
    assert all(p.grad is None for p in student.motor.parameters())
    batch["teacher_actions"] += 100
    batch["controls"][:, 7] += 100
    second = navigation_loss(student, batch, objective="structured_motor")
    torch.testing.assert_close(first["loss"], second["loss"], atol=0, rtol=0)
    assert second["teacher_action_mse"] > first["teacher_action_mse"]


def test_localized_actor_requires_explicit_features():
    student = NavigationMotorStudent(Motor(), Decoder(), width=32, localization=True)
    a = actor()
    with pytest.raises(ValueError, match="causal velocity"):
        student.navigation_step(a)
    with_features = NavigationInput(
        a.proprio, a.navigation_context, a.obstacles_body, a.obstacle_mask, torch.zeros(2, 4)
    )
    assert student.navigation_step(with_features)["actions"].shape == (2, 29)


def test_candidate_queries_cannot_self_qualify(tmp_path):
    from gear_sonic.research.scene_distillation.navigation_queries import NavigationQueryCallback

    task_path = tmp_path / "task.json"
    task_path.write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    (output / "task-result.json").write_text('{"navigation_success": true}')
    callback = object.__new__(NavigationQueryCallback)
    callback.config = dict(
        actor_profile="nav_goal_map_v1",
        task_path=str(task_path),
        teacher_sha256="teacher",
        student_sha256="student",
    )
    callback.rows = [dict(proprio=np.zeros(930, np.float32)) for _ in range(3)]
    callback._complete_task({}, {"control_steps": 3}, output)
    receipt = json.loads((output / "query-collection.json").read_text())
    assert receipt["task_positive_eligible"] is False
    assert receipt["supported_rows"] == 0
    with np.load(output / "navigation-queries.npz") as data:
        assert not data["supported_query_mask"].any()
