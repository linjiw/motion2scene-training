"""Behavioral contracts for masked foundation, residuals, and causal training."""

import json

import pytest
import torch
from torch import nn

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.commands import (
    MaskedMotionFoundation,
    sample_command_mask,
)
from gear_sonic.research.scene_distillation.navigation import (
    FrozenFoundationNavigator,
    navigation_distillation_loss,
)
from gear_sonic.research.scene_distillation.residual_rl import (
    BoundedResidualActorCritic,
    generalized_advantage,
    optimize_residual_ppo,
)
from gear_sonic.research.scene_distillation.scene_teacher import (
    QualifiedSceneRegistry,
    verify_scene_receipt,
)
from gear_sonic.research.scene_distillation.train import (
    load_episodes,
    train_foundation_step,
    train_navigation_step,
)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(29), requires_grad=False)

    def forward(self, tokens, proprio):
        return tokens[:, :29] * self.weight + proprio[:, :29] * 0.01


@pytest.fixture
def public():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return {
        "proprio": torch.randn(2, 930),
        "start_goal_body": torch.randn(2, 6),
        "obstacles_body": torch.randn(2, 5, 15),
        "obstacle_mask": torch.ones(2, 5, dtype=torch.bool),
    }


def navigator():
    return FrozenFoundationNavigator(
        MaskedMotionFoundation(),
        Decoder(),
        command_lower=[-1, -0.5, -1, 0.3],
        command_upper=[1, 0.5, 1, 1.0],
        action_residual_limit=[0.05] * 29,
    )


def test_action_residual_zero_init_bound_and_frozen_gradient(public):
    model = navigator()
    before = {k: v.clone() for k, v in model.foundation.state_dict().items()}
    output = model.forward_step(public)
    torch.testing.assert_close(output["actions"], output["base_actions"], rtol=0, atol=0)
    loss = navigation_distillation_loss(
        output,
        output["actions"].detach() + 0.3,
        torch.ones(2, dtype=torch.bool),
        residual_weight=0.1,
    )
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    opt.zero_grad()
    loss["loss"].backward()
    opt.step()
    assert model.action_residual_head[-1].weight.grad.abs().sum() > 0
    for k, v in model.foundation.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    output = model.forward_step(public)
    assert (output["action_residual"].abs() <= 0.05).all()
    assert output["action_residual"].abs().sum() > 0


def test_masks_remove_unavailable_targets_and_preserve_yaw_pair(public):
    foundation = MaskedMotionFoundation()
    commands = torch.randn(2, 79)
    mask = torch.zeros(2, 79, dtype=torch.bool)
    a = foundation.prior_step(public["proprio"], commands, mask)
    commands.fill_(float("nan"))
    b = foundation.prior_step(public["proprio"], commands, mask)
    torch.testing.assert_close(a["tokens"], b["tokens"])
    available = torch.ones(100, 79, dtype=torch.bool)
    available[:, 7] = False
    sampled = sample_command_mask(available)
    assert not sampled[:, 7].any()
    assert torch.equal(sampled[:, 0], sampled[:, 1])


def test_foundation_real_loss_updates_public_prior_and_posterior(public):
    model = MaskedMotionFoundation()
    decoder = Decoder()
    proprio = public["proprio"]
    tokens = torch.zeros(2, 64)
    controls = torch.zeros(2, 79)
    controls[:, 1] = 1
    batch = {
        "proprio": proprio,
        "controls": controls,
        "control_mask": torch.ones(2, 79, dtype=torch.bool),
        "teacher_tokens": tokens,
        "teacher_actions": decoder(tokens, proprio),
        "privileged_state": torch.randn(2, 1645),
        "future_reference": torch.randn(2, 640),
    }
    loss = train_foundation_step(model, decoder, batch, 0.001, 0.1)
    loss["loss"].backward()
    for component in (model.prior, model.posterior, model.command, model.token_adapter):
        assert sum(p.grad.abs().sum() for p in component.parameters()) > 0
    assert all(p.grad is None for p in decoder.parameters())


def test_recurrent_navigation_training_uses_prefix_and_no_privileged_fields(public):
    model = navigator()
    episode = {k: v[:1].expand(5, *v.shape[1:]).clone() for k, v in public.items()}
    episode.update(
        teacher_actions=torch.zeros(5, 29), qualified_mask=torch.ones(5, dtype=torch.bool)
    )
    config = {
        "latent_kl_weight": 0.0,
        "action_residual_weight": 0.01,
        "action_smoothness_weight": 0.01,
    }
    loss = train_navigation_step(model, episode, 2, 3, config)
    loss["loss"].backward()
    assert model.action_residual_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.foundation.parameters())


def test_smoothness_does_not_cross_reset(public):
    output = navigator().forward_step(public)
    loss = navigation_distillation_loss(
        output,
        output["actions"].detach(),
        torch.ones(2, dtype=torch.bool),
        smoothness_weight=1,
        previous_residual=torch.full((2, 29), 100.0),
        continuation_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert loss["residual_smoothness"] == 0


def test_gae_bootstraps_timeout_but_does_not_cross_episode():
    rewards = torch.tensor([[1.0], [2.0], [100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10.0], [20.0], [30.0]])
    terminated = torch.tensor([[False], [False], [True]])
    end = torch.tensor([[False], [True], [True]])
    adv, returns = generalized_advantage(
        rewards, values, next_values, terminated, end, gamma=1, lam=1
    )
    torch.testing.assert_close(adv, torch.tensor([[33.0], [22.0], [100.0]]))
    torch.testing.assert_close(returns, adv)


def test_residual_ppo_is_bounded_and_updates_on_policy():
    torch.manual_seed(4)
    policy = BoundedResidualActorCritic([0.05] * 29)
    features = torch.randn(12, 128)
    base = torch.randn(12, 29)
    out = policy.act(features, base)
    assert (out["residual"].abs() <= 0.05).all()
    torch.testing.assert_close(policy.act(features, base, deterministic=True)["actions"], base)
    batch = {
        "features": features,
        "pre_tanh": out["pre_tanh"],
        "old_log_prob": out["log_prob"],
        "advantages": torch.linspace(-1, 1, 12),
        "returns": torch.randn(12),
    }
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    assert optimize_residual_ppo(policy, opt, batch, epochs=2)
    assert policy.actor[-1].weight.grad.abs().sum() > 0


def test_missing_scene_evidence_and_changed_goal_are_rejected(tmp_path):
    task = tmp_path / "task.json"
    task.write_text("{}")
    evidence = tmp_path / "trace"
    evidence.write_text("synthetic test fixture")
    receipt = tmp_path / "receipt.json"
    record = {
        "state": "complete",
        "teacher_sha256": "teacher",
        "task_sha256": sha(task),
        "control_dt": 0.02,
        "contact_dt": 0.005,
        "checks": {
            k: True
            for k in ["whole_motion", "environment_contacts", "goal_reached", "terminal_hold"]
        },
        "evidence": [{"path": str(evidence), "sha256": sha(evidence)}],
    }
    receipt.write_text(json.dumps(record))
    binding = {"path": str(receipt), "sha256": sha(receipt)}
    registry = QualifiedSceneRegistry(
        [{"task_path": str(task), "qualification": binding, "continuation_id": "a"}], "teacher"
    )
    assert registry.select(task)["continuation_id"] == "a"
    task.write_text('{"goal":1}')
    with pytest.raises(ValueError, match="No physically qualified"):
        registry.select(task)
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        verify_scene_receipt(binding, "teacher", record["task_sha256"])


def test_training_rejects_scene_labels_from_foundation(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps({"schema": "bfm_executed_foundation_v1", "teacher_sha256": "a", "episodes": []})
    )
    with pytest.raises(ValueError, match="stage or teacher"):
        load_episodes(p, "a", set(), "navigation")
    with pytest.raises(ValueError, match="No eligible"):
        load_episodes(p, "a", set(), "foundation")


def test_live_residual_training_accounts_steps_and_freezes_base(tmp_path, public):
    from gear_sonic.research.scene_distillation.residual_rl import train_live_residual

    class Runtime:
        task = {"reward_profile": "known_goal_progress_v1"}
        teacher_sha256 = "fixture"

        def reset(self):
            return {k: v[:1] for k, v in public.items()}

        def step(self, action):
            assert torch.isfinite(action).all()
            return (
                self.reset(),
                torch.ones(1),
                torch.zeros(1, dtype=torch.bool),
                {"time_outs": torch.zeros(1, dtype=torch.bool)},
            )

    base = navigator()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    policy = BoundedResidualActorCritic([0.05] * 29)
    train_live_residual(
        Runtime(),
        base,
        policy,
        tmp_path / "rl",
        iterations=2,
        horizon=4,
        learning_rate=1e-4,
        reward_profile="known_goal_progress_v1",
    )
    receipt = json.loads((tmp_path / "rl/receipt.json").read_text())
    assert receipt["environment_transitions"] == 8 and receipt["state"] == "complete"
    for k, v in base.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_exploratory_dagger_queries_are_not_silently_qualified(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps(
            {
                "schema": "bfm_executed_foundation_v1",
                "teacher_sha256": "a",
                "source": "queried_native_teacher_on_student_state",
                "episodes": [],
            }
        )
    )
    with pytest.raises(ValueError, match="explicit experiment flag"):
        load_episodes(p, "a", set(), "foundation")


def test_offline_optimizer_continuation_matches_uninterrupted_fit(tmp_path, monkeypatch):
    import argparse

    import numpy as np

    from gear_sonic.research.scene_distillation import train

    teacher = tmp_path / "teacher"
    teacher.write_text("test fixture only")
    arrays = {
        "proprio": np.zeros((4, 930), np.float32),
        "teacher_tokens": np.zeros((4, 64), np.float32),
        "teacher_actions": np.zeros((4, 29), np.float32),
        "privileged_state": np.zeros((4, 1645), np.float32),
        "future_reference": np.zeros((4, 640), np.float32),
        "controls": np.zeros((4, 79), np.float32),
        "control_mask": np.ones((4, 79), bool),
    }
    arrays["controls"][:, 1] = 1
    shard = tmp_path / "episode.npz"
    np.savez(shard, **arrays)
    queried = tmp_path / "queried.npz"
    np.savez(queried, **arrays, query_mask=np.array([True, True, False, True]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "bfm_executed_foundation_v1",
                "teacher_sha256": sha(teacher),
                "episodes": [
                    {
                        "path": str(shard),
                        "sha256": sha(shard),
                        "rows": 4,
                        "motion_id": "a",
                        "split": "train",
                        "eligible": True,
                    },
                    {
                        "path": str(queried),
                        "sha256": sha(queried),
                        "rows": 4,
                        "motion_id": "a",
                        "split": "train",
                        "eligible": True,
                        "source": "queried_native_teacher_on_student_state",
                    },
                ],
            }
        )
    )
    config = {
        "stage": "foundation",
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": sha(teacher),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": sha(manifest),
        "train_ids": ["a"],
        "updates": 3,
        "batch_size": 2,
        "wall_cap_seconds": 60,
        "learning_rate": 0.001,
        "seed": 3,
        "device": "cpu",
        "beta": 0.001,
        "prior_action_weight": 0.1,
        "save_interval": 1,
        "allow_exploratory_queries": True,
    }
    monkeypatch.setattr(train, "make_decoder", lambda checkpoint, device: Decoder())

    def run(name, settings):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(settings))
        train.main(argparse.Namespace(config=path, output=tmp_path / name))

    run("whole", config)
    run("first", dict(config, updates=2))
    checkpoint = tmp_path / "first/step-000002.pt"
    run(
        "resume",
        dict(config, updates=1, resume_checkpoint=str(checkpoint), resume_sha256=sha(checkpoint)),
    )
    whole = torch.load(tmp_path / "whole/step-000003.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resume/step-000003.pt", weights_only=False)
    for k, v in whole["model"].items():
        torch.testing.assert_close(v, resumed["model"][k], atol=0, rtol=0)
    assert json.loads((tmp_path / "resume/receipt.json").read_text())["updates"] == 1


def test_navigation_command_labels_train_director_without_entering_actor(public):
    model = navigator()
    episode = {k: v[:1].expand(3, *v.shape[1:]).clone() for k, v in public.items()}
    episode.update(
        teacher_actions=torch.zeros(3, 29),
        qualified_mask=torch.ones(3, dtype=torch.bool),
        label_controls=torch.zeros(3, 79),
        label_control_mask=torch.ones(3, 79, dtype=torch.bool),
    )
    episode["label_controls"][:, 5] = 0.7
    config = {
        "latent_kl_weight": 0.0,
        "action_residual_weight": 0.0,
        "action_smoothness_weight": 0.0,
        "navigation_command_weight": 1.0,
    }
    loss = train_navigation_step(model, episode, 0, 3, config)
    loss["loss"].backward()
    assert model.command_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.foundation.parameters())
    episode["label_control_mask"][:] = False
    with pytest.raises(ValueError, match="Missing qualified navigation-command"):
        train_navigation_step(model, episode, 0, 3, config)
