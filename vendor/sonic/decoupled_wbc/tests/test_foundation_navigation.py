"""Synthetic architecture checks; none measures trained navigation performance."""

import inspect

import pytest
import torch
from torch import nn

from gear_sonic.research.scene_distillation.foundation import MotionBehaviorFoundation
from gear_sonic.research.scene_distillation.navigation import (
    FOUNDATION_CONTROL_INDICES,
    FrozenFoundationNavigator,
    navigation_distillation_loss,
)


class FixtureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(29))

    def forward(self, tokens, proprio):
        return self.weight * tokens[:, :29] + proprio[:, :29]


@pytest.fixture
def observation():
    torch.set_num_threads(2)
    torch.manual_seed(91147)
    return {
        "proprio": torch.randn(2, 930),
        "start_goal_body": torch.randn(2, 6),
        "obstacles_body": torch.randn(2, 5, 15),
        "obstacle_mask": torch.tensor([[True, True, True, False, False]] * 2),
    }


def navigator(scale=0.0):
    # Fixture bounds only: not measured or approved G1 command limits.
    return FrozenFoundationNavigator(
        MotionBehaviorFoundation(),
        FixtureDecoder(),
        command_lower=[-1, -0.5, -1, 0.4],
        command_upper=[1, 0.5, 1, 0.9],
        residual_scale=scale,
    )


def test_primary_path_uses_exact_frozen_prior(observation):
    model = navigator()
    noise = torch.randn(2, 32)
    output = model.forward_step(observation, epsilon=noise)
    base = model.foundation.prior_step(
        observation["proprio"], output["controls"], output["control_mask"], epsilon=noise
    )
    torch.testing.assert_close(base["tokens"], output["tokens"], rtol=0, atol=0)
    torch.testing.assert_close(output["base_mean"], output["navigation_mean"], rtol=0, atol=0)
    torch.testing.assert_close(output["prior_deviation_kl"], torch.zeros(2), rtol=0, atol=0)
    assert model.residual_head is None
    command = output["controls"][:, list(FOUNDATION_CONTROL_INDICES)]
    assert (command >= model.command_lower).all() and (command <= model.command_upper).all()
    assert output["control_mask"].sum().item() == 8


def test_gradient_crosses_frozen_foundation_without_updating_it(observation):
    model = navigator().train()
    assert not model.foundation.training and not model.decoder.training
    before = {n: p.detach().clone() for n, p in model.foundation.named_parameters()}
    output = model.forward_step(observation)
    loss = navigation_distillation_loss(output, torch.zeros(2, 29), torch.ones(2, dtype=torch.bool))
    loss["loss"].backward()
    assert sum(p.grad.abs().sum().item() for p in model.command_head.parameters()) > 0
    assert sum(p.grad.abs().sum().item() for p in model.encoder.parameters()) > 0
    assert all(p.grad is None and not p.requires_grad for p in model.foundation.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.decoder.parameters())
    # One synthetic optimizer step specifically checks preservation, not dataset fitting.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    for name, parameter in model.foundation.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_optional_residual_starts_at_prior_and_has_analytic_bound(observation):
    scale = 0.2
    model = navigator(scale)
    initial = model.forward_step(observation)
    torch.testing.assert_close(initial["prior_deviation_kl"], torch.zeros(2), atol=1e-6, rtol=0)
    with torch.no_grad():
        model.residual_head[-1].bias.fill_(10)
    changed = model.forward_step(observation)
    expected = 0.5 * changed["standardized_residual"].square().sum(-1)
    torch.testing.assert_close(changed["prior_deviation_kl"], expected, atol=2e-6, rtol=2e-6)
    assert (expected <= 0.5 * model.foundation.latent_dim * scale**2 + 1e-6).all()


def test_posterior_and_reference_are_not_downstream_dependencies(observation):
    model = navigator()
    before = model.forward_step(observation)
    with torch.no_grad():
        for parameter in model.foundation.posterior.parameters():
            parameter.fill_(float("nan"))
    after = model.forward_step(observation)
    torch.testing.assert_close(before["actions"], after["actions"], rtol=0, atol=0)
    assert "future_reference" not in inspect.signature(model.forward_step).parameters
    for field in ("phase", "future_reference", "motion_id", "route_body"):
        with pytest.raises(ValueError, match="Public actor fields"):
            model.forward_step(dict(observation, **{field: torch.zeros(2, 1)}))


def test_foundation_prior_is_scene_independent_and_posterior_is_training_only(observation):
    foundation = MotionBehaviorFoundation()
    proprio = observation["proprio"]
    first = foundation.posterior_step(proprio, torch.zeros(2, 1645), torch.zeros(2, 640))
    second = foundation.posterior_step(proprio, torch.ones(2, 1645), torch.ones(2, 640))
    torch.testing.assert_close(first["prior_mean"], second["prior_mean"])
    assert not torch.allclose(first["posterior_mean"], second["posterior_mean"])
    with pytest.raises(TypeError):
        foundation.prior_step(proprio, obstacles_body=observation["obstacles_body"])
    (first["tokens"].square().mean() + 0.01 * first["kl"].mean()).backward()
    for module in (foundation.prior, foundation.posterior, foundation.token_adapter):
        assert sum(p.grad.abs().sum().item() for p in module.parameters()) > 0


def test_navigation_set_permutation_padding_and_episode_reset(observation):
    model = navigator()
    before = model.forward_step(observation)
    order = [2, 4, 1, 0, 3]
    changed = dict(
        observation,
        obstacles_body=observation["obstacles_body"][:, order],
        obstacle_mask=observation["obstacle_mask"][:, order],
    )
    torch.testing.assert_close(before["controls"], model.forward_step(changed)["controls"])
    observation["obstacles_body"][~observation["obstacle_mask"]] = float("nan")
    torch.testing.assert_close(before["actions"], model.forward_step(observation)["actions"])
    reset = model.forward_step(
        observation, torch.randn(2, 128), episode_start=torch.ones(2, dtype=torch.bool)
    )
    torch.testing.assert_close(before["actions"], reset["actions"])


def test_missing_expert_rows_do_not_become_zero_actions(observation):
    output = navigator().forward_step(observation)
    labels = torch.full((2, 29), float("nan"))
    labels[0] = output["actions"][0].detach()
    loss = navigation_distillation_loss(output, labels, torch.tensor([True, False]))
    assert loss["loss"].item() == 0 and loss["supervised_rows"] == 1
    with pytest.raises(ValueError, match="No qualified"):
        navigation_distillation_loss(output, labels, torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize(
    "lower,upper",
    [
        ([0, 0, 0, 0], [1, 1, 1, 1]),
        ([0, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 0], [1, 1, 1, 2]),
        ([float("nan"), 0, 0, 1], [1, 1, 1, 2]),
    ],
)
def test_invalid_command_profiles_rejected(lower, upper):
    with pytest.raises(ValueError, match="command bounds"):
        FrozenFoundationNavigator(
            MotionBehaviorFoundation(), FixtureDecoder(), command_lower=lower, command_upper=upper
        )
