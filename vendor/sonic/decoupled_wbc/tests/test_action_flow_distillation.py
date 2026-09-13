"""Flow transport, deployable context, migration and independent task scoring."""

import numpy as np
import pytest
import torch

from gear_sonic.research.scene_distillation.action_flow import ActionStudent, euler_flow, flow_path
from gear_sonic.research.scene_distillation.direct_context import (
    score_navigation_task,
    task_context,
)
from gear_sonic.research.scene_distillation.train_action_flow import (
    initialize_student,
    profile_mask,
)
from gear_sonic.research.scene_distillation.transformer import TransformerMotionFoundation


def task():
    return dict(
        start_xyz=[0, 0, 1],
        goal_xyz=[2, 0, 1],
        obstacles=[],
        goal_tolerance_m=0.25,
        terminal_speed_mps=0.1,
        hold_ticks=3,
    )


def batch():
    torch.manual_seed(71)
    controls = torch.randn(4, 79)
    controls[:, 0] = 0
    controls[:, 1] = 1
    mask = torch.ones(4, 79, dtype=torch.bool)
    mask[:, 7] = False
    return dict(
        proprio=torch.randn(4, 930),
        controls=controls,
        control_mask=mask,
        teacher_actions=torch.randn(4, 29) * 0.2,
    )


def context():
    row = task_context(task(), [0, 0, 1], [1, 0, 0, 0])
    return {k: torch.as_tensor(np.stack([v] * 4)) for k, v in row.items()}


def test_flow_transport_endpoints_and_integration_direction():
    target = torch.randn(4, 29)
    noise = torch.randn_like(target)
    start, velocity = flow_path(target, noise, torch.zeros(4, 1))
    end, _ = flow_path(target, noise, torch.ones(4, 1))
    torch.testing.assert_close(start, noise)
    torch.testing.assert_close(end, target)
    for steps in [1, 4, 8]:
        torch.testing.assert_close(euler_flow(lambda x, t: velocity, noise, steps), target)
    with pytest.raises(ValueError):
        flow_path(target, noise, torch.full((4, 1), 1.1))


def test_context_transforms_use_current_pose_and_preserve_obstacle_rotation():
    t = task()
    t["obstacles"] = [
        dict(
            shape="box",
            center_xyz=[2, 0, 1],
            quaternion_wxyz=[1, 0, 0, 0],
            full_dimensions_xyz=[1, 2, 3],
        )
    ]
    row = task_context(t, [1, 0, 1], [2**-0.5, 0, 0, 2**-0.5])
    np.testing.assert_allclose(row["navigation_context"][3:6], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(row["obstacles_body"][0, :3], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(row["obstacles_body"][0, 6:12], [0, -1, 0, 1, 0, 0], atol=1e-6)
    assert row["navigation_context"][-1] == 1


def test_context_only_masks_and_padded_geometry_do_not_leak():
    b = batch()
    c = context()
    m = ActionStudent(width=32, layers=1, heads=4, context=True).eval()
    # Exercise a trained/nonzero context path, not just its zero initialization.
    torch.nn.init.normal_(m.condition.context_output.weight, std=0.1)
    mask = profile_mask(b["control_mask"], "context")
    a = m.actions(b["proprio"], b["controls"], mask, context=c)["actions"]
    c["obstacles_body"][:] = float("nan")
    z = m.actions(b["proprio"], torch.full_like(b["controls"], float("nan")), mask, context=c)[
        "actions"
    ]
    torch.testing.assert_close(a, z, rtol=0, atol=0)
    c["navigation_context"][:, -1] = 0
    with pytest.raises(ValueError, match="known map"):
        m.actions(b["proprio"], b["controls"], mask, context=c)


def test_flow_loss_has_gradients_and_sampler_requires_explicit_noise():
    b = batch()
    m = ActionStudent(kind="flow", width=32, layers=1, heads=4)
    loss = m.imitation_loss(b, b["control_mask"])["loss"]
    loss.backward()
    assert m.head[-1].weight.grad.abs().sum() > 0
    assert m.condition.command_projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="noise"):
        m.actions(b["proprio"], b["controls"], b["control_mask"])
    assert torch.isfinite(
        m.actions(b["proprio"], b["controls"], b["control_mask"], noise=torch.zeros(4, 29))[
            "actions"
        ]
    ).all()


def test_encoder_and_context_migration_preserve_outputs():
    b = batch()
    old = TransformerMotionFoundation(width=32, layers=1, heads=4).eval()
    m = ActionStudent(width=32, layers=1, heads=4, context=True).eval()
    m.condition.initialize_motion_encoder(old.state_dict())
    torch.testing.assert_close(
        m.condition(b["proprio"], b["controls"], b["control_mask"], context()),
        old._condition(b["proprio"], b["controls"], b["control_mask"]),
        rtol=0,
        atol=0,
    )
    baseline = ActionStudent(width=32, layers=1, heads=4).eval()
    initialize_student(m, {"model": baseline.state_dict()})
    torch.testing.assert_close(
        m.actions(b["proprio"], b["controls"], b["control_mask"], context=context())["actions"],
        baseline.actions(b["proprio"], b["controls"], b["control_mask"])["actions"],
        rtol=0,
        atol=0,
    )


def test_residual_zero_identity_and_bound():
    b = batch()
    m = ActionStudent(width=32, layers=1, heads=4, residual_limit=0.05)
    result = m.actions(b["proprio"], b["controls"], b["control_mask"])
    torch.testing.assert_close(result["actions"], result["base_actions"], rtol=0, atol=0)
    with torch.no_grad():
        m.residual[-1].bias.fill_(100)
    result = m.actions(b["proprio"], b["controls"], b["control_mask"])
    assert result["residual"].abs().max() <= 0.05


def test_navigation_scoring_accepts_alternative_path_and_rejects_contact_or_no_hold():
    t = task()
    root = np.array([[0, 1, 1], [1, 1, 1], [2, 0, 1], [2, 0, 1], [2, 0, 1]])
    speed = np.array([1, 1, 0, 0, 0])
    force = np.zeros(5)
    assert score_navigation_task(t, root, speed, force)["navigation_success"]
    force[2] = 2
    assert not score_navigation_task(t, root, speed, force)["navigation_success"]
    assert not score_navigation_task(t, root[:3], speed[:3], np.zeros(3))["terminal_hold"]
    assert not score_navigation_task(t, root, speed, np.zeros(5), fell=True)["navigation_success"]


def test_token_residual_keeps_base_frozen():
    from gear_sonic.research.scene_distillation.token_residual import TokenActionResidual

    class Foundation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(29))

        def prior_step(self, proprio, controls, mask):
            return {"tokens": proprio[:, :29] * self.weight}

    class Decoder(torch.nn.Module):
        def forward(self, tokens, proprio):
            return tokens

    b = batch()
    base = Foundation()
    model = TokenActionResidual(base, Decoder())
    out = model.actions(b["proprio"], b["controls"], b["control_mask"])
    torch.testing.assert_close(out["actions"], out["base_actions"], atol=0, rtol=0)
    (out["actions"] - b["teacher_actions"]).square().mean().backward()
    assert base.weight.grad is None and not base.weight.requires_grad
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_context_bfm_preserves_pretrained_tokens_and_masks_reference_commands():
    from gear_sonic.research.scene_distillation.context_token import ContextTokenFoundation

    b = batch()
    original = TransformerMotionFoundation(width=32, layers=1, heads=4).eval()
    model = ContextTokenFoundation(width=32, layers=1, heads=4).eval()
    model.initialize_foundation(original.state_dict())
    old = original.prior_step(b["proprio"], b["controls"], b["control_mask"])["tokens"]
    current = model.context_step(b["proprio"], b["controls"], b["control_mask"], context())[
        "tokens"
    ]
    torch.testing.assert_close(old, current, rtol=0, atol=0)
    mask = torch.zeros_like(b["control_mask"])
    one = model.context_step(b["proprio"], b["controls"], mask, context())["tokens"]
    two = model.context_step(
        b["proprio"], torch.full_like(b["controls"], float("nan")), mask, context()
    )["tokens"]
    torch.testing.assert_close(one, two, rtol=0, atol=0)
