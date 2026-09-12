"""Synthetic contract tests, not executed navigation or training-performance evidence."""

import copy
import inspect

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence

from gear_sonic.research.scene_distillation.contracts import (
    audit_scene_pairing,
    public_observation_sha256,
    validate_query_record,
)
from gear_sonic.research.scene_distillation.cvae import (
    VariationalNavigationStudent,
    diagonal_gaussian_kl,
    masked_controls,
)
from gear_sonic.research.scene_distillation.losses import (
    admissible_imitation_loss,
    endpoint_auxiliary_loss,
    variational_imitation_loss,
)
from gear_sonic.research.scene_distillation.observations import navigation_observation
from gear_sonic.research.scene_distillation.policy import NavigationTokenStudent


@pytest.fixture
def observation():
    torch.set_num_threads(2)
    torch.manual_seed(91146)
    return {
        "proprio": torch.randn(2, 930),
        "start_goal_body": torch.randn(2, 6),
        "obstacles_body": torch.randn(2, 5, 15),
        "obstacle_mask": torch.tensor([[True, True, True, False, False]] * 2),
    }


@pytest.mark.parametrize("variational", [False, True])
def test_obstacle_permutation_padding_empty_and_fsq(observation, variational):
    model = VariationalNavigationStudent() if variational else NavigationTokenStudent()
    act = model.prior_step if variational else model.forward_step
    expected = act(observation)["tokens"]
    permutation = [2, 4, 0, 3, 1]
    changed = dict(
        observation,
        obstacles_body=observation["obstacles_body"][:, permutation],
        obstacle_mask=observation["obstacle_mask"][:, permutation],
    )
    torch.testing.assert_close(act(changed)["tokens"], expected)
    observation["obstacles_body"][~observation["obstacle_mask"]] = float("nan")
    torch.testing.assert_close(act(observation)["tokens"], expected)
    assert torch.all(expected >= -1) and torch.all(expected <= 15 / 16)
    torch.testing.assert_close(expected * 16, (expected * 16).round())
    observation["obstacle_mask"][:] = False
    output = act(observation)
    assert torch.isfinite(output["tokens"]).all()
    torch.testing.assert_close(output["obstacle_attention"][:, -1], torch.ones(2))


def test_future_cannot_change_past_and_memory_resets(observation):
    model = NavigationTokenStudent()
    sequence = {
        key: value[:, None].repeat(1, 4, *([1] * (value.ndim - 1)))
        for key, value in observation.items()
    }
    resets = torch.zeros(2, 4, dtype=torch.bool)
    before = model.forward_sequence(sequence, resets)
    sequence["start_goal_body"][:, 2:] += 100
    after = model.forward_sequence(sequence, resets)
    torch.testing.assert_close(before["tokens"][:, :2], after["tokens"][:, :2])
    first = model.forward_step(observation)
    reset = model.forward_step(
        observation, torch.randn(2, 128), episode_start=torch.ones(2, dtype=torch.bool)
    )
    torch.testing.assert_close(first["hidden"], reset["hidden"])


@pytest.mark.parametrize("field", ["motion_id", "phase", "route_body", "future_reference"])
def test_actor_rejects_privileged_fields(observation, field):
    observation[field] = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="Public actor fields"):
        VariationalNavigationStudent().prior_step(observation)


def test_posterior_parameters_cannot_affect_prior(observation):
    model = VariationalNavigationStudent()
    before = model.prior_step(observation)
    with torch.no_grad():
        for parameter in model.posterior.parameters():
            parameter.add_(torch.randn_like(parameter))
    after = model.prior_step(observation)
    torch.testing.assert_close(before["tokens"], after["tokens"])
    torch.testing.assert_close(before["prior_mean"], after["prior_mean"])
    assert "privileged_state" not in inspect.signature(model.prior_step).parameters
    assert "future_reference" not in inspect.signature(model.prior_step).parameters


def test_posterior_uses_privileged_state_without_changing_public_memory(observation):
    model = VariationalNavigationStudent()
    a = model.posterior_step(observation, torch.zeros(2, 1645), torch.zeros(2, 640))
    b = model.posterior_step(observation, torch.ones(2, 1645), torch.ones(2, 640))
    assert not torch.allclose(a["posterior_mean"], b["posterior_mean"])
    torch.testing.assert_close(a["prior_mean"], b["prior_mean"])
    torch.testing.assert_close(a["hidden"], b["hidden"])


def test_masked_command_is_not_zero_command(observation):
    proprio = observation["proprio"]
    absent = masked_controls(proprio)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    hidden_values = torch.full((2, 8), float("nan"))
    torch.testing.assert_close(absent, masked_controls(proprio, hidden_values, mask))
    mask[:, 2] = True
    given_zero = masked_controls(proprio, torch.zeros(2, 8), mask)
    assert not torch.equal(absent, given_zero)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="masked together"):
        masked_controls(proprio, torch.zeros(2, 8), mask)
    mask[:, 1] = True
    with pytest.raises(ValueError, match="unit"):
        masked_controls(proprio, torch.zeros(2, 8), mask)


def test_kl_matches_torch_and_reparameterized_gradients(observation):
    values = [torch.randn(2, 32) for _ in range(4)]
    qm, ql, pm, pl = values
    expected = kl_divergence(
        Independent(Normal(qm, (ql / 2).exp()), 1),
        Independent(Normal(pm, (pl / 2).exp()), 1),
    )
    torch.testing.assert_close(diagonal_gaussian_kl(*values), expected)
    model = VariationalNavigationStudent()
    output = model.posterior_step(
        observation, torch.randn(2, 1645), torch.randn(2, 640), epsilon=torch.randn(2, 32)
    )
    (output["tokens"].square().mean() + 0.01 * output["kl"].mean()).backward()
    for module in (model.prior, model.posterior, model.token_adapter, model.memory):
        assert sum(p.grad.abs().sum().item() for p in module.parameters()) > 0


class SyntheticDecoder(nn.Module):
    """Tiny differentiable stand-in; actual SONIC parity is a separate artifact check."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(29), requires_grad=False)

    def forward(self, tokens, proprio):
        return tokens[..., :29] * self.scale + proprio[..., :29]


def test_multi_response_loss_does_not_average_incompatible_actions():
    decoder = SyntheticDecoder()
    student = torch.zeros(1, 64, requires_grad=True)
    teacher = torch.stack([-torch.ones(64), torch.ones(64) * 15 / 16])[None]
    proprio = torch.zeros(1, 930)
    actions = decoder(teacher, proprio[:, None])
    valid = torch.ones(1, 2, dtype=torch.bool)
    losses = admissible_imitation_loss(student, proprio, teacher, actions, valid, decoder)
    assert losses["selected_mode"].item() == 1
    assert losses["action_mse"].item() == pytest.approx((15 / 16) ** 2)
    losses["loss"].backward()
    assert student.grad.sum() < 0
    assert decoder.scale.grad is None
    with pytest.raises(ValueError, match="at least one"):
        admissible_imitation_loss(student, proprio, teacher, actions, ~valid, decoder)
    with pytest.raises(ValueError, match="inconsistent"):
        admissible_imitation_loss(student, proprio, teacher, actions + 1, valid, decoder)


def test_variational_objective_and_decoder_freeze(observation):
    decoder = SyntheticDecoder()
    model = VariationalNavigationStudent()
    output = model.posterior_step(observation, torch.randn(2, 1645), torch.randn(2, 640))
    teacher = torch.zeros(2, 64)
    actions = decoder(teacher, observation["proprio"])
    result = variational_imitation_loss(
        output, observation["proprio"], teacher, actions, decoder, beta=0.01
    )
    torch.testing.assert_close(result["loss"], result["action_mse"] + 0.01 * result["kl"])
    result["loss"].backward()
    assert decoder.scale.grad is None
    decoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        variational_imitation_loss(
            output, observation["proprio"], teacher, actions, decoder, beta=0.01
        )


def test_pre_quantized_teacher_tokens_are_rejected(observation):
    decoder = SyntheticDecoder()
    proprio = observation["proprio"]
    teacher = torch.full((2, 1, 64), 0.01)
    with pytest.raises(ValueError, match="post-FSQ"):
        admissible_imitation_loss(
            torch.zeros(2, 64),
            proprio,
            teacher,
            decoder(teacher, proprio[:, None]),
            torch.ones(2, 1, dtype=torch.bool),
            decoder,
        )


def test_censored_future_is_missing_not_zero():
    prediction = torch.zeros(2, 3, requires_grad=True)
    actual = torch.tensor([[1.0, 1, 1], [float("nan")] * 3])
    mask = torch.tensor([True, False])
    result = endpoint_auxiliary_loss(prediction, actual, mask)
    assert result.item() == 0.5
    result.backward()
    assert torch.equal(prediction.grad[1], torch.zeros(3))
    assert endpoint_auxiliary_loss(prediction, actual, torch.zeros(2, dtype=torch.bool)) == 0


@pytest.fixture
def query_record(observation):
    actor = {k: v[0].numpy() for k, v in observation.items()}
    return {
        "motion_id": "synthetic-train",
        "split": "train",
        "teacher_checkpoint_sha256": "a" * 64,
        "sample_origin": "student_rollout",
        "label_source": "queried_teacher_on_student_state",
        "actor_observation": actor,
        "teacher_query_observation_sha256": public_observation_sha256(actor),
        "state_sha256": "b" * 64,
        "teacher_query_state_sha256": "b" * 64,
        "scene_sha256": "c" * 64,
        "teacher_query_scene_sha256": "c" * 64,
        "goal_sha256": "d" * 64,
        "teacher_query_goal_sha256": "d" * 64,
        "decision_time_s": 1.0,
        "observation_time_s": 1.0,
        "teacher_query_state_time_s": 1.0,
        "teacher_qualification": "scene_teacher_qualified",
        "qualification_receipt_sha256": "e" * 64,
        "state_snapshot_receipt_sha256": "f" * 64,
    }


def check_query(record):
    return validate_query_record(
        record, allowed_motion_ids={"synthetic-train"}, teacher_sha256="a" * 64
    )


def test_query_state_history_binding(query_record):
    assert check_query(query_record) == query_record["teacher_query_observation_sha256"]
    changed = copy.deepcopy(query_record)
    changed["actor_observation"]["proprio"][0] += 1
    with pytest.raises(ValueError, match="history"):
        check_query(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "development"),
        ("motion_id", "dev"),
        ("teacher_checkpoint_sha256", "0" * 64),
        ("label_source", "executed_teacher"),
        ("teacher_query_state_sha256", "0" * 64),
        ("teacher_query_scene_sha256", "0" * 64),
        ("teacher_query_goal_sha256", "0" * 64),
        ("observation_time_s", 2.0),
        ("teacher_query_state_time_s", 0.5),
        ("teacher_qualification", "tracking_only"),
        ("qualification_receipt_sha256", None),
    ],
)
def test_query_rejects_stale_or_unqualified_labels(query_record, field, value):
    query_record[field] = value
    with pytest.raises(ValueError):
        check_query(query_record)


def test_nested_scenes_do_not_become_switch_or_physics_evidence():
    shared = {"motion_id": "synthetic", "teacher_eligible": False, "teacher_actions_present": False}
    result = audit_scene_pairing(
        [
            dict(shared, requested_obstacles=3, cells=[1, 2, 3]),
            dict(shared, requested_obstacles=5, cells=[1, 2, 3, 4, 5]),
        ]
    )
    assert result["nested_scene_pairs"] == 1
    assert result["validated_obstacle_switch_pairs"] is None
    assert result["executed_scene_teacher_labels"] == 0
    assert np.isfinite(result["scenes"])


def test_known_map_frame_and_primitive_contract():
    obstacle = {
        "shape": "beam",
        "center_xyz": [10, 22, 1],
        "full_dimensions_xyz": [0.3, 1.6, 0.2],
        "quaternion_wxyz": [2**-0.5, 0, 0, 2**-0.5],
    }
    inputs = dict(
        proprio=np.zeros(930),
        root_xyz=[10, 20, 1],
        root_wxyz=[2**-0.5, 0, 0, 2**-0.5],
        start_xyz=[10, 19, 1],
        goal_xyz=[10, 23, 1],
        obstacles=[obstacle],
    )
    observation = navigation_observation(**inputs)
    assert "route_body" not in observation
    np.testing.assert_allclose(observation["start_goal_body"], [-1, 0, 0, 3, 0, 0], atol=1e-6)
    np.testing.assert_allclose(observation["obstacles_body"][0, :3], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(observation["obstacles_body"][0, -3:], [1, 0, 0])
    obstacle["shape"] = "mesh"
    with pytest.raises(ValueError, match="primitives only"):
        navigation_observation(**inputs)
    obstacle["shape"] = "sphere"
    with pytest.raises(ValueError, match="equal full diameters"):
        navigation_observation(**inputs)
    obstacle["full_dimensions_xyz"] = [0.4, 0.4, 0.4]
    observation = navigation_observation(**inputs)
    np.testing.assert_allclose(observation["obstacles_body"][0, -3:], [0, 0, 1])
    inputs["obstacles"] = [obstacle] * 6
    with pytest.raises(ValueError, match="More than five"):
        navigation_observation(**inputs)
