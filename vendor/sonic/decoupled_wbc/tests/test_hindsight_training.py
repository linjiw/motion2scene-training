"""Contract checks use synthetic tensors, never claim physical training evidence."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from gear_sonic.research.hindsight_training.observations import scene_features
from gear_sonic.research.hindsight_training.records import load_teacher_shard
from gear_sonic.research.hindsight_training.runtime import cuda_preflight, sha
from gear_sonic.research.hindsight_training.student import (
    FrozenSonicDecoder,
    SceneTokenStudent,
    distillation_loss,
    mlp,
)


@pytest.fixture
def observation():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    return {
        "proprio": torch.randn(2, 930),
        "start_goal_body": torch.randn(2, 6),
        "obstacles_body": torch.randn(2, 5, 15),
        "obstacle_mask": torch.tensor([[True, True, True, False, False]] * 2),
        "route_body": torch.randn(2, 16, 3),
        "route_mask": torch.ones(2, 16, dtype=torch.bool),
    }


def test_obstacle_set_permutation_padding_and_fsq(observation):
    model = SceneTokenStudent()
    expected = model(**observation)
    order = torch.tensor([2, 4, 0, 3, 1])
    changed = dict(
        observation,
        obstacles_body=observation["obstacles_body"][:, order],
        obstacle_mask=observation["obstacle_mask"][:, order],
    )
    torch.testing.assert_close(model(**changed), expected)
    observation["obstacles_body"][~observation["obstacle_mask"]] = float("nan")
    torch.testing.assert_close(model(**observation), expected)
    assert torch.all(expected >= -1) and torch.all(expected <= 15 / 16)
    torch.testing.assert_close(expected * 16, (expected * 16).round())
    observation["obstacle_mask"][:] = False
    assert torch.isfinite(model(**observation)).all()


def test_depth_is_optional_and_explicit(observation):
    model = SceneTokenStudent(camera=True)
    with pytest.raises(ValueError, match="Camera profile"):
        model(**observation)
    assert model(**observation, depth=torch.zeros(2, 2, 120, 160)).shape == (2, 64)
    with pytest.raises(ValueError, match="camera-disabled"):
        SceneTokenStudent()(**observation, depth=torch.zeros(2, 2, 120, 160))


def test_world_body_frames_agree_after_ninety_degree_turn():
    values = scene_features(
        root_xyz=[10, 20, 1],
        root_wxyz=[2**-0.5, 0, 0, 2**-0.5],
        start_xyz=[10, 20, 1],
        goal_xyz=[10, 22, 1],
        route_xyz=[[10, 21, 1], [10, 22, 1]],
        obstacles=[
            {
                "center_xyz": [10, 22, 1],
                "quaternion_wxyz": [2**-0.5, 0, 0, 2**-0.5],
                "full_dimensions_xyz": [1, 2, 3],
                "shape": "box",
            }
        ],
    )
    np.testing.assert_allclose(values["start_goal_body"][3:], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(values["obstacles_body"][0, :3], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(values["obstacles_body"][0, 6:12], [1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(values["route_body"][1], [2, 0, 0], atol=1e-6)


def test_decoder_identity_and_gradient_contract(observation):
    reference = mlp([994, 2048, 2048, 1024, 1024, 512, 512, 29])
    weights = {
        "actor_module.decoders.g1_dyn.module." + k: v for k, v in reference.state_dict().items()
    }
    decoder = FrozenSonicDecoder(weights)
    student = SceneTokenStudent()
    prediction = student(**observation)
    teacher = torch.zeros_like(prediction)
    action = reference(torch.cat([teacher, observation["proprio"]], dim=-1)).detach()
    losses = distillation_loss(prediction, teacher, action, observation["proprio"], decoder)
    losses["loss"].backward()
    assert sum(float(p.grad.abs().sum()) for p in student.parameters() if p.grad is not None) > 0
    assert all(p.grad is None and not p.requires_grad for p in decoder.parameters())
    with pytest.raises(ValueError, match="disagree"):
        distillation_loss(prediction, teacher, action + 1, observation["proprio"], decoder)
    with pytest.raises(RuntimeError):
        FrozenSonicDecoder({})


def test_nvml_failure_does_not_veto_working_cuda():
    real_ones = torch.ones
    with patch(
        "subprocess.run", return_value=subprocess.CompletedProcess([], 18, "NVML mismatch", "")
    ), patch("torch.cuda.is_available", return_value=True), patch(
        "torch.ones", side_effect=lambda shape, device: real_ones(shape)
    ), patch(
        "torch.cuda.synchronize"
    ), patch(
        "torch.cuda.empty_cache"
    ), patch(
        "torch.cuda.mem_get_info", return_value=(3 * 1024**3, 16 * 1024**3)
    ), patch(
        "torch.cuda.get_device_name", return_value="fixture"
    ):
        result = cuda_preflight()
    assert result["launch_allowed"] and result["nvml_exit"] == 18
    with patch(
        "subprocess.run", return_value=subprocess.CompletedProcess([], 0, "healthy", "")
    ), patch("torch.ones", side_effect=RuntimeError("CUDA broken")):
        assert not cuda_preflight()["launch_allowed"]


def test_normal_stop_writes_receipt_before_native_early_return(tmp_path):
    from gear_sonic.research.hindsight_training.tracker import TrackingReceiptCallback

    packet = tmp_path / "plan.json"
    packet.write_text(json.dumps({"tracking": {"iterations": 200}}))
    callback = TrackingReceiptCallback(packet, tmp_path)
    callback.old_step = object()
    callback.report = lambda step, status: {"iteration": step, "state": status}
    (tmp_path / "model_step_000200.pt").write_bytes(b"synthetic callback checkpoint fixture")
    control = SimpleNamespace(should_training_stop=True)
    state = SimpleNamespace(global_step=200)
    env = SimpleNamespace(step=None)
    assert callback.on_step_end(None, state, control, env=env) is control
    receipt = tmp_path / "training-receipt.json"
    assert json.loads(receipt.read_text())["state"] == "complete"
    digest = sha(receipt)
    callback.on_train_end(None, state, control, env=env)
    assert sha(receipt) == digest


def test_reference_only_and_future_teacher_data_rejected(tmp_path, observation):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"label_source": "reference_pose"}))
    with pytest.raises(ValueError, match="label_source"):
        load_teacher_shard(
            tmp_path / "missing.npz", meta, allowed_motion_ids={"a"}, decoder_sha256="x"
        )
    data = {k: v.numpy() for k, v in observation.items()}
    data.update(
        teacher_tokens=np.zeros((2, 64), np.float32),
        teacher_action_mean=np.zeros((2, 29), np.float32),
        motion_id=np.array(["a", "a"]),
        observation_time_s=np.array([2.0, 3.0]),
        decision_time_s=np.array([1.0, 2.0]),
    )
    shard = tmp_path / "synthetic.npz"
    np.savez(shard, **data)
    receipt = tmp_path / "fixture-receipt.json"
    receipt.write_text('{"synthetic_test_fixture": true}')
    metadata = {
        "label_source": "executed_teacher",
        "state_source": "measured_simulator_state",
        "route_source": "provided_command",
        "token_representation": "post_fsq_64",
        "decoder_sha256": "x",
        "split": "train",
        "shard_sha256": sha(shard),
        "execution_receipts": [{"path": str(receipt), "sha256": sha(receipt)}],
    }
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Future observations"):
        load_teacher_shard(shard, meta, allowed_motion_ids={"a"}, decoder_sha256="x")
    with pytest.raises(ValueError, match="development or unknown"):
        load_teacher_shard(shard, meta, allowed_motion_ids={"b"}, decoder_sha256="x")
