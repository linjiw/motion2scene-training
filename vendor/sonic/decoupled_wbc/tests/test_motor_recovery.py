import pytest
import torch

from gear_sonic.research.scene_distillation.motor_runtime import hold_reference_tail
from gear_sonic.research.scene_distillation.motor_training import (
    FullCommandFoundation,
    current_frame_extension,
    initialize_motor,
    public_motor_loss,
)
from gear_sonic.research.scene_distillation.reference_layout import pack_reference, unpack_reference
from gear_sonic.research.scene_distillation.transformer import TransformerMotionFoundation


def test_extended_motor_migration_preserves_original_tokens():
    torch.manual_seed(8)
    base = TransformerMotionFoundation(width=32, layers=1, heads=4, latent_dim=8)
    model = FullCommandFoundation(width=32, layers=1, heads=4, latent_dim=8)
    initialize_motor(model, {"model": base.state_dict()})
    p, c = torch.randn(3, 930), torch.randn(3, 114)
    c[:, :2] = torch.tensor([0.0, 1.0])
    mask = torch.ones_like(c, dtype=torch.bool)
    torch.testing.assert_close(
        base.prior_step(p, c[:, :79], mask[:, :79])["tokens"],
        model.prior_step(p, c, mask)["tokens"],
        rtol=0,
        atol=0,
    )
    with torch.no_grad():
        model.current_extension[-1].weight.normal_(0, 0.1)
    mask[:, 79:] = False
    before = model.prior_step(p, c, mask)["tokens"]
    c[:, 79:] = float("nan")
    torch.testing.assert_close(before, model.prior_step(p, c, mask)["tokens"], rtol=0, atol=0)


def test_current_frame_reconstruction_cannot_read_later_frames():
    c = torch.randn(3, 79)
    f = torch.randn(3, 10, 64)
    f[:, 0, :29] = c[:, 50:]
    expected = f[:, 0, 29:].clone()
    f[:, 1:] = float("nan")
    torch.testing.assert_close(current_frame_extension(pack_reference(f), c), expected)
    f[:, 0, 0] += 1
    with pytest.raises(ValueError, match="Current target mismatch"):
        current_frame_extension(pack_reference(f), c)


def test_teacher_horizon_control_preserves_prefix_and_shape():
    x = torch.randn(2, 1, 640)
    torch.testing.assert_close(hold_reference_tail(x, 10), x, rtol=0, atol=0)
    y = unpack_reference(hold_reference_tail(x, 2))
    torch.testing.assert_close(y[:, :, :2], unpack_reference(x)[:, :, :2])
    torch.testing.assert_close(y[:, :, 2:], y[:, :, 1:2].expand(-1, -1, 8, -1))
    with pytest.raises(ValueError):
        hold_reference_tail(x, 0)


def test_public_objective_updates_student_through_frozen_decoder():
    model = TransformerMotionFoundation(width=32, layers=1, heads=4, latent_dim=8)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(994, 29)
            self.requires_grad_(False)

        def forward(self, tokens, proprio):
            return self.layer(torch.cat([tokens, proprio], -1))

    decoder = Decoder()
    batch = dict(
        proprio=torch.randn(4, 930),
        controls=torch.randn(4, 79),
        control_mask=torch.ones(4, 79, dtype=torch.bool),
        teacher_tokens=torch.randn(4, 64),
        teacher_actions=torch.randn(4, 29),
    )
    batch["controls"][:, :2] = torch.tensor([0.0, 1.0])
    loss = public_motor_loss(model, decoder, batch)["loss"]
    loss.backward()
    assert model.history_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in decoder.parameters())


def test_recovery_tracker_requires_unbroken_teacher_takeover():
    from gear_sonic.research.scene_distillation.online_motor import RecoveryTracker

    tracker = RecoveryTracker(3, "cpu", horizon=4, stable_ticks=2)
    active = tracker.begin(torch.tensor([True, True, True]))
    assert active.all()
    for tick in range(4):
        boundary = torch.tensor([False, tick == 1, False])
        supported = torch.tensor([True, True, False])
        tracker.observe(boundary, supported)
    assert tracker.report() == dict(
        started=3,
        recovered=1,
        failed=2,
        censored=0,
        horizon_ticks=4,
        stable_ticks=2,
        qualification="local_tracking_takeover_only",
    )
    tracker.begin(torch.tensor([True, False, False]))
    assert tracker.report()["censored"] == 1


def test_reference_layout_matches_native_separate_position_velocity_packing():
    q = torch.arange(290, dtype=torch.float32).reshape(1, 10, 29)
    v = q + 1000
    rotation = torch.arange(60, dtype=torch.float32).reshape(1, 10, 6) + 2000
    native_command = torch.cat([q.reshape(1, -1), v.reshape(1, -1)], dim=1)
    native_input = torch.cat([native_command.reshape(1, 10, 58), rotation], dim=-1).reshape(1, 640)
    physical = unpack_reference(native_input)
    torch.testing.assert_close(physical[..., :29], q)
    torch.testing.assert_close(physical[..., 29:58], v)
    torch.testing.assert_close(physical[..., 58:], rotation)
    torch.testing.assert_close(pack_reference(physical), native_input)
    assert native_input.reshape(1, 10, 64)[0, 0, 29] == q[0, 1, 0]
    assert physical[0, 0, 29] == v[0, 0, 0]


def test_current_orientation_reader_cannot_read_future_observations():
    from types import SimpleNamespace

    from gear_sonic.research.scene_distillation.motor_runtime import current_orientation_observation

    metadata = SimpleNamespace(
        tokenizer_obs_names=["unused", "motion_anchor_ori_b_mf_nonflat"],
        tokenizer_obs_dims={"unused": (12,), "motion_anchor_ori_b_mf_nonflat": (10, 6)},
    )
    x = torch.full((2, 72), float("nan"))
    x[:, 12:18] = torch.arange(6, dtype=torch.float32)
    out = current_orientation_observation(metadata, {"tokenizer": x})
    torch.testing.assert_close(out, torch.arange(6, dtype=torch.float32)[None].expand(2, -1))


@pytest.mark.parametrize("success", [False, True])
def test_stopping_collection_eligibility_requires_measured_task_success(tmp_path, success):
    import json

    import numpy as np

    from gear_sonic.research.scene_distillation.stopping_teacher import StoppingTeacherCallback

    callback = object.__new__(StoppingTeacherCallback)
    task_file = tmp_path / "request.json"
    task_file.write_text("{}")
    callback.config = {"teacher_sha256": "test", "task_path": str(task_file)}
    callback.rows = [{"proprio": np.zeros(930, np.float32)} for _ in range(2)]
    callback.handles = []
    score = dict(control_steps=2, navigation_success=success, fell=False)
    (tmp_path / "task-result.json").write_text(json.dumps(score))
    callback._complete_task({"motion_id": "00001"}, score, tmp_path)
    result = json.loads((tmp_path / "collection.json").read_text())
    assert result["episodes"][0]["eligible"] == success
    assert result["scene_task_success"] == success
    assert result["scene_teacher_qualified"] is False
    with np.load(tmp_path / "teacher-episode.npz") as data:
        assert data["query_mask"].tolist() == [success, success]
