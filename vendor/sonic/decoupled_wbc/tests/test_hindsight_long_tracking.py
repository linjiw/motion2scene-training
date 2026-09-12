"""Continuation and online telemetry contracts; no simulator or remote writes."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gear_sonic.research.hindsight_training.audit_long import finish
from gear_sonic.research.hindsight_training.long_tracker import (
    TrackingWandbCallback,
    restore_optimizer_history,
)
from gear_sonic.research.hindsight_training.runtime import sha


def test_optimizer_continuation_matches_uninterrupted_update():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    inputs = torch.randn(8, 3)
    for _ in range(4):
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        optimizer.step()
        scheduler.step()
    restored = copy.deepcopy(model)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-5)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=3, gamma=0.9)
    checkpoint = copy.deepcopy(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": scheduler.state_dict(),
        }
    )
    assert restore_optimizer_history(restored_optimizer, restored_scheduler, checkpoint) == 2
    for network, opt, sched in (
        (model, optimizer, scheduler),
        (restored, restored_optimizer, restored_scheduler),
    ):
        opt.zero_grad()
        network(inputs).square().mean().backward()
        opt.step()
        sched.step()
    for original, continued in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, continued, rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]
    with pytest.raises(ValueError, match="uninitialized"):
        restore_optimizer_history(restored_optimizer, restored_scheduler, checkpoint)


def test_telemetry_joins_native_metrics_in_one_wandb_step(tmp_path):
    receipt = SimpleNamespace(
        output=tmp_path,
        exposure={"motion": 24},
        report=lambda *args: {
            "observed_env_transitions": 24,
            "observed_env_physics_steps": 96,
            "cuda_peak_allocated_bytes": 1024**2,
            "cuda_free_total_bytes": [2 * 1024**2, 4 * 1024**2],
            "wall_seconds": 3.0,
        },
    )
    env = SimpleNamespace(_hindsight_receipt_callback=receipt)
    state = SimpleNamespace(global_step=10, is_world_process_zero=True)
    callback = TrackingWandbCallback()
    with patch("wandb.run", SimpleNamespace()), patch("wandb.log") as log:
        callback.on_log(None, state, None, logs={"reward": 2.0}, env=env)
    log.assert_called_once()
    assert log.call_args.kwargs == {"step": 10}
    assert log.call_args.args[0]["reward"] == 2.0
    assert log.call_args.args[0]["hindsight/motions_seen"] == 1
    assert (tmp_path / "metrics.jsonl").is_file()


def test_final_audit_keeps_unmeasured_quality_na(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "attempt-1").mkdir()
    checkpoint = run / "model.pt"
    checkpoint.write_bytes(b"synthetic audit fixture")
    plan = {
        "run_dir": str(run),
        "tracking": {
            "iterations": 2,
            "wall_time_cap_seconds": 1,
            "maximum_rollout_env_transitions": 48,
        },
        "train_ids": ["a"],
        "development_ids": ["b"],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "attempt-1/exit.json").write_text('{"state":"complete","exit_code":0}')
    (run / "progress.jsonl").write_text('{"iteration":1}\n{"iteration":2}\n')
    (run / "training-receipt.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "iteration": 2,
                "observed_env_transitions": 48,
                "motion_exposure_transitions": {"hindsight_a": 48},
                "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
            }
        )
    )
    finish(tmp_path)
    result = json.loads((tmp_path / "FINAL-RECEIPT.json").read_text())
    assert result["state"] == "complete" and result["teacher_quality"].startswith("UNTESTED")
    assert "b,development,0,NA,NA" in (tmp_path / "motion-training-outcomes.csv").read_text()
    manifest = json.loads((tmp_path / "FINAL-MANIFEST.json").read_text())
    assert all(sha(tmp_path / row["path"]) == row["sha256"] for row in manifest["files"])
