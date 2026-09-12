"""Keep failure censoring and saved simulator trajectories auditable."""

import json
from types import SimpleNamespace

import numpy as np

from gear_sonic.research.hindsight_training.qualify import TrackingQualificationCallback
from gear_sonic.research.hindsight_training.render_qualification import valid_steps
from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback


def test_failure_prefix_excludes_reset_and_handles_float_roundoff():
    # Native float32 progress encodes 65 surviving steps of a 399-step motion.
    progress = float(np.float32(65 / 399))
    assert valid_steps(progress, 399, 398) == 65
    assert valid_steps(1.0, 399, 398) == 398
    assert valid_steps(0.0, 399, 398) == 0


def test_export_preserves_motion_order_positions_and_timing(tmp_path, monkeypatch):
    callback = TrackingQualificationCallback(eval_frequency=1, output_dir=str(tmp_path))
    callback.env = SimpleNamespace(
        env=SimpleNamespace(step_dt=0.02),
        motion_command=SimpleNamespace(cmd_body_names=["pelvis", "torso_link"]),
    )
    callback.gt_pos_all = [np.ones((3, 2, 3)), np.full((4, 2, 3), 2.0)]
    callback.pred_pos_all = [x + 0.1 for x in callback.gt_pos_all]
    result = {"all_metrics_dict": {"motion_keys": ["motion_b", "motion_a"]}}
    monkeypatch.setattr(ImEvalCallback, "_post_evaluate_policy", lambda self, result: result)
    assert callback._post_evaluate_policy(result) is result
    for index, key in enumerate(result["all_metrics_dict"]["motion_keys"]):
        data = np.load(tmp_path / f"{key}.npz")
        np.testing.assert_array_equal(data["reference"], callback.gt_pos_all[index])
        np.testing.assert_array_equal(data["tracked"], callback.pred_pos_all[index])
    assert json.loads((tmp_path / "trajectory-contract.json").read_text())["dt"] == 0.02
