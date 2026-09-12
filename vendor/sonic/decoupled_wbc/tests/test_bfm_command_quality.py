import numpy as np
import pytest

from gear_sonic.research.scene_distillation.command_quality import command_metrics


def test_command_errors_exclude_reset_and_do_not_imply_navigation_success():
    reference = np.zeros((3, 4))
    measured = np.array([[3.0, 4.0, 2.0, 0.1], [0.0, 0.0, 0.0, 0.1], [999.0, 999.0, 999.0, 999.0]])
    metrics = command_metrics(reference, measured, np.array([True, True, False]))
    assert metrics["velocity_xy_rmse_mps"] == pytest.approx(np.sqrt(12.5))
    assert metrics["yaw_rate_rmse_radps"] == pytest.approx(np.sqrt(2))
    assert metrics["height_rmse_m"] == pytest.approx(0.1)
    assert metrics["measured_seconds"] == 0.04
    assert metrics["navigation_success"] is None


def test_missing_or_nonfinite_command_telemetry_is_not_a_zero_error():
    with pytest.raises(ValueError):
        command_metrics(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError):
        command_metrics(np.zeros((2, 4)), np.full((2, 4), np.nan), np.ones(2, dtype=bool))
