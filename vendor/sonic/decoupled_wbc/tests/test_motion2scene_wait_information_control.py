import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/motion2scene_wait_information_control.py"
)
spec = importlib.util.spec_from_file_location("wait_information", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_one_time_keeps_late_schedules_but_cannot_use_future_information():
    passed = np.array([[False, True, False], [False, False, True]])
    times = np.where(passed, 2.0, np.nan)
    known = np.ones_like(passed)
    initial = module.one_time_schedule_limit(["same", "same"], passed, times, known)
    assert initial["maximum_passages"] == 1
    assert initial["groups"][0]["optimal_schedule_indices"] == [1, 2]
    # Both passing schedules enter only at the second decision.
    sequential = module.maximum_causal_passages(
        np.array([["same", "a"], ["same", "b"]]),
        [1, 2],
        [None, 2, 2],
        passed,
        times,
        known,
    )
    assert sequential == 2
    assert module.one_time_schedule_limit(["a", "b"], passed, times, known)["maximum_passages"] == 2


def test_all_failing_context_retained_and_success_time_breaks_equal_count_tie():
    passed = np.array([[True, True], [False, False]])
    result = module.one_time_schedule_limit(
        ["same", "same"], passed, [[3.0, 2.0], [np.nan, np.nan]], np.ones_like(passed)
    )
    assert result["assigned_contexts"] == 2
    assert result["maximum_passages"] == 1
    assert result["groups"][0]["optimal_schedule_indices"] == [1]


def test_unknown_cannot_be_an_exact_information_limit():
    with pytest.raises(ValueError, match="complete measured"):
        module.one_time_schedule_limit(["a"], [[True, False]], [[2.0, np.nan]], [[True, False]])
