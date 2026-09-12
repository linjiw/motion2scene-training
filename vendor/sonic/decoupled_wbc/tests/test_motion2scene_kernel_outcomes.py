import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_kernel_outcomes import fit_kernel_outcomes, predict_kernel_outcomes


def test_nonlinear_feasibility_and_conditional_time_keep_complementary_responses():
    x = np.array([[-1.0], [0.0], [1.0]])
    passed = np.array([[1, 0], [0, 1], [1, 0]], bool)
    times = np.where(passed, 4.0, np.nan)
    mask = np.ones_like(passed)
    model = fit_kernel_outcomes(x, [15] * 3, passed, times, mask, mask)
    assert [predict_kernel_outcomes(model, row, 15, mask[0])["action"] for row in x] == [0, 1, 0]
    # Failure costs can contain arbitrary placeholders; only successes fit time.
    changed = times.copy()
    changed[~passed] = -999
    other = fit_kernel_outcomes(x, [15] * 3, passed, changed, mask, mask)
    for row in x:
        a = predict_kernel_outcomes(model, row, 15, mask[0])
        b = predict_kernel_outcomes(other, row, 15, mask[0])
        np.testing.assert_array_equal(a["passage_time_s"], b["passage_time_s"])


def test_unknown_feasibility_is_not_silently_imputed():
    model = fit_kernel_outcomes(
        [[0]], [15], [[True, False]], [[4, np.nan]], [[True, False]], [[True, True]]
    )
    with pytest.raises(ValueError, match="no observed"):
        predict_kernel_outcomes(model, [0], 15, [True, True])


def test_fast_failure_does_not_displace_predicted_success():
    model = fit_kernel_outcomes(
        [[0]], [15], [[False, True]], [[0.01, 4]], [[True, True]], [[True, True]]
    )
    assert predict_kernel_outcomes(model, [0], 15, [True, True])["action"] == 1
