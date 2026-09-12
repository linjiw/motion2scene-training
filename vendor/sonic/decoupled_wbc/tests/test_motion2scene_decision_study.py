from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_decision_study import OutcomeTrees, feature_keys


def row(x, passed, times):
    return dict(
        features=[x],
        phase_tick=15,
        pass_labels=passed,
        passage_time_s=times,
        admitted=[True, True],
        legal_mask=[True, True],
    )


def test_outcome_head_preserves_complementarity_and_successful_cost():
    rows = [
        row(0, [True, False], [4, None]),
        row(1, [True, True], [4, 3]),
        row(2, [False, True], [None, 5]),
    ]
    learner = OutcomeTrees(rows)
    assert [learner.choose(r) for r in rows] == [0, 1, 1]


def test_all_failed_labels_are_finite_and_do_not_invent_time():
    failed = row(0, [False, False], [None, None])
    assert OutcomeTrees([failed]).choose(failed) == 0


def test_exact_keys_treat_signed_zero_as_equal_but_preserve_small_changes():
    keys = feature_keys([[[0.0]], [[-0.0]], [[np.nextafter(0.0, 1.0)]]])
    assert keys[0, 0] == keys[1, 0]
    assert keys[0, 0] != keys[2, 0]
