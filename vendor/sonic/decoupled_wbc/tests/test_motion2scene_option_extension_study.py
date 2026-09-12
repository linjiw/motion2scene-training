from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_option_extension_study import SEEDS, WINDOWS, specification
from motion2scene_project_splice_prior import project_and_splice


def test_candidate_and_physical_qualification_budgets_are_matched():
    spec = specification()
    assert len(SEEDS) == len(set(SEEDS)) == len(WINDOWS) == 4
    assert spec["qualification"]["attempts_per_arm"] == len(SEEDS) * len(spec["schedules"])
    assert spec["schedules"] == [
        dict(entry_tick=15, return_tick=265),
        dict(entry_tick=50, return_tick=265),
    ]
    assert spec["primary_bank_changes"] is False
    assert spec["reserved_evaluation_queries"] == 0


def test_capability_tasks_are_fixed_without_a_candidate_or_outcome_input():
    tasks = specification()["independent_capability_tasks"]
    assert len(tasks) == len({t["task_id"] for t in tasks}) == 12
    assert len({(tuple(t["center_xy_m"]), t["length_m"], t["underside_m"]) for t in tasks}) == 12
    assert {t["length_m"] for t in tasks} == {0.10, 0.75}
    assert {t["underside_m"] for t in tasks} == {1.24, 1.30}


def test_common_repair_preserves_neutral_prefix_and_tail_for_both_input_families():
    neutral = np.zeros((180, 36))
    neutral[:, 3] = 1
    limits = np.tile([-1.0, 1.0], (29, 1))
    rng = np.random.default_rng(8)
    for amplitude in (0.4, 1.3):
        raw = neutral.copy()
        raw[:, 7:] = rng.normal(size=(180, 29)) * amplitude
        repaired, projected, weight = project_and_splice(neutral, raw, limits)
        np.testing.assert_array_equal(repaired[weight == 0], neutral[weight == 0])
        np.testing.assert_array_equal(repaired[weight == 1], projected[weight == 1])
        assert np.max(abs(repaired[:, 7:])) <= 1
        # Both declared entries and the return lie in copied neutral intervals.
        assert np.all(weight[:31] == 0) and np.all(weight[150:] == 0)
