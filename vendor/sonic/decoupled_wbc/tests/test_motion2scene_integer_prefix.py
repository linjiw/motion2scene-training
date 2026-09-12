import copy

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_integer_prefix import (
    PREFIX_KEYS,
    paired_prefix_on_ticks,
)


def fixture():
    values = {key: np.zeros((100, 3)) for key in PREFIX_KEYS}
    values["motion_time_s"] = np.arange(100) * 0.02
    return values


def test_later_prefix_accepts_numeric_grid_roundoff_without_changing_history():
    first = fixture()
    assert first["motion_time_s"][35] != 35 / 50
    for entry in (1.0, 1.4):
        result = paired_prefix_on_ticks(first, copy.deepcopy(first), entry)
        assert result["exact_match"]
        assert result["frames"] == round(entry * 50)
    changed = copy.deepcopy(first)
    changed["dof_pos"][45, 1] = 1e-8
    assert not paired_prefix_on_ticks(first, changed, 1.0)["exact_match"]


def test_repeated_missing_or_offgrid_clock_is_rejected():
    first = fixture()
    for modification in ("repeat", "shift", "short"):
        changed = copy.deepcopy(first)
        if modification == "repeat":
            changed["motion_time_s"][35] = changed["motion_time_s"][34]
        elif modification == "shift":
            changed["motion_time_s"][35] += 0.001
        else:
            changed["motion_time_s"] = changed["motion_time_s"][:20]
        with pytest.raises(ValueError, match="clock"):
            paired_prefix_on_ticks(first, changed, 1.0)
