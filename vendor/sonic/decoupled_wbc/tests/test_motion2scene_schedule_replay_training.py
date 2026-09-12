"""Replay weights retain their original encounter/phase positions before fitting."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_train_timed_schedules import weighted_targets  # noqa: E402


def groups():
    def target(tick, *, available=True, complete=True):
        return dict(
            phase_tick=tick,
            available=available,
            complete_legal_action_table=complete,
            teacher_action=0 if complete else None,
            legal_mask=[True, True],
        )

    return [
        dict(targets=[target(15), target(50, available=False), target(70)]),
        dict(targets=[target(15, complete=False), target(50), target(70)]),
    ]


def test_unavailable_middle_slot_does_not_shift_later_weights():
    rows, weights = weighted_targets(groups(), [0.1, 0, 0.2, 0, 0.3, 0.4])
    assert [r["phase_tick"] for r in rows] == [15, 70, 50, 70]
    assert np.array_equal(weights, [0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize("bad", [[1] * 5, [1, 0, 1, 0, float("nan"), 1], [1, 0, 1, 0, -1, 1]])
def test_bad_shape_and_nonfinite_or_negative_mass_rejected(bad):
    with pytest.raises(ValueError, match="every recorded target slot"):
        weighted_targets(groups(), bad)


def test_uniform_mass_cannot_drop_complete_decisions_or_invent_missing_ones():
    with pytest.raises(ValueError, match="unavailable phase"):
        weighted_targets(groups(), [1, 1, 1, 0, 1, 1])
    with pytest.raises(ValueError, match="uniform replay component"):
        weighted_targets(groups(), [0, 0, 1, 0, 1, 1])


def test_no_replay_preserves_legacy_available_rows():
    rows, weights = weighted_targets(groups())
    assert len(rows) == 5 and weights is None
