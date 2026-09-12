"""Readout readiness requires every model and assigned measured encounter."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_watch_primary_readouts import missing_inputs


def plan(tmp_path):
    return dict(
        runs=[
            dict(
                rounds=[
                    dict(
                        round_index=i,
                        **{
                            name: str(tmp_path / f"{i}_{name}")
                            for name in (
                                "expected_postupdate_training_result_path",
                                "expected_postupdate_policy_path",
                                "expected_teacher_result_path",
                                "expected_student_result_path",
                            )
                        },
                    )
                    for i in range(3)
                ]
            )
        ]
    )


def test_bootstrap_has_no_new_student_requirement(tmp_path):
    p = plan(tmp_path)
    missing = missing_inputs(p, 2)
    assert len(missing) == 11
    assert not any("0_expected_student" in path for path in missing)
    for path in missing:
        Path(path).touch()
    assert missing_inputs(p, 2) == []


def test_late_missing_model_prevents_partial_arm_comparison(tmp_path):
    p = plan(tmp_path)
    missing = missing_inputs(p, 2)
    for path in missing[:-1]:
        Path(path).touch()
    assert missing_inputs(p, 2) == [missing[-1]]


def test_unassigned_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        missing_inputs(plan(tmp_path), 4)
