import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/motion2scene_response_diversity.py"
spec = importlib.util.spec_from_file_location("response_diversity", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

OPTIONS = ["neutral", "short", "sustained"]


def test_adaptation_required_does_not_imply_complementarity():
    result = module.response_summary(
        [
            dict(neutral="failure", short="pass", sustained="pass"),
            dict(neutral="failure", short="failure", sustained="pass"),
        ],
        OPTIONS,
    )
    assert result["adaptation_required_lower"] == 2
    assert result["minimum_schedule_covers"] == [["sustained"]]
    assert result["capability_minus_best_fixed_lower"] == 0
    assert result["one_fixed_covers_every_solvable_task"] is True


def test_complementary_choices_and_unsolvable_task_stay_in_denominator():
    result = module.response_summary(
        [
            dict(neutral="failure", short="pass", sustained="failure"),
            dict(neutral="failure", short="failure", sustained="pass"),
            dict.fromkeys(OPTIONS, "failure"),
        ],
        OPTIONS,
    )
    assert result["assigned_tasks"] == 3
    assert result["bank_unsolvable"] == 1
    assert result["bank_solvable_lower"] == result["bank_solvable_upper"] == 2
    assert result["capability_minus_best_fixed_lower"] == 1
    assert result["minimum_schedule_covers"] == [["short", "sustained"]]


def test_unknown_cannot_establish_complementarity_or_no_fixed_cover():
    result = module.response_summary(
        [
            dict(neutral="failure", short="pass", sustained="unknown"),
            dict(neutral="failure", short="unknown", sustained="pass"),
            dict.fromkeys(OPTIONS, "unknown"),
        ],
        OPTIONS,
    )
    assert result["bank_solvable_lower"] == 2
    assert result["bank_solvable_upper"] == 3
    assert result["capability_minus_best_fixed_lower"] == 0
    assert result["minimum_cover_size"] is None
    assert result["one_fixed_covers_every_solvable_task"] is None


def test_empty_success_set_does_not_claim_dominant_successful_policy():
    result = module.response_summary([dict.fromkeys(OPTIONS, "failure")], OPTIONS)
    assert result["minimum_cover_size"] == 0
    assert result["minimum_schedule_covers"] == [[]]
    assert result["one_fixed_covers_every_solvable_task"] is None


@pytest.mark.parametrize(
    "rows", [[], [{"neutral": "failure"}], [dict.fromkeys(OPTIONS, "missing")]]
)
def test_reject_incomplete_schema(rows):
    with pytest.raises(ValueError):
        module.response_summary(rows, OPTIONS)


def test_replay_order_is_checked_before_reading_results():
    row = dict(candidate_id="a", scene_definition={"path": "same"}, teacher_branch_order=OPTIONS)
    plan = dict(
        schema="primary",
        runs=[
            dict(physics_seed=93201, arm="analytic_contrast", rounds=[row, row]),
            dict(
                physics_seed=93201,
                arm="observation_curriculum",
                rounds=[row, dict(row, teacher_branch_order=list(reversed(OPTIONS)))],
            ),
        ],
    )
    with pytest.raises(ValueError, match="ordering differ"):
        module.validate_paired_queues(plan, 1)
