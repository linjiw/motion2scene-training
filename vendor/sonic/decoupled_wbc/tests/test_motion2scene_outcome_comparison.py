"""Keep passage denominators and mutually successful time comparisons separate."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_outcome_comparison import paired_summary


def pair(context, ridge, tree):
    return dict(
        context=context,
        ridge={"pass": ridge is not None, "time_s": ridge},
        outcome_tree={"pass": tree is not None, "time_s": tree},
    )


def test_compare_only_mutually_successful_costs_keep_all_assigned_contexts():
    result = paired_summary([pair("a", 4, 3), pair("b", None, 7), pair("c", 2, None)])
    assert result["assigned_contexts"] == 3
    assert result["passage_counts"] == dict(ridge=2, outcome_tree=2)
    assert result["mutually_successful_contexts"] == 1
    assert result["mean_paired_time_difference_s"] == -1


def test_disjoint_success_sets_have_no_time_comparison():
    assert (
        paired_summary([pair("a", 4, None), pair("b", None, 3)])["mean_paired_time_difference_s"]
        is None
    )


def test_missing_success_time_and_duplicate_context_are_rejected():
    row = pair("a", 4, None)
    row["outcome_tree"]["pass"] = True
    with pytest.raises(ValueError, match="finite passage time"):
        paired_summary([row])
    with pytest.raises(ValueError, match="distinct assigned contexts"):
        paired_summary([pair("a", 4, 3), pair("a", 4, 3)])
