"""Acquired response complementarity does not impute unknowns or sensor differences."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_acquired_response_witness import (
    disjoint_response_pairs,
    distinct_conditions,
    phase_feature_contrast,
)


def task(name, outcomes, geometry=None, seed=1):
    return dict(candidate_id=name, geometry_key=geometry or name, seed=seed, outcomes=outcomes)


def test_disjoint_success_sets_need_two_responses_not_just_adaptation():
    a = task("a", dict(neutral="failure", short="pass", long="failure"))
    b = task("b", dict(neutral="failure", short="failure", long="pass"))
    c = task("c", dict(neutral="failure", short="pass", long="pass"))
    pairs = disjoint_response_pairs([a, b, c])
    assert len(pairs) == 1
    assert pairs[0]["left"] == "a" and pairs[0]["right"] == "b"
    b["outcomes"]["short"] = "unknown"
    assert disjoint_response_pairs([a, b]) == []


def test_repeated_conditions_do_not_become_independent_tasks():
    a = task("a", dict(neutral="pass"))
    b = task("b", dict(neutral="pass"), geometry="a")
    c = task("c", dict(neutral="pass"), geometry="a", seed=2)
    assert distinct_conditions([a, b, c]) == [a, c]
    b["outcomes"]["neutral"] = "failure"
    with pytest.raises(ValueError, match="disagree"):
        distinct_conditions([a, b])


def group():
    return dict(
        option_ids=["neutral"],
        targets=[
            dict(
                phase_tick=15,
                available=True,
                feature_names=[f"x{i}" for i in range(114)],
                features=[0.0] * 114,
            )
        ],
    )


def test_exact_proprioception_difference_is_not_called_a_perception_difference():
    a, b = group(), group()
    b["targets"][0]["features"][40] = 1e-10
    result = phase_feature_contrast(a, b)[0]
    assert result["exact_full_feature_equality"] is False
    assert result["exact_perception_feature_equality"] is True
    assert result["differing_features"][0]["index"] == 40
    b["targets"][0]["available"] = False
    assert phase_feature_contrast(a, b) == [dict(phase_tick=15, available=False)]
