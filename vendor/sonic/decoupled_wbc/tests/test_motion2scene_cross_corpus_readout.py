"""Cross-corpus readouts keep paired arms on the same uncontaminated tasks."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_cross_corpus_readout import geometry_key, paired_summary, select_validation


def candidate(seed, arm, geometry):
    return dict(seed=seed, arm=arm, geometry_key=geometry, run_id=f"{seed}_{arm}", round=1)


def test_selection_removes_leakage_and_replay_duplicate_without_outcome_filter():
    rows = [
        candidate(1, "uniform", "a"),
        candidate(2, "uniform", "shared"),
        candidate(2, "target_only", "b"),
        candidate(2, "analytic_contrast", "c"),
        candidate(2, "observation_curriculum", "c"),
        candidate(2, "uniform", "b"),
    ]
    selected, omitted = select_validation(rows, 1, {"a", "shared"})
    assert [r["geometry_key"] for r in selected] == ["b", "c"]
    assert len(omitted) == 4


def test_geometry_key_ignores_names_metadata_and_disabled_beams():
    beam = dict(
        center_xy_m=[2, 0],
        yaw_rad=0,
        length_m=1,
        width_m=1,
        thickness_m=0.1,
        underside_m=1.3,
        route_progress=0.4,
    )
    a = dict(beams=[beam], beam_collision_enabled=[True], scene_id="one")
    b = dict(beams=[dict(beam, route_progress=0.8), beam], beam_collision_enabled=[True, False])
    assert geometry_key(a) == geometry_key(b)
    b["beams"][0]["underside_m"] = 1.2
    assert geometry_key(a) != geometry_key(b)


def outcome(name, passed, time):
    return dict(scene_id=name, branch_proxy_passed=passed, branch_proxy_time_s=time)


def test_time_only_uses_mutually_successful_ordered_contexts():
    a = [outcome("a", True, 2), outcome("b", False, None), outcome("c", True, 10)]
    b = [outcome("a", True, 3), outcome("b", True, 1), outcome("c", False, None)]
    result = paired_summary(a, b)
    assert result["passing_branch_proxies"] == result["constant_prior_passages"] == 2
    assert result["mutually_successful_contexts"] == 1
    assert result["mean_paired_proxy_time_difference_s"] == -1
    with pytest.raises(ValueError, match="identical ordered"):
        paired_summary(a, b[::-1])


def test_disjoint_success_sets_have_no_time_estimate():
    result = paired_summary([outcome("a", False, None)], [outcome("a", True, 2)])
    assert result["mean_paired_proxy_time_difference_s"] is None
