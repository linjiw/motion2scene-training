"""Synthetic tests of selection accounting; no fixture is physical evidence."""

import copy
import json
from pathlib import Path
import sys

import pytest

from decoupled_wbc.tests.test_motion2scene_schedule_script_review import bank, observation
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_baselines import (
    context_oracle,
    rank_baselines,
    script_context,
    summarize,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_script import DEFAULT_CONFIG
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    schedule_layout,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_select_schedule_baselines import (  # noqa: E402
    check_unique_episodes,
    result_paths,
    unavailable_group,
    unknown_rows,
)


def panel(*, hazards=None, outcomes=None):
    b = bank()
    phases, masks, _ = schedule_layout(b)
    targets, prefixes = [], []
    for tick, mask in zip(phases.tolist(), masks, strict=True):
        names, x = observation((hazards or {}).get(tick, ()))
        targets.append(
            dict(
                phase_tick=tick,
                available=True,
                features=x.tolist(),
                feature_names=list(names),
                legal_mask=mask.tolist(),
                recorded_history_sha256=f"actual_history_{tick}",
            )
        )
        for option in b.option_ids:
            prefixes.append(
                dict(
                    phase_tick=tick,
                    option_id=option,
                    matched=True,
                    prefix=dict(exact_match=True, frames=tick),
                )
            )
    assessments, rows = [], []
    for option in b.option_ids:
        outcome, time = (outcomes or {}).get(option, ("pass", 3.0))
        assessments.append(
            dict(
                option_id=option,
                task_outcome_admitted=outcome != "unknown",
                task_outcome=outcome,
                physics_steps=1192,
                outcome=dict(physical_events=[]),
            )
        )
        rows.append(
            dict(
                forced_option_id=option,
                **{"pass": outcome == "pass"},
                costs=dict(passage_time_s=time),
            )
        )
    group = dict(
        scene_id="synthetic",
        targets=targets,
        prefix_comparisons=prefixes,
        branch_assessments=assessments,
    )
    return b, group, rows


def test_first_commitment_uses_actual_branch_and_never_future_neutral_observations():
    b, group, rows = panel(hazards={15: (0,)}, outcomes={"prior_15": ("pass", 4.0)})
    # These would raise if an implementation looked beyond the actual commitment.
    group["targets"][1]["features"] = [float("nan")]
    group["targets"][2]["features"] = None
    row = script_context(b, group, context_oracle(b, group, rows), DEFAULT_CONFIG)
    assert row["known"] and row["selected_option_id"] == "prior_15"
    assert len(row["decision_trace"]) == 1 and row["passage_time_s"] == 4
    assert row["relative_time_regret"] == 0.25
    json.dumps(row, allow_nan=False)


def test_selected_branch_prefix_mismatch_cannot_supply_a_counterfactual_score():
    b, group, rows = panel(hazards={50: (1,)})
    for receipt in group["prefix_comparisons"]:
        if receipt["option_id"] == "prior_50" and receipt["phase_tick"] == 50:
            receipt["matched"] = False
    row = script_context(b, group, context_oracle(b, group, rows), DEFAULT_CONFIG)
    assert not row["known"] and row["passed"] is None
    assert "pre-entry neutral prefix" in row["unknown_reason"]


def test_missing_branch_invalidates_even_other_known_successful_choices():
    b, group, rows = panel(outcomes={"short_15": ("unknown", None)})
    oracle = context_oracle(b, group, rows)
    ranking = rank_baselines(b, [group], [oracle], [DEFAULT_CONFIG])
    assert oracle["unknown_option_ids"] == ["short_15"]
    assert ranking["selected_setting_index"] is None
    assert ranking["preferred_constant_option_id"] is None
    assert len(ranking["constant_candidates"]) == 7


def test_missing_later_phase_needs_a_strictly_earlier_timed_terminal_event():
    b, group, rows = panel(outcomes={"neutral": ("failure", None)})
    group["targets"][1].update(available=False, features=None)
    events = group["branch_assessments"][0]["outcome"]["physical_events"]
    events.append(dict(kind="measured_undesired_environment_contact", maximum_force_n=100))
    oracle = context_oracle(b, group, rows)
    assert not script_context(b, group, oracle, DEFAULT_CONFIG)["known"]
    events.append(dict(kind="recorded_fall_or_upright_threshold_failure", first_index=49))
    assert not script_context(b, group, oracle, DEFAULT_CONFIG)["known"]
    events[-1]["first_index"] = 48
    scored = script_context(b, group, oracle, DEFAULT_CONFIG)
    assert scored["known"] and not scored["passed"]
    assert len(scored["decision_trace"]) == 1


def test_known_failures_stay_in_denominator_and_passage_precedes_cost():
    b, group, rows = panel(hazards={15: (0,)}, outcomes={"prior_15": ("failure", None)})
    # A second setting dismisses a measured 1.30m gap and waits to a successful neutral branch.
    for target in group["targets"]:
        target["features"] = observation((0,), gap=1.30)[1].tolist()
    low = dict(DEFAULT_CONFIG, minimum_observed_free_height_m=1.28)
    ranking = rank_baselines(b, [group], [context_oracle(b, group, rows)], [DEFAULT_CONFIG, low])
    assert ranking["script_candidates"][0]["selected_failure_count"] == 1
    assert ranking["selected_setting_index"] == 1
    assert ranking["preferred_constant_option_id"] == "neutral"


def test_exact_ties_prefer_default_then_original_setting_and_option_order():
    b, group, rows = panel()
    other = dict(DEFAULT_CONFIG, minimum_observed_free_height_m=1.28)
    ranking = rank_baselines(
        b, [group], [context_oracle(b, group, rows)], [other, DEFAULT_CONFIG, copy.deepcopy(other)]
    )
    assert ranking["selected_setting_index"] == 1
    assert ranking["preferred_constant_option_id"] == "neutral"
    ranking = rank_baselines(b, [group], [context_oracle(b, group, rows)], [other, other])
    assert ranking["selected_setting_index"] == 0


def test_all_fail_context_is_known_with_unit_regret_for_every_schedule():
    b = bank()
    b, group, rows = panel(outcomes={option: ("failure", None) for option in b.option_ids})
    oracle = context_oracle(b, group, rows)
    assert oracle["complete_known_panel"]
    assert all(branch["relative_time_regret"] == 1 for branch in oracle["branches"])
    ranking = rank_baselines(b, [group], [oracle], [DEFAULT_CONFIG])
    assert ranking["script_candidates"][0]["selected_failure_count"] == 1
    assert ranking["selected_setting_index"] == 0


def test_duplicate_phase_or_assessment_and_nonfinite_summary_are_rejected():
    b, group, rows = panel()
    group["targets"].append(copy.deepcopy(group["targets"][0]))
    with pytest.raises(ValueError, match="unique registered"):
        script_context(b, group, context_oracle(b, group, rows), DEFAULT_CONFIG)
    group["branch_assessments"].append(group["branch_assessments"][0])
    with pytest.raises(ValueError, match="one actual assessment"):
        context_oracle(b, group, rows)
    with pytest.raises(ValueError, match="finite physical regret"):
        summarize([dict(known=True, passed=True, relative_time_regret=float("nan"))], 1)


def test_unfinished_panel_barrier_opens_no_result_content(tmp_path, monkeypatch):
    paths = []
    for index in range(5):
        folder = tmp_path / str(index)
        folder.mkdir()
        paths.append(dict(path=str(folder / "manifest.json"), sha256="irrelevant"))
        if index < 4:
            (folder / "result.json").write_text("deliberately invalid JSON")

    def forbidden_read(*args, **kwargs):
        raise AssertionError("result content must remain unread behind the existence barrier")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(FileNotFoundError, match="complete assigned panel"):
        result_paths(dict(manifests=paths))
    (tmp_path / "4/result.json").write_text("still no inspection needed")
    assert len(result_paths(dict(manifests=paths))) == 5


def test_hard_audit_failure_preserves_all_assigned_unknown_branches():
    b = bank()
    group = unavailable_group(dict(scene_id="bad"), {"sha256": "raw-result"}, b, "bad hash")
    oracle = context_oracle(b, group, unknown_rows(b))
    ranking = rank_baselines(b, [group], [oracle], [DEFAULT_CONFIG])
    assert len(oracle["unknown_option_ids"]) == 7
    assert group["assigned_branches"] == group["physics_steps_unknown_branches"] == 7
    assert ranking["selected_setting_index"] is None
    json.dumps(dict(group=group, oracle=oracle, ranking=ranking), allow_nan=False)


def test_duplicate_episode_identity_cannot_be_an_independent_context(tmp_path):
    ref = dict(path=str(tmp_path / "record.pkl"), sha256="sha256:123")
    group = dict(trajectory_identities=[ref])
    with pytest.raises(ValueError, match="multiple assigned branches"):
        check_unique_episodes([group, copy.deepcopy(group)])
