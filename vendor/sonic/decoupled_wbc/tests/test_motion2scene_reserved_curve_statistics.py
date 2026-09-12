"""Synthetic complete/paused matrices; never reserved physical measurements."""

import copy
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
import motion2scene_reserved_curve_statistics as statistics  # noqa: E402
import motion2scene_reserved_learning_curve as runner  # noqa: E402
from test_motion2scene_development_learning_curve import save  # noqa: E402
from test_motion2scene_reserved_learning_curve import build_proposal  # noqa: E402


@pytest.fixture
def proposal(tmp_path, monkeypatch):
    return build_proposal(tmp_path, monkeypatch)


def synthetic(proposal, tmp_path):
    _, original, _, _ = proposal
    protocol = copy.deepcopy(original)
    protocol["status"] = runner.ADOPTED
    models = {}
    for p in protocol["policies"]:
        if p["mode"] == "learned":
            p["model"] = dict(path=f'/synthetic_only/{p["policy_id"]}.npz', sha256="a" * 64)
            models[p["policy_id"]] = dict(
                policy=p["model"],
                acquisition_cost=dict(total_recorded_steps=(7 + 8 * p["checkpoint"]) * 1192),
            )
    policies = {p["policy_id"]: p for p in protocol["policies"]}
    assignment = runner.assignments(protocol, tmp_path / "synthetic_execution")
    scenes = {s["scene_id"]: s for s in protocol["scenes"]}
    first_layout = min(a["layout_id"] for a in assignment)
    rows = []
    for a in assignment:
        p = policies[a["policy_id"]]
        status = (
            "failure" if p.get("arm") == "uniform" and a["layout_id"] == first_layout else "pass"
        )
        rows.append(
            dict(
                **a,
                status=status,
                successful_passage_time_s=3.0 if status == "pass" else None,
                recorded_physics_steps=1192,
                policy_mode=p["mode"],
                model=p["model"],
                checkpoint=p.get("checkpoint"),
                training_arm=p.get("arm"),
                acquisition_seed=p.get("seed"),
                acquisition_actual_physics_steps=(
                    (7 + 8 * p["checkpoint"]) * 1192 if p["mode"] == "learned" else None
                ),
                acquisition_assigned_maximum_steps=p.get("acquisition_assigned_maximum_steps"),
                layout_family=scenes[a["scene_id"]]["family"],
                failure_categories=["invalid_initial_approach"] if status == "failure" else [],
            )
        )
    receipt = dict(
        schema=runner.SCHEMA, status="complete", report=dict(rows=rows, assigned_episodes=len(rows))
    )
    inventory = dict(policies=protocol["policies"], models=models)
    return receipt, protocol, assignment, inventory


def test_all_curve_points_and_shared_reference_counts_are_preserved(proposal, tmp_path):
    values = synthetic(proposal, tmp_path)
    result = statistics.summarize(*values, draws=100)
    assert result["assigned_unique_episodes"] == 1908
    assert len(result["learned_curve_points"]) == 45
    assert len(result["comparisons"]) == 142
    assert result["capability_count_lower"] == result["capability_count_upper"] == 36
    contrast = next(
        c
        for c in result["comparisons"]
        if c["left"] == "analytic_contrast"
        and c["right"] == "uniform"
        and c["left_checkpoint"] == 32
    )
    assert contrast["completion"]["mean_difference"] == pytest.approx(1 / 18)
    script = next(c for c in result["comparisons"] if c["right"] == "script")
    assert script["unique_left_episodes"] == 108
    assert script["unique_right_episodes"] == 36
    assert script["paired_success_time"]["unique_right_successful_episodes_used"] == 34
    assert result["failure_categories"]["seed93201_uniform__M32"] == {"invalid_initial_approach": 2}
    assert result["evaluation_recorded_steps"] == 1908 * 1192


def test_unknown_outcomes_withhold_intervals_without_becoming_failures(proposal, tmp_path):
    values = synthetic(proposal, tmp_path)
    row = next(
        r
        for r in values[0]["report"]["rows"]
        if r["policy_id"] == "seed93201_analytic_contrast__M32"
    )
    row.update(
        status="technical_missing", successful_passage_time_s=None, recorded_physics_steps=80
    )
    result = statistics.summarize(*values, draws=100)
    contrast = next(
        c
        for c in result["comparisons"]
        if c["left"] == "analytic_contrast"
        and c["right"] == "uniform"
        and c["left_checkpoint"] == 32
    )
    assert contrast["unknown_matched_pairs"] == 1
    assert contrast["completion"]["mean_difference"] is None
    assert contrast["completion"]["crossed_95_percentile"] is None
    assert result["policy_summaries"][row["policy_id"]]["assigned"] == 36
    assert result["policy_summaries"][row["policy_id"]]["failure"] == 0
    assert result["evaluation_recorded_steps"] == 1907 * 1192 + 80


def test_bank_unsolvable_conditions_remain_in_policy_denominators(proposal, tmp_path):
    values = synthetic(proposal, tmp_path)
    rows = values[0]["report"]["rows"]
    target = rows[0]["layout_id"], rows[0]["physics_seed"]
    for row in rows:
        if row["policy_mode"] == "forced" and (row["layout_id"], row["physics_seed"]) == target:
            row.update(
                status="failure",
                successful_passage_time_s=None,
                failure_categories=["verified_environment_contact"],
            )
    result = statistics.summarize(*values, draws=100)
    assert result["capability_count_lower"] == result["capability_count_upper"] == 35
    p = "seed93201_analytic_contrast__M32"
    assert result["policy_summaries"][p]["assigned"] == 36
    assert result["selection"][p]["policy_only_successes"] == 1


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "cost", "model", "seed", "failure_time"]
)
def test_partial_or_inconsistent_reporting_is_rejected(proposal, tmp_path, change):
    values = synthetic(proposal, tmp_path)
    rows = values[0]["report"]["rows"]
    row = next(r for r in rows if r["policy_mode"] == "learned")
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "cost":
        row["acquisition_actual_physics_steps"] += 1
    elif change == "model":
        row["model"] = None
    elif change == "seed":
        row["physics_seed"] = 8732
    else:
        row["status"] = "failure"
        row["successful_passage_time_s"] = 2.0
    with pytest.raises(ValueError):
        statistics.summarize(*values, draws=100)


def test_full_draw_count_cli_writes_portable_curves_without_opening_models(proposal, tmp_path):
    receipt, protocol, assignment, inventory = synthetic(proposal, tmp_path)
    registered = save(
        tmp_path / "registration.json",
        dict(
            protocol=save(tmp_path / "protocol.json", protocol),
            inventory=save(tmp_path / "inventory.json", inventory),
            assignments=save(tmp_path / "assignments.json", assignment),
        ),
    )
    receipt["registration"] = registered
    source = save(tmp_path / "synthetic_complete.json", receipt)
    result = statistics.run(Path(source["path"]), tmp_path / "statistics")
    reported = json.loads(Path(result["result"]["path"]).read_text())
    assert reported["bootstrap"]["draws"] == 20000
    assert reported["bootstrap"]["seed"] == 202609081822
    with Path(result["learning_curves"]["path"]).open() as handle:
        points = list(csv.DictReader(handle))
    assert len(points) == 45
    assert {r["pass"] for r in points} == {"34", "36"}
    assert result["new_physics_steps"] == 0
    assert not Path("/synthetic_only").exists()
