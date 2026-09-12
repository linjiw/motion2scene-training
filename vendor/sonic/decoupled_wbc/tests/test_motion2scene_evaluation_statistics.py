"""Portable statistics preserve paired blocks, unknown slots and shared baselines."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_evaluation_statistics as stats  # noqa: E402


def receipt():
    families = [family for family, n in stats.FAMILY_COUNTS.items() for _ in range(n)]
    policies = []
    for arm in stats.ARMS:
        for corpus in stats.CORPORA:
            for checkpoint, maximum in stats.CHECKPOINTS.items():
                policy = f"{arm}__seed{corpus}__{checkpoint}"
                policies.append(
                    dict(
                        policy_id=policy,
                        training_arm=arm,
                        acquisition_seed=corpus,
                        checkpoint=checkpoint,
                        acquisition_assigned_maximum_steps=maximum,
                        acquisition_actual_physics_steps=maximum - 100,
                        model={
                            "path": "/public/models/" + policy + ".npz",
                            "sha256": "sha256:" + hashlib.sha256(policy.encode()).hexdigest(),
                        },
                    )
                )
    policies.extend(
        dict(
            policy_id=p,
            training_arm=None,
            acquisition_seed=None,
            checkpoint=None,
            acquisition_actual_physics_steps=None,
            acquisition_assigned_maximum_steps=None,
            model=None,
        )
        for p in stats.BASELINES
    )
    rows = [
        dict(
            **policy,
            layout_id=f"layout_{i:02d}",
            layout_family=family,
            physics_seed=seed,
            status="pass",
            successful_passage_time_s=2 + i / 100,
            recorded_physics_steps=1192,
            variant_id="nominal",
        )
        for policy in policies
        for i, family in enumerate(families)
        for seed in stats.EVALUATION_SEEDS
    ]
    return {
        "schema": stats.SCHEMA,
        "status": "complete",
        "report": {"assigned_episodes": 972, "complete": True, "rows": rows},
    }


def test_full_and_paused_wrapped_receipts_keep_all_assigned_rows():
    value = receipt()
    before = copy.deepcopy(value)
    result = stats.summarize(value, draws=32)
    assert value == before
    assert result["assigned_outcomes"]["assigned"] == len(result["assigned_rows"]) == 972
    assert len(result["policy_summaries"]) == 27
    assert len(result["paired_comparisons"]) == 34
    assert len(result["per_corpus_learning_curves"]) == 24
    assert len(result["mean_learning_curves"]) == 8
    assert all(
        r["completion"]["crossed_95_percentile"] == [0, 0] for r in result["paired_comparisons"]
    )
    row = value["report"]["rows"][0]
    row.update(status="not_run", successful_passage_time_s=None, recorded_physics_steps=None)
    value["status"], value["report"]["complete"] = "paused", False
    wrapped = {"kind": "evaluation_status", "receipt": value}
    result = stats.summarize(wrapped, draws=32)
    assert result["assigned_outcomes"]["not_run"] == 1
    assert result["assigned_outcomes"]["completion_lower"] == 971 / 972
    assert result["assigned_outcomes"]["completion_upper"] == 1
    assert any(
        r["completion"]["crossed_95_percentile"] is None for r in result["paired_comparisons"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "model",
        "budget",
        "family",
        "seed",
        "baseline_copy",
        "failure_time",
        "complete_flag",
        "same_model_path",
    ],
)
def test_incomplete_matrix_and_inconsistent_metadata_are_rejected(change):
    value = receipt()
    rows = value["report"]["rows"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif change == "model":
        rows[0]["model"] = dict(rows[0]["model"], sha256="a" * 64)
    elif change == "budget":
        rows[0]["acquisition_actual_physics_steps"] = 1
    elif change == "family":
        rows[0]["layout_family"] = "short_then_short"
    elif change == "seed":
        rows[0]["physics_seed"] = 0
    elif change == "baseline_copy":
        rows[-1]["acquisition_seed"] = stats.CORPORA[0]
    elif change == "failure_time":
        rows[0]["status"] = "failure"
    elif change == "complete_flag":
        value["report"]["complete"] = False
    elif change == "same_model_path":
        owner = rows[0]["model"]
        other = rows[36]["policy_id"]
        for row in rows:
            if row["policy_id"] == other:
                row["model"] = copy.deepcopy(owner)
    with pytest.raises(ValueError):
        stats.validate_report(value)


def test_byte_identical_models_at_distinct_registered_paths_are_allowed():
    value = receipt()
    for row in value["report"]["rows"]:
        if row["model"] is not None:
            row["model"]["sha256"] = "a" * 64
    stats.validate_report(value)


def test_checkpoint_curve_budgets_are_cumulative_and_not_summed_over_evaluations():
    value = receipt()
    result = stats.summarize(value, draws=16)
    for point in result["per_corpus_learning_curves"]:
        assert (
            point["acquisition_actual_physics_steps"]
            == stats.CHECKPOINTS[point["checkpoint"]] - 100
        )
        assert point["assigned"] == 36
    policy = f"{stats.ARMS[0]}__seed{stats.CORPORA[0]}__M4"
    for row in value["report"]["rows"]:
        if row["policy_id"] == policy:
            row["acquisition_actual_physics_steps"] = 0
    with pytest.raises(ValueError, match="cumulative"):
        stats.validate_report(value)


def test_product_weights_equal_explicit_crossed_block_sampling():
    values = np.arange(12).reshape(3, 4)
    corpora = np.array([[2, 2, 0], [0, 1, 2]])
    layouts = np.array([[1, 1, 3, 0], [3, 2, 2, 2]])
    cw, lw = np.eye(3)[corpora].mean(1), np.eye(4)[layouts].mean(1)
    expected = [values[c[:, None], layout[None, :]].mean() for c, layout in zip(corpora, layouts)]
    np.testing.assert_allclose(stats.weighted_draws(values, cw, lw), expected)


def test_rng_order_and_paired_evaluation_seed_blocks_are_exact():
    cw, lw = stats.bootstrap_weights(3, 18, draws=25)
    rng = np.random.default_rng(stats.RANDOM_SEED)
    np.testing.assert_array_equal(cw, np.eye(3)[rng.integers(3, size=(25, 3))].mean(1))
    np.testing.assert_array_equal(lw, np.eye(18)[rng.integers(18, size=(25, 18))].mean(1))
    left = np.tile([0.0, 1.0], (3, 18, 1))
    right = 1 - left
    times_l = np.where(left == 1, 2.0, np.nan)
    times_r = np.where(right == 1, 2.0, np.nan)
    result = stats.paired_comparison(left, right, times_l, times_r, (cw, lw))
    assert result["completion"]["crossed_95_percentile"] == [0, 0]
    assert result["paired_success_time"]["mutually_successful_pairs"] == 0
    assert result["paired_success_time"]["mean_difference_s"] is None


def test_shared_baseline_has36_unique_rows_and_no_independent_corpus_copies():
    left = np.zeros((3, 18, 2))
    right = np.repeat((np.arange(18) % 2)[:, None], 2, axis=1)
    result = stats.paired_comparison(
        left,
        right,
        np.full_like(left, np.nan),
        np.where(right == 1, 2.0, np.nan),
        stats.bootstrap_weights(3, 18, draws=500),
        shared_right=True,
    )
    assert result["unique_left_episodes"] == 108 and result["unique_right_episodes"] == 36
    assert result["assigned_matched_pairs"] == 108
    np.testing.assert_allclose(
        result["completion"]["crossed_95_percentile"],
        result["completion"]["layout_only_95_percentile"],
        atol=1e-15,
    )


def test_unknown_comparison_bounds_and_intervals_never_drop_assigned_pairs():
    left, right = np.ones((3, 2, 2)), np.zeros((3, 2, 2))
    left[0, 0, 0] = np.nan
    result = stats.paired_comparison(
        left,
        right,
        np.where(left == 1, 3.0, np.nan),
        np.full_like(right, np.nan),
        stats.bootstrap_weights(3, 2, draws=100),
    )
    assert result["assigned_matched_pairs"] == 12 and result["unknown_matched_pairs"] == 1
    assert result["completion"]["difference_lower"] == 11 / 12
    assert result["completion"]["difference_upper"] == 1
    assert result["completion"]["mean_difference"] is None
    assert result["completion"]["crossed_95_percentile"] is None
    assert result["paired_success_time"]["crossed_95_percentile_s"] is None


def test_cost_uses_only_mutually_successful_pairs_and_exposes_empty_draws():
    left, right = np.zeros((3, 2, 2)), np.zeros((3, 2, 2))
    left[:, 0, :] = 1
    right[:, 0, 0] = right[:, 1, 0] = 1
    lt, rt = np.full_like(left, np.nan), np.full_like(right, np.nan)
    lt[:, 0, 0], lt[:, 0, 1] = 3, 100
    rt[:, 0, 0], rt[:, 1, 0] = 2, 0.1
    result = stats.paired_comparison(left, right, lt, rt, stats.bootstrap_weights(3, 2, draws=500))
    cost = result["paired_success_time"]
    assert cost["mutually_successful_pairs"] == 3 and cost["mean_difference_s"] == 1
    assert cost["crossed_zero_support_draws"] > 0
    assert cost["crossed_95_percentile_s"] is None
    assert result["completion"]["mean_difference"] == 0  # Every known pair remains included.


def test_complete_mutual_cost_difference_has_paired_interval():
    y = np.ones((3, 18, 2))
    result = stats.paired_comparison(y, y, 3 * y, 2 * y, stats.bootstrap_weights(3, 18, draws=100))
    cost = result["paired_success_time"]
    assert cost["mutually_successful_pairs"] == 108 and cost["mean_difference_s"] == 1
    np.testing.assert_allclose(cost["crossed_95_percentile_s"], [1, 1])
    assert cost["crossed_zero_support_draws"] == 0


def test_cli_outputs_are_fresh_and_input_hash_bound(tmp_path, monkeypatch):
    source = tmp_path / "input" / "report.json"
    source.parent.mkdir()
    source.write_text(json.dumps(receipt()))
    before = source.read_bytes()
    original = stats.summarize
    monkeypatch.setattr(stats, "summarize", lambda value: original(value, draws=32))
    out = tmp_path / "statistics"
    result = stats.run(source, out, expected_sha256=hashlib.sha256(before).hexdigest())
    assert result["schema"] == stats.OUTPUT_SCHEMA and source.read_bytes() == before
    assert {p.name for p in out.iterdir()} == {
        "registration.json",
        "result.json",
        "learning_curves.csv",
        "paired_comparisons.csv",
    }
    with pytest.raises(FileExistsError):
        stats.run(source, out)
    with pytest.raises(ValueError, match="digest"):
        stats.run(source, tmp_path / "wrong", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="outside"):
        stats.run(source, source.parent / "nested")


def test_comparison_inventory_matches_immutable_reporting_clarification():
    path = Path(__file__).resolve().parents[2] / "docs/motion2scene"
    clarification = json.loads(
        (path / "TRAVERSAL_PROTOCOL_V4_REPORTING_CLARIFICATION_PROPOSED.json").read_text()
    )
    result = stats.summarize(receipt(), draws=16)
    actual = [
        {key: row[key] for key in clarification["comparisons"][0]}
        for row in result["paired_comparisons"]
    ]
    assert actual == clarification["comparisons"]
    assert stats.DRAWS == clarification["bootstrap"]["draws"]
    assert stats.RANDOM_SEED == clarification["bootstrap"]["seed"]
