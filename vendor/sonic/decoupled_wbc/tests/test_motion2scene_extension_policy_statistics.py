"""The held-center comparison separates selection, capability and successful cost."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_extension_coverage import summarize as bank_summary
import motion2scene_extension_policy_statistics as statistics
from motion2scene_extension_policy_statistics import summarize


def example():
    ids = ["neutral", "generated_a", "authored_a"]
    conditions = [
        dict(task_id="a", states=[0, 1, 0], times_s=[None, 3, None]),
        dict(task_id="b", states=[0, 1, 1], times_s=[None, 4, 5]),
        dict(task_id="c", states=[0, 0, 0], times_s=[None, None, None]),
    ]
    policies = [
        dict(
            arm=arm,
            task_id=t,
            fold_index=i,
            training_constant_id=arm + "_a",
            state=state,
            time_s=time,
        )
        for arm, outcomes in [
            ("generated", [(0, None), (1, 3), (1, 6)]),
            ("authored", [(0, None), (1, 5), (0, None)]),
        ]
        for i, (t, (state, time)) in enumerate(zip("abc", outcomes, strict=True))
    ]
    return ids, conditions, policies


def test_own_bank_capability_and_selection_are_distinct():
    result = summarize(*example())
    g, a = result["arms"]["generated"], result["arms"]["authored"]
    assert g["capability_count_lower"] == 2
    assert a["capability_count_lower"] == 1  # Cannot borrow the generated success on a.
    policy = g["policies"]["learned"]
    assert policy["selection_failures"] == 1
    assert policy["policy_only_successes"] == 1  # Repeat disagreement is not clipped away.
    assert policy["assigned"] == 3
    assert policy["paired_with_reference"] == dict(
        mutually_successful_contexts=1, mean_time_difference_s=-1
    )


def test_cross_arm_cost_uses_only_matched_success_intersection():
    result = summarize(*example())
    assert result["generated_minus_authored"] == dict(
        assigned_tasks=3,
        passage_count_difference=1,
        mutually_successful_contexts=1,
        mean_time_difference_s=-2,
    )
    assert len(result["held_center_comparisons"]) == 3


def test_unknown_policy_keeps_denominator_and_withholds_passage_difference():
    ids, conditions, rows = example()
    rows[0]["state"] = -1
    result = summarize(ids, conditions, rows)
    assert result["generated_minus_authored"]["passage_count_difference"] is None
    p = result["arms"]["generated"]["policies"]["learned"]
    assert p["assigned"] == 3 and p["technical_missing"] == 1
    assert p["measured_capability_minus_policy_passages"] is None


def test_unknown_bank_retains_capability_bounds():
    ids, conditions, rows = example()
    conditions[2]["states"][1] = -1
    result = summarize(ids, conditions, rows)["arms"]["generated"]
    assert (result["capability_count_lower"], result["capability_count_upper"]) == (2, 3)
    assert result["policies"]["learned"]["policy_only_successes"] == 0


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: r.pop(), "complete"),
        (lambda r: r.append(deepcopy(r[0])), "distinct"),
        (lambda r: r[0].update(training_constant_id="authored_a"), "own motion bank"),
        (lambda r: r[0].update(fold_index=4), "same held-center"),
        (lambda r: r[0].update(time_s=3), "null time"),
        (lambda r: r[1].update(time_s=None), "positive time"),
    ],
)
def test_invalid_assignments_or_costs_rejected(change, match):
    ids, conditions, rows = example()
    change(rows)
    with pytest.raises(ValueError, match=match):
        summarize(ids, conditions, rows)


def test_constant_cannot_change_across_tasks_of_one_fold():
    ids, conditions, rows = example()
    for row in rows:
        row["fold_index"] = 0
    rows[1]["training_constant_id"] = "neutral"
    with pytest.raises(ValueError, match="one training-selected"):
        summarize(ids, conditions, rows)


def reader_fixture(tmp_path, monkeypatch):
    ids, conditions, rows = example()
    assignments, documents = [], {}
    study = dict(capability_prepared="capability", registry="registry")
    for i, row in enumerate(rows):
        model, episode = f"model_{i}", dict(path=str(tmp_path / f"episode_{i}/manifest.json"))
        documents[model] = dict(
            analysis=dict(constant=dict(selected_option_id=row["training_constant_id"]))
        )
        documents[episode["path"]] = dict(model_result=model, nominal_template=row["task_id"])
        assignments.append(
            dict(
                **{k: row[k] for k in ("arm", "task_id", "fold_index")},
                model_result=model,
                episode=episode,
                nominal_template=row["task_id"],
            )
        )
    documents[str(tmp_path / "prepared.json")] = dict(assignments=assignments)
    documents[str(tmp_path / "coverage.json")] = dict(
        prepared="capability",
        results=[],
        conditions=conditions,
        summary=bank_summary(ids, conditions),
    )
    documents["registry"] = dict(request=dict(options=[dict(option_id=i) for i in ids[1:]]))
    monkeypatch.setattr(statistics, "verify_study", lambda panel: ("study", study, {}))
    monkeypatch.setattr(statistics, "bind_models", lambda *a: {})
    monkeypatch.setattr(statistics, "check_prepared", lambda *a: None)
    monkeypatch.setattr(statistics, "artifact", lambda p: dict(path=str(p)))
    monkeypatch.setattr(
        statistics,
        "read_checked",
        lambda ref: documents[ref["path"] if isinstance(ref, dict) else ref],
    )
    monkeypatch.setattr(statistics, "write_new", lambda path, value: value)
    return documents, assignments


def test_reader_keeps_all_unexecuted_policy_slots(tmp_path, monkeypatch):
    reader_fixture(tmp_path, monkeypatch)
    result = statistics.run(tmp_path, tmp_path / "coverage.json", tmp_path / "stats.json")
    assert result["results"] == []
    assert len(result["conditions"]) == 6
    for arm in ("generated", "authored"):
        assert result["summary"]["arms"][arm]["policies"]["learned"]["technical_missing"] == 3


def test_reader_rejects_swapped_episode_model(tmp_path, monkeypatch):
    documents, assignments = reader_fixture(tmp_path, monkeypatch)
    documents[assignments[0]["episode"]["path"]]["model_result"] = "different"
    with pytest.raises(ValueError, match="model or matched task"):
        statistics.run(tmp_path, tmp_path / "coverage.json", tmp_path / "stats.json")


def test_reader_rejects_unverified_passing_trace(tmp_path, monkeypatch):
    documents, assignments = reader_fixture(tmp_path, monkeypatch)
    result_path = Path(assignments[0]["episode"]["path"]).parent / "result.json"
    monkeypatch.setattr(Path, "exists", lambda path: path == result_path)
    documents[str(result_path)] = dict(
        manifest=assignments[0]["episode"],
        row=dict(outcome=dict(task_outcome="pass"), **{"pass": True}),
        complete_policy_trace_verified=False,
    )
    with pytest.raises(ValueError, match="complete decision trace"):
        statistics.run(tmp_path, tmp_path / "coverage.json", tmp_path / "stats.json")
