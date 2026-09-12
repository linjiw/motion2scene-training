"""Summarize complete M7 bindings without repeating the raw-record audit."""

import datetime
import json
from pathlib import Path
import sys

ROOT = Path("/home/linjiw/groot-wbc-sonic-sim-trackb")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

D = Path("/home/linjiw/research-data/groot-wbc")
RUNROOT = D / "m2s-expanded-acquisition-20260909-v1"
OUTDIR = Path(__file__).parent
OUT = OUTDIR / "all_M7_bound_response_tables.json"


def main():
    from motion2scene_build_timed_replay import read_bound
    import motion2scene_response_diversity as diversity
    from motion2scene_timing_diagnostic import artifact, write_new

    assert not OUT.exists()
    previous_ref = artifact(OUTDIR / "all_M6_bound_response_tables.json")
    assert (
        previous_ref["sha256"]
        == "sha256:b17365f5df889bdd8942b5e7ba59a8f5a862b3ba7dce91076e02f93aaedf5989"
    )
    previous = read_bound(previous_ref)
    plan_ref = artifact(D / "m2s-expanded-acquisition-plan-20260909-v1/plan.json")
    assert (
        plan_ref["sha256"]
        == "sha256:bf551021c82fd42d0e6b002a2941a71b1358d72a469903105a2230bdfae3a4ed"
    )
    plan = read_bound(plan_ref)
    completions = sorted(RUNROOT.glob("*/controller/complete_007.json"))
    assert len(completions) == len(plan["runs"]) == 15
    reports = []
    for path in completions:
        completion_ref = artifact(path)
        c = read_bound(completion_ref)
        run = next(r for r in plan["runs"] if r["run_id"] == c["run_id"])
        prior = next(r for r in previous["corpora"] if r["run_id"] == c["run_id"])
        slot = run["rounds"][7]
        training = read_bound(c["training_result"])
        registration = read_bound(training["registration"])
        groups = read_bound(training["teachers"])
        prior_groups = read_bound(prior["teachers"])
        assert c["completed_through_round"] == 7
        assert c["reserved_evaluation_started"] is False
        assert len(groups) == 8 and groups[:-1] == prior_groups
        assert registration["collections"] == [g["collection"] for g in groups]
        assert registration["l2"] == 10.0
        expected_weighting = (
            "historical_observation_gap"
            if run["arm"] == "observation_curriculum"
            else "uniform_per_phase"
        )
        assert registration["weighting"] == expected_weighting
        assert c["model"] == training["policy"] == artifact(Path(c["model"]["path"]))
        tasks = []
        for i, group in enumerate(groups[1:], 1):
            collection = read_bound(group["collection"])
            table = {
                r["forced_option_id"]: r["outcome"]["task_outcome"]
                for r in collection["rows"]
            }
            assert len(table) == len(collection["rows"]) == 7
            if i <= 6:
                assert table == prior["tasks"][i - 1]["outcomes"]
                assert group["scene_id"] == prior["tasks"][i - 1]["scene_id"]
            tasks.append(
                dict(
                    round=i,
                    scene_id=group["scene_id"],
                    collection=group["collection"],
                    outcomes=table,
                )
            )
        root = path.parent.parent
        pre_ref = artifact(root / "round_007/preupdate.json")
        pre = read_bound(pre_ref)
        release = read_bound(artifact(root / "round_007/student_complete.json"))
        assert release["preupdate"] == pre_ref
        assert pre["training_result"] == prior["training_result"]
        assert pre["earlier_collections"] == [g["collection"] for g in prior_groups]
        student = read_bound(release["student"])
        student_manifest = read_bound(student["manifest"])
        assert student_manifest["policy"] == pre["model"]
        assert pre["model"] == artifact(Path(pre["model"]["path"]))
        assert len(student["rows"]) == 1
        row = student["rows"][0]
        teacher = read_bound(groups[-1]["collection"])
        teacher_manifest = read_bound(teacher["manifest"])
        assert (
            student_manifest["scene_definition"]
            == teacher_manifest["scene_definition"]
            == slot["scene"]
        )
        assert [r["forced_option_id"] for r in teacher["rows"]] == slot["branch_order"]
        assert [r["forced_option_id"] for r in teacher_manifest["cells"]] == slot[
            "branch_order"
        ]
        response = diversity.response_summary(
            [t["outcomes"] for t in tasks], groups[0]["option_ids"]
        )
        reports.append(
            dict(
                run_id=c["run_id"],
                arm=run["arm"],
                seed=run["seed"],
                completion=completion_ref,
                training_result=c["training_result"],
                teachers=training["teachers"],
                new_model=c["model"],
                weighting=registration["weighting"],
                learner_l2=registration["l2"],
                M6_bound_teacher_prefix_unchanged=True,
                tasks=tasks,
                responses=response,
                current_student=dict(
                    collection=release["student"],
                    preupdate=pre_ref,
                    generating_model=pre["model"],
                    generating_checkpoint=6,
                    task_outcome=row["outcome"]["task_outcome"],
                    passage_time_s=row["costs"]["passage_time_s"],
                    physics_steps=row["physics_steps"],
                    outcome_measurement=row["outcome"],
                ),
                current_teacher_outcomes={
                    r["forced_option_id"]: dict(
                        outcome=r["outcome"]["task_outcome"],
                        passage_time_s=r["costs"]["passage_time_s"],
                        steps=r["physics_steps"],
                    )
                    for r in teacher["rows"]
                },
                current_teacher_measurements={
                    r["forced_option_id"]: r["outcome"] for r in teacher["rows"]
                },
                current_encounter_physics_steps=row["physics_steps"]
                + sum(r["physics_steps"] for r in teacher["rows"]),
                recorded_captures=c["accounting"][
                    "distinct_original_recorded_captures"
                ],
                actual_recorded_physics_steps=c["accounting"][
                    "actual_recorded_physics_steps"
                ],
            )
        )
    by_id = {r["run_id"]: r for r in reports}
    pairs = []
    for seed in sorted({r["seed"] for r in reports}):
        run_a = next(
            r for r in plan["runs"] if r["run_id"] == f"seed{seed}_analytic_contrast"
        )
        run_b = next(
            r
            for r in plan["runs"]
            if r["run_id"] == f"seed{seed}_observation_curriculum"
        )
        for a, b in zip(run_a["rounds"][:8], run_b["rounds"][:8], strict=True):
            assert all(
                a[k] == b[k] for k in ["index", "candidate_id", "scene", "branch_order"]
            )
        a, b = by_id[run_a["run_id"]], by_id[run_b["run_id"]]
        pairs.append(
            dict(
                seed=seed,
                corpora=[a["run_id"], b["run_id"]],
                candidate_order_through_M7_unchanged=True,
                current_teacher_tables_identical=a["current_teacher_outcomes"]
                == b["current_teacher_outcomes"],
            )
        )
    report = dict(
        schema="motion2scene_completed_checkpoint_bound_tables_v1",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        budget=7,
        scope=(
            "Interim acquisition readout from hash-bound completed collection/training records. "
            "Uses original collection scoring; no new all-capture raw audit or common-set policy evaluation."
        ),
        implementation=[artifact(Path(__file__)), artifact(Path(diversity.__file__))],
        plan=plan_ref,
        prior_M6_bound_readout=previous_ref,
        corpora=reports,
        paired_constructor_checks=pairs,
        total_recorded_captures=sum(r["recorded_captures"] for r in reports),
        total_recorded_physics_steps=sum(
            r["actual_recorded_physics_steps"] for r in reports
        ),
        total_postbootstrap_encounter_assignments=sum(
            r["responses"]["assigned_tasks"] for r in reports
        ),
        corpora_without_one_passing_fixed_schedule=[
            r["run_id"]
            for r in reports
            if r["responses"]["one_fixed_covers_every_solvable_task"] is False
        ],
        new_physics_from_readout=0,
        new_fits_from_readout=0,
        interpretation=(
            "Finite-bank coverage concerns acquired tasks, including assigned bank-unsolvable tasks "
            "and explicit teacher unknowns. Student outcomes belong to generating M6 policies on each "
            "arm's acquired scene, not the new M7 fits or a common evaluation set. Earlier failures "
            "and the reconciled reference student unknown remain in acquisition histories and cost. "
            "Corpus complementarity does not establish sensor realizability or downstream benefit. "
            "Passage time is crossing plus stabilization, not full adaptation/recovery completion. "
            "No queue, teacher, learner, baseline or reserved assignment changes; "
            "M8 common-set performance remains pending."
        ),
    )
    write_new(OUT, report)
    print(
        json.dumps(
            dict(
                output=artifact(OUT),
                captures=report["total_recorded_captures"],
                steps=report["total_recorded_physics_steps"],
                encounter_assignments=report[
                    "total_postbootstrap_encounter_assignments"
                ],
                corpora_without_one_passing_fixed_schedule=report[
                    "corpora_without_one_passing_fixed_schedule"
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
