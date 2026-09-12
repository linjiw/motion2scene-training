#!/usr/bin/env python3
"""Matched 8/16/32 encounter continuation with a reference-envelope comparator."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_prepare_acquisition_pools import read_bound  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

STRATA = ("short", "sustained", "early_constraint", "short_then_short", "short_then_sustained")


def balanced_extension(queues, prefixes, count):
    """Balance counts jointly, preserving queue ranks within each eligible stratum.

    No physical outcome or positive-option identity enters the choice. Existing
    four-encounter prefixes are retained. Exhausted strata remain reported;
    never duplicate a scene or replace one arm's missing scene with another arm.
    """
    if set(queues) != set(prefixes) or count < 4:
        raise ValueError("aligned constructors and budget >= four required")
    selected = {arm: list(prefixes[arm]) for arm in queues}
    used = {arm: {row["candidate_id"] for row in rows} for arm, rows in selected.items()}
    composition = [Counter(row["stratum"] for row in rows) for rows in selected.values()]
    if any(c != composition[0] for c in composition) or any(
        len(rows) != 4 for rows in selected.values()
    ):
        raise ValueError("four matched-stratum initial encounters required")
    for arm, rows in queues.items():
        if len({r["candidate_id"] for r in rows}) != len(rows) or len(used[arm]) != 4:
            raise ValueError("distinct scene identities required")
    counts = composition[0].copy()
    common = [s for s in STRATA if all(any(r["stratum"] == s for r in q) for q in queues.values())]
    while len(next(iter(selected.values()))) < count:
        available = [
            s
            for s in common
            if all(
                any(r["stratum"] == s and r["candidate_id"] not in used[arm] for r in rows)
                for arm, rows in queues.items()
            )
        ]
        if not available:
            break
        stratum = min(available, key=lambda s: (counts[s], STRATA.index(s)))
        for arm, rows in queues.items():
            rank, row = next(
                (i, r)
                for i, r in enumerate(rows)
                if r["stratum"] == stratum and r["candidate_id"] not in used[arm]
            )
            selected[arm].append(dict(row, original_queue_rank=rank))
            used[arm].add(row["candidate_id"])
        counts[stratum] += 1
    return dict(
        selected=selected,
        requested=count,
        obtained=sum(counts.values()),
        common_strata=common,
        composition=dict(counts),
        shortfall=count - sum(counts.values()),
        physical_outcomes_consulted=False,
    )


def build(old_plan, expanded_pools, references, execution_root, out):
    old = read_bound(artifact(old_plan))
    pool_result = read_bound(artifact(expanded_pools / "result.json"))
    reference = read_bound(artifact(references / "result.json"))
    if read_bound(reference["experiment"])["pools"] != artifact(expanded_pools / "result.json"):
        raise ValueError("reference and executed construction must use the same candidate pool")
    if execution_root.exists():
        raise ValueError("a new expanded execution root is required")
    reference_by_seed = {p["seed"]: p for p in reference["pools"]}
    runs, selection_records = [], []
    for pool in pool_result["pools"]:
        seed = pool["seed"]
        queues = {arm: read_bound(ref)["rows"] for arm, ref in pool["arms"].items()}
        queues["reference_contrast"] = read_bound(reference_by_seed[seed]["queue"])["rows"]
        old_runs = {r["arm"]: r for r in old["runs"] if r["physics_seed"] == seed}
        prefixes = {
            arm: [
                dict(
                    candidate_id=r["candidate_id"],
                    stratum=r["stratum"],
                    definition=r["scene_definition"],
                    future_branch_order=r["teacher_branch_order"],
                )
                for r in run["rounds"][1:]
            ]
            for arm, run in old_runs.items()
        }
        strata = [r["stratum"] for r in prefixes["analytic_contrast"]]
        ref_prefix = []
        for stratum in strata:
            candidates = [r for r in queues["reference_contrast"] if r["stratum"] == stratum]
            if not candidates:
                raise ValueError(
                    f"reference comparator lacks initial stratum {stratum} at seed {seed}"
                )
            ref_prefix.append(candidates[0])
        prefixes["reference_contrast"] = ref_prefix
        selected = balanced_extension(queues, prefixes, 32)
        selection_records.append(dict(seed=seed, **selected))
        if selected["shortfall"]:
            raise ValueError(
                f"matched 32-scene support shortfall at seed {seed}: {selected['shortfall']}"
            )
        if [r["candidate_id"] for r in selected["selected"]["analytic_contrast"]] != [
            r["candidate_id"] for r in selected["selected"]["observation_curriculum"]
        ]:
            raise ValueError("paired analytic/replay scenes must remain identical")
        candidate_rows = {r["candidate_id"]: r for r in read_bound(pool["candidates"])["rows"]}
        for arm, prefix in prefixes.items():
            for row in prefix:
                expanded = candidate_rows[row["candidate_id"]]
                old_scene, new_scene = read_bound(row["definition"]), read_bound(
                    expanded["definition"]
                )
                if (
                    row["stratum"] != expanded["stratum"]
                    or old_scene["beams"] != new_scene["beams"]
                    or old_scene["beam_collision_enabled"] != new_scene["beam_collision_enabled"]
                    or row.get("future_branch_order", expanded["future_branch_order"])
                    != expanded["future_branch_order"]
                ):
                    raise ValueError(
                        f"expanded draw changed an inherited scene or branch order: {arm}"
                    )
        for arm, rows in selected["selected"].items():
            run_id = f"seed{seed}_{arm}"
            inherited = old_runs.get(arm)
            bootstrap = old_runs["analytic_contrast"]["rounds"][0]["scene_definition"]
            rounds = []
            for index, row in enumerate([None, *rows]):
                inherited_row = inherited["rounds"][index] if inherited and index <= 4 else None
                location = (
                    execution_root / run_id / ("bootstrap" if index == 0 else f"round_{index:03d}")
                )
                rounds.append(
                    dict(
                        index=index,
                        candidate_id=None if row is None else row["candidate_id"],
                        stratum="empty_bootstrap" if row is None else row["stratum"],
                        scene=bootstrap if row is None else row["definition"],
                        branch_order=(
                            old["option_ids"]
                            if row is None
                            else candidate_rows[row["candidate_id"]]["future_branch_order"]
                        ),
                        inherited_teacher=(
                            None
                            if inherited_row is None
                            else inherited_row["expected_teacher_result_path"]
                        ),
                        inherited_student=(
                            None
                            if inherited_row is None
                            else inherited_row["expected_student_result_path"]
                        ),
                        inherited_model=(
                            None
                            if inherited_row is None
                            else inherited_row["expected_postupdate_training_result_path"]
                        ),
                        teacher_directory=str(location / "teachers"),
                        student_directory=None if index == 0 else str(location / "student"),
                        model_directory=str(
                            execution_root / run_id / "models" / f"model_{index:03d}"
                        ),
                    )
                )
            runs.append(dict(run_id=run_id, seed=seed, arm=arm, rounds=rounds))
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "selection.json", dict(rows=selection_records))
    return write_new(
        out / "plan.json",
        dict(
            schema="motion2scene_expanded_acquisition_v1",
            predecessor=artifact(old_plan),
            candidate_pools=artifact(expanded_pools / "result.json"),
            reference_pools=artifact(references / "result.json"),
            registry=old["registry"],
            execution_root=str(execution_root.resolve()),
            runs=runs,
            checkpoints=[2, 4, 8, 16, 32],
            selection=artifact(out / "selection.json"),
            maximum_assigned_episodes_per_corpus=263,
            maximum_assigned_steps_per_corpus=313496,
            maximum_charged_steps_per_corpus=320000,
            maximum_new_episodes=3477,
            inherited_assigned_episodes=468,
            learner="same phase-specific ridge l2=10; passage regret and measured-tie initialization",
            primary_construction_contrast=(
                "executed versus reference envelope with identical native shapes, "
                "schedules and learner"
            ),
            prerequisite="all original twelve M4 corpora complete before expanded physics",
            authority="User explicitly requested continuing until the research goal is complete",
            physical_outcomes_used_for_new_selection=False,
            reserved_evaluation_started=False,
            implementation=[
                artifact(ROOT / "scripts/research" / name)
                for name in (
                    "motion2scene_expanded_plan.py",
                    "motion2scene_expanded_acquisition.py",
                    "motion2scene_storage.py",
                )
            ],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-plan", "expanded-pools", "references", "execution-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.old_plan, args.expanded_pools, args.references, args.execution_root, args.out
            )
        )
    )
