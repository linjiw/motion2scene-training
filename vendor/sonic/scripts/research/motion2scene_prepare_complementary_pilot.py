#!/usr/bin/env python3
"""Freeze and select a paired M2-to-M4 development pilot; never launch physics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_complementary_acquisition as selector  # noqa: E402
from motion2scene_cross_corpus_readout import geometry_key  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination import (  # noqa: E402
    motion2scene_acquisition_pool as original_constructor,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

SEEDS = (93201, 93202, 93203)
STRATA = ("early_constraint", "short_then_sustained")
PROPOSALS_PER_STRATUM = 256


def declaration(expanded_plan, m2_audit, development_study):
    plan_ref, audit_ref, study_ref = map(artifact, (expanded_plan, m2_audit, development_study))
    plan, audit, study = map(read_checked, (plan_ref, audit_ref, study_ref))
    if audit["budget"] != 2 or audit["plan"] != plan["predecessor"]:
        raise ValueError("the original complete M2 acquisition audit is required")
    if study["expanded_plan"] != plan_ref or study["assigned_episodes"] != 138:
        raise ValueError("the original common-set M8 development study is required")
    corpora = [c for c in audit["corpora"] if c["arm"] == "analytic_contrast"]
    if sorted(c["seed"] for c in corpora) != list(SEEDS):
        raise ValueError("the three original executed-contrast M2 prefixes are required")
    for corpus in corpora:
        if len(corpus["tasks"]) != 2 or len(corpus["source_results"]) != 3:
            raise ValueError("exact M2 encounter prefix, including bootstrap, required")
    return dict(
        schema="motion2scene_complementary_pilot_declaration_v1",
        declared_utc=datetime.now(timezone.utc).isoformat(),
        expanded_plan=plan_ref,
        earlier_M2_audit=audit_ref,
        development_study=study_ref,
        registry=plan["registry"],
        candidate_pool=plan["candidate_pools"],
        implementation=[
            artifact(Path(__file__)),
            artifact(Path(selector.__file__)),
            artifact(Path(original_constructor.__file__)),
            artifact(ROOT / "scripts/research/motion2scene_cross_corpus_readout.py"),
        ],
        hypothesis="removing a common fixed response can make acquisition teach useful choices",
        changed_component="acquisition selection only; separately versioned development pilot",
        control="ordinary executed contrast from the identical M2 prefix",
        intervention="ordinary third encounter; fourth encounter targets the earlier common response",
        fixed_response_rule=(
            "complete solvable M1/M2 intersection, then mean real passage time, then registry order"
        ),
        learner="unchanged phase-specific ridge l2=10; uniform per phase; measured-tie initialization",
        teacher="original scene-wise complete-continuation teacher; seven full schedules",
        unchanged="frozen tracker, seven schedules, 114D causal sensing, legal actions, recovery/contact scoring",
        execution_order="each pre-update student, then all seven teachers in frozen draw order, then fit",
        paired_strata=list(STRATA),
        candidate_budget_per_seed=2 * PROPOSALS_PER_STRATUM,
        proposals_per_stratum=PROPOSALS_PER_STRATUM,
        draw_order="first 256 original draws within each declared stratum; exclusions consume budget",
        geometry_rule="positive clearance >=0.01m over all 81 offsets; distinct negative nominal <=-0.01m",
        ranking="same maximum-minimum geometric slack and registry/draw tie order as original constructor",
        exclusion="all original/expanded assigned geometries through M32 and all six development geometries",
        fallback="no targeted proposal: retain ordinary selection, record inactive; never refill after physics",
        candidate_source_disclosure=(
            "reused precomputed geometry; new policy/teacher executions required; "
            "no claim of new geometric search"
        ),
        fixed_seed_pairs=list(SEEDS),
        additional_encounters_per_arm_corpus=2,
        maximum_new_acquisition_episodes=96,
        maximum_new_acquisition_physics_steps=96 * 1192,
        maximum_new_teacher_branches=84,
        maximum_new_preupdate_student_executions=12,
        maximum_new_cpu_fits=12,
        final_policy_evaluation=dict(
            policies=6,
            scenes=6,
            assigned_episodes=36,
            maximum_physics_steps=36 * 1192,
            matching="same six scene/seed/sensor/stress assignments as M8, seed 8732",
            comparators="reuse the 48 original fixed/script captures on these exact conditions",
        ),
        execution_gate=(
            "complete original 138-assignment M8 panel and development decision; "
            "no pilot physics before gate"
        ),
        missing_outcome_rule=(
            "retain failure and unknown, include every assigned task and charged attempt; "
            "no unregistered retry"
        ),
        primary_outcome="paired actual closed-loop passage, all assigned tasks in denominator",
        secondary_outcome="real passage-time differences on mutually successful conditions, with counts",
        mechanism_outcome=(
            "passing-set intersection on complete bank-solvable tasks; "
            "all-failing/unknown reported separately"
        ),
        decision_rule=(
            "retain for larger development only if at least two of three corpus pairs improve actual passage "
            "without a passage decrease in the third; for equal passage in all pairs require lower mean "
            "mutually-successful passage time in at least two and no increase in the third. "
            "Otherwise discard as a performance component or investigate one separately declared bottleneck. "
            "Geometric contrast or training coverage alone is insufficient. "
            "Report all effects; no significance promise."
        ),
        repeated_data_rule=(
            "three paired corpora and six reused development layouts; "
            "episodes are not independent layouts"
        ),
        maximum_budget_rule="stop at M4 even for null/negative effects; no sixth full M32 arm or reserved tuning",
        new_physical_executions=0,
    )


def earlier_response(corpus, option_ids, registry_ref):
    model = read_checked(corpus["checkpoint"])
    registration = read_checked(model["registration"])
    collections = [r["teacher"] for r in corpus["source_results"]]
    if (
        model["status"] != "complete"
        or registration["collections"] != collections
        or registration["registry"] != registry_ref
        or registration["l2"] != 10.0
        or not registration["allow_measured_tie_initialization"]
        or registration["replay_weights"] is not None
        or model["replay_audit"] is not None
    ):
        raise ValueError("exact completed common-learner M2 prefix required")
    checked(Path(model["policy"]["path"]), model["policy"]["sha256"])
    tasks = []
    for task, source in zip(corpus["tasks"], corpus["source_results"][1:], strict=True):
        rows = read_checked(source["teacher"])["rows"]
        if len(rows) != len(option_ids) or {r["forced_option_id"] for r in rows} != set(option_ids):
            raise ValueError("complete earlier teacher branch assignment required")
        outcomes = {r["forced_option_id"]: r["outcome"]["task_outcome"] for r in rows}
        if outcomes != task["outcomes"]:
            raise ValueError("teacher outcomes differ from the existing independent M2 audit")
        tasks.append(
            dict(
                outcomes=outcomes,
                passage_time_s={r["forced_option_id"]: r["costs"]["passage_time_s"] for r in rows},
            )
        )
    return dict(
        seed=corpus["seed"],
        original_model=corpus["checkpoint"],
        original_policy=model["policy"],
        original_collections=corpus["source_results"],
        original_physical_cost=corpus["physical_cost"],
        response=selector.common_response(tasks, option_ids),
    )


def excluded_geometry(plan, study):
    refs = [r["scene"] for run in plan["runs"] for r in run["rounds"]]
    refs += [r["scene_definition"] for r in study["assignments"]]
    unique = {(r["path"], r["sha256"]): r for r in refs}
    return sorted({geometry_key(read_checked(ref)) for ref in unique.values()})


def read_geometry(pool):
    candidates = read_checked(pool["candidates"])["rows"]
    ref = pool["geometry"]
    checked(Path(ref["path"]), ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as archive:
        option_ids = archive["option_ids"].tolist()
        if archive["candidate_ids"].tolist() != [c["candidate_id"] for c in candidates]:
            raise ValueError("cached geometry and original candidate order differ")
        minimum = np.nanmin(archive["outer_clearance_by_offset_m"], axis=1)
        offsets = archive["evaluated_offset_counts"].copy()
        negative = archive["nominal_inner_clearance_m"].copy()
        queries = archive["outer_beam_queries"].sum(axis=1) + archive["inner_beam_queries"].sum(
            axis=1
        )
    for values, key in (
        (minimum, "minimum_evaluated_positive_clearance_m"),
        (offsets, "positive_offset_counts"),
        (negative, "nominal_negative_clearance_m"),
    ):
        if not np.array_equal(values, np.asarray([c[key] for c in candidates])):
            raise ValueError(f"cached geometry disagrees with frozen candidate {key}")
    for candidate in candidates:
        candidate["geometry_key"] = geometry_key(
            dict(beams=candidate["beams"], beam_collision_enabled=[True] * len(candidate["beams"]))
        )
    return candidates, option_ids, minimum, offsets, negative, queries


def select(out):
    declaration_ref = artifact(out / "declaration.json")
    config = read_checked(declaration_ref)
    for ref in config["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    if (out / "selection.json").exists():
        raise ValueError("selection already exists; retain the original bounded run")
    plan = read_checked(config["expanded_plan"])
    study = read_checked(config["development_study"])
    audit = read_checked(config["earlier_M2_audit"])
    pools = read_checked(config["candidate_pool"])
    excluded = excluded_geometry(plan, study)
    results = []
    for seed in SEEDS:
        pool = next(p for p in pools["pools"] if p["seed"] == seed)
        candidates, options, minimum, offsets, negative, queries = read_geometry(pool)
        corpus = next(
            c for c in audit["corpora"] if c["seed"] == seed and c["arm"] == "analytic_contrast"
        )
        prefix = earlier_response(corpus, options, config["registry"])
        fixed = prefix["response"]["selected_fixed_schedule"]
        if fixed is None:
            raise ValueError(
                "the declared common-response intervention is inactive for this prefix"
            )
        start = time.perf_counter()
        selection = selector.paired_selection(
            candidates,
            options,
            minimum,
            offsets,
            negative,
            fixed,
            excluded,
            strata=STRATA,
            proposals_per_stratum=PROPOSALS_PER_STRATUM,
        )
        elapsed = time.perf_counter() - start
        by_id = {c["candidate_id"]: (i, c) for i, c in enumerate(candidates)}
        considered = {cid for row in selection["rounds"] for cid in row["proposed_candidate_ids"]}
        for row in selection["rounds"]:
            for arm in ("ordinary", "complementary"):
                if row[arm] is not None:
                    candidate = by_id[row[arm]["candidate_id"]][1]
                    scene = read_checked(candidate["definition"])
                    if geometry_key(scene) != candidate["geometry_key"]:
                        raise ValueError("selected native definition differs from geometric query")
                    row[arm].update(
                        scene_definition=candidate["definition"],
                        geometry_key=candidate["geometry_key"],
                        teacher_branch_order=candidate["future_branch_order"],
                    )
        results.append(
            dict(
                **prefix,
                selection=selection,
                geometric_cache=dict(candidates=pool["candidates"], geometry=pool["geometry"]),
                previous_clearance_queries_in_considered_draws=int(
                    sum(queries[by_id[cid][0]] for cid in considered)
                ),
                new_selector_wall_seconds=elapsed,
                new_clearance_queries=0,
                new_physics_steps=0,
            )
        )
    result = dict(
        schema="motion2scene_complementary_pilot_selection_v1",
        declaration=declaration_ref,
        corpora=results,
        excluded_geometry_keys=excluded,
        active_distinct_pairs=sum(
            c["selection"]["rounds"][1]["distinct_intervention"] for c in results
        ),
        proposal_budget=3 * 2 * PROPOSALS_PER_STRATUM,
        original_shared_pool_cost=dict(
            proposed_candidates=sum(p["proposed_candidates"] for p in pools["pools"]),
            measured_clearance_queries=pools["actual_shared_clearance_queries"],
            measured_clearance_seconds=sum(
                p["shared_clearance_search_seconds"] for p in pools["pools"]
            ),
            rule="previous shared geometric cost; neither newly executed nor physical cost",
        ),
        inherited_unique_physics_steps=sum(
            c["original_physical_cost"]["total_recorded_steps"] for c in results
        ),
        new_physical_executions=0,
        new_physics_steps=0,
        physical_execution_ready=False,
        pending=config["execution_gate"],
    )
    write_new(out / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    declare_parser = sub.add_parser("declare")
    for name in ("expanded-plan", "m2-audit", "development-study", "out"):
        declare_parser.add_argument(f"--{name}", type=Path, required=True)
    sub.add_parser("select").add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "declare":
        result = declaration(args.expanded_plan, args.m2_audit, args.development_study)
        args.out.mkdir(parents=True, exist_ok=False)
        print(json.dumps(write_new(args.out / "declaration.json", result)))
    else:
        result = select(args.out)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "active_distinct_pairs",
                        "proposal_budget",
                        "new_physical_executions",
                        "pending",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
