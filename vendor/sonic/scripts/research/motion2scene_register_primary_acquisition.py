#!/usr/bin/env python3
"""Register a proposed, unexecuted 50k online acquisition/replay comparison."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_collect_timed_schedules import validate_scene  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_coverage import (  # noqa: E402
    coverage_selection,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    ARMS,
    PLAN_SCHEMA,
    SEEDS,
    artifact,
    budget_contract,
    checked_path,
    read_bound,
    validate_plan,
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)


def run_slots(root, run_id, seed, selections, option_ids, bootstrap):
    folder = root / run_id
    rounds = []
    for index in range(5):
        current = folder / ("bootstrap" if index == 0 else f"round_{index:03d}")
        model = folder / "models" / f"model_{index - 1:03d}"
        selection = None if index == 0 else selections[index - 1]
        rounds.append(
            dict(
                round_index=index,
                candidate_id=None if selection is None else selection["candidate_id"],
                scene_definition=bootstrap if selection is None else selection["definition"],
                stratum="empty_bootstrap" if selection is None else selection["stratum"],
                physics_seed=seed,
                teacher_branch_order=(
                    list(option_ids) if selection is None else selection["future_branch_order"]
                ),
                allowed_teacher_rounds=list(range(index)),
                student_before_current_teachers=bool(index),
                teacher_directory=str(current / "teachers"),
                student_directory=None if index == 0 else str(current / "student"),
                expected_teacher_result_path=str(current / "teachers" / "result.json"),
                expected_student_result_path=(
                    None if index == 0 else str(current / "student" / "result.json")
                ),
                expected_preupdate_policy_path=(None if index == 0 else str(model / "policy.npz")),
                expected_preupdate_training_result_path=(
                    None if index == 0 else str(model / "result.json")
                ),
                expected_preupdate_binding_path=(
                    None if index == 0 else str(current / "preupdate_binding.json")
                ),
                expected_teacher_release_path=(
                    None if index == 0 else str(current / "student_complete_release.json")
                ),
                expected_replay_result_path=(
                    None if index == 0 else str(current / "replay" / "result.json")
                ),
                expected_postupdate_policy_path=str(
                    folder / "models" / f"model_{index:03d}" / "policy.npz"
                ),
                expected_postupdate_training_result_path=str(
                    folder / "models" / f"model_{index:03d}" / "result.json"
                ),
                observation_availability=None,
                actual_student_gap=None,
                physical_steps=None,
            )
        )
    return rounds


def build_plan(args):
    pool_ref = artifact(args.pools / "result.json")
    pools = read_bound(pool_ref)
    registration_ref = artifact(args.pools / "registration.json")
    registration = read_bound(registration_ref)
    checked_path(registration["draws"])
    registry = registration["registry"]
    bank = load_verified_registry(registry["path"], registry["sha256"])
    if bank.frame_count != 299 or len(bank.option_ids) != 7:
        raise ValueError("qualified four-reference/seven-schedule six-second bank required")
    if pools["physical_steps"] != 0 or pools["actual_sensor_queries"] != 0:
        raise ValueError("the frozen geometry-only pools must precede primary execution")
    if [p["seed"] for p in pools["pools"]] != list(SEEDS):
        raise ValueError("three registered independent candidate seeds required")
    if args.execution_root.exists():
        raise ValueError("proposed future execution root must not already exist")
    args.out.mkdir(parents=True, exist_ok=False)
    old_empty = read_bound(artifact(args.empty))
    empty = dict(
        schema="motion2scene_timed_schedule_scene_v1",
        scene_id="primary_empty_bootstrap",
        split="development",
        acquisition_role="new primary-corpus bootstrap; no old development labels reused",
        scene=old_empty["scene"],
        beams=[old_empty["beam"]],
        beam_collision_enabled=[False],
        physical_labels=None,
        provenance=dict(template_definition=artifact(args.empty)),
    )
    validate_scene(empty)
    bootstrap = write_new(args.out / "bootstrap_definition.json", empty)
    runs = []
    coverage_receipts = []
    for pool in pools["pools"]:
        candidates = {r["candidate_id"]: r for r in read_bound(pool["candidates"])["rows"]}
        pool_queues = {arm: read_bound(pool["arms"][arm]) for arm in ARMS}
        coverage = coverage_selection(
            {arm: queue["rows"] for arm, queue in pool_queues.items()}, pool["seed"]
        )
        if coverage["planned_group_shortfall"]:
            raise ValueError(
                "common-support shortfall retained; a four-group plan cannot be filled"
            )
        coverage_ref = write_new(args.out / f"coverage_{pool['seed']}.json", coverage)
        coverage_receipts.append(coverage_ref)
        for arm in ARMS:
            queue_ref = pool["arms"][arm]
            queue = pool_queues[arm]
            if (
                queue["acquisition_seed"] != pool["seed"]
                or queue["future_physics_seed"] != pool["seed"]
                or not queue["pool_registration"] == registration_ref
                or len(queue["rows"]) < 4
            ):
                raise ValueError("bound four-candidate geometric prefix required; no refill")
            selections = []
            for witness in coverage["selected"][arm]:
                candidate = candidates[witness["candidate_id"]]
                if candidate["excluded_reason"] is not None:
                    raise ValueError("excluded candidate cannot enter acquisition")
                if candidate["definition"] != witness["definition"]:
                    raise ValueError("candidate and selected scene identity mismatch")
                definition = read_bound(candidate["definition"])
                validate_scene(definition)
                selections.append(
                    dict(
                        **witness,
                        future_branch_order=candidate["future_branch_order"],
                        preassigned_positive_option_id=candidate["assigned_positive_option_id"],
                        proposal_rng_seed=candidate["proposal_rng_seed"],
                        target_rng_seed=candidate["target_rng_seed"],
                    )
                )
            run_id = f"seed{pool['seed']}_{arm}"
            runs.append(
                dict(
                    run_id=run_id,
                    arm=arm,
                    proposal_seed=pool["seed"],
                    physics_seed=pool["seed"],
                    queue=queue_ref,
                    candidates=pool["candidates"],
                    queue_prefix_length=None,
                    common_support_coverage=coverage_ref,
                    selected_candidates=selections,
                    selected_strata=dict(Counter(r["stratum"] for r in selections)),
                    standalone_equivalent_query_costs=queue["standalone_equivalent_query_costs"],
                    rounds=run_slots(
                        args.execution_root,
                        run_id,
                        pool["seed"],
                        selections,
                        bank.option_ids,
                        bootstrap,
                    ),
                )
            )
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    plan = dict(
        schema=PLAN_SCHEMA,
        proposal_version=3,
        previous_proposal=(
            None if args.previous_proposal is None else artifact(args.previous_proposal)
        ),
        adoption_status="proposed_not_adopted",
        execution_authorized=False,
        current_physics_steps=0,
        current_acquired_primary_corpora=0,
        intended_execution_root=str(args.execution_root),
        registry=registry,
        option_ids=list(bank.option_ids),
        source_ancestry="41002 development ancestry; independently drawn layouts, not unseen motion source",
        candidate_pool_registration=registration_ref,
        candidate_pool_result=pool_ref,
        frozen_draws=registration["draws"],
        common_support_coverage=coverage_receipts,
        coverage_rule=dict(
            schema="motion2scene_common_support_coverage_v1",
            support="Intersection of eligible scene strata across all four constructors, per seed",
            slots="Common short/sustained/early singles, then common courses; four distinct strata",
            course_rotation=(
                "[short_then_short,short_then_sustained] rotated left by seed index modulo2"
            ),
            within_stratum="First original constructor queue row; no outcome or reference reranking",
            shortages="Report unsupported/omitted strata; refuse duplicate/refill for fewer than four",
            estimand="Constructor comparison conditional on common stratum support and coverage quotas",
            reference_diversity="Report positive witnesses only; every group measures all seven schedules",
            original_pool_coverage="Retain all original eligibility and unconstrained queue coverage",
        ),
        domain_lock=registration["domain_lock"],
        runtime_review_snapshot=artifact(args.runtime_review),
        implementation=sources,
        budget=budget_contract(),
        physical_accounting=dict(
            scope="New bootstrap, pre-update student and forced teacher physics only",
            fresh_bootstrap_per_arm=True,
            aggregate_planned_episodes=12 * 39,
            aggregate_maximum_planned_physics_steps=12 * 46488,
            shared_captures_between_arms=False,
            planned_steps_are_actual_only_after_measurement=True,
            inherited_qualification_inputs=registration["option_qualification_inputs"],
            inherited_qualification_cost=(
                "Report unique original receipts separately and equal allocation; not free, not new50k "
                "steps"
            ),
            acquisition_attempts=(
                "Every launched successful, failed or partial teacher/student attempt consumes budget"
            ),
            unknown_counts="Reserve full1192 for a launched unmeasured attempt; preserve null actual count",
            resource_pauses="Unlaunched resource preflight costs zero physics; log wall time",
            retry_policy=(
                "Never repeat a launched attempt. Known physical failures continue the scheduled "
                "collection; unknown infrastructure failures stop the run pending a registered amendment"
            ),
            fit_failure_policy=(
                "Retain incomplete/unsolvable labels; if no valid common learner can be fitted, stop "
                "before next student, no fabricated fallback"
            ),
            slack_policy="3512 left unallocated; no hidden fifth group, student refresh or uncharged sensing",
            later_budgets=(
                "150k/450k are followup designs; old teacher-only pool prefixes do not describe this50k "
                "online plan"
            ),
        ),
        proposal_accounting=dict(
            actual_shared_candidates=sum(p["proposed_candidates"] for p in pools["pools"]),
            actual_shared_clearance_queries=pools["actual_shared_clearance_queries"],
            actual_shared_clearance_search_seconds=sum(
                p["shared_clearance_search_seconds"] for p in pools["pools"]
            ),
            actual_sensor_queries=0,
            semantics=(
                "Count the full frozen searches, including every rejected/nonselected proposal, once; "
                "standalone arm-equivalent query counts are logical costs, not extra actual work"
            ),
            acquisition_selection=(
                "First queue row in each matched common-support stratum; no outcome filtering/refill"
            ),
            target_only=(
                "Positive reference assignment was drawn independently before any clearance query; target "
                "ranking does not consume negative outcomes"
            ),
        ),
        common_student=dict(
            features=114,
            decision_ticks=[15, 50, 70],
            observations=(
                "Actual mounted65-ray2s/101-frame floor-ceiling history, robot state, active option and "
                "legality only"
            ),
            architecture="Common separate-phase linear measured-regret ridge readout; masked by legal schedules",
            penalty=None,
            final_learner_contract=None,
            pending=(
                "Root's registered leave-one-development-layout-out penalty study, then one frozen common "
                "learner artifact before adoption; no primary labels used for choice"
            ),
            bootstrap="FitM0 only from seven freshly executed empty teachers for this arm/seed",
            round_update=(
                "FitMr on bootstrap plus rounds1..r only; same target admission, features, solver and "
                "penalty across arms"
            ),
            labels=(
                "Complete physical future-schedule regret targets; unknown/illegal/missing branches never "
                "become failures"
            ),
            complete_student_teacher_matching=(
                "Audit actual student-visited neutral prefix, sensor history, legal mask and physical "
                "seed exactly; unmatched continuation has no verified gap"
            ),
        ),
        curriculum=dict(
            method_scope=(
                "Observation-aware replay of acquired encounters; analytic and observation arms use "
                "identical candidate IDs/order"
            ),
            baseline_weights=(
                "Uniform encounter mass, then complete consequential phase targets; no extra mass for "
                "additional WAIT visits"
            ),
            observation_weights=dict(uniform=0.2, coverage=0.2, verified_gap=0.6),
            availability=(
                "Pending actual causal sensor delivery at the decision tick; no true-scene visibility "
                "proxy or uncharged sensor preview"
            ),
            eligibility=(
                "Complete physical outcomes plus exact pre-action student/teacher state and sensor "
                "history; surface visibility by each current decision deadline for all enabled beams"
            ),
            gap=(
                "Measured full-continuation teacher versus the pre-update student's actual passage/time "
                "cost; missing or unobservable evidence contributes no gap mass"
            ),
            historical_gap=(
                "Retain generating model identity and age; no free refresh, no assertion of current "
                "final-model error"
            ),
            no_gap_fallback="Uniform encounter weights, then complete phase targets",
            coverage=(
                "Report acquired strata; balance measured teacher-option/length/phase/entry/return replay "
                "strata where available; GT metadata is not a policy feature"
            ),
            no_corpus_discard=(
                "Physical failures and both-pass branches stay in the corpus; visibility gates replay "
                "priority only"
            ),
            learned_generator="Optional future acceleration only; retain existing negative development result",
        ),
        chronology=dict(
            sequence=[
                "Execute seven empty bootstrap teachers; fitM0",
                "CommitM(r-1) and exactly rounds0..r-1 labels before preparing current student",
                "Execute one current student; preserve actual failure as a valid measurement",
                "Write student-complete release while current teacher directory is absent",
                "Execute all seven current teacher schedules in frozen candidate order",
                "Audit complete measured prefixes/outcomes; write completed-prefix receipt",
                "Compute historical replay weights and fitMr; proceed to next round",
            ],
            markers=(
                "bind_preupdate_round and release_current_teachers write exclusive immutable markers; "
                "adopted controller must enforce them before dispatch"
            ),
            receipt_prefix="round0 plus rounds1..r, statuscomplete_prefix; final statuscomplete requiresr4",
            limitation=(
                "SHA commitments verify recorded provenance, not external attestation or absence of "
                "unregistered experiments"
            ),
        ),
        intended_comparisons=[
            (
                "Uniform/target-only/analytic compare geometric training selection under a common learner "
                "and actual-step budget"
            ),
            (
                "Observation curriculum versus analytic isolates actual observation/gap replay weighting "
                "under identical geometry and counts"
            ),
            (
                "Three seeds provide independently acquired corpora when all fresh executions complete; "
                "reused development groups are excluded"
            ),
        ],
        risks=[
            (
                "Four layouts provide limited coverage; common-support conditioning excludes strata "
                "unavailable to any one constructor"
            ),
            (
                "Near interpolation in an overparameterized ridge model can make replay weights nearly "
                "ineffective; no advantage is promised"
            ),
            (
                "Surface evidence establishes measured cue delivery, not semantic distinguishability or "
                "removal of every partial-observation alias"
            ),
            (
                "Geometric positivity does not guarantee physical success; zero complete feasible targets "
                "can stop a run"
            ),
            (
                "Historical pre-update gaps can become stale after learning; this protocol does not pay "
                "to refresh them"
            ),
            (
                "Final traversal benefit needs actual locked evaluation with separately reported "
                "evaluation steps; training fit is not downstream performance"
            ),
        ],
        unresolved_before_adoption=[
            "Freeze common regularization/learner contract after the separately registered development study",
            "Bind final runtime asset/source closure, teacher/admission and replay rule code hashes",
            (
                "Bind and test an orchestration controller that enforces causal markers and budgets; this "
                "registrar launches no physics"
            ),
            "Issue separate adopted record naming this exact proposed plan and the final common contracts",
            "Bind actual locked evaluation protocol and resource accounting; larger budgets require a new plan",
        ],
        runs=runs,
    )
    validate_plan(plan)
    ref = write_new(args.out / "plan.json", plan)
    write_new(
        args.out / "registration_result.json",
        dict(
            schema="motion2scene_primary_acquisition_registration_result_v1",
            plan=ref,
            status="proposed_not_adopted",
            actual_new_physics_steps=0,
            selected_primary_layouts=sum(len(r["selected_candidates"]) for r in runs),
            twelve_fresh_bootstraps_required=True,
        ),
    )
    print(json.dumps(ref))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--empty", type=Path, required=True)
    parser.add_argument("--runtime-review", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous-proposal", type=Path)
    build_plan(parser.parse_args())


if __name__ == "__main__":
    main()
