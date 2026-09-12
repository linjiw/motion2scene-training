#!/usr/bin/env python3
"""Write a review-only protocol amendment without evaluating reserved geometry."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from gear_sonic.dataset_generation.hallucination.motion2scene_evaluation_protocol import (
    ADOPTION_GATES,
    SCHEMA,
    _canonical_beam,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    DEFAULT_AZIMUTHS_DEG,
    DEFAULT_ELEVATIONS_DEG,
    HistoryGrid,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    expected_feature_names,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/linjiw/research-data/groot-wbc")


def artifact(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def create(out):
    if out.exists() or out.with_suffix(".sha256").exists():
        raise FileExistsError("Use a new proposal version; do not replace an existing receipt")
    lock_path = ROOT / "docs/motion2scene/TRAVERSAL_EVALUATION_LOCK_V3.json"
    lock_ref = artifact(lock_path)
    if lock_ref["sha256"] != "509817600075888ef5cf681c1033c9dcd6796d7a08561e30ee4157049aff5181":
        raise ValueError("original V3 lock changed")
    lock = json.loads(lock_path.read_text())
    request_path = (
        DATA / "m2s-prior-splice-seven-schedule-preparation-v1/qualification_request.json"
    )
    request = json.loads(request_path.read_text())
    materialized_path = DATA / "m2s-reserved-geometry-materialization-v3/result.json"
    materialized = json.loads(materialized_path.read_text())
    usd = {Path(ref["path"]).stem: ref for ref in materialized["scene_files"]}
    scenes = []
    for layout in lock["evaluation"]["layouts"]:
        for variant in layout["fixed_world_variants"]:
            name = layout["layout_id"] + "__" + variant["offset_id"]
            scenes.append(
                dict(
                    layout_id=layout["layout_id"],
                    variant_id=variant["offset_id"],
                    scene_id=name,
                    family=layout["family"],
                    scene=usd[name],
                    beams=[_canonical_beam(b) for b in variant["beams"]],
                    beam_collision_enabled=[True] * len(variant["beams"]),
                )
            )
    modules = ROOT / "gear_sonic/dataset_generation/hallucination"
    runtime_names = [
        "motion2scene_timed_schedule_execution",
        "motion2scene_timed_schedule_policy",
        "motion2scene_timed_options",
        "motion2scene_long_schedule_execution",
        "motion2scene_timed_history_execution",
        "motion2scene_environment_contact_execution",
        "motion2scene_observation_history",
        "motion2scene_observation_delay",
        "motion2scene_closed_loop_policy",
        "motion2scene_evaluation_protocol",
        "motion2scene_schedule_script",
    ]
    scoring_names = [
        "motion2scene_course",
        "motion2scene_passage",
        "motion2scene_environment_contacts",
        "motion2scene_timed_course_measurements",
        "motion2scene_sensor_alignment",
    ]
    implementation = dict(
        registry=artifact(DATA / "m2s-prior-splice-seven-schedule-analysis-v2/registry.json"),
        request=artifact(request_path),
        controller=request["controller"],
        runtime_artifacts=[artifact(modules / (name + ".py")) for name in runtime_names]
        + [artifact(ROOT / "scripts/research/motion2scene_collect_timed_schedules.py")],
        scoring_artifacts=[artifact(modules / (name + ".py")) for name in scoring_names],
        qualification_result=artifact(
            DATA / "m2s-prior-splice-seven-schedule-analysis-v2/environment_result.json"
        ),
        current_files_are_review_snapshots_not_final_closure=True,
        final_adoption_requires=(
            "Transitive source closure, native asset/config and environment hashes; current review"
            " snapshots must be refreshed after fixes before adoption."
        ),
    )
    option_ids = ["neutral"] + [o["option_id"] for o in request["options"]]
    policies = []

    def add_policy(name, mode, option="neutral", reference="sustained", **extra):
        policies.append(
            dict(
                policy_id=name,
                mode=mode,
                preferred_option_id=option,
                preferred_reference_id=reference,
                model=None,
                **extra,
            )
        )

    add_policy("always_walk", "always_walk")
    add_policy(
        "constant_development_selected",
        "constant_option",
        option=None,
        selection_artifact=None,
        artifact_status="PENDING_DEVELOPMENT_CONSTANT_SELECTION",
    )
    add_policy(
        "scripted_multi_option",
        "scripted_multi",
        script_parameters=None,
        artifact_status="CPU_CANDIDATE_EXISTS_RUNTIME_AND_DEVELOPMENT_VALIDATION_PENDING",
    )
    for arm in lock["comparison_matrix"][:4]:
        for corpus_seed in lock["acquisition"]["acquisition_seeds"]:
            for budget in [50000]:
                add_policy(
                    f"{arm['arm']}__seed{corpus_seed}__budget{budget}",
                    "learned",
                    training_arm=arm["arm"],
                    acquisition_seed=corpus_seed,
                    acquisition_budget_steps=budget,
                    artifact_status="PENDING_DEVELOPMENT_TRAINING",
                )
    proposal = dict(
        schema=SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="PROPOSED_NOT_ADOPTED_NO_EXECUTION_AUTHORITY",
        adoption_receipt=None,
        adoption_requirements={
            key: key in ("seven_schedules_physically_qualified", "no_reserved_outcomes_inspected")
            for key in ADOPTION_GATES
        },
        geometry_lock=lock_ref,
        previous_lock=lock["relationship_to_v2"]["original"],
        materialization_result=artifact(materialized_path),
        native_geometry_audit=materialized["native_geometry_audit"],
        source_ancestry=(
            "Source41002 development ancestry; same locked180frame neutral, authored"
            " short/sustained and explicitly projected/spliced conditioned prior. No source-heldout"
            " or raw-prior-feasibility claim."
        ),
        geometry_change=False,
        replaces_v2=False,
        completes_v2_72_episodes=False,
        physics_seeds=lock["evaluation"]["physics_seeds"],
        execution_order_seed=lock["evaluation"]["execution_order_seed"],
        execution_order=(
            "Before execution enumerate policy_id x layout_id x physics_seed x variant_id; shuffle"
            " with Python random.Random(execution_order_seed), hash full assignment table. Do not"
            " reorder from outcomes."
        ),
        nominal_episodes_per_policy=36,
        stress_episodes_per_policy=288,
        scenes=scenes,
        implementation=implementation,
        schedules=dict(
            option_ids=option_ids,
            definitions=request["options"],
            references=request["references"],
            decision_ticks=[15, 50, 70],
            reference_phase_s=[0.30, 1.00, 1.40],
            capture_elapsed_s=[0.28, 0.98, 1.38],
            nominal_qualified_indices_by_phase=[[0, 1, 2, 3], [0, 4], [0, 5, 6]],
            waiting_continuation_indices_by_phase=[[0, 4, 5, 6], [0, 5, 6], [0]],
            maximum_entries=1,
            no_reentry=True,
            automatic_return=True,
            maximum_reference_joint_jump_rad=0.05,
            maximum_reference_root_jump_m=0.01,
            interpretation=(
                "Neutral at an earlier phase means wait with later commitments retained; at tick70"
                " it commits to terminal walking. A complete option fixes entry and return; no"
                " adaptation-to-adaptation switch or return learning."
            ),
            blocked_guard=(
                "A nonneutral current guard may remove that action. Mandatory return refusal is a"
                " recorded task failure, never a silent hold or protective stop."
            ),
        ),
        sensor=dict(
            schema="motion2scene_timed_schedule_value_v1",
            feature_dimension=114,
            feature_names=list(expected_feature_names(7)),
            rate_hz=50,
            range_m=4.0,
            origin_offset_in_root_body_m=[0.2, 0.0, 0.4],
            elevations_deg=list(DEFAULT_ELEVATIONS_DEG),
            azimuths_deg=list(DEFAULT_AZIMUTHS_DEG),
            ray_count=65,
            pose="measured root position and quaternion; ideal known pose",
            model=(
                "nearest PhysX environment hit plus measured normal; robot excluded, no semantic"
                " object selection"
            ),
            noise="none",
            delivery_delay_s=0.0,
            history=asdict(HistoryGrid()),
            feature_order=(
                "100 existing sensor/state/phase values,7 preaction active one-hot indicators,7"
                " legal indicators"
            ),
            exclusions=lock["evaluation"]["student_input_exclusions"],
            unknown_space=(
                "Keep independent observed/unknown floor and ceiling masks; no unseen space filled"
                " as free."
            ),
            timing_audit=(
                "Retain measured per-beam surface and underside first delivery at every decision;"
                " geometric endpoint association is audit-only. Never exclude an evaluation scene"
                " for late/unseen evidence."
            ),
        ),
        policies=policies,
        enabled_variant_ids=["nominal"],
        experiment_scope_amendment=dict(
            status="PROPOSED_PRIMARY_SCOPE_NOT_ADOPTED",
            primary_arms=[arm["arm"] for arm in lock["comparison_matrix"][:4]],
            primary_acquisition_seeds=lock["acquisition"]["acquisition_seeds"],
            primary_budget_checkpoints=[50000],
            primary_variants=["nominal"],
            primary_baselines=[
                "always_walk",
                "constant_development_selected",
                "scripted_multi_option",
            ],
            nominal_layouts_removed=0,
            seeds_removed=0,
            original_fullmatrix_completed=False,
            optional_followups=dict(
                status="PLANNED_UNEXECUTED_NOT_PART_OF_PRIMARY_ADOPTION",
                budget_checkpoints=[150000, 450000],
                constructor_arms=[arm["arm"] for arm in lock["comparison_matrix"][4:]],
                stress_variants=[
                    v["id"] for v in lock["evaluation"]["perturbation_set"] if v["id"] != "nominal"
                ],
                isolated_ablations=lock["isolating_ablations"],
                finite_capability_oracle=(
                    "Optional separately costed seven-complete-schedule execution on every"
                    " layout/seed. The primary three baselines do not supply a complete oracle."
                ),
                authorization=(
                    "Require a separately frozen follow-up protocol before outcomes are used for"
                    " its tuning; do not treat prior reserved nominal outcomes as untouched"
                    " follow-up evidence."
                ),
            ),
        ),
        baseline_protocol=dict(
            strong_multi_option_script=dict(
                status="CPU_CANDIDATE_IMPLEMENTED_NOT_RUNTIME_VALIDATED",
                candidate_source=artifact(modules / "motion2scene_schedule_script.py"),
                input_features=(
                    "Current near/far upper_hit bands, measured same-band floor/ceiling fractions"
                    " and heights, phase and current/future schedule legality; no scene identities."
                ),
                rule=(
                    "A band is hazardous when upper_hit>0 unless both floor and ceiling are"
                    " observed and their gap is at least the free-height threshold. At tick15, a"
                    " hazard within the immediate near-band count selects legal prior_splice; at"
                    " tick50 use the later near-band count. At tick70 any hazard selects sustained"
                    " when enough bands are hazardous, otherwise short. If the chosen reference has"
                    " no current legal option, keep neutral. Mandatory return bypasses the script."
                ),
                parameter_grid=dict(
                    minimum_observed_free_height_m=[1.20, 1.25, 1.30, 1.36, 1.40],
                    immediate_prior_band_count=[1, 2],
                    later_prior_band_count=[1, 2, 3],
                    sustained_minimum_hazard_bands=[1, 2, 3],
                ),
                clearance_rule=(
                    "Use minimum_ceiling_m minus maximum_floor_m only within a band with both"
                    " observed fractions greater than zero; unknown clearance never becomes an"
                    " observed number."
                ),
                selection=(
                    "Rank90 fixed rule configurations solely on a named complete matched"
                    " seven-schedule development corpus: passage first then measured time,"
                    " canonical parameter-tuple ties. Offline finite replay requires exact neutral"
                    " histories and all required complete continuations; validate the selected rule"
                    " through actual one/two-beam development execution before adoption."
                ),
                freeze=(
                    "Bind the selected parameter JSON and complete runtime source closure; learning"
                    " over a one-reference script alone never establishes curriculum superiority."
                ),
                action_coverage_limit=(
                    "Current candidate never selects the early authored short/sustained schedules."
                    " If development evidence shows this is consequential, revise the script and"
                    " its finite tuning grid before adoption; do not change it from V3 outcomes."
                ),
            ),
            constant=(
                "Evaluate all six exact constants only on a named development calibration corpus."
                " Choose one by development passage, then measured paired-success time, then"
                " canonical option ID; pin selection receipt and use that one unchanged on all36"
                " primary assignments. The other constants remain development evidence, not"
                " mandatory held-out policies."
            ),
            scripted=(
                "The three existing one-reference scripts remain simple development baselines. They"
                " are not the strong primary comparator. Optional future held-out evaluations must"
                " be explicitly registered."
            ),
            development_tuning=(
                "Freeze these scripts with threshold0 before evaluation; any replacement threshold"
                " must be selected solely on named development corpora and recorded in a new"
                " proposal before adoption."
            ),
            capability_comparator=(
                "Union of physically successful complete schedules, with continuation and"
                " actual-action audit. Reuse constant/walk recordings only when they actually"
                " executed the exact schedule under the identical runtime/state/seed; otherwise"
                " keep comparator missing or acquire an explicitly registered branch. No per-scene"
                " best script presented as a deployable policy."
            ),
            no_privileged_script_inputs=True,
        ),
        learning=dict(
            arms=lock["comparison_matrix"][:4],
            common_model="phase-specific ridge value heads",
            schema="motion2scene_timed_schedule_value_v1",
            l2=1e-6,
            normalization=(
                "Each phase uses only its admitted training rows; zero-variance coordinates scale1."
                " Normalization, heads, masks and request digest frozen before evaluation."
            ),
            values=(
                "Measured physical relative regret from complete legal tables; passage-first finite"
                " teacher then measured passage time. Regression does not guarantee lexicographic"
                " success under partial observability."
            ),
            teacher_wait=(
                "Require every expected neutral/future full schedule, common seed and exact"
                " preaction physical/sensor/controller-history digest. Missing/failed/unadmitted"
                " continuations never become optimistic complete WAIT labels."
            ),
            model_gate=(
                "Every potentially legal phase head must have measured training support; missing"
                " heads block model registration. No evaluation-time fallback or fitting."
            ),
            corpus_scope=(
                "Different scene layouts only; original V3 nominal and stress geometry excluded"
                " from clearance, teacher, policy, threshold and model-selection development"
                " queries."
            ),
            isolated_ablations=lock["isolating_ablations"],
        ),
        acquisition={**lock["acquisition"], "budget_checkpoints": [50000]},
        acquisition_six_second_accounting=dict(
            physics_dt_s=0.005,
            control_dt_s=0.02,
            maximum_steps_per_complete_branch=1192,
            maximum_steps_per_seven_branch_group=8344,
            maximum_steps_per_group_plus_one_student_visit=9536,
            policy=(
                "Reserve a complete group's maximum before launch. Count actual all-branch steps"
                " including failures/rejections/resets; never truncate a group to meet a"
                " checkpoint. Acquisition remainders are reported, not spent beyond cap."
            ),
            teacher_replay=(
                "Source prefixes must be replayed with the same full4-reference bank and seed;"
                " equal feature vectors alone are insufficient."
            ),
        ),
        scoring=dict(
            reference_frames=299,
            recorded_control_steps=298,
            physics_steps_per_complete_episode=1192,
            last_capture_elapsed_s=5.94,
            last_captured_command_phase_s=5.96,
            control_dt_s=0.02,
            physics_dt_s=0.005,
            run_to_full_horizon=True,
            early_success_stop=False,
            padding_or_wrapping=False,
            root_height_min_m=0.5,
            minus_projected_gravity_z_min=0.5,
            downstream_margin_m=0.1,
            stability_hold_s=0.3,
            stability_hold_samples=16,
            crossing=(
                "For each beam, all recorded body origins project beyond its downstream edge+0.1m"
                " for16 consecutive samples, with rootZ>=.5 and minusgravityZ>=.5. Initially all"
                " origins must be upstream of every beam; complete beam holds in beam order."
                " Collider-extents passage is not claimed."
            ),
            course=(
                "One continuous first episode across both enabled constraints with all beam forces"
                " recorded; no state reset, concatenation or independent per-beam lookup."
            ),
            full_horizon=(
                "No fall/reset, undesired environment normal contact, illegal transition, refusal"
                " or missed return throughout all298 rows; actual neutral return and at least15"
                " recorded ticks afterward for an adaptation. No unmeasured neutral-tail tracking"
                " guarantee."
            ),
            contact=(
                "All30 subjects against exact robot/floor/four-wall/one-or-two-beam paths at200Hz."
                " Foot-floor support allowed; all other environment normal norms>1N fail."
                " Self-contacts and strict nonfoot-net diagnostic separately retained. Include"
                " every beam's every substep, including after first crossing."
            ),
            environment_normal_force_threshold_n=1.0,
            internal_contact_policy=(
                "Report all pairs and new pairs relative to qualification; self-contact alone is"
                " not environmental collision under this explicitly amended criterion."
            ),
            terminal=(
                "Incomplete crossing, incomplete16sample hold or incomplete actual return/recovery"
                " at5.94s is a task failure with a distinct finite_horizon_* flag. Completion time"
                " is right-censored; success is not censored out of the denominator. Never pad,"
                " loop, relocate or extend a scene-dependent horizon."
            ),
            time_cost=(
                "For successful episodes only, physical elapsed time at final required beam hold"
                " completion minus first physical timestamp; separately report full recorded"
                " span5.94 and integrated duration5.96. Failure success-time=null; retain last"
                " observed time and crossing/hold events."
            ),
            cost_weights=dict(passage_time_s=1.0, mechanical_work_j=0.0, switches=0.0),
            mechanical_work=(
                "unknown with current implicit-actuator effort; never zero, battery energy or a"
                " cost benefit inferred from request count"
            ),
            late_decisions=(
                "Audit label only using measured beam visibility and exact legal phase; no"
                " evaluation removal for late/unobservable situations."
            ),
        ),
        failure_accounting=dict(
            orthogonal_fields=[
                "task_status",
                "measurement_admitted",
                "infrastructure_status",
                "supervision_eligible",
                "all_failure_flags",
            ],
            verified_physical_failure=(
                "A source-bound observed fall/reset/contact/refusal/missed return/horizon failure"
                " is a task failure even when later sensor history is ineligible or the process"
                " subsequently aborts."
            ),
            missing=(
                "A startup or recording failure without a source-bound physical outcome is"
                " technical_missing, not a fabricated physical failure or success. Retain assigned"
                " slot and every attempt."
            ),
            primary_denominator=(
                "36 nominal assigned episodes per policy; 288 stress separately. Report successes,"
                " verified physical failures and technical_missing/not_run explicitly. Incomplete"
                " panels are incomplete, not complete-case passage rankings."
            ),
            missing_bounds=(
                "For N assigned, S successes and M technical missing/not-run: completion lower=S/N,"
                " upper=(S+M)/N; additionally report S/(N-M) as measured-only with its denominator,"
                " never as the complete benchmark."
            ),
            retries=(
                "At most one separately recorded exact-configuration retry for a preclassified"
                " transient infrastructure failure, with same seed/model/scene and no tuning. No"
                " automatic retry of physical failure; systematic code faults suspend the whole"
                " panel and require a disclosed amendment, not selective replacement."
            ),
            selection=(
                "Use earliest valid attempt; retain earlier technical wall/step costs. Any verified"
                " physical failure fixes the slot outcome and is never replaced by a favorable"
                " retry."
            ),
        ),
        reporting=dict(
            primary=(
                "paired layout-level completion; two seeds are repeated measurements, not"
                " independent scene samples"
            ),
            strata=[
                "all18_nominal",
                "12_single",
                "6_two_beam",
                "short",
                "sustained",
                "early_constraint",
                "short_then_short",
                "short_then_sustained",
            ],
            uncertainty=(
                "Pair by layout/source/seed; cluster-bootstrap18layout means with20000 draws"
                " seed202609081822, and independently report variability across3 acquired corpora."
                " Courses with6layouts retain small-sample limitation."
            ),
            cost=(
                "Per-scene paired times only on jointly successful episodes with denominator;"
                " report each policy's success count beside cost. No arbitrary failure-time"
                " imputation or energy claim."
            ),
            stress=(
                "All8 fixed world perturbations retained and reported separately; no nominal/stress"
                " pooling or geometry feasibility filtering."
            ),
            scope=(
                "Unseen locked layouts on reused development ancestry, ideal sensing and a finite"
                " one-adaptation route. Not general navigation, source transfer, hardware transfer,"
                " global-optimality or full repeated-adaptation capability."
            ),
        ),
        planned_resource_size=dict(
            primary_fixed_baselines=3,
            primary_learned_checkpoints=12,
            primary_nominal_episodes=540,
            primary_max_evaluation_physics_steps=643680,
            primary_acquisition_cap_steps=600000,
            optional_stress_all15primary_policies_episodes=4320,
            optional_stress_all15primary_policies_max_physics_steps=5149440,
            original_fullmatrix_learned_checkpoints=54,
            original_fullmatrix_nominal_per_learned_checkpoint=36,
            original_fullmatrix_stress_per_policy=288,
            note=(
                "Large full matrix, not execution authorization or a resource allocation. Isolated"
                " ablations and additional nonreusable oracle branches add cost. Any reduced"
                " comparison must be explicitly amended before outcomes; never reduce the18scene"
                " panel."
            ),
        ),
        open_implementation_issues=[
            (
                "Collector currently only SHA-checks an opaque protocol file; integrate"
                " validate_reserved_execution before prepare/run/analyze after current development"
                " smoke sources unfreeze."
            ),
            (
                "No physically completed114D seven-bank runtime smoke or two-beam full-horizon"
                " qualification yet."
            ),
            (
                "Current measurement admission can fail after a real physical reset; implement"
                " orthogonal task failure versus history-supervision status."
            ),
            (
                "Use explicit finite-horizon crossing/hold/return categories consistently in single"
                " and course scores; current single scorer lacks explicit terminal reasons."
            ),
            (
                "Current course scorer checks initial upstream approach, while single scorer does"
                " not; apply the same registered initial-approach rule to both without scene"
                " filtering."
            ),
            (
                "Compile exact full runtime/native asset/environment closure and all learner"
                " checkpoints; current file hashes are review snapshots only."
            ),
            (
                "No late-online policy, repeated adaptation, arbitrary holding, stopping or"
                " steering result is implied by schedule qualification."
            ),
        ],
        prior_inspection=dict(
            reserved_clearance_queries=0,
            reserved_physics_steps=0,
            reserved_outcomes_inspected=False,
            development_evidence_used=(
                "Existing9timed teachers and interim seed8732 development horizon failure informed"
                " the finite-horizon distinction; no reserved geometry was selected from outcomes."
            ),
        ),
        amendments=(
            "Do not overwrite V2/V3 geometry locks or this proposal after publication. Adoption is"
            " a new artifact with all gates true, exact final hashes and a separately hash-bound"
            " adoption receipt; neither elapsed time nor a registry file adopts this proposal."
        ),
        builder=artifact(Path(__file__)),
    )
    out.write_text(json.dumps(proposal, indent=2, sort_keys=True, allow_nan=False) + "\n")
    out.with_suffix(".sha256").write_text(artifact(out)["sha256"] + "  " + out.name + "\n")
    print(
        json.dumps(
            {
                "proposal": artifact(out),
                "scenes": len(scenes),
                "policies": len(policies),
                "status": proposal["status"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    create(parser.parse_args().out)
