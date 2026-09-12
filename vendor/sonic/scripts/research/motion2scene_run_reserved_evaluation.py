#!/usr/bin/env python3
"""Prepare and serially execute the separately adopted V4 nominal evaluation.

V4 changes checkpoint scope, retaining the V3 geometry/validator wire schema.
Proposals and incomplete checkpoint inventories cannot prepare a rollout.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import motion2scene_collect_schedule_batch as batch  # noqa: E402
import motion2scene_collect_timed_schedules as collector  # noqa: E402
import motion2scene_run_primary_acquisition as primary  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    ARMS,
    SEEDS,
    artifact,
    checked_path,
    read_bound,
    same_artifact,
    source_identities,
    validate_completed_replay_order,
    validate_plan,
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_evaluation_protocol import (  # noqa: E402
    ADOPTED,
    SCHEMA as PROTOCOL_SCHEMA,
    _validate_panel,
    validate_reserved_execution,
)

SCHEMA = "motion2scene_reserved_execution_v4"
INVENTORY_SCHEMA = "motion2scene_reserved_checkpoint_inventory_v4"
CHECKPOINTS = {"M2": (2, 23, 27416), "M4": (4, 39, 46488)}
BASELINES = ("always_walk", "constant_development_selected", "scripted_multi_option")
ARM_NAMES = {
    arm: ("analytic_observation_curriculum" if arm == "observation_curriculum" else arm)
    for arm in ARMS
}
CONTRACT = dict(
    scope_revision=4,
    enabled_variant_ids=["nominal"],
    learned_checkpoints=24,
    baselines=3,
    episodes_per_policy=36,
    assigned_episodes=972,
    maximum_physics_steps_per_episode=1192,
    maximum_physics_steps=1158624,
    automatic_retries=False,
    unknown_attempt_policy="retain_in_denominator_and_pause_without_retry",
    execution_order_seed=202609081821,
    all_assignments_required_before_launch=True,
)


def model_slots(plan):
    validate_plan(plan)
    result = []
    for arm in ARMS:
        for seed in SEEDS:
            run_id = f"seed{seed}_{arm}"
            run = next(row for row in plan["runs"] if row["run_id"] == run_id)
            folder = Path(plan["intended_execution_root"]) / run_id
            for checkpoint, (index, episodes, maximum) in CHECKPOINTS.items():
                row = run["rounds"][index]
                result.append(
                    dict(
                        policy_id=f"{ARM_NAMES[arm]}__seed{seed}__{checkpoint}",
                        run_id=run_id,
                        training_arm=ARM_NAMES[arm],
                        acquisition_seed=seed,
                        checkpoint=checkpoint,
                        completed_through_round=index,
                        acquisition_assigned_episodes=episodes,
                        acquisition_assigned_maximum_steps=maximum,
                        expected_model_path=row["expected_postupdate_policy_path"],
                        expected_training_result_path=row[
                            "expected_postupdate_training_result_path"
                        ],
                        expected_completion_path=str(folder / "controller" / "completion.json"),
                        expected_checkpoint_receipt_path=str(
                            folder
                            / "controller"
                            / ("prefix_002.json" if index == 2 else "completion.json")
                        ),
                    )
                )
    return result


def propose_v4(previous_ref, plan_ref):
    """New immutable scope proposal; never edits or adopts the V3 predecessor."""
    previous, plan = read_bound(previous_ref), read_bound(plan_ref)
    if previous["schema"] != PROTOCOL_SCHEMA or previous["status"] == ADOPTED:
        raise ValueError("the preserved unadopted V3 proposal is required")
    _validate_panel(previous)
    result = copy.deepcopy(previous)
    result.update(
        created_utc=primary.utc(),
        scope_amendment_builder=artifact(Path(__file__)),
        execution_scope_revision=4,
        previous_scope_proposal=previous_ref,
        primary_acquisition_plan=plan_ref,
        status="PROPOSED_V4_SCOPE_NOT_ADOPTED_NO_EXECUTION_AUTHORITY",
        adoption_receipt=None,
        execution_runner=None,
        scope_change="M2 and M4 on each same corpus trajectory; no additional acquisition physics",
    )
    result["adoption_requirements"] = {key: False for key in previous["adoption_requirements"]}
    result["adoption_requirements"]["no_reserved_outcomes_inspected"] = True
    baseline_policies = [
        copy.deepcopy(row) for row in previous["policies"] if row["policy_id"] in BASELINES
    ]
    if {row["policy_id"] for row in baseline_policies} != set(BASELINES):
        raise ValueError("three explicitly retained baseline definitions required")
    result["policies"] = baseline_policies + [
        dict(
            **slot,
            mode="learned",
            model=None,
            preferred_option_id="neutral",
            preferred_reference_id="sustained",
            artifact_status="PENDING_ACTUAL_PRIMARY_CHECKPOINT",
        )
        for slot in model_slots(plan)
    ]
    result["checkpoint_protocol"] = dict(
        checkpoints=[
            dict(
                name=name,
                completed_through_round=values[0],
                assigned_episodes=values[1],
                assigned_maximum_physics_steps=values[2],
            )
            for name, values in CHECKPOINTS.items()
        ],
        acquisition_cap_per_corpus=50000,
        independent_corpus_seeds=list(SEEDS),
        checkpoint_dependence="M2/M4 share one corpus trajectory; only three seeds are independent",
        horizontal_axis="actual measured acquisition steps through each checkpoint; maxima separately disclosed",
        no_additional_acquisition_physics=True,
        scope="two available early checkpoints, not the proposed150k/450k extensions",
    )
    result["planned_resource_size"].update(
        primary_learned_checkpoints=24,
        primary_fixed_baselines=3,
        primary_nominal_episodes=972,
        primary_max_evaluation_physics_steps=1158624,
        primary_acquisition_cap_steps=600000,
    )
    for key in (
        "optional_stress_all15primary_policies_episodes",
        "optional_stress_all15primary_policies_max_physics_steps",
    ):
        result["planned_resource_size"].pop(key, None)
    result["planned_resource_size"].update(
        optional_stress_all27primary_policies_episodes=7776,
        optional_stress_all27primary_policies_max_physics_steps=9268992,
        optional_stress_status="retained_unexecuted_followup_not_enabled",
    )
    result["acquisition"].update(
        budget_checkpoints=[27416, 46488],
        corpus_budget_cap_steps=50000,
        checkpoint_axis="assigned maxima only; plot actual measured steps from M2/M4 receipts",
    )
    result["experiment_scope_amendment"].update(
        status="PROPOSED_V4_CHECKPOINT_SCOPE_NOT_ADOPTED",
        primary_budget_checkpoints=[27416, 46488],
        checkpoint_axis_kind="assigned maxima; actual measured acquisition steps reported",
    )
    result["learning"]["l2"] = None
    result["learning"]["common_learner_status"] = "bind actual frozen common learner at adoption"
    result["execution_contract"] = CONTRACT
    return result


def assignments(protocol, output):
    """Complete immutable pairing; no feasibility, visibility or outcome filtering."""
    _validate_panel(protocol)
    if protocol.get("execution_scope_revision") != 4 or protocol.get("enabled_variant_ids") != [
        "nominal"
    ]:
        raise ValueError("V4 nominal-only scope is required; follow-ups need their own protocol")
    scenes = sorted(
        (row for row in protocol["scenes"] if row["variant_id"] == "nominal"),
        key=lambda row: row["layout_id"],
    )
    policies = protocol["policies"]
    ids = [row["policy_id"] for row in policies]
    if len(ids) != 27 or len(set(ids)) != 27 or len(scenes) != 18:
        raise ValueError("all27 policies and18 nominal layouts are required")
    if protocol["execution_order_seed"] != CONTRACT["execution_order_seed"]:
        raise ValueError("original fixed execution order seed must be retained")
    rows = [
        dict(
            policy_id=policy_id,
            layout_id=scene["layout_id"],
            variant_id="nominal",
            scene_id=scene["scene_id"],
            physics_seed=seed,
        )
        for policy_id in sorted(ids)
        for scene in scenes
        for seed in protocol["physics_seeds"]
    ]
    random.Random(CONTRACT["execution_order_seed"]).shuffle(rows)
    for index, row in enumerate(rows):
        row.update(
            index=index,
            assignment_id=f"nominal_{index:04d}",
            collection_directory=str(Path(output).resolve() / "batch" / f"collection_{index:03d}"),
        )
    if len(rows) != 972:
        raise ValueError("the complete972-assignment denominator is required")
    return rows


def inventory_template(plan_ref, protocol_ref):
    plan, protocol = read_bound(plan_ref), read_bound(protocol_ref)
    return dict(
        schema=INVENTORY_SCHEMA,
        status="PENDING_ACTUAL_ARTIFACTS_NOT_EXECUTABLE",
        primary_plan=plan_ref,
        primary_adoption=None,
        models=[
            dict(**row, completion=None, checkpoint_receipt=None, model=None, training_result=None)
            for row in model_slots(plan)
        ],
        baselines=[
            dict(policy=policy, development_validation=None)
            for policy in protocol["policies"]
            if policy["policy_id"] in BASELINES
        ],
    )


def same(actual, expected):
    checked_path(actual)
    checked_path(expected)
    if not same_artifact(actual, expected):
        raise ValueError("actual checkpoint identity differs from its registered slot")


def recording_counts(complete, run):
    """Reconcile every assigned acquisition recording, including known failures."""
    captures, totals, measured = set(), {}, 0
    for entry, slot in zip(complete["entries"], run["rounds"], strict=True):
        for kind in (("teacher",) if slot["round_index"] == 0 else ("teacher", "student")):
            result = read_bound(entry[kind + "_collection"])
            manifest = read_bound(result["manifest"])
            count = 7 if kind == "teacher" else 1
            if (
                result.get("schema") != collector.COLLECTION_SCHEMA
                or len(result["rows"]) != count
                or len(manifest["cells"]) != count
                or result.get("unmeasured_failed_attempts") != 0
                or manifest["scene_definition"] != slot["scene_definition"]
            ):
                raise ValueError(
                    "each checkpoint must retain every assigned measured acquisition slot"
                )
            steps = 0
            expected_mode = "forced" if kind == "teacher" else "learned"
            if (
                kind == "teacher"
                and [cell["forced_option_id"] for cell in manifest["cells"]]
                != slot["teacher_branch_order"]
            ):
                raise ValueError("all seven assigned teacher schedules must remain in frozen order")
            for cell, row in zip(manifest["cells"], result["rows"], strict=True):
                attempt = read_bound(row["attempt"])
                actual_steps = row.get("physics_steps")
                if (
                    row["cell_id"] != cell["cell_id"]
                    or cell["timed_schedule_mode"] != expected_mode
                    or row["mode"] != expected_mode
                    or cell.get("runtime_seed") != slot["physics_seed"]
                    or row.get("task_outcome_admitted") is not True
                    or row.get("outcome", {}).get("task_outcome") not in ("pass", "failure")
                    or type(actual_steps) is not int
                    or not 0 < actual_steps <= 1192
                    or row["outcome"]["exit_status"] != attempt["exit_status"]
                    or attempt["command"] != cell["command"]
                ):
                    raise ValueError(
                        "unknown/missing acquisition attempts cannot supply a complete checkpoint"
                    )
                same(row["attempt"], artifact(Path(cell["output"]) / "attempt.json"))
                expected_output = Path(slot[kind + "_directory"]) / "rollouts" / cell["cell_id"]
                if Path(cell["output"]).resolve() != expected_output.resolve():
                    raise ValueError(
                        "each independent corpus requires its own registered capture slot"
                    )
                raw = row.get("raw_artifacts", {})
                capture = (
                    row.get("trajectory")
                    or raw.get("trajectory")
                    or raw.get("aborted_raw_recording")
                )
                identity = str(checked_path(capture).resolve())
                if not Path(identity).is_relative_to(expected_output.resolve()):
                    raise ValueError("acquisition capture is outside its original rollout slot")
                if identity in captures:
                    raise ValueError("one original capture cannot fill two acquisition slots")
                captures.add(identity)
                steps += actual_steps
            if steps != result["physics_steps"]:
                raise ValueError("acquisition step total differs from actual assigned rows")
            measured += steps
        totals[slot["round_index"]] = measured
    if len(captures) != 39 or measured > 50000:
        raise ValueError("complete39-slot corpus under the common50k cap required")
    return totals


def validate_inventory(inventory, protocol):
    if inventory.get("schema") != INVENTORY_SCHEMA or inventory.get("status") != "FROZEN_COMPLETE":
        raise ValueError("all real primary checkpoints must exist before reserved preparation")
    plan = read_bound(inventory["primary_plan"])
    expected = model_slots(plan)
    slots = {row["policy_id"]: row for row in expected}
    models = inventory["models"]
    if len(models) != 24 or {row["policy_id"] for row in models} != set(slots):
        raise ValueError("exactly24 M2/M4 checkpoints from all12 independent corpus runs required")
    policies = {row["policy_id"]: row for row in protocol["policies"]}
    if set(policies) != set(slots) | set(BASELINES) or len(protocol["policies"]) != 27:
        raise ValueError("partial or substituted comparison policy inventory")
    if not same_artifact(protocol["primary_acquisition_plan"], inventory["primary_plan"]):
        raise ValueError("primary plan differs between inventory and evaluation protocol")
    adoption = read_bound(inventory["primary_adoption"])
    learner = read_bound(adoption["common_learner"])
    bank = collector.load_verified_registry(plan["registry"]["path"], plan["registry"]["sha256"])
    current = {}
    for item in models:
        slot = slots[item["policy_id"]]
        if any(item.get(key) != value for key, value in slot.items()):
            raise ValueError("checkpoint metadata must equal the explicit primary plan slots")
        for field, expected_path in (
            ("completion", slot["expected_completion_path"]),
            ("checkpoint_receipt", slot["expected_checkpoint_receipt_path"]),
            ("model", slot["expected_model_path"]),
            ("training_result", slot["expected_training_result_path"]),
        ):
            if checked_path(item[field]).resolve() != Path(expected_path).resolve():
                raise ValueError("checkpoint artifact is outside its canonical primary slot")
        complete = read_bound(item["completion"])
        same(complete["plan"], inventory["primary_plan"])
        same(complete["adoption"], inventory["primary_adoption"])
        if complete["run_id"] != slot["run_id"] or complete["status"] != "complete":
            raise ValueError("each checkpoint requires its actual completed primary corpus")
        if slot["run_id"] not in current:
            order = validate_completed_replay_order(plan, complete)
            run = next(row for row in plan["runs"] if row["run_id"] == slot["run_id"])
            order["measured_steps_by_round"] = recording_counts(complete, run)
            current[slot["run_id"]] = order
        prefix = read_bound(item["checkpoint_receipt"])
        index = slot["completed_through_round"]
        if index == 2:
            if (
                prefix["status"] != "complete_prefix"
                or prefix["completed_through_round"] != 2
                or prefix["entries"] != complete["entries"][:3]
                or prefix["run_id"] != complete["run_id"]
            ):
                raise ValueError("M2 must retain the exact historical round2 completion prefix")
            same(prefix["plan"], complete["plan"])
            same(prefix["adoption"], complete["adoption"])
            same(complete["entries"][3]["student_model"], item["model"])
            same(complete["entries"][3]["model_training_result"], item["training_result"])
        else:
            same(complete["final_model"], item["model"])
            same(complete["final_model_training_result"], item["training_result"])
        trained = read_bound(item["training_result"])
        registration = read_bound(trained["registration"])
        registered_ties = registration.get("allow_measured_tie_initialization", False)
        adopted_ties = learner.get("allow_measured_tie_initialization", False)
        if (
            type(registered_ties) is not bool
            or type(adopted_ties) is not bool
            or registered_ties != adopted_ties
        ):
            raise ValueError("measured-tie initialization must match the adopted boolean setting")
        if trained["status"] != "complete":
            raise ValueError("an incomplete fit is not an evaluation checkpoint")
        same(trained["policy"], item["model"])
        expected_teachers = [
            entry["teacher_collection"] for entry in complete["entries"][: index + 1]
        ]
        if (
            registration["collections"] != expected_teachers
            or registration["l2"] != learner["l2"]
            or source_identities(registration["implementation"])
            != source_identities(learner["training_implementation"])
        ):
            raise ValueError(
                "checkpoint labels/penalty/training closure differ from the common learner"
            )
        same(registration["registry"], plan["registry"])
        same(policies[item["policy_id"]]["model"], item["model"])
        collector.load_schedule_policy(checked_path(item["model"]), item["model"]["sha256"], bank)
    baseline = inventory["baselines"]
    if len(baseline) != 3 or {row["policy"]["policy_id"] for row in baseline} != set(BASELINES):
        raise ValueError("exactly the three declared baseline policies are required")
    for row in baseline:
        if row["policy"] != policies[row["policy"]["policy_id"]]:
            raise ValueError("development-selected baseline differs from the evaluation protocol")
        checked_path(row["development_validation"])
    return dict(
        plan=plan,
        learner=learner,
        bank=bank,
        primary_adoption=adoption,
        corpus_runs=12,
        checkpoint_policies=24,
        paired_checkpoints=True,
        acquisition_counts=current,
    )


def runtime_context(protocol, execution, bank, policy, seed):
    runtime_sources = closure([ROOT / (collector.RUNTIME.replace(".", "/") + ".py")]) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    assets = read_bound(execution["runtime_assets_declaration"])
    return dict(
        physics_seed=seed,
        policy_id=policy["policy_id"],
        mode=policy["mode"],
        preferred_option_id=policy["preferred_option_id"],
        preferred_reference_id=policy["preferred_reference_id"],
        model=policy.get("model"),
        script_parameters=policy.get("script_parameters"),
        registry=protocol["implementation"]["registry"],
        request=protocol["implementation"]["request"],
        controller=bank.request["controller"],
        runtime_artifacts=[artifact(path) for path in sorted(runtime_sources)] + assets,
        scoring_artifacts=[artifact(path) for path in sorted(closure([Path(collector.__file__)]))],
        feature_names=list(collector.expected_feature_names(len(bank.option_ids))),
        reference_frames=bank.frame_count,
        recorded_control_steps=bank.frame_count - 1,
    )


def load_context(protocol_ref, output):
    protocol = read_bound(protocol_ref)
    if protocol.get("status") != ADOPTED or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("proposed/unadopted protocols cannot prepare or execute evaluation")
    if protocol.get("execution_contract") != CONTRACT:
        raise ValueError(
            "explicit adopted V4 checkpoint and972-episode execution contract required"
        )
    execution = protocol["execution_runner"]
    if (
        execution.get("schema") != SCHEMA
        or Path(execution["output_directory"]).resolve() != Path(output).resolve()
    ):
        raise ValueError("adopted evaluation output slot required")
    primary.current_sources(execution["implementation"], closure([Path(__file__)]))
    primary.current_sources(execution["batch_implementation"], closure([Path(batch.__file__)]))
    inventory = read_bound(execution["inventory"])
    verified = validate_inventory(inventory, protocol)
    if protocol["learning"]["l2"] != verified["learner"]["l2"]:
        raise ValueError("the common penalty must be finalized in the adopted protocol")
    bank = verified["bank"]
    template = read_bound(execution["template"])
    if template["implementation"]["checkpoint"] != bank.request["controller"] or not any(
        row["cell_id"] == "neutral" and row["motion"] == bank.request["references"][0]["motion"]
        for row in template["cells"]
    ):
        raise ValueError("frozen neutral template does not match the qualified source/controller")
    assets = read_bound(execution["runtime_assets_declaration"])
    if execution["environment"] not in assets:
        raise ValueError("explicit environment must be among the frozen native assets")
    table = assignments(protocol, output)
    first_scene = next(row for row in protocol["scenes"] if row["variant_id"] == "nominal")
    definition = dict(
        first_scene,
        schema=collector.SCENE_SCHEMA,
        split="reserved_evaluation_v3",
        evaluation_protocol=protocol_ref,
    )
    contexts = [
        runtime_context(protocol, execution, bank, policy, 94301) for policy in protocol["policies"]
    ]
    for actual in contexts:
        validate_reserved_execution(definition, actual)
    # Every scene's immutable USD is checked, including retained disabled stress
    # assignments; no robot clearance or physical outcome is queried here.
    for scene in protocol["scenes"]:
        checked_path(scene["scene"])
    expected_lock = Path(verified["plan"]["intended_execution_root"]) / ".primary-acquisition.lock"
    if Path(execution["shared_lock_path"]).resolve() != expected_lock.resolve():
        raise ValueError("evaluation must share the adopted primary acquisition serial lock")
    return dict(
        protocol=protocol,
        protocol_ref=protocol_ref,
        execution=execution,
        inventory=inventory,
        learner=verified["learner"],
        bank=bank,
        table=table,
        output=Path(output).resolve(),
        actual_context=contexts[0],
        acquisition_counts=verified["acquisition_counts"],
    )


def prepare(protocol_ref, output):
    context = load_context(protocol_ref, output)
    output = context["output"]
    if output.exists():
        raise ValueError("retain existing registration; preparation does not overwrite")
    protocol, execution = context["protocol"], context["execution"]
    output.mkdir(parents=True, exist_ok=False)
    definitions = {}
    for scene in protocol["scenes"]:
        if scene["variant_id"] == "nominal":
            definitions[scene["scene_id"]] = write_new(
                output / "definitions" / (scene["scene_id"] + ".json"),
                dict(
                    scene,
                    schema=collector.SCENE_SCHEMA,
                    split="reserved_evaluation_v3",
                    evaluation_protocol=protocol_ref,
                ),
            )
    assigned = write_new(output / "assignments.json", context["table"])
    policies = {row["policy_id"]: row for row in protocol["policies"]}
    collections = []
    for row in context["table"]:
        policy = policies[row["policy_id"]]
        collections.append(
            dict(
                scene_definition=definitions[row["scene_id"]],
                seed=row["physics_seed"],
                policy_mode=policy["mode"],
                policy_id=policy["policy_id"],
                policy=policy.get("model"),
                script_parameters=policy.get("script_parameters"),
                preferred_option_id=policy["preferred_option_id"],
                preferred_reference_id=policy["preferred_reference_id"],
                runtime_assets=execution["runtime_assets_declaration"],
            )
        )
    specification = write_new(
        output / "batch_specification.json",
        dict(
            schema=batch.SCHEMA,
            registry=protocol["implementation"]["registry"],
            request=protocol["implementation"]["request"],
            template=execution["template"],
            collections=collections,
            maximum_physics_steps=CONTRACT["maximum_physics_steps"],
            scope="adopted V4 nominal paired checkpoints; all972 assignments, no stress or oracle branches",
        ),
    )
    batch.prepare(checked_path(specification), output / "batch")
    registration = write_new(
        output / "registration.json",
        dict(
            schema=SCHEMA,
            protocol=protocol_ref,
            inventory=execution["inventory"],
            assignments=assigned,
            batch_registration=artifact(output / "batch" / "registration.json"),
            execution_contract=CONTRACT,
            source_implementation=execution["implementation"],
            new_physics_steps=0,
        ),
    )
    return dict(
        status="registered_not_executed",
        registration=registration,
        assigned_episodes=972,
        maximum_physics_steps=1158624,
        new_physics_steps=0,
    )


def summarize(table, assessments):
    """Assigned denominators stay fixed; unknown/missing never becomes failure."""
    rows, policies = [], {}
    measured = reserved = 0
    for assignment in table:
        item = assessments.get(assignment["index"])
        row = None if item is None else item.get("row")
        status = "not_run" if item is None else "technical_missing"
        if row is not None and row.get("task_outcome_admitted") is True:
            outcome = row.get("outcome", {}).get("task_outcome")
            if outcome in ("pass", "failure"):
                status = outcome
        steps = None if row is None else row.get("physics_steps")
        if type(steps) is int:
            if not 0 <= steps <= 1192:
                raise ValueError("invalid recorded step count")
            measured += steps
        if item is not None and status == "technical_missing":
            reserved += 1192 - (steps or 0)
        cost = row.get("costs", {}).get("passage_time_s") if status == "pass" else None
        failure_categories = []
        if status == "failure":
            failure_categories = sorted(
                {event["kind"] for event in row.get("outcome", {}).get("physical_events", [])}
            )
            if not failure_categories:
                failure_categories = ["verified_task_failure_unclassified"]
        rows.append(
            dict(
                **assignment,
                status=status,
                measurement_admitted=(None if row is None else row.get("measurement_admitted")),
                recorded_physics_steps=steps,
                successful_passage_time_s=cost,
                failure_categories=failure_categories,
                recorded_classification=(
                    None if row is None else row.get("outcome", {}).get("classification")
                ),
                evidence=None if item is None else item.get("evidence"),
            )
        )
        counts = policies.setdefault(
            assignment["policy_id"],
            dict(
                assigned=0,
                passed=0,
                verified_task_failed=0,
                technical_missing=0,
                not_run=0,
                failure_categories={},
            ),
        )
        counts["assigned"] += 1
        counts[{"pass": "passed", "failure": "verified_task_failed"}.get(status, status)] += 1
        for category in failure_categories:
            counts["failure_categories"][category] = (
                counts["failure_categories"].get(category, 0) + 1
            )
    for counts in policies.values():
        n, successes = counts["assigned"], counts["passed"]
        missing = counts["technical_missing"] + counts["not_run"]
        counts.update(
            completion_lower=successes / n,
            completion_upper=(successes + missing) / n,
            measured_only_denominator=n - missing,
            measured_only_pass_fraction=successes / (n - missing) if n != missing else None,
        )
    return dict(
        assigned_episodes=len(table),
        rows=rows,
        policies=policies,
        actual_recorded_physics_steps=measured,
        unknown_reserved_steps=reserved,
        conservative_charged_steps=measured + reserved,
        complete=all(row["status"] in ("pass", "failure") for row in rows),
    )


class EvaluationController(primary.Controller):
    """Use registered batch children with the existing per-cell guarded runner."""

    def __init__(self, evaluation, backend=None):
        execution = evaluation["execution"]
        runtime = dict(
            environment=execution["environment"],
            runtime_assets_declaration=execution["runtime_assets_declaration"],
            controller_implementation=execution["implementation"],
            collection_implementation=[
                artifact(path) for path in sorted(closure([Path(collector.__file__)]))
            ],
        )
        context = dict(
            run_root=evaluation["output"],
            execution_root=Path(execution["shared_lock_path"]).parent,
            run=dict(run_id="reserved_v4_nominal"),
            learner=evaluation["learner"],
            runtime=runtime,
            plan=dict(budget=dict(budget_physics_steps=1158624)),
        )
        super().__init__(context, backend)
        self.evaluation = evaluation

    def accounting(self):
        value = super().accounting()
        count = len(value["attempts"])
        value.update(
            unlaunched_assigned_episode_slots=972 - count,
            reserved_future_assigned_steps=(972 - count) * 1192,
        )
        return value

    def report(self):
        assessments = {}
        for path in (self.folder / "attempts").glob("*/intent.json"):
            intent = primary.read(path)
            index = int(intent["attempt_id"].split("_")[0].removeprefix("round"))
            assessment_path = path.parent / "assessment.json"
            assessments[index] = dict(
                row=primary.read(assessment_path)["row"] if assessment_path.exists() else None,
                evidence=artifact(assessment_path) if assessment_path.exists() else artifact(path),
            )
        protocol = self.evaluation.get("protocol", {})
        policies = {row["policy_id"]: row for row in protocol.get("policies", [])}
        scenes = {row["scene_id"]: row for row in protocol.get("scenes", [])}
        expanded = []
        for assignment in self.evaluation["table"]:
            policy = policies.get(assignment["policy_id"], {})
            checkpoint = policy.get("checkpoint")
            index = CHECKPOINTS[checkpoint][0] if checkpoint in CHECKPOINTS else None
            counts = (
                self.evaluation.get("acquisition_counts", {})
                .get(policy.get("run_id"), {})
                .get("measured_steps_by_round", {})
            )
            expanded.append(
                dict(
                    **assignment,
                    policy_mode=policy.get("mode"),
                    model=policy.get("model"),
                    training_arm=policy.get("training_arm"),
                    acquisition_seed=policy.get("acquisition_seed"),
                    checkpoint=checkpoint,
                    acquisition_actual_physics_steps=counts.get(index, counts.get(str(index))),
                    acquisition_assigned_maximum_steps=policy.get(
                        "acquisition_assigned_maximum_steps"
                    ),
                    layout_family=scenes.get(assignment.get("scene_id"), {}).get("family"),
                )
            )
        return summarize(expanded, assessments)

    def run(self, *, max_new_episodes=None):
        if max_new_episodes is not None and (
            type(max_new_episodes) is not int or max_new_episodes < 1
        ):
            raise ValueError("positive pause boundary required; assignments remain unchanged")
        evaluation = self.evaluation
        with primary.acquisition_lock(evaluation["execution"]["shared_lock_path"]):
            registration = primary.read(evaluation["output"] / "registration.json")
            if (
                registration["protocol"] != evaluation["protocol_ref"]
                or read_bound(registration["assignments"]) != evaluation["table"]
            ):
                raise ValueError("registered evaluation protocol/assignment table differs")
            batch_registration = read_bound(registration["batch_registration"])
            batch_plan = read_bound(batch_registration["plan"])
            specification = read_bound(batch_plan["specification"])
            if (
                batch_plan["expected_episodes"] != 972
                or batch_plan["planned_maximum_physics_steps"] != 1158624
                or batch_plan["budget_physics_steps"] != 1158624
                or len(specification["collections"]) != 972
                or source_identities(batch_plan["implementation"])
                != source_identities(evaluation["execution"]["batch_implementation"])
            ):
                raise ValueError("batch specification, complete budget or implementation differs")
            children = batch_registration["children"]
            if len(children) != 972 or [row["index"] for row in children] != list(range(972)):
                raise ValueError("complete explicitly assigned batch is required")
            for assignment, child in zip(evaluation["table"], children, strict=True):
                if (
                    checked_path(child["manifest"]).parent.resolve()
                    != Path(assignment["collection_directory"]).resolve()
                ):
                    raise ValueError("complete frozen child inventory differs before execution")
            before = len(list((self.folder / "attempts").glob("*/intent.json")))
            try:
                for assignment, child in zip(evaluation["table"], children, strict=True):
                    folder = checked_path(child["manifest"]).parent
                    if folder.resolve() != Path(assignment["collection_directory"]).resolve():
                        raise ValueError("batch child occupies another assignment slot")
                    manifest, bank, scene = collector.verify_manifest(folder)
                    if len(manifest["cells"]) != 1:
                        raise ValueError("one actual policy episode per assigned child required")
                    cell = manifest["cells"][0]
                    if (
                        Path(cell["output"]).resolve()
                        != (folder / "rollouts" / cell["cell_id"]).resolve()
                    ):
                        raise ValueError("actual rollout output escapes its assigned batch child")
                    if (
                        cell["policy_id"] != assignment["policy_id"]
                        or cell["runtime_seed"] != assignment["physics_seed"]
                        or scene["scene_id"] != assignment["scene_id"]
                    ):
                        raise ValueError("actual model/layout/seed assignment differs")
                    count = len(list((self.folder / "attempts").glob("*/intent.json")))
                    if (
                        max_new_episodes is not None
                        and count - before >= max_new_episodes
                        and not (Path(cell["output"]) / "attempt.json").exists()
                    ):
                        raise primary.Paused(
                            "explicit episode pause boundary reached; complete denominator retained"
                        )
                    result = self.step(
                        dict(round_index=assignment["index"], teacher_directory=str(folder)),
                        "teacher",
                        cell,
                        manifest,
                        bank,
                        scene,
                    )
                    if not (folder / "result.json").exists():
                        self.backend.analyze(folder)
                    published = primary.read(folder / "result.json")
                    if published["rows"] != [result] or published["manifest"] != child["manifest"]:
                        raise ValueError(
                            "published result differs from independently audited assigned episode"
                        )
                status, reason = "complete", None
            except primary.Paused as error:
                status, reason = "paused", str(error)
            report = self.report()
            receipt = dict(
                schema=SCHEMA,
                status=status,
                reason=reason,
                registration=artifact(evaluation["output"] / "registration.json"),
                ledger=self.ledger(),
                report=report,
            )
            self.event("evaluation_status", receipt=receipt)
            if status == "complete":
                if not report["complete"]:
                    raise ValueError("missing assignments cannot complete the evaluation")
                primary.commit(evaluation["output"] / "result.json", receipt)
            return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("propose-v4", "preview", "check", "prepare", "run"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--primary-plan", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-new-episodes", type=int)
    args = parser.parse_args()
    ref = artifact(args.protocol)
    if args.action in ("propose-v4", "preview") and args.primary_plan is None:
        parser.error("proposal/preview requires --primary-plan")
    if args.action == "propose-v4":
        result = dict(
            proposal=write_new(args.out, propose_v4(ref, artifact(args.primary_plan))),
            adopted=False,
            new_physics_steps=0,
        )
    elif args.action == "preview":
        value = read_bound(ref)
        args.out.mkdir(parents=True, exist_ok=False)
        result = dict(
            assignments=write_new(args.out / "assignments.json", assignments(value, args.out)),
            inventory_template=write_new(
                args.out / "inventory_template.json",
                inventory_template(artifact(args.primary_plan), ref),
            ),
            status="PROPOSED_NOT_EXECUTABLE",
            new_physics_steps=0,
        )
    elif args.action == "prepare":
        result = prepare(ref, args.out)
    else:
        context = load_context(ref, args.out)
        result = (
            dict(
                status="adopted_configuration_verified", assigned_episodes=972, new_physics_steps=0
            )
            if args.action == "check"
            else EvaluationController(context).run(max_new_episodes=args.max_new_episodes)
        )
    print(json.dumps(result, indent=2))
    if result.get("status") == "paused":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
