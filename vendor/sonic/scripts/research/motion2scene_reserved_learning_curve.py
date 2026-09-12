#!/usr/bin/env python3
"""Reserved nominal comparison for the five-arm M8/M16/M32 acquisition study.

Proposals cannot launch physics. Exact acquired model files and completed
development evaluation must precede an independently evidenced protocol adoption.
The existing reserved validator, collector and charged-attempt controller remain
unchanged. All seven fixed schedules are measured on the same conditions.
"""

import argparse
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
import motion2scene_collect_schedule_batch as batch  # noqa: E402
import motion2scene_development_learning_curve as development  # noqa: E402
import motion2scene_run_primary_acquisition as primary  # noqa: E402
import motion2scene_run_reserved_evaluation as original  # noqa: E402
from motion2scene_storage import deduplicate_inventories  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    read_bound,
    same_artifact,
    source_identities,
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_evaluation_protocol import (  # noqa: E402
    ADOPTED,
    ADOPTION_GATES,
    SCHEMA as PROTOCOL_SCHEMA,
    _validate_panel,
    protocol_spec_digest,
    validate_reserved_execution,
)

collector = development.collection
SCHEMA = "motion2scene_reserved_learning_curve_v1"
CHECKPOINTS = (8, 16, 32)
ORDER_SEED = 202609081821


def source_refs():
    return [
        artifact(p)
        for p in sorted(
            closure(
                [
                    Path(__file__),
                    Path(__file__).with_name("motion2scene_reserved_curve_statistics.py"),
                ]
            )
        )
    ]


def design_models(plan):
    """Every declared corpus at every fixed checkpoint, regardless of its outcomes."""
    result = []
    for budget in CHECKPOINTS:
        for run in plan["runs"]:
            path = development.checkpoint_slots(plan, run["run_id"], budget)[-1]["model"]
            result.append(
                dict(
                    policy_id=f'{run["run_id"]}__M{budget}',
                    run_id=run["run_id"],
                    arm=run["arm"],
                    seed=run["seed"],
                    checkpoint=budget,
                    expected_training_result=str(path),
                    acquisition_assigned_maximum_steps=(7 + 8 * budget) * 1192,
                )
            )
    return result


def baseline_policies(bank, script):
    return [
        dict(
            policy_id="script",
            mode="scripted_multi",
            model=None,
            script_parameters=script,
            preferred_option_id="neutral",
            preferred_reference_id="sustained",
        ),
        *[
            dict(
                policy_id="fixed_" + option,
                mode="forced",
                model=None,
                forced_option_id=option,
                preferred_option_id="neutral",
                preferred_reference_id="sustained",
            )
            for option in bank.option_ids
        ],
    ]


def contract(protocol):
    _validate_panel(protocol)
    if protocol["enabled_variant_ids"] != ["nominal"]:
        raise ValueError("only the unchanged nominal allocation is enabled")
    scenes = [s for s in protocol["scenes"] if s["variant_id"] == "nominal"]
    ids = [p["policy_id"] for p in protocol["policies"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate comparison policy")
    count = len(ids) * len(scenes) * len(protocol["physics_seeds"])
    return dict(
        schema=SCHEMA,
        checkpoints=list(CHECKPOINTS),
        final_checkpoint=32,
        learned_policies=sum(p["mode"] == "learned" for p in protocol["policies"]),
        shared_baselines=sum(p["mode"] != "learned" for p in protocol["policies"]),
        layouts=len(scenes),
        physics_seeds=protocol["physics_seeds"],
        enabled_variant_ids=["nominal"],
        episodes_per_policy=len(scenes) * len(protocol["physics_seeds"]),
        assigned_episodes=count,
        maximum_physics_steps_per_episode=1192,
        maximum_physics_steps=count * 1192,
        automatic_retries=False,
        unknown_attempt_policy="retain_in_denominator_and_pause_without_retry",
        execution_order_seed=ORDER_SEED,
        all_assignments_required_before_launch=True,
        development_gate="all M32 acquisition and all 318 development evaluation episodes complete",
    )


def assignments(protocol, output):
    design = contract(protocol)
    if protocol["execution_contract"] != design or protocol["execution_order_seed"] != ORDER_SEED:
        raise ValueError("declared comparison size or ordering differs")
    scenes = sorted(
        (s for s in protocol["scenes"] if s["variant_id"] == "nominal"),
        key=lambda s: s["layout_id"],
    )
    rows = [
        dict(
            policy_id=policy_id,
            layout_id=scene["layout_id"],
            variant_id="nominal",
            scene_id=scene["scene_id"],
            physics_seed=seed,
        )
        for policy_id in sorted(p["policy_id"] for p in protocol["policies"])
        for scene in scenes
        for seed in protocol["physics_seeds"]
    ]
    random.Random(ORDER_SEED).shuffle(rows)
    for index, row in enumerate(rows):
        row.update(
            index=index,
            assignment_id=f"nominal_{index:04d}",
            collection_directory=str(Path(output).resolve() / "batch" / f"collection_{index:03d}"),
        )
    return rows


def declare(previous_path, development_root, output):
    previous_ref = artifact(previous_path)
    previous = read_bound(previous_ref)
    if previous["schema"] != PROTOCOL_SCHEMA or previous["status"] == ADOPTED:
        raise ValueError("the preserved unadopted predecessor proposal is required")
    _validate_panel(previous)
    curve_ref, curve, study, plan, bank = development.read_study(development_root)
    for key in ("registry", "request"):
        if not same_artifact(previous["implementation"][key], study[key]):
            raise ValueError("the qualified bank and sensor interface must stay common")
    if previous["schedules"]["option_ids"] != list(bank.option_ids):
        raise ValueError("the complete qualified schedule repertoire must be preserved")
    result = copy.deepcopy(previous)
    result.update(
        created_utc=primary.utc(),
        proposal_revision=7,
        execution_scope_revision=7,
        status="PROPOSED_EXPANDED_CURVE_NOT_ADOPTED_NO_EXECUTION_AUTHORITY",
        previous_scope_proposal=previous_ref,
        development_curve=curve_ref,
        expanded_acquisition_plan=curve["expanded_plan"],
        adoption_receipt=None,
        execution_runner=None,
        curve_implementation=source_refs(),
        selection="all five arms, three corpora and M8/M16/M32; no outcome-based model selection",
        scope_change="expanded acquisition learning curves and all seven matched fixed schedules",
        scope="nominal reserved layout generalization on the existing development carrier",
        policies=baseline_policies(bank, study["script"])
        + [
            dict(
                **m,
                mode="learned",
                model=None,
                preferred_option_id="neutral",
                preferred_reference_id="sustained",
            )
            for m in design_models(plan)
        ],
    )
    result["adoption_requirements"] = {k: False for k in previous["adoption_requirements"]}
    result["adoption_requirements"]["no_reserved_outcomes_inspected"] = True
    result["execution_contract"] = contract(result)
    result["checkpoint_protocol"] = dict(
        checkpoints=list(CHECKPOINTS),
        final_checkpoint=32,
        corpus_budget_cap_steps=320000,
        independent_corpus_seeds=[93201, 93202, 93203],
        checkpoint_dependence="three nested prefixes of each corpus, not independent fits",
        horizontal_axis="actual recorded cumulative acquisition steps; assigned maxima separately",
    )
    result["planned_resource_size"] = dict(
        primary_nominal_episodes=result["execution_contract"]["assigned_episodes"],
        primary_max_evaluation_physics_steps=result["execution_contract"]["maximum_physics_steps"],
        stress_status="all eight original offsets retained; none enabled by this comparison",
        optional_stress_episodes_per_policy=previous["stress_episodes_per_policy"],
    )
    result["acquisition"] = dict(
        plan=curve["expanded_plan"],
        checkpoint_indices=list(CHECKPOINTS),
        actual_cost_source="each exact teacher/bootstrap/pre-update student receipt",
        proposal_computation="separate original candidate-pool receipts; not physical cost",
    )
    result["learning"].update(
        l2=10.0,
        arms=sorted({r["arm"] for r in plan["runs"]}),
        teacher="original scene-wise complete-continuation teacher",
        replay="historical observation-gap weighting only in observation_curriculum",
        common_learner_status="the existing common phase ridge, unchanged measured-tie initialization",
        isolated_ablations="separate development studies; excluded from this fixed primary comparison",
    )
    result["baseline_protocol"] = dict(
        strong_script=study["script"],
        fixed_schedules=list(bank.option_ids),
        selection="all seven fixed schedules, with no held-out selection of a deployable baseline",
        capability="union of the seven matched measured complete schedules, separately from policy passage",
        no_privileged_script_inputs=True,
        shared_references="one capture per baseline/layout/seed, reused across checkpoint analyses",
    )
    result["experiment_scope_amendment"] = dict(
        previous=previous_ref,
        changed="model inventory, checkpoint budgets and full fixed-schedule capability comparison",
        retained="all world geometry, execution seeds, offsets, tracker, sensor and physical scorer",
    )
    result["reporting"] = dict(
        primary="M32 nominal passage with all assigned tasks in denominator",
        curves="M8/M16/M32 actual closed-loop passage versus recorded acquisition steps",
        comparisons=[
            ["analytic_contrast", "uniform"],
            ["analytic_contrast", "target_only"],
            ["analytic_contrast", "reference_contrast"],
            ["observation_curriculum", "analytic_contrast"],
        ],
        uncertainty="paired corpus-by-layout descriptive bootstrap; both execution seeds stay together",
        bootstrap_draws=20000,
        bootstrap_seed=202609081822,
        shared_baselines="same 36 physical recordings, never independently replicated per corpus or checkpoint",
        incomplete="unknown bounds, no completion interval while any paired outcome is unknown",
        time="mutually successful matched conditions only, with counts; crossing plus stabilization",
        limitations="three corpus seeds, one carrier; no source-transfer or continuous-perturbation claim",
    )
    output.mkdir(parents=True, exist_ok=False)
    proposal_ref = write_new(output / "proposal.json", result)
    table_ref = write_new(
        output / "assignments_preview.json", assignments(result, output / "execution")
    )
    return dict(proposal=proposal_ref, assignments=table_ref, new_physics_steps=0)


def read_proposal(path):
    ref = artifact(path)
    protocol = read_bound(ref)
    if protocol.get("execution_scope_revision") != 7:
        raise ValueError("the expanded evaluation scope is required")
    if source_identities(protocol["curve_implementation"]) != source_identities(source_refs()):
        raise ValueError("declared evaluation source closure changed")
    curve_path = Path(protocol["development_curve"]["path"])
    checked(curve_path, protocol["development_curve"]["sha256"])
    _, curve, study, plan, bank = development.read_study(curve_path.parent)
    if protocol["expanded_acquisition_plan"] != curve["expanded_plan"]:
        raise ValueError("a different acquisition trajectory cannot fill evaluation slots")
    expected = baseline_policies(bank, study["script"]) + [
        dict(
            **m,
            mode="learned",
            model=None,
            preferred_option_id="neutral",
            preferred_reference_id="sustained",
        )
        for m in design_models(plan)
    ]
    normalized = copy.deepcopy(protocol["policies"])
    for p in normalized:
        if p["mode"] == "learned":
            p["model"] = None
    if normalized != expected or protocol["execution_contract"] != contract(protocol):
        raise ValueError("every original model and fixed baseline must remain assigned")
    previous = read_bound(protocol["previous_scope_proposal"])
    for key in ("geometry_lock", "scenes", "physics_seeds", "schedules", "sensor", "scoring"):
        if protocol[key] != previous[key]:
            raise ValueError(
                "the reserved geometry, action, sensor and scoring contracts must be retained"
            )
    return ref, protocol, curve, study, plan, bank


def frozen_models(protocol, curve, study, plan, bank):
    if development.acquisition_ready(curve, plan):
        raise FileNotFoundError("all fifteen M32 acquisition boundaries must finish first")
    development.completed_anchor(curve, study)
    bindings = {}
    for budget in CHECKPOINTS:
        assigned = [m for m in design_models(plan) if m["checkpoint"] == budget]
        if budget == 8:
            models = development.anchor.bind_models(study)
        else:
            models = development.bind_models(
                plan, bank, development.model_slots(plan, budget), budget
            )
        for model in assigned:
            bound = models[model["run_id"]]
            result = read_bound(bound["training_result"])
            registration = read_bound(result["registration"])
            expected_weight = (
                "historical_observation_gap"
                if model["arm"] == "observation_curriculum"
                else "uniform_per_phase"
            )
            if (
                registration["schema"] != "motion2scene_expanded_fit_v1"
                or registration["l2"] != 10.0
                or registration["allow_measured_tie_initialization"] is not True
                or registration["weighting"] != expected_weight
                or registration["plan"] != protocol["expanded_acquisition_plan"]
            ):
                raise ValueError(
                    "the common original learner, teacher prefix and arm weighting are required"
                )
            bindings[model["policy_id"]] = dict(
                **model,
                **bound,
                acquisition_cost=development.acquisition_cost(plan, model, budget),
            )
    return bindings


def freeze_inventory(proposal_path, development_statistics, output):
    ref, protocol, curve, study, plan, bank = read_proposal(proposal_path)
    models = frozen_models(protocol, curve, study, plan, bank)
    stats_ref = artifact(development_statistics)
    stats = read_bound(stats_ref)
    if (
        stats["study"] != protocol["development_curve"]
        or stats["implementation"] != artifact(Path(development.__file__))
        or [p["checkpoint"] for p in stats["panels"]] != list(CHECKPOINTS)
        or stats["evaluation_accounting"]["unique_assigned_episodes"] != 318
        or stats["evaluation_accounting"]["known_outcomes"] != 318
        or any(not p["summary"]["complete"] for p in stats["panels"])
    ):
        raise ValueError(
            "complete fixed development evaluation must precede reserved model freezing"
        )
    # Recompute the summary from the exact prepared captures. A claimed summary
    # alone cannot freeze a policy or conceal missing development measurements.
    check_dir = output / "development_recheck"
    output.mkdir(parents=True, exist_ok=False)
    actual = development.analyze(Path(protocol["development_curve"]["path"]).parent, check_dir)
    if read_bound(actual) != stats:
        raise ValueError("development statistics differ from their actual assigned captures")
    policies = copy.deepcopy(protocol["policies"])
    for p in policies:
        if p["mode"] == "learned":
            p["model"] = models[p["policy_id"]]["policy"]
    return write_new(
        output / "inventory.json",
        dict(
            schema=SCHEMA,
            status="FROZEN_COMPLETE_MODELS_NOT_EVALUATION_ADOPTION",
            proposal=ref,
            development_statistics=stats_ref,
            development_recheck=actual,
            models=models,
            policies=policies,
            new_physics_steps=0,
        ),
    )


def adopt(proposal_path, inventory_path, runtime_path, evidence_path, output):
    """Bind a reviewed, complete gate-evidence bundle after actual model freezing.

    Evidence reviews remain explicit inputs. This command cannot infer that a
    missing gate passed, or bind pending checkpoint paths as actual policies.
    """
    ref, protocol, curve, study, plan, bank = read_proposal(proposal_path)
    if protocol["status"] == ADOPTED:
        raise ValueError("retain the existing adoption; a new proposal is required for a change")
    models = frozen_models(protocol, curve, study, plan, bank)
    inventory_ref = artifact(inventory_path)
    inventory = read_bound(inventory_ref)
    if inventory["proposal"] != ref or inventory["models"] != models:
        raise ValueError("the exact complete pre-evaluation inventory is required")
    evidence_ref = artifact(evidence_path)
    evidence = read_bound(evidence_ref)
    if (
        evidence.get("evaluation_outcomes_inspected") is not False
        or evidence.get("reviewed") is not True
        or set(evidence.get("gate_evidence", {})) != set(ADOPTION_GATES)
    ):
        raise ValueError(
            "explicit pre-evaluation review and evidence for all nine gates are required"
        )
    for refs in evidence["gate_evidence"].values():
        if not refs:
            raise ValueError("missing gate evidence cannot be inferred from another check")
        for item in refs:
            checked(Path(item["path"]), item["sha256"])
    runtime_ref = artifact(runtime_path)
    runtime = read_bound(runtime_ref)
    predecessor = read_bound(plan["predecessor"])
    execution = dict(
        schema=SCHEMA,
        output_directory=str((proposal_path.parent / "execution").resolve()),
        inventory=inventory_ref,
        runtime_freeze=runtime_ref,
        implementation=source_refs(),
        batch_implementation=[artifact(p) for p in sorted(closure([Path(batch.__file__)]))],
        shared_lock_path=str(
            Path(predecessor["intended_execution_root"]) / ".primary-acquisition.lock"
        ),
        **{k: runtime[k] for k in ("environment", "runtime_assets_declaration", "template")},
    )
    protocol["policies"] = inventory["policies"]
    context = original.runtime_context(protocol, execution, bank, protocol["policies"][0], 94301)
    protocol["implementation"] = dict(
        registry=study["registry"],
        request=study["request"],
        controller=bank.request["controller"],
        **{k: context[k] for k in ("runtime_artifacts", "scoring_artifacts")},
    )
    protocol.update(
        status=ADOPTED,
        execution_runner=execution,
        adoption_requirements={k: True for k in ADOPTION_GATES},
    )
    output.mkdir(parents=True, exist_ok=False)
    protocol["adoption_receipt"] = write_new(
        output / "adoption_receipt.json",
        dict(
            status="ADOPTED",
            created_utc=primary.utc(),
            evaluation_outcomes_inspected=False,
            protocol_spec_sha256=protocol_spec_digest(protocol),
            gate_evidence=evidence["gate_evidence"],
            evidence_review=evidence_ref,
            authority=(
                "user-requested protected evaluation after complete acquisition "
                "and frozen development decisions"
            ),
        ),
    )
    adopted = write_new(output / "protocol.json", protocol)
    # The ordinary launch preflight must accept the complete produced artifact.
    # Any failure leaves all files intact and cannot register or launch physics.
    load_context(Path(adopted["path"]), Path(execution["output_directory"]))
    return write_new(
        output / "adoption_check.json", dict(protocol=adopted, verified=True, new_physics_steps=0)
    )


def load_context(protocol_path, output):
    ref, protocol, curve, study, plan, bank = read_proposal(protocol_path)
    if protocol["status"] != ADOPTED:
        raise ValueError("a proposal cannot prepare or execute reserved physics")
    execution = protocol["execution_runner"]
    if (
        execution["schema"] != SCHEMA
        or Path(execution["output_directory"]).resolve() != output.resolve()
        or source_identities(execution["implementation"]) != source_identities(source_refs())
    ):
        raise ValueError("exact adopted output and evaluation implementation are required")
    inventory = read_bound(execution["inventory"])
    if (
        inventory["schema"] != SCHEMA
        or inventory["status"] != "FROZEN_COMPLETE_MODELS_NOT_EVALUATION_ADOPTION"
        or inventory["policies"] != protocol["policies"]
        or inventory["models"] != frozen_models(protocol, curve, study, plan, bank)
    ):
        raise ValueError("all exact frozen models and completed development evidence required")
    for key in ("development_statistics", "development_recheck"):
        read_bound(inventory[key])
    if read_bound(inventory["development_statistics"]) != read_bound(
        inventory["development_recheck"]
    ):
        raise ValueError("the independently regenerated development summary changed")
    original_proposal = read_bound(inventory["proposal"])
    if (
        original_proposal["development_curve"] != protocol["development_curve"]
        or original_proposal["execution_contract"] != protocol["execution_contract"]
        or assignments(original_proposal, output) != assignments(protocol, output)
    ):
        raise ValueError("adoption changed the predeclared assignment or selection rule")
    mutable_at_adoption = {
        "status",
        "adoption_receipt",
        "adoption_requirements",
        "execution_runner",
        "implementation",
        "policies",
    }
    if {k: v for k, v in protocol.items() if k not in mutable_at_adoption} != {
        k: v for k, v in original_proposal.items() if k not in mutable_at_adoption
    }:
        raise ValueError("adoption cannot change the declared analysis or method contract")
    runtime = read_bound(execution["runtime_freeze"])
    for key in ("environment", "runtime_assets_declaration", "template"):
        if execution[key] != runtime[key]:
            raise ValueError("the acquired native environment and assets must stay common")
    if execution["environment"] not in read_bound(execution["runtime_assets_declaration"]):
        raise ValueError("the complete native environment must be frozen among runtime assets")
    acquired = development.checkpoint_slots(plan, plan["runs"][0]["run_id"], 32)[-1]["teacher"]
    acquisition_manifest = read_bound(read_bound(artifact(acquired))["manifest"])
    if (
        execution["runtime_assets_declaration"]
        != acquisition_manifest["runtime_assets_declaration"]
    ):
        raise ValueError("evaluation must retain the native assets used for acquisition")
    for key in ("runtime_artifacts", "scoring_artifacts"):
        if source_identities(protocol["implementation"][key]) != source_identities(
            acquisition_manifest[key]
        ):
            raise ValueError("the physical runtime and scoring must stay common with acquisition")
    if source_identities(execution["batch_implementation"]) != source_identities(
        [artifact(p) for p in sorted(closure([Path(batch.__file__)]))]
    ):
        raise ValueError("the unchanged batch preparation implementation is required")
    predecessor = read_bound(plan["predecessor"])
    lock = Path(predecessor["intended_execution_root"]) / ".primary-acquisition.lock"
    if Path(execution["shared_lock_path"]).resolve() != lock.resolve():
        raise ValueError("the original serial acquisition lock must remain shared")
    first = next(s for s in protocol["scenes"] if s["variant_id"] == "nominal")
    definition = dict(
        first,
        schema=collector.SCENE_SCHEMA,
        split="reserved_evaluation_v3",
        evaluation_protocol=ref,
    )
    # Existing independent protocol validator checks all nine adopted gates,
    # original geometry, effective runtime identity and exact policy artifacts.
    for policy in protocol["policies"]:
        context = original.runtime_context(protocol, execution, bank, policy, 94301)
        validate_reserved_execution(definition, context)
    for scene in protocol["scenes"]:
        checked(Path(scene["scene"]["path"]), scene["scene"]["sha256"])
    learner = dict(
        l2=10.0,
        training_implementation=[
            artifact(p) for p in sorted(closure([Path(development.__file__)]))
        ],
    )
    return dict(
        protocol=protocol,
        protocol_ref=ref,
        execution=execution,
        inventory=inventory,
        learner=learner,
        bank=bank,
        table=assignments(protocol, output),
        output=output.resolve(),
    )


def check_assignment(assignment, manifest, protocol):
    if len(manifest["cells"]) != 1 or manifest["split"] != "reserved_evaluation_v3":
        raise ValueError("one reserved physical episode per exact assignment is required")
    cell = manifest["cells"][0]
    policy = next(p for p in protocol["policies"] if p["policy_id"] == assignment["policy_id"])
    scene = read_bound(manifest["scene_definition"])
    if (
        cell["policy_id"] != assignment["policy_id"]
        or cell["runtime_seed"] != assignment["physics_seed"]
        or cell["timed_schedule_mode"] != policy["mode"]
        or scene["scene_id"] != assignment["scene_id"]
        or scene["layout_id"] != assignment["layout_id"]
        or scene["variant_id"] != assignment["variant_id"]
        or manifest["policy"] != policy["model"]
        or manifest["script_parameters"] != policy.get("script_parameters")
        or manifest["expected_physics_steps"] != 1192
        or (policy["mode"] == "forced" and cell["forced_option_id"] != policy["forced_option_id"])
        or Path(cell["output"]).resolve()
        != (Path(assignment["collection_directory"]) / "rollouts" / cell["cell_id"]).resolve()
    ):
        raise ValueError(
            "actual model, fixed schedule, layout, seed or output differs from assignment"
        )
    return cell


def prepare(protocol_path, output):
    context = load_context(protocol_path, output)
    protocol, execution = context["protocol"], context["execution"]
    output.mkdir(parents=True, exist_ok=False)
    definitions = {
        scene["scene_id"]: write_new(
            output / "definitions" / (scene["scene_id"] + ".json"),
            dict(
                scene,
                schema=collector.SCENE_SCHEMA,
                split="reserved_evaluation_v3",
                evaluation_protocol=context["protocol_ref"],
            ),
        )
        for scene in protocol["scenes"]
        if scene["variant_id"] == "nominal"
    }
    policies = {p["policy_id"]: p for p in protocol["policies"]}
    collections = []
    for row in context["table"]:
        p = policies[row["policy_id"]]
        collections.append(
            dict(
                scene_definition=definitions[row["scene_id"]],
                seed=row["physics_seed"],
                policy_mode=p["mode"],
                policy_id=p["policy_id"],
                policy=p["model"],
                script_parameters=p.get("script_parameters"),
                forced_option_ids=[p["forced_option_id"]] if p["mode"] == "forced" else None,
                preferred_option_id=p["preferred_option_id"],
                preferred_reference_id=p["preferred_reference_id"],
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
            maximum_physics_steps=protocol["execution_contract"]["maximum_physics_steps"],
            scope="expanded nominal learning curve; complete fixed denominator; all seven matched schedules",
        ),
    )
    batch.prepare(Path(specification["path"]), output / "batch")
    registered = read_bound(artifact(output / "batch/registration.json"))
    for row, child in zip(context["table"], registered["children"], strict=True):
        check_assignment(row, read_bound(child["manifest"]), protocol)
    return write_new(
        output / "registration.json",
        dict(
            schema=SCHEMA,
            protocol=context["protocol_ref"],
            inventory=execution["inventory"],
            assignments=write_new(output / "assignments.json", context["table"]),
            batch_registration=artifact(output / "batch/registration.json"),
            execution_contract=protocol["execution_contract"],
            new_physics_steps=0,
        ),
    )


class EvaluationController(primary.Controller):
    """Reuse the charged-attempt controller with the actual complete allocation."""

    def __init__(self, evaluation, backend=None):
        execution = evaluation["execution"]
        super().__init__(
            dict(
                run_root=evaluation["output"],
                execution_root=Path(execution["shared_lock_path"]).parent,
                run=dict(run_id="reserved_expanded_curve"),
                learner=evaluation["learner"],
                runtime=dict(
                    environment=execution["environment"],
                    runtime_assets_declaration=execution["runtime_assets_declaration"],
                    controller_implementation=execution["implementation"],
                    collection_implementation=[
                        artifact(p) for p in sorted(closure([Path(collector.__file__)]))
                    ],
                ),
                plan=dict(budget=dict(budget_physics_steps=len(evaluation["table"]) * 1192)),
            ),
            backend,
        )
        self.evaluation = evaluation

    def accounting(self):
        value = super().accounting()
        remaining = len(self.evaluation["table"]) - len(value["attempts"])
        if remaining < 0:
            raise ValueError("physical attempts exceed the fixed allocation")
        return dict(
            value,
            unlaunched_assigned_episode_slots=remaining,
            reserved_future_assigned_steps=remaining * 1192,
        )

    def report(self):
        assessments = {}
        for path in (self.folder / "attempts").glob("*/intent.json"):
            intent = primary.read(path)
            index = int(intent["attempt_id"].split("_")[0].removeprefix("round"))
            assessment = path.parent / "assessment.json"
            assessments[index] = dict(
                row=primary.read(assessment)["row"] if assessment.exists() else None,
                evidence=artifact(assessment) if assessment.exists() else artifact(path),
            )
        policies = {p["policy_id"]: p for p in self.evaluation["protocol"]["policies"]}
        models = self.evaluation["inventory"]["models"]
        scenes = {s["scene_id"]: s for s in self.evaluation["protocol"]["scenes"]}
        table = []
        for a in self.evaluation["table"]:
            p, m = policies[a["policy_id"]], models.get(a["policy_id"])
            table.append(
                dict(
                    **a,
                    policy_mode=p["mode"],
                    model=p["model"],
                    training_arm=p.get("arm"),
                    acquisition_seed=p.get("seed"),
                    checkpoint=p.get("checkpoint"),
                    acquisition_actual_physics_steps=(
                        None if m is None else m["acquisition_cost"]["total_recorded_steps"]
                    ),
                    acquisition_assigned_maximum_steps=p.get("acquisition_assigned_maximum_steps"),
                    layout_family=scenes[a["scene_id"]]["family"],
                )
            )
        if any(i not in range(len(table)) for i in assessments):
            raise ValueError("retained attempt lies outside the declared assignment matrix")
        return original.summarize(table, assessments)

    def run(self, max_new_episodes=None):
        if max_new_episodes is not None and (
            type(max_new_episodes) is not int or max_new_episodes < 1
        ):
            raise ValueError("positive episode pause boundary required")
        e = self.evaluation
        with ExitStack() as stack:
            stack.enter_context(primary.acquisition_lock(e["execution"]["shared_lock_path"]))
            stack.enter_context(
                primary.acquisition_lock(
                    Path(e["execution"]["shared_lock_path"]).with_name(".expanded_acquisition.lock")
                )
            )
            registration = primary.read(e["output"] / "registration.json")
            if (
                registration["protocol"] != e["protocol_ref"]
                or read_bound(registration["assignments"]) != e["table"]
                or registration["execution_contract"] != e["protocol"]["execution_contract"]
            ):
                raise ValueError("adopted protocol and complete registered assignment differ")
            registered = read_bound(registration["batch_registration"])
            batch_plan = read_bound(registered["plan"])
            spec = read_bound(batch_plan["specification"])
            n = len(e["table"])
            if (
                batch_plan["expected_episodes"] != n
                or batch_plan["planned_maximum_physics_steps"] != n * 1192
                or batch_plan["budget_physics_steps"] != n * 1192
                or len(spec["collections"]) != n
                or source_identities(batch_plan["implementation"])
                != source_identities(e["execution"]["batch_implementation"])
                or [c["index"] for c in registered["children"]] != list(range(n))
            ):
                raise ValueError("complete registered batch and charged allocation required")
            for row, child in zip(e["table"], registered["children"], strict=True):
                if (
                    Path(child["manifest"]["path"]).parent.resolve()
                    != Path(row["collection_directory"]).resolve()
                ):
                    raise ValueError("collection is outside its assigned output slot")
                check_assignment(row, read_bound(child["manifest"]), e["protocol"])
            before = len(list((self.folder / "attempts").glob("*/intent.json")))
            try:
                for row, child in zip(e["table"], registered["children"], strict=True):
                    folder = Path(child["manifest"]["path"]).parent
                    common, bank, scene = collector.verify_manifest(folder)
                    cell = check_assignment(row, common, e["protocol"])
                    count = len(list((self.folder / "attempts").glob("*/intent.json")))
                    if (
                        max_new_episodes is not None
                        and count - before >= max_new_episodes
                        and not (Path(cell["output"]) / "attempt.json").exists()
                    ):
                        raise primary.Paused(
                            "explicit pause boundary reached; full denominator retained"
                        )
                    if (
                        not (Path(cell["output"]) / "attempt.json").exists()
                        and shutil.disk_usage(e["output"]).free < 30 * 1024**3
                    ):
                        raise primary.Paused("resource pause before launch; fewer than 30 GiB free")
                    result = self.step(
                        dict(round_index=row["index"], teacher_directory=str(folder)),
                        "teacher",
                        cell,
                        common,
                        bank,
                        scene,
                    )
                    if not (folder / "result.json").exists():
                        self.backend.analyze(folder)
                    saved = primary.read(folder / "result.json")
                    if saved["rows"] != [result] or saved["manifest"] != child["manifest"]:
                        raise ValueError(
                            "published result differs from the re-audited physical assignment"
                        )
                    if not (folder / "storage.json").exists():
                        primary.commit(folder / "storage.json", deduplicate_inventories([folder]))
                status, reason = "complete", None
            except primary.Paused as error:
                status, reason = "paused", str(error)
            report = self.report()
            receipt = dict(
                schema=SCHEMA,
                status=status,
                reason=reason,
                registration=artifact(e["output"] / "registration.json"),
                ledger=self.ledger(),
                report=report,
            )
            self.event("evaluation_status", receipt=receipt)
            if status == "complete":
                if not report["complete"]:
                    raise ValueError("unknown or missing outcomes cannot complete the evaluation")
                primary.commit(e["output"] / "result.json", receipt)
            return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("declare", "freeze-inventory", "adopt", "check", "prepare", "run")
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--development-statistics", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--runtime-freeze", type=Path)
    parser.add_argument("--gate-evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-episodes", type=int)
    args = parser.parse_args()
    if args.action == "declare":
        result = declare(args.protocol, args.development, args.out)
    elif args.action == "freeze-inventory":
        result = freeze_inventory(args.protocol, args.development_statistics, args.out)
    elif args.action == "adopt":
        result = adopt(
            args.protocol, args.inventory, args.runtime_freeze, args.gate_evidence, args.out
        )
    elif args.action == "check":
        _, protocol, curve, _, plan, _ = read_proposal(args.protocol)
        result = dict(
            status=protocol["status"],
            contract=protocol["execution_contract"],
            missing_M32_corpora=development.acquisition_ready(curve, plan),
            new_physics_steps=0,
        )
    elif args.action == "prepare":
        result = prepare(args.protocol, args.out)
    else:
        result = EvaluationController(load_context(args.protocol, args.out)).run(
            args.max_new_episodes
        )
    print(json.dumps(result))
