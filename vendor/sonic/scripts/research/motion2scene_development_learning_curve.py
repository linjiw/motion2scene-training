#!/usr/bin/env python3
"""Native M16/M32 evaluation on the unchanged M8 development conditions.

The registered M8 panel is preserved. Its 48 script/forced-schedule measurements
are shared references, counted once. Ninety new policy episodes are assigned at
each later checkpoint. Evaluation waits for completed M32 acquisition, so it
cannot affect scene queues, early stopping, or acquisition checkpoint selection.
"""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_checkpoint_controls import checkpoint_slots  # noqa: E402
import motion2scene_collect_timed_schedules as collection  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
import motion2scene_development_checkpoint_panel as anchor  # noqa: E402
from motion2scene_development_panel_statistics import (  # noqa: E402
    construction_comparisons,
    measure,
    paired_time,
    summarize,
)
from motion2scene_expanded_acquisition import validate_training_prefix  # noqa: E402
from motion2scene_run_primary_acquisition import acquisition_lock, commit  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    load_schedule_policy,
)

CHECKPOINTS = (16, 32)
SOURCE_NAMES = (
    "motion2scene_development_learning_curve.py",
    "motion2scene_development_checkpoint_panel.py",
    "motion2scene_development_panel_statistics.py",
    "motion2scene_evaluation_statistics.py",
    "motion2scene_checkpoint_controls.py",
    "motion2scene_expanded_acquisition.py",
    "motion2scene_collect_timed_schedules.py",
    "motion2scene_run_extension_tasks.py",
    "motion2scene_run_primary_acquisition.py",
    "motion2scene_storage.py",
)


def canonical_anchor(study):
    plan = read_checked(study["expanded_plan"])
    bank = load_verified_registry(study["registry"]["path"], study["registry"]["sha256"])
    scenes = {
        row["scene_id"]: {k: row[k] for k in ("scene_id", "scene_definition")}
        for row in study["assignments"]
    }
    expected = anchor.assignments(plan["runs"], list(scenes.values()), bank.option_ids)
    for scene in scenes.values():
        definition = read_checked(scene["scene_definition"])
        if definition["split"] != "development" or definition["scene_id"] != scene["scene_id"]:
            raise ValueError("only the original development scene definitions are permitted")
    if (
        study["checkpoint"] != 8
        or study["assignments"] != expected
        or study["assigned_episodes"] != len(expected)
        or study["registry"] != plan["registry"]
    ):
        raise ValueError("the full original M8 panel and its acquisition plan are required")
    checked(Path(study["implementation"]["path"]), study["implementation"]["sha256"])
    return plan, bank


def model_slots(plan, budget):
    if budget not in CHECKPOINTS:
        raise ValueError("only the prespecified M16/M32 continuation is supported")
    return [
        dict(
            run_id=r["run_id"],
            arm=r["arm"],
            seed=r["seed"],
            expected_training_result=str(checkpoint_slots(plan, r["run_id"], budget)[-1]["model"]),
        )
        for r in plan["runs"]
    ]


def declare(anchor_path, out):
    anchor_ref = artifact(anchor_path / "study.json")
    study = read_checked(anchor_ref)
    plan, _ = canonical_anchor(study)
    sources = [artifact(Path(__file__).with_name(name)) for name in SOURCE_NAMES]
    learned = [r for r in study["assignments"] if r["mode"] == "learned"]
    references = [r for r in study["assignments"] if r["mode"] != "learned"]
    if len(learned) != 90 or len(references) != 48:
        raise ValueError("all 15 models, strong script and seven complete schedules required")
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "study.json",
        dict(
            schema="motion2scene_development_learning_curve_v1",
            anchor=anchor_ref,
            expanded_plan=study["expanded_plan"],
            implementations=sources,
            checkpoints=[8, *CHECKPOINTS],
            evaluation_gate_acquisition_checkpoint=32,
            panels=[dict(checkpoint=b, models=model_slots(plan, b)) for b in CHECKPOINTS],
            assignments=study["assignments"],
            new_policy_episodes_per_checkpoint=len(learned),
            shared_reference_episodes=len(references),
            unique_assigned_episodes=len(study["assignments"]) + len(CHECKPOINTS) * len(learned),
            new_assigned_episodes=len(CHECKPOINTS) * len(learned),
            maximum_new_evaluation_physics_steps=1192 * len(CHECKPOINTS) * len(learned),
            maximum_total_evaluation_physics_steps=1192
            * (len(study["assignments"]) + len(CHECKPOINTS) * len(learned)),
            selection="all five arms and three corpora at each declared checkpoint; no outcome-based selection",
            common_conditions="unchanged six M8 development scenes, physics seed 8732, bank, sensor and scorer",
            baseline_reuse=(
                "exact M8 script and all seven forced-schedule captures; "
                "no new acquisitions or replicates"
            ),
            physical_accounting=(
                "unique collection references counted once across checkpoints; "
                "evaluation separate from acquisition"
            ),
            primary_axis="actual recorded cumulative acquisition steps, not assigned maxima or episode count",
            uncertainty=(
                "paired corpus effects on six reused development layouts; "
                "no claim of independent held-out layouts"
            ),
            stopping=(
                "finite M32 acquisition ceiling retained; "
                "no early stopping or additional checkpoints from outcomes"
            ),
            scope="development learning curve only; no reserved geometry or outcomes",
        ),
    )


def read_study(root):
    ref = artifact(root / "study.json")
    curve = read_checked(ref)
    if curve["schema"] != "motion2scene_development_learning_curve_v1":
        raise ValueError("unknown curve declaration")
    if curve["implementations"] != [
        artifact(Path(__file__).with_name(name)) for name in SOURCE_NAMES
    ]:
        raise ValueError("the complete declared implementation must be retained")
    for source in curve["implementations"]:
        checked(Path(source["path"]), source["sha256"])
    study = read_checked(curve["anchor"])
    plan, bank = canonical_anchor(study)
    if (
        curve["expanded_plan"] != study["expanded_plan"]
        or curve["assignments"] != study["assignments"]
        or curve["panels"] != [dict(checkpoint=b, models=model_slots(plan, b)) for b in CHECKPOINTS]
        or curve["evaluation_gate_acquisition_checkpoint"] != 32
        or curve["checkpoints"] != [8, 16, 32]
        or curve["new_assigned_episodes"] != 180
        or curve["unique_assigned_episodes"] != 318
        or curve["new_policy_episodes_per_checkpoint"] != 90
        or curve["shared_reference_episodes"] != 48
        or curve["maximum_new_evaluation_physics_steps"] != 214560
        or curve["maximum_total_evaluation_physics_steps"] != 379056
    ):
        raise ValueError("checkpoint, assignment, or final acquisition gate changed")
    return ref, curve, study, plan, bank


def acquisition_ready(curve, plan):
    missing = []
    for run in plan["runs"]:
        path = Path(plan["execution_root"]) / run["run_id"] / "controller/complete_032.json"
        if not path.is_file():
            missing.append(run["run_id"])
            continue
        complete = read_checked(artifact(path))
        if complete["expanded_plan"] != curve["expanded_plan"]:
            raise ValueError("final acquisition boundary belongs to a different trajectory")
        final = checkpoint_slots(plan, run["run_id"], 32)[-1]["model"]
        if complete["training_result"] != artifact(final):
            raise ValueError("M32 completion must bind its exact assigned checkpoint")
    return missing


def bind_models(plan, bank, models, budget):
    if models != model_slots(plan, budget):
        raise ValueError("every exact model slot must match the declared checkpoint")
    bindings = {}
    for model in models:
        ref = artifact(Path(model["expected_training_result"]))
        result = read_checked(ref)
        slots = checkpoint_slots(plan, model["run_id"], budget)
        teachers = [artifact(s["teacher"]) for s in slots]
        if result["status"] != "complete" or result["audit_errors"]:
            raise ValueError("a complete acquired model is required")
        registration = read_checked(result["registration"])
        validate_training_prefix(registration, teachers, plan["registry"])
        weighting = (
            "historical_observation_gap"
            if model["arm"] == "observation_curriculum"
            else "uniform_per_phase"
        )
        if (
            registration["schema"] != "motion2scene_expanded_fit_v1"
            or registration["l2"] != 10.0
            or registration["allow_measured_tie_initialization"] is not True
            or registration["weighting"] != weighting
            or read_checked(registration["plan"]) != plan
        ):
            raise ValueError(
                "the common acquired ridge learner and assigned replay arm must be retained"
            )
        if [g["collection"] for g in read_checked(result["teachers"])] != teachers:
            raise ValueError("the model must retain its exact cumulative teacher prefix")
        if (
            Path(result["policy"]["path"]).resolve()
            != Path(ref["path"]).with_name("policy.npz").resolve()
        ):
            raise ValueError("policy substitution outside the assigned model slot")
        load_schedule_policy(result["policy"]["path"], result["policy"]["sha256"], bank)
        bindings[model["run_id"]] = dict(training_result=ref, policy=result["policy"])
    return bindings


def completed_anchor(curve, study):
    directory = Path(curve["anchor"]["path"]).parent
    prepared_ref = artifact(directory / "prepared.json")
    prepared = read_checked(prepared_ref)
    result = read_checked(artifact(directory / "result.json"))
    if (
        prepared["study"] != curve["anchor"]
        or prepared["models"] != anchor.bind_models(study)
        or result["prepared"] != prepared_ref
        or [{k: v for k, v in r.items() if k != "collection"} for r in prepared["assignments"]]
        != study["assignments"]
    ):
        raise ValueError("complete unchanged M8 preparation required")
    results = []
    for row in prepared["assignments"]:
        anchor.check_assignment(row, read_checked(row["collection"]), study, prepared["models"])
        ref = artifact(Path(row["collection"]["path"]).with_name("result.json"))
        known_result(ref, row["collection"])
        results.append(dict(assignment_id=row["assignment_id"], result=ref))
    if result["assignments"] != results:
        raise ValueError("all 138 known M8 outcomes must precede the continuation")
    return prepared


def known_result(ref, collection_ref):
    result = read_checked(ref)
    if result["manifest"] != collection_ref or len(result["rows"]) != 1:
        raise ValueError("one physical result for this exact collection is required")
    row = measure(result["rows"][0])
    if row["status"] not in ("pass", "failure"):
        raise RuntimeError("unknown outcome retained; no automatic retry or substitution")
    validate_measurement(row)
    return row


def validate_measurement(row):
    cost, steps = row["time_s"], row["recorded_physics_steps"]
    if (
        (
            row["status"] == "pass"
            and (type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0)
        )
        or (row["status"] != "pass" and cost is not None)
        or (steps is not None and (type(steps) is not int or steps < 0))
    ):
        raise ValueError(
            "successful time and recorded physical cost must retain their measurement definitions"
        )


def comparison_contract(manifest):
    """Runtime/scoring/observation equality, allowing different model and output paths."""
    keys = (
        "registry",
        "request",
        "template",
        "feature_names",
        "feature_schema",
        "sensor",
        "option_ids",
        "phase_ticks",
        "implementation",
        "runtime_artifacts",
        "scoring_artifacts",
        "runtime_assets_declaration",
        "scoring",
        "limits",
        "expected_physics_steps",
        "expected_recorded_control_steps",
        "qualified_environment_audits",
    )
    return dict(
        **{k: manifest[k] for k in keys},
        dependencies=[{k: r[k] for k in ("path", "sha256")} for r in manifest["dependencies"]],
    )


def prepare(root, budget):
    ref, curve, study, plan, bank = read_study(root)
    if budget not in CHECKPOINTS:
        raise ValueError("M16 or M32 required")
    missing = acquisition_ready(curve, plan)
    if missing:
        raise FileNotFoundError(f"final M32 acquisition is incomplete for {len(missing)} corpora")
    previous = completed_anchor(curve, study)
    panel = next(p for p in curve["panels"] if p["checkpoint"] == budget)
    models = bind_models(plan, bank, panel["models"], budget)
    folder = root / f"M{budget}"
    folder.mkdir(exist_ok=True)
    binding = commit(folder / "models.json", dict(study=ref, checkpoint=budget, models=models))
    rows = []
    for assigned, old in zip(curve["assignments"], previous["assignments"], strict=True):
        if assigned["mode"] != "learned":
            rows.append(dict(**assigned, collection=old["collection"], reused_reference=True))
            continue
        target = folder / "episodes" / assigned["assignment_id"]
        policy = models[assigned["run_id"]]["policy"]
        if not (target / "manifest.json").is_file():
            if target.exists():
                raise RuntimeError(
                    "partial collection preparation retained; inspect before continuation"
                )
            collection.prepare(
                SimpleNamespace(
                    registry=Path(study["registry"]["path"]),
                    request=Path(study["request"]["path"]),
                    template=Path(study["template"]["path"]),
                    cell="neutral",
                    scene_definition=Path(assigned["scene_definition"]["path"]),
                    policy_mode="learned",
                    policy=Path(policy["path"]),
                    script_parameters=None,
                    forced_option_ids=None,
                    preferred_option_id="neutral",
                    preferred_reference_id="sustained",
                    seed=anchor.PHYSICS_SEED,
                    out=target,
                )
            )
        manifest = artifact(target / "manifest.json")
        anchor.check_assignment(assigned, read_checked(manifest), study, models)
        if comparison_contract(read_checked(manifest)) != comparison_contract(
            read_checked(old["collection"])
        ):
            raise ValueError(
                "M8 and later policies must share the exact runtime, sensor and scoring contract"
            )
        collection.verify_manifest(target, execution=False)
        rows.append(dict(**assigned, collection=manifest, reused_reference=False))
    return commit(
        folder / "prepared.json",
        dict(study=ref, checkpoint=budget, models=binding, assignments=rows),
    )


def run(root, budget):
    _, curve, study, plan, _ = read_study(root)
    if budget not in CHECKPOINTS:
        raise ValueError("M16 or M32 required")
    if acquisition_ready(curve, plan):
        raise FileNotFoundError(
            "all final M32 acquisition boundaries must complete before evaluation"
        )
    # No second acquisition process is launched. Both original acquisition locks
    # are held while evaluating, after their final completion gate has passed.
    predecessor = read_checked(plan["predecessor"])
    lock_root = Path(predecessor["intended_execution_root"])
    with ExitStack() as stack:
        for name in (".primary-acquisition.lock", ".expanded_acquisition.lock"):
            stack.enter_context(acquisition_lock(lock_root / name))
        stack.enter_context(acquisition_lock(root / ".curve.lock"))
        prepared_ref = prepare(root, budget)
        prepared = read_checked(prepared_ref)
        models = read_checked(prepared["models"])["models"]
        results = []
        for row in prepared["assignments"]:
            folder = Path(row["collection"]["path"]).parent
            anchor.check_assignment(row, read_checked(row["collection"]), study, models)
            path = folder / "result.json"
            if not path.exists():
                if row["reused_reference"]:
                    raise ValueError("a shared baseline must retain its original completed capture")
                if not anchor.resource_ready(root):
                    return dict(
                        status="waiting_for_native_capacity",
                        checkpoint=budget,
                        current_assignment_not_launched=row["assignment_id"],
                        processed_assignments=len(results),
                    )
                common, bank, scene = collection.verify_manifest(folder, execution=True)
                cell = anchor.check_assignment(row, common, study, models)
                anchor.execute_cell(cell, common, bank, scene, folder / "launch.json")
                collection.analyze(folder)
                commit(folder / "storage.json", anchor.deduplicate_inventories([folder]))
            result_ref = artifact(path)
            outcome = known_result(result_ref, row["collection"])
            results.append(
                dict(
                    assignment_id=row["assignment_id"],
                    result=result_ref,
                    reused_reference=row["reused_reference"],
                )
            )
            print(
                json.dumps(
                    dict(
                        checkpoint=budget,
                        assignment=row["assignment_id"],
                        outcome=outcome["status"],
                        reused_reference=row["reused_reference"],
                    )
                ),
                flush=True,
            )
        return commit(
            root / f"M{budget}" / "result.json", dict(prepared=prepared_ref, assignments=results)
        )


def acquisition_cost(plan, model, budget):
    costs = dict(bootstrap_teacher_steps=0, encounter_teacher_steps=0, preupdate_student_steps=0)
    sources = []
    for index, slot in enumerate(checkpoint_slots(plan, model["run_id"], budget)):
        for kind in ("teacher", "student"):
            if slot[kind] is None:
                continue
            ref = artifact(slot[kind])
            result = read_checked(ref)
            manifest = read_checked(result["manifest"])
            if (
                manifest["split"] != "development"
                or manifest["scene_definition"] != slot["scene"]
                or any(c["runtime_seed"] != model["seed"] for c in manifest["cells"])
                or len(result["rows"]) != (7 if kind == "teacher" else 1)
                or any(
                    r["outcome"]["task_outcome"] not in ("pass", "failure") for r in result["rows"]
                )
                or type(result["physics_steps"]) is not int
                or result["physics_steps"] < 0
            ):
                raise ValueError(
                    "complete measured acquisition accounting on the assigned prefix required"
                )
            key = (
                "preupdate_student_steps"
                if kind == "student"
                else "bootstrap_teacher_steps" if index == 0 else "encounter_teacher_steps"
            )
            costs[key] += result["physics_steps"]
            sources.append(ref)
    return dict(
        **costs,
        total_recorded_steps=sum(costs.values()),
        sources=sources,
        scope="recorded acquisition captures; historical reservations and evaluation excluded",
    )


def evaluation_accounting(panels):
    """Count reused M8 references once, retaining unknown measured-step counts."""
    unique = {}
    for panel in panels:
        for row in panel["rows"]:
            key = (row["collection"]["path"], row["collection"]["sha256"])
            value = {k: row[k] for k in ("status", "time_s", "recorded_physics_steps")}
            validate_measurement(value)
            if key in unique and unique[key] != value:
                raise ValueError("one reused physical capture has inconsistent outcome accounting")
            unique[key] = value
    return dict(
        unique_assigned_episodes=len(unique),
        analysis_rows=sum(len(p["rows"]) for p in panels),
        measured_physics_steps=sum(v["recorded_physics_steps"] or 0 for v in unique.values()),
        unknown_physics_step_counts=sum(
            v["recorded_physics_steps"] is None for v in unique.values()
        ),
        known_outcomes=sum(v["status"] in ("pass", "failure") for v in unique.values()),
        unknown_outcomes=sum(v["status"] == "technical_missing" for v in unique.values()),
        unexecuted=sum(v["status"] == "not_run" for v in unique.values()),
        scope="evaluation only; shared references are the same episodes, not additional replicates",
    )


def analyze(root, out):
    ref, curve, study, plan, bank = read_study(root)
    old = completed_anchor(curve, study)
    old_rows = {r["assignment_id"]: r for r in old["assignments"]}
    reports = []
    for budget in (8, *CHECKPOINTS):
        if budget == 8:
            assignments = [dict(r, reused_reference=False) for r in old["assignments"]]
            slots = study["models"]
            models = old["models"]
        else:
            prepared = read_checked(artifact(root / f"M{budget}" / "prepared.json"))
            slots = next(p["models"] for p in curve["panels"] if p["checkpoint"] == budget)
            models = bind_models(plan, bank, slots, budget)
            binding = read_checked(prepared["models"])
            if (
                prepared["study"] != ref
                or prepared["checkpoint"] != budget
                or binding != dict(study=ref, checkpoint=budget, models=models)
            ):
                raise ValueError("prepared models differ from their declared acquired checkpoints")
            assignments = prepared["assignments"]
        if (
            len(assignments) != len(curve["assignments"])
            or [
                {k: v for k, v in r.items() if k not in ("collection", "reused_reference")}
                for r in assignments
            ]
            != curve["assignments"]
        ):
            raise ValueError("all declared context/model assignments must remain in the analysis")
        rows = []
        for assigned in assignments:
            previous = old_rows[assigned["assignment_id"]]
            reused = budget != 8 and assigned["mode"] != "learned"
            if assigned["reused_reference"] != reused:
                raise ValueError("only original M8 baseline episodes may be reused")
            if reused or budget == 8:
                if assigned["collection"] != previous["collection"]:
                    raise ValueError("shared reference was replaced or reexecuted")
            elif (
                Path(assigned["collection"]["path"]).resolve()
                != (
                    root / f"M{budget}" / "episodes" / assigned["assignment_id"] / "manifest.json"
                ).resolve()
            ):
                raise ValueError("new policy measurement is outside its assigned output slot")
            manifest = read_checked(assigned["collection"])
            anchor.check_assignment(assigned, manifest, study, models)
            if comparison_contract(manifest) != comparison_contract(
                read_checked(previous["collection"])
            ):
                raise ValueError("cross-checkpoint runtime, sensing or scoring contract differs")
            directory = Path(assigned["collection"]["path"]).parent
            path = directory / "result.json"
            measured = dict(
                status="not_run",
                time_s=None,
                measurement_admitted=False,
                recorded_physics_steps=None,
            )
            source = None
            if (directory / "launch.json").exists() or Path(
                manifest["cells"][0]["output"]
            ).exists():
                measured["status"] = "technical_missing"
            if path.is_file():
                source = artifact(path)
                result = read_checked(source)
                if result["manifest"] != assigned["collection"] or len(result["rows"]) != 1:
                    raise ValueError("result and assignment identities differ")
                measured = measure(result["rows"][0])
                validate_measurement(measured)
            rows.append(
                dict(
                    assignment_id=assigned["assignment_id"],
                    policy_id=assigned["policy_id"],
                    scene_id=assigned["scene_id"],
                    physics_seed=anchor.PHYSICS_SEED,
                    **measured,
                    collection=assigned["collection"],
                    result=source,
                    reused_reference=reused,
                )
            )
        summary = summarize(rows, ["fixed_" + s for s in bank.option_ids])
        reports.append(
            dict(
                checkpoint=budget,
                rows=rows,
                summary=summary,
                construction_comparisons=construction_comparisons(rows, slots),
                corpora=[
                    dict(
                        **model,
                        physical_cost=acquisition_cost(plan, model, budget),
                        policy_summary=summary["policies"][model["run_id"]],
                    )
                    for model in slots
                ],
            )
        )
    changes = []
    for earlier, later in zip(reports, reports[1:], strict=False):
        for model in earlier["corpora"]:
            run_id = model["run_id"]
            a = {
                (r["scene_id"], r["physics_seed"]): r
                for r in later["rows"]
                if r["policy_id"] == run_id
            }
            b = {
                (r["scene_id"], r["physics_seed"]): r
                for r in earlier["rows"]
                if r["policy_id"] == run_id
            }
            complete = all(r["status"] in ("pass", "failure") for r in [*a.values(), *b.values()])
            changes.append(
                dict(
                    run_id=run_id,
                    earlier_checkpoint=earlier["checkpoint"],
                    later_checkpoint=later["checkpoint"],
                    passage_count_difference=(
                        (
                            sum(r["status"] == "pass" for r in a.values())
                            - sum(r["status"] == "pass" for r in b.values())
                        )
                        if complete
                        else None
                    ),
                    **paired_time(a, b),
                )
            )
    accounting = evaluation_accounting(reports)
    if accounting["unique_assigned_episodes"] != curve["unique_assigned_episodes"]:
        raise ValueError("evaluation denominator differs from its frozen unique-episode manifest")
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "result.json",
        dict(
            study=ref,
            implementation=artifact(Path(__file__)),
            panels=reports,
            within_corpus_checkpoint_changes=changes,
            evaluation_accounting=accounting,
            proposal_computation={k: plan[k] for k in ("candidate_pools", "reference_pools")},
            new_physics_steps=0,
            interpretation=(
                "fixed development contexts and nested acquisition prefixes; "
                "no held-out or independent-layout claim"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "check", "prepare", "run", "analyze"))
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--budget", choices=CHECKPOINTS, type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "declare":
        result = declare(args.anchor, args.study)
    elif args.action == "check":
        _, curve, _, plan, _ = read_study(args.study)
        missing = acquisition_ready(curve, plan)
        result = dict(
            status="waiting_for_M32_acquisition" if missing else "acquisition_ready",
            missing_corpora=missing,
            new_physics_steps=0,
            unique_assigned_episodes=curve["unique_assigned_episodes"],
        )
    elif args.action == "analyze":
        result = analyze(args.study, args.out)
    else:
        result = {"prepare": prepare, "run": run}[args.action](args.study, args.budget)
    print(json.dumps(result))
