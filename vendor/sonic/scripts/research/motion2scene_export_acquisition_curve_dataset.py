#!/usr/bin/env python3
"""Export complete declared five-arm acquisition prefixes without changing history.

The existing native dataset exporter audits recorded physics. Expanded replay
sidecars are additionally checked against the original teacher/student histories.
No simulation, selection, new fitting, or replacement of an existing release occurs.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_acquisition_curve_dataset_baseline import (  # noqa: E402
    ARMS,
    CURVE_SCHEMA,
    SEEDS,
    relative_path,
)
from motion2scene_checkpoint_controls import checkpoint_slots, load_checkpoint  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_expanded_acquisition import priority_records  # noqa: E402
from motion2scene_export_timed_schedule_dataset import export  # noqa: E402
from motion2scene_response_diversity import validate_paired_queues  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    encounter_replay_weights,
)


def validate_audit(audit, plan):
    budget = audit["budget"]
    expected = {f"seed{s}_{a}" for s in SEEDS for a in (*ARMS, "reference_contrast")}
    if (
        audit["schema"] != "motion2scene_response_diversity_v1"
        or plan["schema"] != "motion2scene_expanded_acquisition_v1"
        or type(budget) is not int
        or budget not in (8, 16, 32)
        or budget not in plan["checkpoints"]
        or len(plan["runs"]) != 15
        or {r["run_id"] for r in plan["runs"]} != expected
        or len(audit["corpora"]) != 15
        or {c["run_id"] for c in audit["corpora"]} != expected
    ):
        raise ValueError("a complete independently audited five-arm M8/M16/M32 prefix is required")
    by_id = {r["run_id"]: r for r in plan["runs"]}
    for corpus in audit["corpora"]:
        run = by_id[corpus["run_id"]]
        if (
            run["run_id"] != f"seed{run['seed']}_{run['arm']}"
            or run["seed"] not in SEEDS
            or run["arm"] not in (*ARMS, "reference_contrast")
            or corpus["arm"] != run["arm"]
            or corpus["seed"] != run["seed"]
            or corpus["physical_cost"]["assigned_episodes"] != 7 + 8 * budget
            or len(corpus["tasks"]) != budget
            or [t["candidate_id"] for t in corpus["tasks"]]
            != [r["candidate_id"] for r in run["rounds"][1 : budget + 1]]
        ):
            raise ValueError("audited corpus identity, task order or acquisition budget changed")
    validate_paired_queues(plan, budget)
    return budget


def preflight(audit_path, adoption_path):
    audit_ref, adoption_ref = artifact(audit_path), artifact(adoption_path)
    audit, adoption = read_checked(audit_ref), read_checked(adoption_ref)
    plan = read_checked(audit["plan"])
    budget = validate_audit(audit, plan)
    if adoption["status"] != "adopted" or adoption["plan"] != plan["predecessor"]:
        raise ValueError("original adoption must bind the inherited acquisition trajectory")
    runtime = read_checked(adoption["runtime_freeze"])
    rule_ref = runtime["replay_rule"]
    read_checked(rule_ref)
    snapshots = []
    for corpus in audit["corpora"]:
        slots = checkpoint_slots(plan, corpus["run_id"], budget)
        models = [artifact(s["model"]) for s in slots]
        collections = [artifact(s["teacher"]) for s in slots]
        collections += [artifact(s["student"]) for s in slots[1:]]
        if len({r["path"] for r in collections}) != 2 * budget + 1:
            raise ValueError("every teacher collection and actual student visit must stay distinct")
        final = read_checked(models[-1])
        if (
            models[-1] != corpus["checkpoint"]
            or final["status"] != "complete"
            or final["audit_errors"]
        ):
            raise ValueError("final model differs from the independently audited checkpoint")
        for index, model in enumerate(models):
            result = read_checked(model)
            registration = read_checked(result["registration"])
            groups = read_checked(result["teachers"])
            if (
                result["status"] != "complete"
                or result["audit_errors"]
                or registration["collections"] != collections[: index + 1]
                or [g["collection"] for g in groups] != collections[: index + 1]
                or registration["l2"] != 10.0
                or registration["allow_measured_tie_initialization"] is not True
            ):
                raise ValueError(
                    "retain every exact historical teacher prefix and common ridge fit"
                )
            # The inherited physical schema requires complete recorded neutral
            # histories. Refuse unsupported data before writing any release;
            # never remove an assigned task to make the export succeed.
            if any(not t.get("available", True) for g in groups for t in g["targets"]):
                raise ValueError(
                    "unavailable neutral histories require a newer physical export schema; no task may be dropped"
                )
        for source in collections:
            result = read_checked(source)
            if any(
                row.get("status") in ("failed_attempt", "invalid_or_partial_measurement")
                for row in result["rows"]
            ):
                raise ValueError(
                    "partial captures require a newer physical export schema; "
                    "preserve the complete assigned prefix"
                )
        snapshots.append(
            dict(
                run_id=corpus["run_id"],
                arm=corpus["arm"],
                seed=corpus["seed"],
                models=models,
                collections=collections,
                teachers=final["teachers"],
                physical_cost=corpus["physical_cost"],
            )
        )
    return dict(
        audit=audit_ref,
        adoption=adoption_ref,
        plan=audit["plan"],
        replay_rule=rule_ref,
        budget=budget,
        corpora=snapshots,
    )


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def audit_expanded_replay(sidecar, bank, groups, students, rule):
    records = priority_records(bank, groups, students, rule)
    expected = dict(
        records=records,
        weights=encounter_replay_weights(records),
        signal="historical measured generating-policy outcomes",
    )
    if normalized(sidecar) != normalized(expected):
        raise ValueError(
            "expanded replay sidecar differs from independently re-audited historical outcomes"
        )
    return expected


def copy_ref(source, destination):
    read = Path(source["path"])
    if artifact(read) != source:
        raise ValueError("source artifact changed during export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(read.read_bytes())


def history_sources(history_path):
    """Copy only explicitly inventoried text receipts, never a directory's extras."""
    history_ref = artifact(history_path)
    history = read_checked(history_ref)
    if history["schema"] != "motion2scene_portable_historical_startup_v1":
        raise ValueError("the retained original startup provenance is required")
    sources = {"history.json": history_ref}
    for item in history["sources"].values():
        name = item["file"]
        source = relative_path(history_path.parent, name)
        if Path(name).suffix not in (".json", ".log", ".sh") or name in sources:
            raise ValueError("startup provenance must inventory distinct text receipts only")
        ref = artifact(source)
        if ref["sha256"] != item["source"]["sha256"]:
            raise ValueError("original retained bootstrap/startup history changed")
        sources[name] = ref
    return history, sources


def export_curve(audit_path, adoption_path, history_path, out):
    snapshot = preflight(audit_path, adoption_path)
    budget = snapshot["budget"]
    plan = read_checked(snapshot["plan"])
    rule = read_checked(snapshot["replay_rule"])
    history_ref = artifact(history_path)
    history, retained_sources = history_sources(history_path)
    out.mkdir(parents=True, exist_ok=False)
    for name, source in (
        ("source_audit.json", snapshot["audit"]),
        ("source_plan.json", snapshot["plan"]),
        ("source_adoption.json", snapshot["adoption"]),
        ("replay_rule.json", snapshot["replay_rule"]),
    ):
        copy_ref(source, out / name)
    for name, source in retained_sources.items():
        copy_ref(source, out / "historical_startup" / name)
    write_new(
        out / "registration.json",
        dict(
            **snapshot,
            history=history_ref,
            implementation=artifact(Path(__file__)),
            exporter=artifact(
                ROOT / "scripts/research/motion2scene_export_timed_schedule_dataset.py"
            ),
            new_physics_steps=0,
        ),
    )
    records = []
    for item in snapshot["corpora"]:
        print(json.dumps(dict(status="exporting_corpus", run_id=item["run_id"])), flush=True)
        started = time.monotonic()
        bank, groups, students, checkpoint = load_checkpoint(plan, item["run_id"], budget)
        if checkpoint != item["models"][-1]:
            raise ValueError("audited source checkpoint changed before export")
        package = out / "corpora" / item["run_id"]
        export(
            package, [Path(r["path"]) for r in item["collections"]], Path(item["teachers"]["path"])
        )
        manifest = json.loads((package / "manifest.json").read_text())
        if (
            manifest["counts"]["episodes"] != 7 + 8 * budget
            or manifest["counts"]["physics_steps"] != item["physical_cost"]["total_recorded_steps"]
        ):
            raise ValueError("portable data omitted or duplicated an assigned measured acquisition")
        if item["run_id"] == history["charged_to_run_id"]:
            captures = {
                e["source_trajectory"]["path"]: e["source_trajectory"]["sha256"]
                for e in manifest["episodes"]
            }
            if any(
                captures.get(r["path"]) != r["sha256"]
                for r in history["reused_bootstrap_trajectories"]
            ):
                raise ValueError(
                    "the reused bootstrap must remain in its original corpus and be counted once"
                )
        models = []
        for index, model_ref in enumerate(item["models"]):
            result = read_checked(model_ref)
            registration = read_checked(result["registration"])
            folder = out / "fit_provenance" / item["run_id"] / f"model_{index:03d}"
            refs = dict(
                result=model_ref,
                registration=result["registration"],
                teachers=result["teachers"],
                policy=result["policy"],
            )
            schema = registration["schema"]
            if schema == "motion2scene_timed_schedule_training_v1":
                if registration["replay_weights"] is not None:
                    refs["replay_weights"] = registration["replay_weights"]
            elif schema == "motion2scene_expanded_fit_v1":
                if registration["plan"] != snapshot["plan"]:
                    raise ValueError("expanded model belongs to a different acquisition plan")
                if registration["weighting"] == "historical_observation_gap":
                    sidecar_ref = artifact(Path(model_ref["path"]).with_name("replay.json"))
                    sidecar = read_checked(sidecar_ref)
                    audit_expanded_replay(
                        sidecar, bank, groups[: index + 1], students[: index + 1], rule
                    )
                    refs["replay"] = sidecar_ref
                    write_new(
                        folder / "replay_verification.json",
                        dict(
                            original_sidecar=sidecar_ref,
                            replay_rule=snapshot["replay_rule"],
                            teacher_collections=[g["collection"] for g in groups[: index + 1]],
                            historical_outcomes_independently_reaudited=True,
                            new_physics_steps=0,
                        ),
                    )
                elif registration["weighting"] != "uniform_per_phase":
                    raise ValueError("unsupported expanded weighting rule")
            else:
                raise ValueError("unsupported historical model schema")
            for name, source in refs.items():
                copy_ref(source, folder / (name + (".npz" if name == "policy" else ".json")))
            models.append(
                dict(
                    checkpoint=index,
                    original=model_ref,
                    files={
                        p.name: dict(path=str(p.relative_to(out)), sha256=artifact(p)["sha256"])
                        for p in sorted(folder.iterdir())
                        if p.is_file()
                    },
                )
            )
        record = dict(
            run_id=item["run_id"],
            arm=item["arm"],
            seed=item["seed"],
            dataset=str(package.relative_to(out)),
            dataset_manifest_sha256=artifact(package / "manifest.json")["sha256"],
            counts=manifest["counts"],
            physical_cost=item["physical_cost"],
            models=models,
            export_wall_seconds=time.monotonic() - started,
        )
        records.append(record)
        write_new(out / "receipts" / (item["run_id"] + ".json"), record)
    toolkit = out / "tools/portable_baseline"
    for source in sorted(
        closure([ROOT / "scripts/research/motion2scene_acquisition_curve_dataset_baseline.py"])
    ):
        target = toolkit / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in (
        "gear_sonic",
        "gear_sonic/dataset_generation",
        "gear_sonic/dataset_generation/hallucination",
    ):
        (toolkit / name / "__init__.py").write_text(
            '"""Portable pure Python subset; no simulator assets."""\n'
        )
    (toolkit / "requirements.txt").write_text("numpy>=1.24\n")
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")
    shutil.copytree(ROOT / "legal", out / "legal")
    (out / "DATA_CARD.md").write_text(
        f"# Completed M{budget} acquisition prefixes\n\n"
        "Fifteen separate development corpora retain all bootstrap, teacher and "
        "actual pre-update student records. "
        "Every historical M0-through-checkpoint model is preserved, including inherited original-prefix files. "
        "These are acquisition data, not held-out policy evaluations. "
        "Physical interaction costs come from measurements; "
        "source_audit.json keeps proposal computation separate. "
        "Prefix costs must not be summed across checkpoints.\n\n"
        "Original models retain their bound replay artifacts. Expanded replay sidecars, "
        "which the original fit result does not itself hash-bind, are hashed at export "
        "and independently checked against the historical teacher/student records and original replay rule. "
        "The NumPy reader reconstructs weighting and fits; it does not generate new "
        "physical gaps. Student outcomes remain attached to their generating earlier models.\n\n"
        "historical_startup retains the original unknown startup assessment, zero measured startup steps, "
        "separate conservative reservation and failed initial fit. "
        "The seven reused bootstrap captures stay in their original corpus and are counted once. "
        "No raw pretrained weights or full motion/reference bank is included. Recorded reference "
        "commands remain execution data; existing repository and upstream licensing applies.\n\n"
        "Reconstruct with Python and NumPy:\n\n"
        "    OPENBLAS_NUM_THREADS=1 python3 "
        "tools/portable_baseline/scripts/research/motion2scene_acquisition_curve_dataset_baseline.py "
        "--release . --out /absolute/path/to/new-reconstruction\n"
    )
    return write_new(
        out / "release.json",
        dict(
            schema=CURVE_SCHEMA,
            registration=artifact(out / "registration.json"),
            source_audit=snapshot["audit"],
            budget=budget,
            reported_checkpoints=[n for n in (8, 16, 32) if n <= budget],
            corpora=records,
            total_episodes=sum(r["counts"]["episodes"] for r in records),
            total_physics_steps=sum(r["counts"]["physics_steps"] for r in records),
            toolkit_files=[
                dict(path=str(p.relative_to(out)), sha256=artifact(p)["sha256"])
                for p in sorted(toolkit.rglob("*"))
                if p.is_file()
            ],
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = preflight(args.audit, args.adoption)
        print(
            json.dumps(
                dict(
                    status="complete_export_inputs_checked",
                    budget=result["budget"],
                    corpora=len(result["corpora"]),
                    new_physics_steps=0,
                )
            )
        )
    else:
        if args.history is None or args.out is None:
            parser.error("--history and --out are required for export")
        print(json.dumps(export_curve(args.audit, args.adoption, args.history, args.out)))
