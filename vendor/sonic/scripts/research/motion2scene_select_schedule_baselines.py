#!/usr/bin/env python3
"""Freeze and run a registered offline selection of sensor scripts and constant schedules."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_baselines import (  # noqa: E402
    context_oracle,
    rank_baselines,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_script import (  # noqa: E402
    DEFAULT_CONFIG,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    load_script_parameters,
)

SCHEMA = "motion2scene_schedule_baseline_development_selection_v1"


def read_ref(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def validate_rules(registration):
    """Validate only registered inputs here; no physical result file is opened."""
    original = read_ref(registration["setting_source"])
    batch = read_ref(registration["batch"])
    if (
        registration["schema"] != SCHEMA
        or registration["assigned_contexts"] != 5
        or registration["assigned_branches"] != 35
        or registration["command_ticks"] != [15, 50, 70]
        or registration["configurations"] != original["configurations"]
        or len(registration["configurations"]) != 72
        or registration["provisional_default"] != DEFAULT_CONFIG
        or registration["manifests"] != [row["manifest"] for row in batch["children"]]
        or len(registration["manifests"]) != 5
    ):
        raise ValueError(
            "original ordered 72 settings and exact five-context registration required"
        )
    checked(Path(registration["script_readout"]["path"]), registration["script_readout"]["sha256"])
    manifests = [read_ref(ref) for ref in registration["manifests"]]
    scenes = [read_ref(row["scene_definition"]) for row in manifests]
    if (
        any(row["split"] != "development" for row in manifests + scenes)
        or any(row["registry"] != registration["registry"] for row in manifests)
        or any(len(row["cells"]) != 7 for row in manifests)
        or len({row["scene_id"] for row in scenes}) != 5
        or len({row["scene"]["sha256"] for row in scenes}) != 5
        or len(
            {
                json.dumps(
                    {key: row[key] for key in ("beams", "beam_collision_enabled")}, sort_keys=True
                )
                for row in scenes
            }
        )
        != 5
    ):
        raise ValueError("five distinct native development layouts with common registry required")
    for scene in scenes:
        checked(Path(scene["scene"]["path"]), scene["scene"]["sha256"])
    ref = registration["registry"]
    bank = load_verified_registry(Path(ref["path"]), ref["sha256"])
    phases = sorted({option["entry_tick"] for option in bank.request["options"]})
    if phases != registration["command_ticks"] or len(bank.option_ids) != 7:
        raise ValueError("actual qualified seven-schedule phase interface differs")
    return bank, scenes


def freeze(out):
    registration = json.loads((out / "registration.json").read_text())
    validate_rules(registration)
    if (out / "implementation_registration.json").exists():
        raise FileExistsError("selection implementation was already frozen")
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as handle:
            handle.write(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        out / "implementation_registration.json",
        dict(
            schema=SCHEMA,
            rules=artifact(out / "registration.json"),
            implementation=artifact(Path(__file__)),
            source_closure=sources,
            result_read_barrier="all five registered result.json files must exist before any is read",
            registration_changes="none; original settings, objective, panel and tie rules retained",
            scope="offline finite actual-branch proxy; no new physical execution",
        ),
    )


def result_paths(registration):
    paths = [Path(ref["path"]).parent / "result.json" for ref in registration["manifests"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "complete assigned panel is not yet published: " + ", ".join(missing)
        )
    return paths


def unavailable_group(scene, ref, bank, error):
    """Keep every assigned slot when evidence fails independent audit."""
    return dict(
        scene_id=scene["scene_id"],
        collection=ref,
        assigned_branches=len(bank.option_ids),
        branch_assessments=[
            dict(
                option_id=option,
                task_outcome_admitted=False,
                task_outcome="unknown",
                physics_steps=None,
                outcome=None,
            )
            for option in bank.option_ids
        ],
        targets=[],
        prefix_comparisons=[],
        trajectory_identities=[],
        physics_steps=0,
        physics_steps_unknown_branches=len(bank.option_ids),
        audit_error=error,
    )


def unknown_rows(bank):
    return [dict(forced_option_id=option) for option in bank.option_ids]


def check_unique_episodes(groups):
    identities = [
        (str(Path(ref["path"]).resolve()), ref["sha256"])
        for group in groups
        for ref in group["trajectory_identities"]
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("one recorded physical episode supplied in multiple assigned branches")


def run(out):
    frozen = json.loads((out / "implementation_registration.json").read_text())
    registration = read_ref(frozen["rules"])
    checked(Path(__file__), frozen["implementation"]["sha256"])
    for ref in frozen["source_closure"]:
        checked(Path(ref["path"]), ref["sha256"])
        checked(Path(ref["snapshot"]["path"]), ref["sha256"])
    bank, scenes = validate_rules(registration)
    # All assigned results exist before any outcomes are inspected; an unfinished
    # acquisition remains pending, rather than becoming a favorable smaller panel.
    paths = result_paths(registration)
    write_new(
        out / "run_registration.json",
        dict(
            implementation_registration=artifact(out / "implementation_registration.json"),
            results=[artifact(path) for path in paths],
            assigned_contexts=5,
            assigned_branches=35,
            new_physics_steps=0,
        ),
    )
    groups, oracles, errors = [], [], []
    for scene, path, manifest_ref in zip(scenes, paths, registration["manifests"], strict=True):
        ref = artifact(path)
        try:
            raw = json.loads(path.read_text())
            if raw["manifest"] != manifest_ref:
                raise ValueError("stored result does not bind the registered child manifest")
            group = audit_collection(path, bank, registration["registry"])
            if group["collection"] != ref or group["scene_id"] != scene["scene_id"]:
                raise ValueError("independent group identifies a different physical acquisition")
            oracle = context_oracle(bank, group, raw["rows"])
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            errors.append(dict(scene_id=scene["scene_id"], result=ref, reason=str(error)))
            group = unavailable_group(scene, ref, bank, str(error))
            oracle = context_oracle(bank, group, unknown_rows(bank))
        groups.append(group)
        oracles.append(oracle)
    write_new(out / "actual_reaudits.json", dict(groups=groups, oracles=oracles, errors=errors))
    try:
        check_unique_episodes(groups)
    except ValueError as error:
        errors.append(dict(reason=str(error)))
    ranking = rank_baselines(bank, groups, oracles, registration["configurations"])
    if errors:
        ranking["selected_setting_index"] = None
        ranking["preferred_constant_option_id"] = None
    selected = ranking["selected_setting_index"]
    parameters = None
    if selected is not None:
        path = out / "script_parameters.json"
        write_new(path, registration["configurations"][selected])
        parameters = artifact(path)
        if (
            load_script_parameters(path, parameters["sha256"], bank)
            != registration["configurations"][selected]
        ):
            raise ValueError("selected configuration differs from actual runtime loading")
    complete = selected is not None and ranking["preferred_constant_option_id"] is not None
    result = dict(
        schema=SCHEMA,
        status=(
            "audit_failed"
            if errors
            else "completed_selection" if complete else "insufficient_panel_evidence"
        ),
        implementation_registration=artifact(out / "implementation_registration.json"),
        run_registration=artifact(out / "run_registration.json"),
        actual_reaudits=artifact(out / "actual_reaudits.json"),
        assigned_contexts=5,
        assigned_branches=35,
        known_branch_outcomes=sum(
            branch["known"] for oracle in oracles for branch in oracle["branches"]
        ),
        unknown_branch_outcomes=sum(
            not branch["known"] for oracle in oracles for branch in oracle["branches"]
        ),
        source_physics_steps=sum(group["physics_steps"] for group in groups),
        source_physics_steps_unknown_branches=sum(
            group["physics_steps_unknown_branches"] for group in groups
        ),
        new_physics_steps=0,
        errors=errors,
        script_parameters=parameters,
        scope=registration["scope"],
        **ranking,
    )
    write_new(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.out.resolve()) if args.action == "freeze" else run(args.out.resolve())
    print(
        json.dumps(
            dict(action=args.action, status="frozen" if result is None else result["status"])
        )
    )


if __name__ == "__main__":
    main()
