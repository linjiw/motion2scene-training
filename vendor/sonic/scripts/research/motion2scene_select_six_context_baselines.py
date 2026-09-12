#!/usr/bin/env python3
"""Extend the existing fixed script grid to six audited development contexts."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_select_schedule_baselines import (  # noqa: E402
    check_unique_episodes,
    read_ref,
    unavailable_group,
    unknown_rows,
    validate_rules as validate_original_rules,
)
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_baselines import (  # noqa: E402
    context_oracle,
    rank_baselines,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    load_script_parameters,
)

SCHEMA = "motion2scene_six_context_baseline_development_selection_v1"
UNCHANGED_RULES = (
    "setting_source",
    "configurations",
    "provisional_default",
    "registry",
    "command_ticks",
    "script_readout",
    "regret_rule",
    "script_rule",
    "terminal_rule",
    "tie_rule",
    "unknown_rule",
)


def validate_extension(registration):
    """Only the assigned panel changes; the original ranking implementation is pinned."""
    old = read_ref(registration["original_rules"])
    bank, scenes = validate_original_rules(old)
    complement = read_ref(registration["complementary_batch"])
    if (
        registration["schema"] != SCHEMA
        or registration["assigned_contexts"] != 6
        or registration["assigned_branches"] != 42
        or registration["selection_runs"] != 1
        or registration["new_physics_steps"] != 0
        or registration["original_batch"] != old["batch"]
        or any(registration[key] != old[key] for key in UNCHANGED_RULES)
        or len(complement["children"]) != 1
        or registration["manifests"] != old["manifests"] + [complement["children"][0]["manifest"]]
        or len(registration["collections"]) != 6
    ):
        raise ValueError("exact original rules and one complementary seven-branch context required")
    for key in ("original_selection", "original_model", "original_comparison", "complement_audit"):
        ref = registration[key]
        checked(Path(ref["path"]), ref["sha256"])
    original_implementation = read_ref(
        read_ref(registration["original_selection"])["implementation_registration"]
    )
    ranking = registration["ranking_implementation"]
    if ranking not in [
        {key: row[key] for key in ("path", "sha256")}
        for row in original_implementation["source_closure"]
    ]:
        raise ValueError("ranking must be byte-identical to the original frozen implementation")
    checked(Path(ranking["path"]), ranking["sha256"])
    extra = read_ref(registration["manifests"][-1])
    scene = read_ref(extra["scene_definition"])
    if (
        extra["split"] != "development"
        or scene["split"] != "development"
        or extra["registry"] != registration["registry"]
        or len(extra["cells"]) != len(bank.option_ids)
    ):
        raise ValueError("complement requires an actual common-bank development acquisition")
    checked(Path(scene["scene"]["path"]), scene["scene"]["sha256"])
    scenes.append(scene)
    geometries = [
        json.dumps({key: row[key] for key in ("beams", "beam_collision_enabled")}, sort_keys=True)
        for row in scenes
    ]
    if any(
        len(set(values)) != 6
        for values in (
            [row["scene_id"] for row in scenes],
            [row["scene"]["sha256"] for row in scenes],
            geometries,
        )
    ):
        raise ValueError("six distinct native development layouts required")
    for manifest, result in zip(
        registration["manifests"], registration["collections"], strict=True
    ):
        if Path(result["path"]) != Path(manifest["path"]).parent / "result.json":
            raise ValueError("result must belong to its assigned manifest")
    return bank, scenes


def freeze(out):
    registration = json.loads((out / "registration.json").read_text())
    validate_extension(registration)
    if (out / "implementation_registration.json").exists():
        raise FileExistsError("implementation already frozen")
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
            scope=registration["scope"],
            prior_evidence=registration["prior_evidence"],
        ),
    )


def run(out):
    frozen = json.loads((out / "implementation_registration.json").read_text())
    registration = read_ref(frozen["rules"])
    checked(Path(__file__), frozen["implementation"]["sha256"])
    for ref in frozen["source_closure"]:
        checked(Path(ref["path"]), ref["sha256"])
        checked(Path(ref["snapshot"]["path"]), ref["sha256"])
    bank, scenes = validate_extension(registration)
    for ref in registration["collections"]:
        checked(Path(ref["path"]), ref["sha256"])
    # Exclusive registration prevents a second selection run in this artifact directory.
    write_new(
        out / "run_registration.json",
        dict(
            implementation_registration=artifact(out / "implementation_registration.json"),
            results=registration["collections"],
            assigned_contexts=6,
            assigned_branches=42,
            new_physics_steps=0,
        ),
    )
    groups, oracles, errors = [], [], []
    for scene, ref, manifest_ref in zip(
        scenes, registration["collections"], registration["manifests"], strict=True
    ):
        try:
            raw = read_ref(ref)
            if raw["manifest"] != manifest_ref:
                raise ValueError("result differs from its assigned manifest")
            group = audit_collection(Path(ref["path"]), bank, registration["registry"])
            if group["collection"] != ref or group["scene_id"] != scene["scene_id"]:
                raise ValueError("audit identifies a different physical acquisition")
            oracle = context_oracle(bank, group, raw["rows"])
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            errors.append(dict(scene_id=scene["scene_id"], result=ref, reason=str(error)))
            group = unavailable_group(scene, ref, bank, str(error))
            oracle = context_oracle(bank, group, unknown_rows(bank))
        groups.append(group)
        oracles.append(oracle)
    try:
        check_unique_episodes(groups)
    except ValueError as error:
        errors.append(dict(reason=str(error)))
    write_new(out / "actual_reaudits.json", dict(groups=groups, oracles=oracles, errors=errors))
    ranking = rank_baselines(bank, groups, oracles, registration["configurations"])
    if errors:
        ranking["selected_setting_index"] = None
        ranking["preferred_constant_option_id"] = None
    selected, parameters = ranking["selected_setting_index"], None
    if selected is not None:
        path = out / "script_parameters.json"
        write_new(path, registration["configurations"][selected])
        parameters = artifact(path)
        if (
            load_script_parameters(path, parameters["sha256"], bank)
            != registration["configurations"][selected]
        ):
            raise ValueError("selected settings differ from actual runtime loader")
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
        assigned_contexts=6,
        assigned_branches=42,
        known_branch_outcomes=sum(b["known"] for o in oracles for b in o["branches"]),
        unknown_branch_outcomes=sum(not b["known"] for o in oracles for b in o["branches"]),
        source_physics_steps=sum(g["physics_steps"] for g in groups),
        source_physics_steps_unknown_branches=sum(
            g["physics_steps_unknown_branches"] for g in groups
        ),
        new_physics_steps=0,
        errors=errors,
        script_parameters=parameters,
        scope=registration["scope"],
        prior_evidence=registration["prior_evidence"],
        **ranking,
    )
    # Verify dependencies and immutable input artifacts again before publishing selection.
    for ref in frozen["source_closure"]:
        checked(Path(ref["path"]), ref["sha256"])
    validate_extension(registration)
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
