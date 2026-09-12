#!/usr/bin/env python3
"""Qualify every assigned extension schedule using a common loaded reference bank."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from hallucination.run_approved_manifest import free_gpu_mib, run_with_process_group  # noqa: E402
from motion2scene_option_extension_study import DATA, checked_json, load_plan  # noqa: E402
import motion2scene_qualify_environment_schedules as environment  # noqa: E402
import motion2scene_qualify_long_schedules as strict  # noqa: E402
from motion2scene_scene_resolution import validate_scene_handoff  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_integer_prefix import (  # noqa: E402
    paired_prefix_on_ticks,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    definition_digest,
    validate_request,
)

RUNTIME = (
    "gear_sonic.dataset_generation.hallucination.motion2scene_schedule_bank_qualification_execution"
)


def build_request(plan, candidates, neutral, controller, raw_motions, *, plan_ref, candidates_ref):
    if [r["candidate_id"] for r in candidates["rows"]] != [
        f"{arm}_{i:02d}" for arm in ("generated", "authored") for i in range(4)
    ]:
        raise ValueError("exact four-candidate assignment per arm required")
    references = [
        dict(
            reference_id="neutral",
            motion=neutral,
            construction="generated",
            expected_loaded_frames=299,
        )
    ]
    options = []
    for row in candidates["rows"]:
        name = row["candidate_id"]
        reference = dict(
            reference_id=name,
            motion=row["motion"],
            parent_motion=neutral,
            expected_loaded_frames=299,
            construction="authored_local_crouch",
            common_repair=plan["specification"]["repair"],
        )
        if row["arm"] == "generated":
            reference.update(
                construction="authored_prior_projection_splice",
                prior_motion=raw_motions[name],
                derivation_registration=plan_ref,
                derivation_result=candidates_ref,
                boundary_diagnostic=artifact(Path(row["motion"]["path"] + ".manifest.json")),
            )
        references.append(reference)
        for schedule in plan["specification"]["schedules"]:
            options.append(
                dict(
                    option_id=f"{name}_e{schedule['entry_tick']:03d}_r265",
                    reference_id=name,
                    profile_label=name,
                    **schedule,
                    recovery_end_tick=297,
                    maintenance=dict(start_tick=100, end_tick=175, maximum_body_height_m=None),
                )
            )
    request = dict(
        schema="motion2scene_timed_option_request_v1",
        request_id="matched_extension_qualification_v1",
        split="development",
        max_entries_per_episode=1,
        reference_fps=50,
        expected_loaded_frames=299,
        support_plane_z_m=0.0,
        height_measurement="executed_outer_collision_height_above_support_plane",
        minimum_recovery_ticks=15,
        legal_entry_decision_ticks=[15, 50],
        references=references,
        options=options,
        controller=controller,
        qualification_study=plan_ref,
        candidate_results=candidates_ref,
        scope="common nine-reference bank, fresh neutral and all sixteen assigned schedules",
    )
    validate_request(request)
    return request


def prepare(study, out):
    from gear_sonic.dataset_generation.kimodo_motion_adapter import (
        qpos_to_sonic_motion_entry,
        save_sonic_motion_file,
    )

    plan = load_plan(study)
    candidates_ref = artifact(study / "candidate_results.json")
    candidates = checked_json(candidates_ref)
    if candidates["plan"] != artifact(study / "plan.json") or len(candidates["rows"]) != 8:
        raise ValueError("all eight assigned candidates required")
    template_path = DATA / "m2s-prior-splice-seven-schedule-qualification-v1/manifest.json"
    template = json.loads(template_path.read_text())
    original_request = checked_json(template["request"])
    out.mkdir(parents=True, exist_ok=False)
    raw_motions = {}
    for row in candidates["rows"]:
        for name in ("motion", "stages", "native_reference"):
            checked(Path(row[name]["path"]), row[name]["sha256"])
        if row["arm"] == "generated":
            with np.load(row["stages"]["path"], allow_pickle=False) as data:
                raw = data["raw"].copy()
            path = out / "raw_motion_ancestry" / (row["candidate_id"] + ".pkl")
            save_sonic_motion_file(
                path,
                motion_key="unrepaired_" + row["candidate_id"],
                motion_entry=qpos_to_sonic_motion_entry(raw, source_fps=30),
            )
            raw_motions[row["candidate_id"]] = artifact(path)
    request = build_request(
        plan,
        candidates,
        original_request["references"][0]["motion"],
        original_request["controller"],
        raw_motions,
        plan_ref=artifact(study / "plan.json"),
        candidates_ref=candidates_ref,
    )
    request_ref = write_new(out / "request.json", request)
    bank = validate_request(request)
    cells = []
    for option_id in bank.option_ids:
        option = next((o for o in request["options"] if o["option_id"] == option_id), None)
        tick = option["entry_tick"] if option else 15
        cell = copy.deepcopy(template["cells"][0])
        cell.update(
            cell_id=option_id,
            forced_option_id=option_id,
            declared_schedule=option,
            output=str(out.resolve() / "rollouts" / option_id),
            encounter_action=int(option is not None),
            decision_time_s=tick / 50,
            required_exact_prefix_frames=tick,
            role="matched_extension_qualification",
            motion=request["references"][0]["motion"],
            alternate_motion=request["references"][1]["motion"],
            runtime_seed=97000,
        )
        updates = dict(seed=97000)
        updates.update(
            {
                "manager_env._target_": RUNTIME + ".ScheduleBankQualificationEnvCfg",
                "manager_env.recorders.trajectory._target_": RUNTIME
                + ".ScheduleBankQualificationRecorderCfg",
            }
        )
        updates.update(
            {
                "manager_env.config." + k: v
                for k, v in dict(
                    qualification_request_path=request_ref["path"],
                    qualification_request_sha256=request_ref["sha256"],
                    forced_option_id=option_id,
                    encounter_action=int(option is not None),
                    decision_time_s=tick / 50,
                    expected_reference_frames=299,
                    reactive_alternate_path=cell["alternate_motion"]["path"],
                ).items()
            }
        )
        cell["hydra_overrides"] = [
            v for v in cell["hydra_overrides"] if v.lstrip("+").split("=", 1)[0] not in updates
        ] + [f"++{k}={v}" for k, v in updates.items()]
        for flag, value in (
            ("--out", cell["output"]),
            ("--max-steps", "299"),
            ("--motion", cell["motion"]["path"]),
            ("--extra", " ".join(cell["hydra_overrides"])),
        ):
            cell["command"][cell["command"].index(flag) + 1] = value
        cells.append(cell)
    sources = closure([Path(__file__), ROOT / (RUNTIME.replace(".", "/") + ".py")]) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    dependencies = []
    for path in sorted(sources):
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        dependencies.append(dict(**artifact(path), snapshot=artifact(target)))
    manifest = {
        key: copy.deepcopy(template[key])
        for key in (
            "geometry",
            "contact_criterion",
            "limits",
            "previous_strict_results",
            "neutral_counterpart_diagnostic",
        )
    }
    manifest["contact_criterion"][
        "replay"
    ] = "fresh neutral plus sixteen candidate schedules, eight per arm"
    manifest.update(
        schema="motion2scene_extension_qualification_v1",
        study=artifact(study / "plan.json"),
        candidate_results=candidates_ref,
        template=artifact(template_path),
        request=request_ref,
        request_digest=definition_digest(request),
        cells=cells,
        dependencies=dependencies,
        expected_physics_steps=1192,
        expected_recorded_control_steps=298,
        expected_total_physics_steps=20264,
        expected_actual_bank_shape=[9, 299, 3],
        physics_seed=97000,
        prefix_comparison="paired_prefix_on_ticks; unchanged exact cross-branch state equality",
        status="prepared forced qualification; no candidate is online-qualified by preparation",
    )
    return write_new(out / "manifest.json", manifest)


def verify(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != "motion2scene_extension_qualification_v1":
        raise ValueError("wrong qualification schema")
    for ref in manifest["dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    request = checked_json(manifest["request"])
    bank = validate_request(request)
    if [c["cell_id"] for c in manifest["cells"]] != list(bank.option_ids):
        raise ValueError("assigned qualification branch missing")
    if len(bank.option_ids) != 17 or len(request["references"]) != 9:
        raise ValueError("matched qualification budget differs")
    for ref in [request["controller"], *(r["motion"] for r in request["references"])]:
        checked(Path(ref["path"]), ref["sha256"])
    for cell in manifest["cells"]:
        command = cell["command"]
        if command[command.index("--extra") + 1] != " ".join(cell["hydra_overrides"]):
            raise ValueError("qualification command differs from overrides")
        effective = dict(value.lstrip("+").split("=", 1) for value in cell["hydra_overrides"])
        if len(effective) != len(cell["hydra_overrides"]):
            raise ValueError("duplicate qualification override")
        expected = {
            "seed": "97000",
            "manager_env._target_": RUNTIME + ".ScheduleBankQualificationEnvCfg",
            "manager_env.config.forced_option_id": cell["cell_id"],
            "manager_env.config.qualification_request_path": manifest["request"]["path"],
            "manager_env.config.qualification_request_sha256": manifest["request"]["sha256"],
            "manager_env.recorders.trajectory._target_": RUNTIME
            + ".ScheduleBankQualificationRecorderCfg",
            "manager_env.config.expected_reference_frames": "299",
            "manager_env.config.reactive_alternate_path": request["references"][1]["motion"][
                "path"
            ],
        }
        if any(effective.get(k) != v for k, v in expected.items()):
            raise ValueError("effective qualification identity differs")
        checked(Path(cell["scene"]["path"]), cell["scene"]["sha256"])
        for flag, value in (
            ("--out", cell["output"]),
            ("--max-steps", "299"),
            ("--motion", request["references"][0]["motion"]["path"]),
            ("--checkpoint", request["controller"]["path"]),
        ):
            if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
                raise ValueError("effective qualification argument differs: " + flag)
        validate_scene_handoff(
            command, dict(scene=cell["scene"], scene_id=cell["scene"]["scene_id"])
        )
    return manifest


def analyze(out):
    verify(out)
    # Use the existing numerically correct clock predicate; preserve exact state
    # equality and every original contact/stability/return predicate.
    previous = strict.paired_prefix
    try:
        strict.paired_prefix = paired_prefix_on_ticks
        environment.analyze(out)
    finally:
        strict.paired_prefix = previous
    result = json.loads((out / "environment_result.json").read_text())
    arms = {
        arm: [r for r in result["rows"] if r["cell_id"].startswith(arm + "_")]
        for arm in ("generated", "authored")
    }
    return write_new(
        out / "yield_result.json",
        dict(
            source=artifact(out / "environment_result.json"),
            arms={
                arm: dict(
                    assigned_candidates=4,
                    assigned_schedules=8,
                    qualified_schedules=sum(r["qualified"] for r in rows),
                    candidates_with_any_qualified_schedule=len(
                        {r["cell_id"].split("_e")[0] for r in rows if r["qualified"]}
                    ),
                )
                for arm, rows in arms.items()
            },
            scope=(
                "single-seed empty-room qualification yield; task coverage "
                "and downstream policy effects are separate"
            ),
        ),
    )


def run(out):
    manifest = verify(out)
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        if (folder / "attempt.json").exists():
            if json.loads((folder / "attempt.json").read_text())["command"] != cell["command"]:
                raise ValueError("recorded attempt command differs")
            continue
        launch = out / (cell["cell_id"] + "_launch.json")
        if folder.exists() or launch.exists():
            raise ValueError("unfinished attempt retained; automatic retry forbidden")
        if free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
            raise RuntimeError("remaining qualification assignments unlaunched for GPU capacity")
        write_new(launch, dict(command=cell["command"], charged_maximum_physics_steps=1192))
        started, interrupted = time.monotonic(), None
        try:
            status = run_with_process_group(
                cell["command"], timeout=manifest["limits"]["timeout_s"]
            )
        except subprocess.TimeoutExpired:
            status = 124
        except BaseException as error:
            status, interrupted = -1, error
        folder.mkdir(parents=True, exist_ok=True)
        write_new(
            folder / "attempt.json",
            dict(
                command=cell["command"], exit_status=status, wall_seconds=time.monotonic() - started
            ),
        )
        print(json.dumps(dict(qualified_attempt=cell["cell_id"], exit_status=status)), flush=True)
        if interrupted is not None:
            raise interrupted
        if status != 0:
            raise RuntimeError(
                "non-complete native attempt retained; inspect before remaining qualification"
            )
    return analyze(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "analyze"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.study, args.out)
    elif args.command == "verify":
        verify(args.out)
        result = dict(verified=True)
    else:
        result = globals()[args.command](args.out)
    print(json.dumps(result))
