#!/usr/bin/env python3
"""Matched candidate budgets for generated versus authored bank extensions.

Candidate generation is CPU-only and separate from the primary traversal bank.
All candidates receive the same repair; no candidate is selected by task outcomes.
"""

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/linjiw/research-data/groot-wbc")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_project_splice_prior import (  # noqa: E402
    canonical,
    limit_report,
    native_model,
    project_and_splice,
)
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

SEEDS = (96001, 96002, 96003, 96004)
WINDOWS = (0.18, 0.28, 0.38, 0.48)


def checked_json(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def specification():
    return dict(
        generated_seeds=list(SEEDS),
        authored_window_half_widths=list(WINDOWS),
        shared_target_drop_m=0.085,
        authored_station_fraction=0.55,
        repair="same native hinge-box projection and neutral-prefix/tail quintic splice for every candidate",
        schedules=[dict(entry_tick=15, return_tick=265), dict(entry_tick=50, return_tick=265)],
        qualification=dict(
            physics_seed=97000,
            attempts_per_arm=8,
            common_neutral_attempts=1,
            criterion=(
                "unchanged full-horizon stability, measured external contacts, "
                "matched prefix, legal joins and recovery"
            ),
            selection="retain every passing schedule; no task-based candidate ranking",
        ),
        independent_capability_tasks=[
            dict(
                task_id=f"extension_fixed_{i:02d}",
                center_xy_m=[x, -0.1],
                yaw_rad=0.0,
                length_m=length,
                width_m=1.2,
                underside_m=height,
                thickness_m=0.1,
            )
            for i, (x, length, height) in enumerate(
                (x, length, height)
                for x in (1.70, 2.25, 2.80)
                for length in (0.10, 0.75)
                for height in (1.24, 1.30)
            )
        ],
        task_physics_seed=97001,
        task_selection=(
            "fixed geometry before new candidate generation; "
            "no new candidate envelope or task outcome used"
        ),
        outcome=(
            "qualification yield and incremental task coverage over the shared neutral; "
            "downstream selectors after equivalent teaching"
        ),
        interpretation=(
            "one shared carrier; four new neural seeds versus a deterministic authored duration grid; "
            "matched proposal/qualification budgets, not identical distributions or equal inference compute"
        ),
        failure_policy=(
            "retain raw outputs, rejected repairs and all physical failures; "
            "one inference per seed; no replacement samples"
        ),
        primary_bank_changes=False,
        reserved_evaluation_queries=0,
    )


def prepare(out):
    parent_path = DATA / "m2s-kimodo-conditioned-lowheight-development-v1/registration.json"
    parent = json.loads(parent_path.read_text())
    for ref in parent["assets"].values():
        checked(Path(ref["path"]), ref["sha256"])
    checked_json(parent["proposal"])
    checked_json(parent["constraints"])
    authored = DATA / "m2s-authored-duration-profiles-v2/registration.json"
    authors = json.loads(authored.read_text())
    neutral = parent["planned_neutral_source_csv"]
    checked(Path(neutral["path"]), neutral["sha256"])
    # Explicit import targets include CPU adaptation/FK used lazily below.
    sources = closure(
        [Path(__file__), ROOT / "scripts/research/motion2scene_author_timed_profiles.py"]
    )
    plan = dict(
        schema="motion2scene_option_extension_study_v1",
        specification=specification(),
        generation_parent=artifact(parent_path),
        authored_parent=artifact(authored),
        neutral_csv=neutral,
        native_mjcf=artifact(
            ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
        ),
        authored_operator_mjcf=authors["mjcf"],
        implementation=[artifact(p) for p in sorted(sources)],
        shared_operator=dict(
            ramp=authors["ramp"],
            range_keep=authors["range_keep"],
            max_excursion_rad=authors["max_excursion_rad"],
            waist_use_fraction=authors["waist_use_fraction"],
        ),
        generation_runtime=parent["driver"],
        generation_environment=parent["environment"],
    )
    out.mkdir(parents=True, exist_ok=False)
    for ref in plan["implementation"]:
        path = Path(ref["path"])
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        ref["snapshot"] = artifact(target)
    return write_new(out / "plan.json", plan)


def load_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    if (
        plan["schema"] != "motion2scene_option_extension_study_v1"
        or plan["specification"] != specification()
    ):
        raise ValueError("candidate study differs from fixed specification")
    for ref in plan["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    for key in (
        "generation_parent",
        "authored_parent",
        "neutral_csv",
        "native_mjcf",
        "authored_operator_mjcf",
        "generation_runtime",
    ):
        ref = plan[key]
        checked(Path(ref["path"]), ref["sha256"])
    return plan


def generate(out):
    plan = load_plan(out)
    parent = checked_json(plan["generation_parent"])
    results = []
    for seed in SEEDS:
        folder = out / f"generated_{seed}"
        completed = folder / "generation_execution.json"
        if completed.exists():
            result = json.loads(completed.read_text())
            if result["status"] != "generated_retained_pending_audit":
                raise ValueError("previous failed generation retained; no automatic replacement")
            results.append(artifact(completed))
            continue
        if folder.exists():
            raise ValueError("unfinished generation folder retained; no automatic retry")
        folder.mkdir()
        (folder / "generated").mkdir()
        shutil.copyfile(parent["constraints"]["path"], folder / "constraints.json")
        registration = copy.deepcopy(parent)
        registration.update(
            seed=seed,
            constraints=artifact(folder / "constraints.json"),
            batch_plan=artifact(out / "plan.json"),
            source_generation_parent=plan["generation_parent"],
            ancestry="new neural seed under the same planned-neutral conditioning; shared development carrier",
            downstream_repair=(
                "separate, common operator applied to generated and authored "
                "candidates after raw output retention"
            ),
        )
        command = [
            str(ROOT / ".venv_kimodo/bin/python"),
            parent["driver"]["path"],
            "generate",
            "--out",
            str(folder),
        ]
        registration["command"] = command
        write_new(folder / "registration.json", registration)
        started = time.monotonic()
        with (folder / "generation.log").open("x") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={**os.environ, **parent["environment"]},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = 124
        write_new(
            folder / "process.json",
            dict(command=command, exit_status=status, wall_seconds=time.monotonic() - started),
        )
        if status != 0 or not completed.exists():
            raise RuntimeError("registered generation failed; all artifacts retained")
        result = json.loads(completed.read_text())
        if result["status"] != "generated_retained_pending_audit":
            raise RuntimeError("generation did not produce the assigned raw output")
        results.append(artifact(completed))
        print(json.dumps(dict(generated_seed=seed, execution=results[-1])), flush=True)
    return write_new(
        out / "generation_results.json",
        dict(
            plan=artifact(out / "plan.json"),
            results=results,
            new_neural_samples=len(results),
            physics_steps=0,
        ),
    )


def reuse_generated(out, source):
    """Reuse fixed raw samples after a conversion-only implementation correction."""
    load_plan(out)
    previous = json.loads((source / "generation_results.json").read_text())
    old_plan = checked_json(previous["plan"])
    if old_plan["specification"] != specification() or len(previous["results"]) != 4:
        raise ValueError("raw generation does not match the fixed four-sample study")
    for ref in old_plan["implementation"]:
        saved = ref["snapshot"]
        checked(Path(saved["path"]), saved["sha256"])
    for seed, ref in zip(SEEDS, previous["results"], strict=True):
        result = checked_json(ref)
        registration = checked_json(result["registration"])
        if registration["seed"] != seed or result["status"] != "generated_retained_pending_audit":
            raise ValueError("incomplete or mismatched raw generation")
        folder = Path(ref["path"]).parent
        metadata = json.loads((folder / "generated/conditioned.json").read_text())
        for key in ("raw_motion", "converted_csv"):
            bound = metadata[key]
            checked(Path(bound["path"]), bound["sha256"])
        (out / f"generated_{seed}").symlink_to(folder, target_is_directory=True)
    return write_new(
        out / "generation_results.json",
        dict(
            plan=artifact(out / "plan.json"),
            results=previous["results"],
            reused_generation=artifact(source / "generation_results.json"),
            new_neural_samples=0,
            reused_neural_samples=4,
            physics_steps=0,
            correction=(
                "separate authored-operator XML from native repair XML; "
                "raw samples and study specification unchanged"
            ),
        ),
    )


def author(out):
    import torch

    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import init_humanoid_fk
    from gear_sonic.dataset_generation.kimodo_motion_adapter import (
        load_kimodo_qpos_csv,
        qpos_to_sonic_motion_entry,
        save_sonic_motion_file,
    )
    from gear_sonic.dataset_generation.local_adaptation import local_crouch

    if torch.cuda.is_available():
        raise ValueError("candidate conversion must use CPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    plan = load_plan(out)
    results = json.loads((out / "generation_results.json").read_text())
    if results["plan"] != artifact(out / "plan.json") or len(results["results"]) != 4:
        raise ValueError("all four assigned raw generations required before matched repair")
    for ref in results["results"]:
        checked_json(ref)
    target = out / "candidates"
    target.mkdir(exist_ok=False)
    write_new(
        target / "started.json",
        dict(
            plan=artifact(out / "plan.json"),
            generation_results=artifact(out / "generation_results.json"),
        ),
    )
    neutral = canonical(load_kimodo_qpos_csv(plan["neutral_csv"]["path"]))
    names, limits, _ = native_model(plan["native_mjcf"]["path"])
    fk = init_humanoid_fk()

    def native(qpos):
        entry = qpos_to_sonic_motion_entry(qpos, source_fps=30)
        with torch.no_grad():
            values = fk.fk_batch(
                torch.as_tensor(entry["pose_aa"])[None],
                torch.as_tensor(entry["root_trans_offset"])[None],
                return_full=True,
                fps=30,
                target_fps=50,
                interpolate_data=True,
            )
        values = {
            k: v.detach().cpu().numpy()[0] for k, v in values.items() if isinstance(v, torch.Tensor)
        }
        return entry, values

    _, baseline = native(neutral)
    rows = []
    for arm, identities in (("generated", SEEDS), ("authored", WINDOWS)):
        for i, identity in enumerate(identities):
            candidate_id = f"{arm}_{i:02d}"
            raw_ref, operator_report = None, None
            if arm == "generated":
                raw_ref = artifact(out / f"generated_{identity}/generated/conditioned.csv")
                raw = canonical(load_kimodo_qpos_csv(raw_ref["path"]))
            else:
                config = plan["shared_operator"]
                raw, report = local_crouch(
                    neutral,
                    0.55,
                    target_drop_m=0.085,
                    window=identity,
                    ramp=config["ramp"],
                    range_keep=config["range_keep"],
                    max_excursion=config["max_excursion_rad"],
                    waist_use_fraction=config["waist_use_fraction"],
                    mjcf_path=plan["authored_operator_mjcf"]["path"],
                )
                operator_report = asdict(report)
            repaired, projected, weight = project_and_splice(neutral, raw, limits)
            motion, loaded = native(repaired)
            save_sonic_motion_file(
                target / f"{candidate_id}.pkl", motion_key=candidate_id, motion_entry=motion
            )
            np.savez_compressed(
                target / f"{candidate_id}_stages.npz",
                raw=raw,
                projected=projected,
                repaired=repaired,
                candidate_weight=weight,
                neutral=neutral,
                joint_names=np.asarray(names),
            )
            np.savez_compressed(target / f"{candidate_id}_native.npz", **loaded)
            joint = abs(loaded["dof_pos"] - baseline["dof_pos"]).max(axis=1)
            root = np.linalg.norm(
                loaded["global_translation"][:, 0] - baseline["global_translation"][:, 0], axis=1
            )
            checks = [
                dict(
                    tick=t,
                    joint_jump_rad=float(joint[t]),
                    root_jump_m=float(root[t]),
                    position_guard=bool(joint[t] <= 0.05 and root[t] <= 0.01),
                )
                for t in (15, 50, 265)
            ]
            row = dict(
                candidate_id=candidate_id,
                arm=arm,
                seed_or_window=identity,
                raw_generation_csv=raw_ref,
                authored_operator_report=operator_report,
                raw_limits=limit_report(raw[:, 7:], limits),
                repaired_limits=limit_report(repaired[:, 7:], limits),
                motion=artifact(target / f"{candidate_id}.pkl"),
                stages=artifact(target / f"{candidate_id}_stages.npz"),
                native_reference=artifact(target / f"{candidate_id}_native.npz"),
                reference_guard=checks,
                loaded_frames=len(loaded["global_translation"]),
                physically_qualified=False,
            )
            write_new(
                target / f"{candidate_id}.pkl.manifest.json",
                dict(plan=artifact(out / "plan.json"), shared_neutral=plan["neutral_csv"], **row),
            )
            rows.append(row)
            print(json.dumps(dict(repaired=candidate_id, guards=checks)), flush=True)
    return write_new(
        out / "candidate_results.json",
        dict(
            plan=artifact(out / "plan.json"),
            rows=rows,
            proposed_candidates_per_arm=4,
            physics_steps=0,
            scope=(
                "common repaired references; physical qualification and task coverage "
                "not inferred from kinematics"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "generate", "author", "reuse-generated"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = (
        reuse_generated(args.out, args.source)
        if args.command == "reuse-generated"
        else globals()[args.command](args.out)
    )
    print(json.dumps(dict(result=result)))
