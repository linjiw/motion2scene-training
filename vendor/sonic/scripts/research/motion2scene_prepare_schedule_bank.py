#!/usr/bin/env python3
"""Bind every declared forced schedule to a fresh execution of the complete bank."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, write_new  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    checked_artifact,
    definition_digest,
    validate_request,
)


def prepare(args):
    request = json.loads(args.request.read_text())
    bank = validate_request(request)
    parent = json.loads(args.template.read_text())
    for reference in request["references"]:
        for name in (
            "motion",
            "parent_motion",
            "prior_motion",
            "derivation_registration",
            "derivation_result",
            "boundary_diagnostic",
        ):
            if name in reference:
                checked_artifact(reference[name])
    checked_artifact(request["controller"])
    if request["controller"] != parent["implementation"]["checkpoint"]:
        raise ValueError("frozen controller changed between banks")
    if any(x["recovery_end_tick"] - x["return_tick"] < 15 for x in request["options"]):
        raise ValueError("full minimum15-tick recovery required")
    args.out.mkdir(parents=True, exist_ok=False)
    cells = []
    for option_id in bank.option_ids:
        option = next((x for x in request["options"] if x["option_id"] == option_id), None)
        entry_tick = (
            option["entry_tick"] if option else min(x["entry_tick"] for x in request["options"])
        )
        cell = copy.deepcopy(parent["cells"][0])
        cell.update(
            cell_id=option_id,
            forced_option_id=option_id,
            output=str(args.out.resolve() / "rollouts" / option_id),
            declared_schedule=option,
            encounter_action=int(option is not None),
            decision_time_s=entry_tick / 50,
            required_exact_prefix_frames=entry_tick,
            role="fresh_full_bank_complete_schedule_qualification",
            motion=request["references"][0]["motion"],
            alternate_motion=request["references"][1]["motion"],
        )
        replacements = {
            "++manager_env._target_=": (
                "gear_sonic.dataset_generation.hallucination."
                "motion2scene_schedule_bank_qualification_execution.ScheduleBankQualificationEnvCfg"
            ),
            "++manager_env.recorders.trajectory._target_=": (
                "gear_sonic.dataset_generation.hallucination."
                "motion2scene_schedule_bank_qualification_execution.ScheduleBankQualificationRecorderCfg"
            ),
            "++manager_env.config.qualification_request_path=": str(args.request.resolve()),
            "++manager_env.config.qualification_request_sha256=": artifact(args.request)["sha256"],
            "++manager_env.config.forced_option_id=": option_id,
            "++manager_env.config.encounter_action=": str(int(option is not None)),
            "++manager_env.config.decision_time_s=": str(entry_tick / 50),
            "++manager_env.config.expected_reference_frames=": str(bank.frame_count),
            "++manager_env.config.reactive_alternate_path=": request["references"][1]["motion"][
                "path"
            ],
        }
        cell["hydra_overrides"] = [
            value
            for value in cell["hydra_overrides"]
            if not any(value.startswith(prefix) for prefix in replacements)
        ]
        cell["hydra_overrides"] += [key + value for key, value in replacements.items()]
        cell["command"][cell["command"].index("--out") + 1] = cell["output"]
        cell["command"][cell["command"].index("--max-steps") + 1] = str(bank.frame_count)
        cell["command"][cell["command"].index("--motion") + 1] = cell["motion"]["path"]
        cell["command"][-1] = " ".join(cell["hydra_overrides"])
        cells.append(cell)
    sources = closure(
        [
            Path(__file__),
            ROOT / "scripts/research/motion2scene_qualify_environment_schedules.py",
            ROOT / "scripts/research/motion2scene_compare_entry_executions.py",
            ROOT
            / "gear_sonic/dataset_generation/hallucination/motion2scene_schedule_bank_qualification_execution.py",
        ]
    ) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    dependencies = []
    for path in sorted(sources):
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        dependencies.append({**artifact(path), "snapshot": artifact(destination)})
    manifest = copy.deepcopy(parent)
    manifest["contact_criterion"]["replay"] = (
        f"All {len(cells)} declared branches receive fresh physical executions with the full bank."
    )
    manifest["analysis_outputs"]["registry.json"] = (
        "Written only if neutral and every declared schedule satisfy complete fresh evidence."
    )
    manifest.update(
        schema="motion2scene_full_schedule_bank_qualification_v1",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        cells=cells,
        dependencies=dependencies,
        request=artifact(args.request),
        request_digest=definition_digest(request),
        template=artifact(args.template),
        bank_preparation=artifact(args.preparation),
        whole_execution_comparison_contract=artifact(args.comparison_contract),
        previous_online_registry=artifact(args.template.parent / "registry.json"),
        expected_actual_bank_shape=[len(request["references"]), bank.frame_count, 3],
        expected_recorded_control_steps=bank.frame_count - 1,
        expected_physics_steps=4 * (bank.frame_count - 1),
        expected_total_physics_steps=4 * (bank.frame_count - 1) * len(cells),
        qualification_scope=(
            "Every schedule is newly executed with identical full bank, seed8731, emptyroom, "
            "finite horizon, own-entry matchedprefix; prior3-bank evidence is not reused for promotion."
        ),
    )
    write_new(args.out / "manifest.json", manifest)
    print(json.dumps({"manifest": artifact(args.out / "manifest.json"), "branches": len(cells)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--comparison-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args())
