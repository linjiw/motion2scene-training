#!/usr/bin/env python3
"""Register new full schedule evidence under an explicit external-contact criterion."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from hallucination.run_approved_manifest import free_gpu_mib  # noqa: E402
import motion2scene_qualify_long_schedules as strict  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (  # noqa: E402
    EXTERNAL_PATHS,
    audit_environment_contacts,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    make_verified_registry,
    validate_evidence,
)

NEUTRAL_SELF_PAIRS = [
    ["left_hip_roll_link", "left_wrist_yaw_link"],
    ["right_hip_roll_link", "right_wrist_yaw_link"],
]


def prepare(args):
    parent = json.loads(args.template.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    cells = copy.deepcopy(parent["cells"])
    for cell in cells:
        cell["output"] = str(args.out.resolve() / "rollouts" / cell["cell_id"])
        cell["role"] = "new_physical_qualification_with_explicit_environment_contact_criterion"
        cell["hydra_overrides"] = [
            value.replace(
                "motion2scene_long_schedule_execution.LongScheduleEnvCfg",
                "motion2scene_environment_contact_execution.EnvironmentContactEnvCfg",
            ).replace(
                "motion2scene_long_schedule_execution.LongScheduleRecorderCfg",
                "motion2scene_environment_contact_execution.EnvironmentContactRecorderCfg",
            )
            for value in cell["hydra_overrides"]
        ]
        cell["command"][cell["command"].index("--out") + 1] = cell["output"]
        cell["command"][-1] = " ".join(cell["hydra_overrides"])
    sources = closure(
        [
            Path(__file__),
            ROOT
            / "gear_sonic/dataset_generation/hallucination/motion2scene_environment_contact_execution.py",
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
    body_names = cells[0]["hydra_overrides"][0].split("=[", 1)[1][:-1].split(",")
    manifest = copy.deepcopy(parent)
    manifest.update(
        schema="motion2scene_environment_contact_qualification_v1",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        template=artifact(args.template),
        dependencies=dependencies,
        cells=cells,
        previous_strict_results=artifact(args.template.parent / "result.json"),
        neutral_counterpart_diagnostic=artifact(args.diagnostic),
        contact_criterion={
            "name": "explicit_environment_normal_contact_v1",
            "all_subject_bodies": body_names,
            "all_counterpart_paths": [f"/World/envs/env_0/Robot/{x}" for x in body_names]
            + EXTERNAL_PATHS,
            "undesired_environment_contact_max_n": 1.0,
            "pair_sum_to_net_tolerance_n": 0.001,
            "allowed_external_contact": "left/right ankle-roll links against floor only",
            "self_contact": (
                "Every internal pair is retained and audited separately. "
                "Self-contact is not classified as an environment collision. "
                "Unexpected pairs and changes in magnitude are explicitly reported."
            ),
            "declared_neutral_self_pairs": NEUTRAL_SELF_PAIRS,
            "prior_failures": "All original strict nonfoot-contact failures remain unchanged",
            "replay": "All three schedules receive new physical executions with the full pair instrumentation",
            "other_checks": (
                "Original whole-horizon transition, stability, matched-prefix, recovery and source checks"
            ),
            "geometry_scope": "Empty room with floor/four walls and disabled beam; no obstacle passage claim",
            "measurement_limits": (
                "Native measured normal forces; tangential friction is not measured by this array"
            ),
        },
        analysis_outputs={
            "result.json": "Counterfactual application of old strict nonfoot gate to these NEW physical captures",
            "environment_result.json": "Declared new environment-contact qualification evidence",
            "registry.json": "Written only if all three fresh branches satisfy the complete evidence contract",
        },
    )
    for ref in [
        manifest["request"],
        manifest["geometry"],
        manifest["neutral_counterpart_diagnostic"],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(args.out / "manifest.json", manifest)
    print(json.dumps({"manifest": artifact(args.out / "manifest.json"), "branches": len(cells)}))


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if not (out / "result.json").exists():
        strict.analyze(out)
    request = json.loads(Path(manifest["request"]["path"]).read_text())
    original = json.loads((out / "result.json").read_text())
    rows = []
    evidence_refs = []
    neutral_self = None
    for row in original["rows"]:
        cell = next(c for c in manifest["cells"] if c["cell_id"] == row["cell_id"])
        path = Path(cell["output"]) / "trajectories"
        evidence_path = path / "qualification_evidence.json"
        if not evidence_path.exists():
            rows.append(row)
            continue
        with np.load(path / "environment_pair_contacts.npz") as handle:
            pairs = {name: handle[name] for name in handle.files}
        with np.load(path / "all_body_contacts.npz") as handle:
            net = {name: handle[name] for name in handle.files}
        mapping = json.loads((path / "environment_contact_mapping.json").read_text())
        audit = audit_environment_contacts(
            pairs,
            net,
            mapping,
            manifest["expected_physics_steps"],
            threshold_n=manifest["contact_criterion"]["undesired_environment_contact_max_n"],
            neutral_self_pairs=manifest["contact_criterion"]["declared_neutral_self_pairs"],
        )
        if row["cell_id"] == "neutral":
            neutral_self = {
                tuple(value["body_pair"]): value["maximum_force_n"]
                for value in audit.get("self_contacts", [])
            }
        audit["self_contact_magnitude_change_vs_fresh_neutral_n"] = (
            [
                {
                    "body_pair": value["body_pair"],
                    "maximum_force_difference_n": value["maximum_force_n"]
                    - neutral_self.get(tuple(value["body_pair"]), 0.0),
                }
                for value in audit.get("self_contacts", [])
            ]
            if neutral_self is not None
            else None
        )
        write_new(path / "environment_contact_audit.json", audit)
        evidence = json.loads(evidence_path.read_text())
        geometric = json.loads((path / "geometry_audit.json").read_text())
        evidence["checks"]["all_contacts_audited"] = (
            audit["complete_synchronized_streams"]
            and audit["no_undesired_measured_contact"]
            and geometric["outer_wall_clearance_m"] >= 0.5
        )
        evidence["contact_criterion"] = manifest["contact_criterion"]
        evidence["artifacts"].update(
            {
                "environment_pair_contacts": artifact(path / "environment_pair_contacts.npz"),
                "environment_mapping": artifact(path / "environment_contact_mapping.json"),
                "environment_contact_audit": artifact(path / "environment_contact_audit.json"),
                "qualification_registration": artifact(out / "manifest.json"),
                "strict_counterfactual_evidence": artifact(evidence_path),
            }
        )
        failure = None
        try:
            validate_evidence(request, evidence)
        except ValueError as error:
            failure = str(error)
        destination = path / "environment_qualification_evidence.json"
        write_new(destination, evidence)
        evidence_refs.append(artifact(destination))
        rows.append(
            {
                **row,
                "qualified": failure is None,
                "qualification_error": failure,
                "checks": evidence["checks"],
                "evidence": artifact(destination),
                "environment_contact_audit": artifact(path / "environment_contact_audit.json"),
                "maximum_undesired_environment_force_n": audit.get(
                    "maximum_undesired_environment_force_n"
                ),
                "unexpected_self_contact_pairs": audit.get("unexpected_self_contact_pairs"),
            }
        )
    registry_ref = None
    if len(rows) == len(manifest["cells"]) and all(row["qualified"] for row in rows):
        registry = make_verified_registry(request, evidence_refs)
        write_new(out / "registry.json", registry)
        registry_ref = artifact(out / "registry.json")
    write_new(
        out / "environment_result.json",
        {
            "schema": "motion2scene_environment_contact_qualification_result_v1",
            "manifest": artifact(out / "manifest.json"),
            "rows": rows,
            "registry": registry_ref,
            "physics_steps": original["physics_steps"],
            "unmeasured_failed_attempts": original["unmeasured_failed_attempts"],
            "old_strict_application_to_new_captures": artifact(out / "result.json"),
            "original_strict_failures": manifest["previous_strict_results"],
        },
    )
    print(
        json.dumps(
            {
                "rows": [{"id": r["cell_id"], "qualified": r["qualified"]} for r in rows],
                "registry": registry_ref,
            }
        )
    )


def run(out):
    manifest = json.loads((out / "manifest.json").read_text())
    # A resource-only pause creates no attempt. Existing completed cells are never rerun.
    if not any(Path(c["output"]).exists() for c in manifest["cells"]):
        readings = []
        for index in range(7):
            free = free_gpu_mib()
            active = subprocess.run(
                ["pgrep", "-af", "[e]val_agent_trl.py"], capture_output=True, text=True, check=False
            ).stdout.strip()
            readings.append({"elapsed_s": index * 5, "free_gpu_mib": free, "active_eval": active})
            print(json.dumps(readings[-1]), flush=True)
            if free < manifest["limits"]["minimum_free_gpu_mib"] or active:
                raise RuntimeError("resource-only preflight pause, no branch launched")
            if index < 6:
                time.sleep(5)
        write_new(out / "preflight.json", readings)
    strict.run(out)
    analyze(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=strict.DATA / "m2s-long-schedule-qualification-v3/manifest.json",
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=strict.DATA / "m2s-neutral-contact-counterpart-diagnostic-v1/result.json",
    )
    args = parser.parse_args()
    {
        "prepare": lambda: prepare(args),
        "run": lambda: run(args.out),
        "analyze": lambda: analyze(args.out),
    }[args.mode]()
