#!/usr/bin/env python3
"""Freeze and execute the 16 post-hoc matched command-audit cells only."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from bundle_motion2scene_sources import closure
from motion2scene_development_bank import DATA
from motion2scene_icra_eval_audit import execute
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

PROTOCOL = ROOT / "docs/motion2scene/SUBMISSION_COMMAND_AUDIT_V1.md"
PARENT = DATA / "m2s-icra-learning-v1"
NOMINAL = DATA / "m2s-icra-nominal-learning-v1"
DEFAULT = DATA / "m2s-submission-command-audit-v1"
EXPECTED = {
    (41001, "layout_02"),
    (41001, "layout_03"),
    (41001, "layout_06"),
    (41001, "layout_08"),
    (41001, "layout_09"),
    (41002, "layout_02"),
    (41002, "layout_09"),
    (41002, "layout_11"),
}


def prepare(out):
    """Select missing commands by availability, never by a desired outcome."""
    inputs = [NOMINAL / "comparison_144.json", PARENT / "comparison_540.json"]
    rows = [
        r
        for p in inputs
        for r in json.loads(p.read_text())["rows"]
        if r["suite"] == "traversal" and r["physics_seed"] == 8511
    ]
    keys = {(r["source"], r["layout"]) for r in rows}
    missing = {
        key
        for key in keys
        if not any((r["source"], r["layout"]) == key and r["action"] == 1 for r in rows)
    }
    assert len(keys) == 36 and missing == EXPECTED
    out.mkdir(exist_ok=False)
    policies = {}
    for action, name in ((0, "always_walk"), (1, "always_d040")):
        path = out / f"{name}.json"
        write_new(
            path,
            {
                "kind": "privileged_geometry",
                "selected_action": action,
                "audit_policy": name,
                "uses_geometry": False,
            },
        )
        policies[name] = artifact(path)
    refs = [artifact(PROTOCOL), *[artifact(p) for p in inputs]]
    refs += [artifact(p) for p in sorted(closure([Path(__file__)]))]
    master = json.loads((NOMINAL / "evaluation_master.json").read_text())
    templates = {}
    for b in master["batches"]:
        p = Path(b["directory"]) / "manifest.json"
        m = json.loads(p.read_text())
        c = m["cells"][0]
        templates[c["generation_seed"], c["layout"]] = (p, m)
    batches = []
    for index, key in enumerate(sorted(missing)):
        original, template = templates[key]
        m = copy.deepcopy(template)
        folder = out.with_name(out.name + f"-eval-{index:02d}")
        folder.mkdir()
        group = f"audit_{key[0]}_{key[1]}_p8511"
        cells = []
        for name, policy in policies.items():
            c = copy.deepcopy(template["cells"][0])
            c.update(
                cell_id=group + "_" + name,
                group_id=group,
                arm=name,
                role="post_hoc_matched_command_audit",
                policy=policy,
                output=str(folder / "rollouts" / (group + "_" + name)),
            )
            c["hydra_overrides"] = [v for v in c["hydra_overrides"] if "learned_policy_" not in v]
            c["hydra_overrides"] += [
                f"++manager_env.config.learned_policy_path={policy['path']}",
                f"++manager_env.config.learned_policy_sha256={policy['sha256']}",
            ]
            cells.append(c)
        m.update(
            experiment=group,
            purpose="Post-hoc missing matched-command audit",
            cells=cells,
            registered_predictions=artifact(PROTOCOL),
            original_nominal_manifest=artifact(original),
        )
        m["new_dependencies"] += refs + [artifact(original)] + list(policies.values())
        m["execution_policy"]["cost_ceiling"].update(rollouts=2, gpu_hours_contended=2 * 375 / 3600)
        m["execution_policy"][
            "timing_override"
        ] = "User-authorized validity audit; two matched cells; original rolling gates"
        m["authorization"]["note"] = "User submission-hardening brief permits targeted audit"
        write_new(folder / "manifest.json", m)
        batches.append(
            {
                "directory": str(folder),
                "assigned": 2,
                "manifest": artifact(folder / "manifest.json"),
            }
        )
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "registration.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "post_hoc": True,
            "references": refs,
            "policies": policies,
            "batches": batches,
            "cells": 16,
            "conditions": sorted(missing),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "scope": "Missing matched commands only; original primary results unchanged",
        },
    )


def run(out, preflight=False):
    registration = json.loads((out / "registration.json").read_text())
    for ref in registration["references"]:
        checked(Path(ref["path"]), ref["sha256"])
    historical = json.loads((NOMINAL / "comparison_144.json").read_text())["rows"]
    for b in registration["batches"]:
        folder = Path(b["directory"])
        checked(Path(b["manifest"]["path"]), b["manifest"]["sha256"])
        if not (folder / "admission.json").exists():
            execute(folder, preflight=preflight)
        if preflight:
            continue
        if not (folder / "admission.json").exists():
            return
        admission = json.loads((folder / "admission.json").read_text())
        assert admission["admitted"], folder
        result = json.loads(
            checked(Path(admission["result"]["path"]), admission["result"]["sha256"]).read_text()
        )
        repeated = next(r for r in result["rows"] if r["arm"] == "always_walk")
        previous = next(
            r
            for r in historical
            if r["source"] == repeated["source"]
            and r["layout"] == repeated["layout"]
            and r["arm"] == "uniform"
        )
        assert repeated["pass"] == previous["pass"], "Repeated-walk outcome mismatch"
        a, b = [
            json.loads(checked(Path(r["decision"]["path"]), r["decision"]["sha256"]).read_text())
            for r in (repeated, previous)
        ]
        fields = (
            "packet",
            "features",
            "state",
            "root_pos_w",
            "root_quat_w",
            "active_before",
            "phase_s",
        )
        assert all(a[k] == b[k] for k in fields), "Historical decision input mismatch"
        payloads = [
            load_reset_capture(checked(Path(r["trajectory"]["path"]), r["trajectory"]["sha256"]))
            for r in (repeated, previous)
        ]
        assert paired_prefix(*payloads, 0.3)["exact_match"], "Historical prefix mismatch"
        print(
            json.dumps(
                {
                    "batch": str(folder),
                    "admitted": True,
                    "outcomes": [r["pass"] for r in result["rows"]],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
    else:
        run(args.out, preflight=args.command == "preflight")
