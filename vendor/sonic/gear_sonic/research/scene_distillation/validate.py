"""Read-only CPU validation of frozen dataset structure and a SONIC decoder interface.

Writes a new receipt directory. No simulator is imported and no optimizer runs.
Checkpoint inspection establishes compatibility, not tracking/scene qualification.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch.nn import functional as F

from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha, write_new
from gear_sonic.research.hindsight_training.student import FrozenSonicDecoder
from gear_sonic.research.scene_distillation.contracts import audit_scene_pairing
from gear_sonic.research.scene_distillation.cvae import VariationalNavigationStudent
from gear_sonic.research.scene_distillation.losses import variational_imitation_loss
from gear_sonic.research.scene_distillation.policy import NavigationTokenStudent


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(91146)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    failures = []
    for row in manifest["files"]:
        path = (args.dataset / row["path"]).resolve()
        if not path.is_relative_to(args.dataset.resolve()):
            failures.append({"path": row["path"], "reason": "outside dataset"})
        elif not path.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
        elif sha(path) != row["sha256"]:
            failures.append({"path": row["path"], "reason": "hash mismatch"})
    write_new(
        args.output / "dataset-integrity.json",
        {
            "manifest_sha256": sha(args.dataset / "manifest.json"),
            "files_checked": len(manifest["files"]),
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError("Dataset integrity failed; see preserved receipt")
    catalog = json.loads((args.dataset / "catalog.json").read_text())
    assignments = json.loads((args.dataset / "assignments.json").read_text())
    motions = {row["id"]: row for row in catalog}
    scenes = []
    ordered_cells = defaultdict(dict)
    assignment_rows = []
    unique_obstacles = set()
    for assignment in assignments:
        path = args.dataset / "scenes" / (assignment["scene_id"] + ".json")
        scene = json.loads(path.read_text())
        labels = scene["physical_labels"]
        cells = scene["sampling"]["uniqueCells"]
        if scene["motion_id"] != assignment["motion_id"]:
            raise ValueError("Scene/assignment motion mismatch")
        if scene["split"] != motions[scene["motion_id"]]["split"]:
            raise ValueError("Scene/motion split mismatch")
        requested = assignment["requested_obstacles"]
        ordered_cells[scene["motion_id"]][requested] = cells
        unique_obstacles.update((scene["motion_id"], cell) for cell in cells)
        scenes.append(
            {
                "motion_id": scene["motion_id"],
                "requested_obstacles": requested,
                "cells": cells,
                "teacher_eligible": scene["teacher_eligible"],
                "teacher_actions_present": labels["teacher_actions"] is not None,
            }
        )
        assignment_rows.append(
            dict(
                assignment,
                scene_sha256=sha(path),
                teacher_actions_present=labels["teacher_actions"] is not None,
            )
        )
    groups = defaultdict(set)
    for row in catalog:
        groups[row["split"]].add(row["group"])
    overlap = sorted(groups["train"] & groups["development"])
    audit = dict(
        audit_scene_pairing(scenes),
        splits=dict(Counter(r["split"] for r in catalog)),
        categories=dict(Counter(r["category"] for r in catalog)),
        reference_frames=sum(r["frames"] for r in catalog),
        prompt_ancestry_groups={k: len(v) for k, v in groups.items()},
        cross_split_prompt_group_overlap=overlap,
        prefix_3_of_5_pairs=sum(p.get(3) == p.get(5, [])[:3] for p in ordered_cells.values()),
        placements=sum(len(s["cells"]) for s in scenes),
        unique_motion_candidate_pairs=len(unique_obstacles),
        kinematic_contrast_scenes=sum(a["kinematic_contrast_obstacles"] > 0 for a in assignments),
        reported_scene_physics_runs=sum(a["physics_runs"] for a in assignments),
        unrun_scenes=sum(a["unrun"] for a in assignments),
        physical_passes=None,
        physical_failures=None,
        p_value=None,
    )
    write_new(args.output / "dataset-audit.json", audit)
    with (args.output / "assignment-outcomes.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        for row in assignment_rows:
            writer.writerow({k: "NA" if v is None else v for k, v in row.items()})

    checkpoint_sha = sha(args.checkpoint)
    policy = load_release_checkpoint(args.checkpoint)["policy_state_dict"]
    decoder = FrozenSonicDecoder(policy)
    proprio = torch.randn(8, 930)
    tokens = torch.randint(-16, 16, (8, 64)).float() / 16
    expected = torch.cat([tokens, proprio], -1)
    prefix = "actor_module.decoders.g1_dyn.module."
    for layer in range(0, 13, 2):
        expected = F.linear(
            expected, policy[f"{prefix}{layer}.weight"], policy[f"{prefix}{layer}.bias"]
        )
        if layer < 12:
            expected = F.silu(expected)
    actual = decoder(tokens, proprio)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    model = VariationalNavigationStudent()
    observation = {
        "proprio": proprio,
        "start_goal_body": torch.randn(8, 6),
        "obstacles_body": torch.randn(8, 5, 15),
        "obstacle_mask": torch.ones(8, 5, dtype=torch.bool),
    }
    posterior = model.posterior_step(
        observation, torch.randn(8, 1645), torch.randn(8, 640), epsilon=torch.randn(8, 32)
    )
    loss = variational_imitation_loss(posterior, proprio, tokens, actual, decoder, beta=0.01)
    loss["loss"].backward()
    gradients = {
        name: sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        for name, module in (
            ("prior", model.prior),
            ("posterior", model.posterior),
            ("adapter", model.token_adapter),
        )
    }
    assert all(v > 0 for v in gradients.values())
    assert all(p.grad is None and not p.requires_grad for p in decoder.parameters())
    if sha(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("Checkpoint changed during read-only validation")
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "CPU_INTERFACE_CHECKS_PASSED",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "device": "cpu",
        },
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_role": "Inspected in-flight teacher checkpoint; not selected or qualified",
        "decoder_parameter_count": sum(p.numel() for p in decoder.parameters()),
        "decoder_parity_max_abs_error": float((actual - expected).abs().max()),
        "deterministic_trainable_parameters": sum(
            p.numel() for p in NavigationTokenStudent().parameters()
        ),
        "variational_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "variational_prior_acting_parameters": sum(
            p.numel() for n, p in model.named_parameters() if not n.startswith("posterior.")
        ),
        "gradient_check_nonzero": {k: v > 0 for k, v in gradients.items()},
        "decoder_frozen": True,
        "synthetic_rows": 8,
        "seed": 91146,
        "new_student_optimizer_updates": 0,
        "new_physics_steps": 0,
        "gpu_measurements": None,
        "wall_seconds": time.perf_counter() - started,
        "physical_navigation_success": None,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_new(args.output / "interface-validation.json", report)
    print(json.dumps({"dataset": audit, "interface": report}, indent=2))


if __name__ == "__main__":
    main()
