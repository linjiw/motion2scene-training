#!/usr/bin/env python3
"""Acceptance checks for one review evaluation run (CPU + MuJoCo kinematics only)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_common import (  # noqa: E402
    DEFAULT_REVIEW, PRIOR_PREVIOUS8000_DEV, SPLITS, Run, load_metrics, split_keys,
)

FK_FAIL_MM = 20.0
FK_WARN_MM = 5.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("split", choices=sorted(SPLITS))
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--allow-partial", action="store_true", help="synthetic fixtures only")
    args = parser.parse_args()
    run_root = args.review / "eval" / args.arm / f"{args.split}{args.suffix}"
    metrics_path = run_root / "metrics" / "metrics_eval.json"
    errors, warnings, report = [], [], {"run": f"{args.arm}/{args.split}{args.suffix}"}

    receipt = json.loads((run_root / "receipt.json").read_text())
    report["receipt"] = receipt
    if receipt.get("exit_code") != 0:
        errors.append(f"exit_code {receipt.get('exit_code')}")
    if not metrics_path.exists():
        errors.append("metrics_eval.json missing (os._exit(0) does not prove success)")
        return finish(run_root, report, errors, warnings)
    metrics = load_metrics(metrics_path)
    keys = [str(k) for k in metrics["motion_keys"]]
    expected = split_keys(args.split)
    report["motions"] = len(keys)
    if keys != expected and not (args.allow_partial and set(keys) <= set(expected)):
        errors.append(f"motion_keys differ from sorted {args.split} folder ({len(keys)} vs {len(expected)})")
    report["completed"] = int(sum(not t for t in metrics["terminated"]))
    report["mean_progress"] = float(np.mean(metrics["progress"]))

    has_pose = all((run_root / "metrics" / f"{k}.pose.npz").exists() for k in keys)
    missing_native = [k for k in keys if not (run_root / "metrics" / f"{k}.npz").exists()]
    if missing_native:
        errors.append(f"native npz missing: {missing_native[:5]}")
    report["pose_capture"] = has_pose
    if has_pose and not missing_native:
        from g1_scene import G1Scene, fk_check

        scene = G1Scene()
        per_motion = {}
        for index, key in enumerate(keys):
            run = Run(args.review, args.arm, f"{args.split}{args.suffix}", key)
            p = run.pose
            row = {"T": run.T, "valid": run.valid, "completed": run.completed}
            if run.T != len(p["robot_root_pos"]):
                errors.append(f"{key}: pose length {len(p['robot_root_pos'])} != native {run.T}")
            if run.completed and run.T != run.n - 1:
                errors.append(f"{key}: completed but T={run.T} != motion_num_steps-1={run.n - 1}")
            if bool(p["native_terminated"]) != bool(metrics["terminated"][index]) or abs(
                float(p["native_progress"]) - float(metrics["progress"][index])
            ) > 1e-9:
                errors.append(f"{key}: pose native flags disagree with metrics_eval.json")
            if not run.joint_order_consistent:
                errors.append(f"{key}: robot joint names and motion-lib dof table disagree")
            frames = np.asarray(p["ref_frame"][: run.valid])
            row["ref_offset"] = run.ref_offset
            if run.valid and not np.array_equal(frames, np.arange(run.valid) + run.ref_offset):
                errors.append(f"{key}: ref_frame is not k+offset inside the valid prefix")
            if run.ref_offset != 1:
                warnings.append(f"{key}: reference offset {run.ref_offset} (expected 1)")
            idx = np.clip(p["ref_frame"][: run.T].astype(int), 0, run.F - 1)
            ref_gap = float(np.abs(p["ref_root_pos"][: run.T] - p["ref_track_root_pos"][idx]).max())
            row["ref_track_alignment_m"] = ref_gap
            if ref_gap > 1e-5:
                errors.append(f"{key}: captured reference != reference track ({ref_gap:.2e} m)")
            pelvis_gap = float(np.abs(
                (run.tracked[: run.valid, 0] - run.origin) - p["robot_root_pos"][: run.valid]
            ).max()) if run.valid else 0.0
            row["root_vs_native_pelvis_m"] = pelvis_gap
            if pelvis_gap > 1e-4:
                errors.append(f"{key}: robot root != native pelvis ({pelvis_gap:.2e} m)")
            if not run.completed and run.term_failure_frame != run.valid:
                warnings.append(
                    f"{key}: first failing term frame {run.term_failure_frame} != "
                    f"round(progress*n) {run.valid}"
                )
            row["failure_terms"] = run.failure_terms
            if run.completed:
                native = float(metrics["mpjpe_g"][index])
                prefix = float(run.err_global_mm[: run.valid].mean())
                row["native_vs_prefix_mpjpe_g_mm"] = abs(native - prefix)
                if abs(native - prefix) > 0.5:
                    warnings.append(f"{key}: native mpjpe_g {native:.2f} vs recomputed {prefix:.2f}")
            fk = fk_check(scene, run)
            row.update(fk)
            for name in ("robot_fk_max_mm", "ghost_fk_max_mm"):
                value = fk[name]
                if value is not None and value > FK_FAIL_MM:
                    errors.append(f"{key}: {name} {value:.1f} mm")
                elif value is not None and value > FK_WARN_MM:
                    warnings.append(f"{key}: {name} {value:.1f} mm")
            per_motion[key] = row
        report["per_motion"] = per_motion

    if args.arm == "previous8000" and args.split == "development" and PRIOR_PREVIOUS8000_DEV.exists():
        prior = load_metrics(PRIOR_PREVIOUS8000_DEV)
        same_keys = [str(k) for k in prior["motion_keys"]] == keys
        term_match = same_keys and list(prior["terminated"]) == list(metrics["terminated"])
        prog_gap = float(np.max(np.abs(np.asarray(prior["progress"]) - np.asarray(metrics["progress"])))) if same_keys else None
        report["reproduction_gate"] = {
            "recorded_completed": int(sum(not t for t in prior["terminated"])),
            "recorded_mean_progress": float(np.mean(prior["progress"])),
            "this_completed": report["completed"],
            "this_mean_progress": report["mean_progress"],
            "terminated_identical": term_match,
            "max_progress_abs_diff": prog_gap,
            "per_motion_outcome_agreement": int(sum(
                a == b for a, b in zip(prior["terminated"], metrics["terminated"]))) if same_keys else None,
            "verdict": "exact" if term_match and prog_gap is not None and prog_gap < 1e-6 else "differs",
        }
        if abs(report["completed"] - report["reproduction_gate"]["recorded_completed"]) > 3:
            errors.append("previous8000/development differs from the recorded baseline by >3 "
                          "completions: stop and run the native-callback control before continuing")
    return finish(run_root, report, errors, warnings)


def finish(run_root, report, errors, warnings):
    report["errors"], report["warnings"] = errors, warnings
    report["accepted"] = not errors
    (run_root / "check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report.get(k) for k in (
        "run", "accepted", "motions", "completed", "mean_progress", "reproduction_gate")}, indent=2))
    for message in errors:
        print("ERROR", message)
    for message in warnings[:20]:
        print("WARN", message)
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
