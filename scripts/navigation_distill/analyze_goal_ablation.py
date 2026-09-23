"""Phase 0.4 goal/map-use readout for one navigation panel (CPU only, no Isaac).

Reads <panel>/<task_id>/{config.json, command.json, task/task-result.json, task/trace.npz},
the task JSON named by config.json and its sha-bound reference. For every episode it reports
where the pelvis ended relative to the start, the original goal, the rotated goal and the
reference endpoint.

- Goal rotation panels: applies the registered rule (docs/ROADMAP_20260923.md §8, 0.4): the
  adapter ignores the goal if its final XY stays within 0.5 m of the reference endpoint in
  >=70% of tasks. With --baseline (the unablated panel, same checkpoint and seed) it adds a
  paired sensitivity check; the registered verdict is reported unchanged.
- Zero-obstacle panels: descriptive paired comparison with --baseline; no registered gate.
- Unablated panels: the same position readout, e.g. to check whether the rule is attainable.

Usage: analyze_goal_ablation.py --panel STAGE [--baseline STAGE] [--output JSON]
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.navigation_ablation import (
    IGNORE_FRACTION,
    IGNORE_RADIUS_M,
    ObservationAblation,
    final_position_row,
    goal_use_decision,
    reference_endpoint,
)


def launch_seed(task_dir):
    command = task_dir / "command.json"
    if not command.exists():
        return None
    seeds = [a.split("=", 1)[1] for a in json.loads(command.read_text()) if a.startswith("++seed=")]
    return int(seeds[0]) if len(seeds) == 1 else None


def load_panel(stage):
    """Completed episodes keyed by task id, plus launched tasks without a result."""
    episodes, failures = {}, []
    for task_dir in sorted(p for p in Path(stage).iterdir() if (p / "config.json").exists()):
        config = json.loads((task_dir / "config.json").read_text())
        result_path = task_dir / "task" / "task-result.json"
        if not result_path.exists():
            failures.append(task_dir.name)
            continue
        result = json.loads(result_path.read_text())
        if sha(config["task_path"]) != result["task_sha256"]:
            raise ValueError(f"{task_dir.name}: task changed since the run")
        task = json.loads(Path(config["task_path"]).read_text())
        with np.load(task_dir / "task" / "trace.npz") as trace:
            roots = np.asarray(trace["root_xyz"], dtype=np.float64)
        if len(roots) != result["control_steps"]:
            raise ValueError(f"{task_dir.name}: trace and result lengths differ")
        ablation = ObservationAblation.from_config(config)
        recorded = result.get("navigation_ablation")
        if ablation.active != (recorded is not None):
            raise ValueError(f"{task_dir.name}: config flags and task-result record disagree")
        if recorded is not None and (
            recorded["goal_rotation_deg"] != ablation.goal_rotation_deg
            or recorded["zero_obstacles"] != ablation.zero_obstacles
            or not np.allclose(recorded["final_root_xyz"], roots[-1], atol=1e-6)
        ):
            raise ValueError(f"{task_dir.name}: navigation_ablation record does not match trace")
        episodes[task["task_id"]] = dict(
            task=task,
            ablation=ablation,
            result=result,
            final_root_xyz=roots[-1],
            reference_endpoint_xyz=reference_endpoint(task),
            student_sha256=config.get("student_sha256"),
            seed=launch_seed(task_dir),
        )
    return episodes, failures


def uniform(episodes, key, stage):
    values = {repr(e[key]) for e in episodes.values()}
    if len(values) > 1:
        raise ValueError(f"{stage}: tasks disagree on {key}")
    return next(iter(episodes.values()))[key] if episodes else None


def position_rows(episodes, goal_rotation_deg, radius_m):
    rows = {}
    for task_id, e in episodes.items():
        row = final_position_row(
            e["task"], e["final_root_xyz"], e["reference_endpoint_xyz"], goal_rotation_deg, radius_m
        )
        r = e["result"]
        row.update(
            has_obstacles=bool(e["task"]["obstacles"]),
            navigation_success=r["navigation_success"],
            stop_reason=r["stop_reason"],
            collision_free=r["collision_free"],
            fell=r["fell"],
            control_steps=r["control_steps"],
            reference_endpoint_equals_goal=bool(
                np.allclose(e["reference_endpoint_xyz"], e["task"]["goal_xyz"], atol=1e-6)
            ),
        )
        rows[task_id] = row
    return rows


def zero_obstacle_summary(rows, baseline_rows):
    def counts(table, ids):
        return dict(
            tasks=len(ids),
            successes=sum(table[t]["navigation_success"] for t in ids),
            contacts=sum(not table[t]["collision_free"] for t in ids),
            falls=sum(table[t]["fell"] for t in ids),
        )

    corridor = sorted(t for t in rows if rows[t]["has_obstacles"])
    summary = dict(
        gate=None,
        note="No registered gate for 0.4(b); descriptive only. Clear tasks see an unchanged "
        "observation, so at the baseline's seed they should reproduce it exactly.",
        all=counts(rows, sorted(rows)),
        corridor=counts(rows, corridor),
    )
    if baseline_rows is None:
        return summary
    paired = sorted(set(rows) & set(baseline_rows))
    shift = {
        t: float(np.linalg.norm(np.subtract(rows[t]["final_xy"], baseline_rows[t]["final_xy"])))
        for t in paired
    }
    clear = [t for t in paired if not rows[t]["has_obstacles"]]
    paired_corridor = [t for t in paired if rows[t]["has_obstacles"]]
    summary["baseline"] = dict(
        paired_tasks=len(paired),
        all=counts(baseline_rows, paired),
        corridor=counts(baseline_rows, paired_corridor),
        changed_outcomes=[
            t
            for t in paired
            if (rows[t]["navigation_success"], rows[t]["stop_reason"])
            != (baseline_rows[t]["navigation_success"], baseline_rows[t]["stop_reason"])
        ],
        clear_max_final_xy_shift_m=max((shift[t] for t in clear), default=None),
        corridor_median_final_xy_shift_m=(
            float(np.median([shift[t] for t in paired_corridor])) if paired_corridor else None
        ),
    )
    return summary


def commit():
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def analyze(panel, baseline=None, radius_m=IGNORE_RADIUS_M, fraction=IGNORE_FRACTION):
    episodes, failures = load_panel(panel)
    if not episodes:
        raise ValueError(f"{panel}: no completed task")
    ablation = uniform(episodes, "ablation", panel)
    student = uniform(episodes, "student_sha256", panel)
    seed = uniform(episodes, "seed", panel)
    rows = position_rows(episodes, ablation.goal_rotation_deg, radius_m)
    report = dict(
        schema="navigation_goal_map_ablation_readout_v1",
        panel=str(Path(panel).resolve()),
        ablation=ablation.as_record(),
        student_sha256=student,
        seed=seed,
        completed_tasks=len(rows),
        without_result=failures,  # process failure, or still running
        scoring="legacy scorer against the ORIGINAL goal (3-D <=0.25 m, 50-tick hold)",
        successes=sum(r["navigation_success"] for r in rows.values()),
        analysis_commit=commit(),
    )
    baseline_rows, warnings = None, []
    if baseline is not None:
        base_episodes, base_failures = load_panel(baseline)
        if uniform(base_episodes, "ablation", baseline).active:
            raise ValueError("The baseline panel must be unablated")
        if uniform(base_episodes, "student_sha256", baseline) != student:
            raise ValueError("Baseline and ablation panels use different checkpoints")
        base_seed = uniform(base_episodes, "seed", baseline)
        if base_seed != seed:
            warnings.append(
                f"baseline seed {base_seed} != panel seed {seed}: paired shifts compare "
                "different physics instances, not a deterministic replay"
            )
        baseline_rows = position_rows(base_episodes, ablation.goal_rotation_deg, radius_m)
        report["baseline"] = dict(
            panel=str(Path(baseline).resolve()),
            seed=base_seed,
            completed_tasks=len(baseline_rows),
            without_result=base_failures,
            successes=sum(r["navigation_success"] for r in baseline_rows.values()),
        )
    if not all(r["reference_endpoint_equals_goal"] for r in rows.values()):
        warnings.append("some tasks' reference endpoint differs from their goal")
    report["goal_use"] = goal_use_decision(rows, baseline_rows, radius_m, fraction)
    if ablation.goal_rotation_deg is None:
        report["goal_use"]["applicable"] = False
        report["goal_use"]["note_unrotated"] = (
            "Goal not rotated: this is the unablated position readout; 'adapter_ignores_goal' "
            "here only shows whether the rule could fire for a goal-blind adapter."
        )
    else:
        report["goal_use"]["applicable"] = True
    if ablation.zero_obstacles:
        report["zero_obstacles"] = zero_obstacle_summary(rows, baseline_rows)
    report["warnings"] = warnings
    report["tasks"] = rows
    if baseline_rows is not None:
        report["baseline_tasks"] = baseline_rows
    return report


def print_report(report):
    def cell(value, digits=2):
        return "-" if value is None else f"{value:.{digits}f}"

    print(f"panel {report['panel']}  ablation {report['ablation']}  seed {report['seed']}")
    print(f"{'task':<22} {'stop':<9} {'ok':<3} {'dRef':>6} {'dRot':>6} {'dStart':>6} "
          f"{'pOrig':>6} {'pRot':>6} {'hdg':>6}")
    for task_id, r in report["tasks"].items():
        print(
            f"{task_id:<22} {r['stop_reason']:<9} {'Y' if r['navigation_success'] else '-':<3} "
            f"{r['final_xy_distance_to_reference_endpoint_m']:>6.2f} "
            f"{r['final_xy_distance_to_rotated_goal_m']:>6.2f} "
            f"{r['final_xy_distance_to_start_m']:>6.2f} {r['progress_along_original']:>6.2f} "
            f"{r['progress_along_rotated']:>6.2f} "
            f"{cell(r['displacement_heading_vs_original_deg'], 0):>6}"
        )
    g = report["goal_use"]
    print(
        f"near reference endpoint {g['near_reference_endpoint']}/{g['tasks']} "
        f"({g['near_reference_endpoint_fraction']:.0%}); "
        f"near rotated goal {g['near_rotated_goal']}; "
        f"near start {g['near_start']}; adapter_ignores_goal={g['adapter_ignores_goal']} "
        f"(applicable={g['applicable']}); heading toward rotated goal "
        f"{g['heading_toward_rotated_goal']}/{g['heading_tasks']}"
    )
    if g["sensitivity"]:
        s = g["sensitivity"]
        print(
            f"baseline near reference {s['baseline_near_reference_endpoint']}/{s['paired_tasks']} "
            f"-> rule_attainable={s['rule_attainable']}; median shift vs baseline "
            f"{cell(s['median_final_xy_shift_vs_baseline_m'])} m; median heading turn "
            f"{cell(s['median_heading_turn_vs_baseline_deg'], 0)} deg "
            f"({s['heading_turned_with_goal']}/{s['heading_turn_tasks']} turned with the goal)"
        )
        print(
            f"paired (proposed, not registered): final XY within {g['radius_m']} m of baseline "
            f"{s['final_xy_within_radius_of_baseline']}/{s['paired_tasks']} -> "
            f"adapter_ignores_goal={s['paired_rule_proposed']['adapter_ignores_goal']}"
        )
    if "zero_obstacles" in report:
        print("zero_obstacles", json.dumps({k: v for k, v in report["zero_obstacles"].items()
                                            if k != "note"}))
    for warning in report["warnings"]:
        print("WARNING", warning)
    if report["without_result"]:
        print("launched without a result (failed or running):", *report["without_result"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--panel", type=Path, required=True, help="run_stage.sh nav stage directory")
    p.add_argument("--baseline", type=Path, help="unablated panel, same checkpoint and seed")
    p.add_argument("--radius-m", type=float, default=IGNORE_RADIUS_M)
    p.add_argument("--fraction", type=float, default=IGNORE_FRACTION)
    p.add_argument("--output", type=Path, help="write the full JSON readout here")
    a = p.parse_args()
    report = analyze(a.panel, a.baseline, a.radius_m, a.fraction)
    print_report(report)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
