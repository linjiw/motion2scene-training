"""Re-score recorded single-env scene episodes with an XY goal metric and back-fill root metrics.

Reads every `<stage>/<episode>/task/{task-result.json,trace.npz}` under --packet (following
directory symlinks, deduplicated by real path) plus the episode's config.json/command.json and
its task JSON. Nothing under --packet is written.

Metrics, all at the 50 Hz control rate of DirectSceneTaskCallback (trace rows are the
uncensored pre-reset measurements after each control step):

* legacy 3-D (reproduction check): an exact port of direct_context.score_navigation_task:
  3-D pelvis distance <= goal_tolerance_m and recorded speed <= terminal_speed_mps for
  hold_ticks consecutive rows anywhere in the trace, no row with undesired force > 1 N, and
  no fall (pelvis z < 0.25 m on the last row). It must reproduce every recorded field.
* XY (primary): XY pelvis distance <= goal_tolerance_m and speed <= terminal_speed_mps for
  hold_ticks consecutive rows; the first such hold must complete no later than the last
  recorded row, with no contact > 1 N and no fall on any row up to and including the
  completion row. This is what an evaluator that stops at the first XY hold would have
  recorded: rollouts are unchanged up to that row, and every legacy stop (3-D hold,
  contact, fall, deadline) happens at or after it.
* Speed: the trace records only the 3-D pelvis linear-velocity norm, an upper bound on the
  planar speed, so the primary XY flag is a lower bound on planar-speed success.
  `xy_fd_success` repeats it with planar speed from XY finite differences over one
  control step (row 0 keeps the recorded speed).
* `xy_whole_trace_success` keeps the legacy whole-trace semantics (hold anywhere, no contact
  or fall on any recorded row) with the XY distance.

Episodes with a takeover_tick > 0 (learner prefix, motor or rewound suffix) also get the
suffix flags: the recorded receipt (recovery.json `supported`, reentry.json
`outcomes.timely_suffix_supported`), its 3-D reproduction, and the XY suffix flag (the hold
counted from the takeover row, same first-hold contact/fall rule).

Usage: rescore_xy.py --packet workspace/nav-8192 --output workspace/phase0/rescore
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CONTROL_DT = 0.02
CONTACT_LIMIT_N = 1.0
FALL_HEIGHT_M = 0.25
# Unassisted policies from the first tick; the others mix a learner prefix with a motor or
# rewound-reference suffix (or are motor demonstrations collected as training rows).
EVALUATION_CALLBACKS = (
    "StoppingTeacherCallback",
    "FullMotorTaskCallback",
    "NavigationMotorCallback",
)
LEGACY_FIELDS = (
    "navigation_success",
    "goal_ever_reached",
    "terminal_hold",
    "collision_free",
    "fell",
    "max_hold_ticks",
    "final_goal_distance_m",
    "max_undesired_force_n",
    "control_steps",
)


def goal_distance(root_xyz, goal_xyz, metric):
    """Pelvis-to-goal distance per row: "xyz" (legacy 3-D) or "xy" (horizontal)."""
    root, goal = np.asarray(root_xyz), np.asarray(goal_xyz)
    if metric == "xyz":
        return np.linalg.norm(root - goal, axis=-1)
    if metric == "xy":
        return np.linalg.norm(root[:, :2] - goal[:2], axis=-1)
    raise ValueError(f"Unknown goal metric {metric!r}")


def planar_speed_fd(root_xyz, recorded_speed, dt=CONTROL_DT):
    """Planar pelvis speed from XY differences of consecutive rows; row 0 keeps the record."""
    root = np.asarray(root_xyz, dtype=np.float64)
    speed = np.asarray(recorded_speed, dtype=np.float64).copy()
    if len(root) > 1:
        speed[1:] = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1) / dt
    return speed


def runs(good):
    """Length of the consecutive-True run ending at each row."""
    out = np.zeros(len(good), dtype=np.int64)
    run = 0
    for i, valid in enumerate(good):
        run = run + 1 if valid else 0
        out[i] = run
    return out


def _check(task, root, speed, force):
    n = len(root)
    if root.shape != (n, 3) or speed.shape != (n,) or force.shape != (n,) or n < 1:
        raise ValueError("Invalid task evidence shape")
    if not all(np.isfinite(x).all() for x in (root, speed, force)):
        raise ValueError("Nonfinite task evidence")
    if (speed < 0).any() or (force < 0).any():
        raise ValueError("Speed and contact magnitude must be nonnegative")


def navigation_score(task, root_xyz, speed, undesired_force, *, fell, metric="xyz"):
    """direct_context.score_navigation_task with a selectable goal metric (xyz = legacy)."""
    root, speed, force = np.asarray(root_xyz), np.asarray(speed), np.asarray(undesired_force)
    _check(task, root, speed, force)
    distance = goal_distance(root, task["goal_xyz"], metric)
    good = (distance <= task["goal_tolerance_m"]) & (speed <= task["terminal_speed_mps"])
    best = int(runs(good).max())
    hold = best >= task["hold_ticks"]
    contacts = bool((force <= CONTACT_LIMIT_N).all())
    return {
        "navigation_success": bool(hold and contacts and not fell),
        "goal_ever_reached": bool((distance <= task["goal_tolerance_m"]).any()),
        "terminal_hold": bool(hold),
        "collision_free": contacts,
        "fell": bool(fell),
        "max_hold_ticks": best,
        "final_goal_distance_m": float(distance[-1]),
        "max_undesired_force_n": float(force.max()),
        "control_steps": len(root),
    }


def first_hold(task, root_xyz, speed, undesired_force, *, metric="xy", start=0):
    """Success as recorded by an evaluator that stops at the first hold.

    The hold counts rows from `start` (a takeover row) on. It succeeds only if no row
    0..completion has undesired force > 1 N or pelvis z < FALL_HEIGHT_M. Returns
    (success, completion_row or None, longest hold run from `start`).
    """
    root, speed, force = np.asarray(root_xyz), np.asarray(speed), np.asarray(undesired_force)
    _check(task, root, speed, force)
    if not 0 <= start < len(root):
        return False, None, 0
    distance = goal_distance(root[start:], task["goal_xyz"], metric)
    good = (distance <= task["goal_tolerance_m"]) & (speed[start:] <= task["terminal_speed_mps"])
    run = runs(good)
    done = np.flatnonzero(run >= task["hold_ticks"])
    if not len(done):
        return False, None, int(run.max())
    tick = start + int(done[0])
    safe = (force[: tick + 1] <= CONTACT_LIMIT_N).all() and (
        root[: tick + 1, 2] >= FALL_HEIGHT_M
    ).all()
    return bool(safe), tick, int(run.max())


def root_metrics(task, root_xyz, speed, undesired_force):
    """Goal errors, time to goal, path efficiency and pelvis height, from the trace alone."""
    root = np.asarray(root_xyz, dtype=np.float64)
    force = np.asarray(undesired_force)
    start, goal = np.asarray(task["start_xyz"], float), np.asarray(task["goal_xyz"], float)
    xy = goal_distance(root, goal, "xy")
    xyz = goal_distance(root, goal, "xyz")
    tol = task["goal_tolerance_m"]
    reach_xy = np.flatnonzero(xy <= tol)
    reach_xyz = np.flatnonzero(xyz <= tol)
    contact = np.flatnonzero(force > CONTACT_LIMIT_N)
    # The robot resets onto the task start pose; the path begins there.
    points = np.concatenate([start[None, :2], root[:, :2]])
    path = float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())
    straight = float(np.linalg.norm(goal[:2] - start[:2]))
    fd = planar_speed_fd(root, speed)
    return dict(
        final_xy_err_m=float(xy[-1]),
        min_xy_err_m=float(xy.min()),
        min_xy_err_row=int(xy.argmin()),
        first_xy_reach_row=int(reach_xy[0]) if len(reach_xy) else None,
        first_xy_reach_s=round((reach_xy[0] + 1) * CONTROL_DT, 4) if len(reach_xy) else None,
        final_3d_err_m=float(xyz[-1]),
        min_3d_err_m=float(xyz.min()),
        first_3d_reach_row=int(reach_xyz[0]) if len(reach_xyz) else None,
        final_z_minus_goal_z_m=float(root[-1, 2] - goal[2]),
        straight_xy_m=straight,
        path_xy_m=path,
        path_ratio=path / straight if straight > 1e-9 else None,
        final_displacement_xy_m=float(np.linalg.norm(root[-1, :2] - start[:2])),
        max_root_height_dev_m=float(np.abs(root[:, 2] - start[2]).max()),
        min_root_z_m=float(root[:, 2].min()),
        final_speed_mps=float(np.asarray(speed)[-1]),
        final_fd_planar_speed_mps=float(fd[-1]),
        first_contact_row=int(contact[0]) if len(contact) else None,
    )


def legacy_mismatches(recorded, reproduced):
    """Fields where the reproduction differs from the recorded task-result.json."""
    bad = []
    for key in LEGACY_FIELDS:
        a, b = recorded[key], reproduced[key]
        if isinstance(a, float) or isinstance(b, float):
            if not np.isclose(a, b, rtol=0, atol=1e-9):
                bad.append(f"{key}:{a}!={b}")
        elif a != b:
            bad.append(f"{key}:{a}!={b}")
    return bad


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(packet):
    """Unique episode dirs (real path) -> packet-relative aliases, the symlink-free one first."""
    packet = Path(os.path.realpath(packet))
    episodes = {}
    for dirpath, dirnames, filenames in os.walk(packet, followlinks=True):
        dirnames.sort()
        path = Path(dirpath)
        if path.name == "task" and "task-result.json" in filenames:
            episode = path.parent
            alias = episode.relative_to(packet).as_posix()
            episodes.setdefault(os.path.realpath(episode), []).append(alias)
    for real, aliases in episodes.items():
        aliases.sort(key=lambda a: (str(packet / a) != real, a))
    return episodes


def _command_value(command, prefix):
    for item in command:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return None


def rescore_episode(episode, aliases, packet):
    task_dir = episode / "task"
    recorded = json.loads((task_dir / "task-result.json").read_text())
    config = json.loads((episode / "config.json").read_text())
    command_path = episode / "command.json"
    command = json.loads(command_path.read_text()) if command_path.exists() else []
    task_path = Path(config["task_path"])
    task = json.loads(task_path.read_text())
    alias = aliases[0]
    panel, name = alias.rsplit("/", 1)
    row = dict(
        panel=panel,
        episode=name,
        aliases=";".join(a.rsplit("/", 1)[0] for a in aliases[1:]),
        task_id=task["task_id"],
        motion_id=task["motion_id"],
        variant=task["task_id"].rsplit("-", 1)[1],
        callback=(_command_value(command, "++callbacks.im_eval._target_=") or "").split(".")[-1],
        seed=_command_value(command, "++seed="),
        teacher_mode=bool(config.get("teacher_mode", False)),
        student_sha256=(config.get("student_sha256") or "")[:12],
        takeover_tick=config.get("takeover_tick"),
        deadline_ticks=task["deadline_ticks"],
        max_steps=config.get("max_steps"),
        stop_reason=recorded.get("stop_reason"),
        task_sha256_ok=sha256(task_path) == recorded.get("task_sha256"),
        trace="present",
    )
    trace_path = task_dir / "trace.npz"
    if not trace_path.exists():
        row["trace"] = "missing"
        row["legacy_3d_success"] = bool(recorded["navigation_success"])
        return row
    with np.load(trace_path) as trace:
        root = trace["root_xyz"]
        speed = trace["speed"]
        force = trace["undesired_force"]
    fell = bool(root[-1, 2] < FALL_HEIGHT_M)
    legacy = navigation_score(task, root, speed, force, fell=fell, metric="xyz")
    mismatch = legacy_mismatches(recorded, legacy)
    xy_ok, xy_row, xy_best = first_hold(task, root, speed, force, metric="xy")
    fd = planar_speed_fd(root, speed)
    fd_ok, fd_row, _ = first_hold(task, root, fd, force, metric="xy")
    whole = navigation_score(task, root, speed, force, fell=fell, metric="xy")
    row.update(
        legacy_3d_success=bool(recorded["navigation_success"]),
        legacy_repro_success=legacy["navigation_success"],
        legacy_repro_match=not mismatch,
        legacy_repro_mismatch=";".join(mismatch),
        legacy_max_hold_ticks=recorded["max_hold_ticks"],
        legacy_goal_ever_reached=bool(recorded["goal_ever_reached"]),
        collision_free=bool(recorded["collision_free"]),
        fell=bool(recorded["fell"]),
        max_undesired_force_n=float(recorded["max_undesired_force_n"]),
        control_steps=int(recorded["control_steps"]),
        xy_success=xy_ok,
        xy_hold_row=xy_row,
        xy_hold_s=round((xy_row + 1) * CONTROL_DT, 4) if xy_row is not None else None,
        xy_max_hold_ticks=xy_best,
        xy_fd_success=fd_ok,
        xy_fd_hold_row=fd_row,
        xy_whole_trace_success=whole["navigation_success"],
    )
    row.update(root_metrics(task, root, speed, force))
    tol = task["goal_tolerance_m"]
    in_xy = goal_distance(root, task["goal_xyz"], "xy") <= tol
    row["rows_xy_only_in_tol"] = int(
        (in_xy & (goal_distance(root, task["goal_xyz"], "xyz") > tol)).sum()
    )
    in_xy[0] = False  # row 0 has no finite difference
    row["_speed_in_tol"] = (speed[in_xy], fd[in_xy])
    switch = config.get("takeover_tick") or 0
    if switch > 0:
        receipt = None
        if (task_dir / "recovery.json").exists():
            receipt = bool(json.loads((task_dir / "recovery.json").read_text())["supported"])
        elif (task_dir / "reentry.json").exists():
            outcomes = json.loads((task_dir / "reentry.json").read_text())["outcomes"]
            receipt = bool(outcomes["timely_suffix_supported"])
        if switch < len(root):
            suffix = navigation_score(
                task, root[switch:], speed[switch:], force[switch:], fell=fell, metric="xyz"
            )
            repro = bool(suffix["navigation_success"] and (force <= CONTACT_LIMIT_N).all())
        else:
            repro = False
        row.update(
            legacy_suffix_supported=receipt,
            legacy_suffix_repro=repro,
            xy_suffix_supported=first_hold(task, root, speed, force, metric="xy", start=switch)[0],
        )
    return row


COLUMNS = [
    "panel", "episode", "aliases", "task_id", "motion_id", "variant", "callback", "seed",
    "teacher_mode", "student_sha256", "takeover_tick", "deadline_ticks", "max_steps",
    "control_steps", "stop_reason", "trace", "task_sha256_ok",
    "legacy_3d_success", "legacy_repro_success", "legacy_repro_match", "legacy_repro_mismatch",
    "xy_success", "xy_fd_success", "xy_whole_trace_success",
    "legacy_suffix_supported", "legacy_suffix_repro", "xy_suffix_supported",
    "legacy_max_hold_ticks", "xy_max_hold_ticks", "xy_hold_row", "xy_hold_s", "xy_fd_hold_row",
    "legacy_goal_ever_reached", "collision_free", "fell", "max_undesired_force_n",
    "first_contact_row", "final_xy_err_m", "min_xy_err_m", "min_xy_err_row",
    "first_xy_reach_row", "first_xy_reach_s", "final_3d_err_m", "min_3d_err_m",
    "first_3d_reach_row", "rows_xy_only_in_tol", "final_z_minus_goal_z_m", "straight_xy_m",
    "path_xy_m", "path_ratio",
    "final_displacement_xy_m", "max_root_height_dev_m", "min_root_z_m", "final_speed_mps",
    "final_fd_planar_speed_mps",
]


def infra_failures(stage):
    """PROCESS_FAILED lines in a stage's results.txt (infra failures, later re-launched)."""
    path = stage / "results.txt"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if "PROCESS_FAILED" in line)


def xy_outcome(row):
    """XY outcome class: success, contact_or_fall, reached_not_held or never_reached."""
    if row["xy_success"]:
        return "success"
    if not row["collision_free"] or row["fell"]:
        return "contact_or_fall"
    return "reached_not_held" if row["first_xy_reach_row"] is not None else "never_reached"


def _count(rows, key):
    return sum(1 for r in rows if r.get(key))


def _median(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else None


def summarize(rows, packet, feasible):
    panels = {}
    for r in rows:
        panels.setdefault(r["panel"], []).append(r)
    table = []
    for panel, items in sorted(panels.items()):
        scored = [r for r in items if r["trace"] == "present"]
        sub = [r for r in scored if r["task_id"] in feasible]
        suffix = [r for r in scored if r.get("legacy_suffix_supported") is not None]
        entry = dict(
            panel=panel,
            aliases=sorted({a for r in items for a in r["aliases"].split(";") if a}),
            episodes=len(items),
            traces_missing=len(items) - len(scored),
            infra_failures_rerun=infra_failures(packet / panel),
            legacy_3d=_count(scored, "legacy_3d_success"),
            legacy_repro_match=_count(scored, "legacy_repro_match"),
            xy=_count(scored, "xy_success"),
            xy_fd=_count(scored, "xy_fd_success"),
            xy_whole_trace=_count(scored, "xy_whole_trace_success"),
            xy_outcomes={
                k: sum(1 for r in scored if xy_outcome(r) == k)
                for k in ("success", "reached_not_held", "never_reached", "contact_or_fall")
            },
            gained=sum(1 for r in scored if r["xy_success"] and not r["legacy_3d_success"]),
            lost=sum(1 for r in scored if r["legacy_3d_success"] and not r["xy_success"]),
            feasible_episodes=len(sub),
            legacy_3d_feasible=_count(sub, "legacy_3d_success"),
            xy_feasible=_count(sub, "xy_success"),
            suffix_episodes=len(suffix),
            legacy_suffix=_count(suffix, "legacy_suffix_supported"),
            legacy_suffix_repro_match=sum(
                1 for r in suffix if r["legacy_suffix_repro"] == r["legacy_suffix_supported"]
            ),
            xy_suffix=_count(suffix, "xy_suffix_supported"),
            median_final_xy_err_m=_median(scored, "final_xy_err_m"),
            median_min_xy_err_m=_median(scored, "min_xy_err_m"),
            median_path_ratio=_median(scored, "path_ratio"),
            median_first_xy_reach_s=_median(scored, "first_xy_reach_s"),
            seeds=sorted({r["seed"] for r in items if r["seed"]}),
            callback=sorted({r["callback"] for r in items}),
            kind="evaluation"
            if all(r["callback"] in EVALUATION_CALLBACKS for r in items)
            else "collection",
        )
        table.append(entry)
    return table


def _fmt(value, digits=2):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def speed_channel_stats(rows, threshold=0.1):
    """Recorded 3-D speed vs XY finite-difference speed on rows within XY goal tolerance."""
    pairs = [r["_speed_in_tol"] for r in rows if "_speed_in_tol" in r]
    rec = np.concatenate([p[0] for p in pairs]) if pairs else np.zeros(0)
    fd = np.concatenate([p[1] for p in pairs]) if pairs else np.zeros(0)
    if not len(rec):
        return dict(rows=0)
    return dict(
        rows=int(len(rec)),
        median_recorded_mps=float(np.median(rec)),
        median_fd_planar_mps=float(np.median(fd)),
        median_abs_diff_mps=float(np.median(np.abs(rec - fd))),
        p90_abs_diff_mps=float(np.percentile(np.abs(rec - fd), 90)),
        correlation=float(np.corrcoef(rec, fd)[0, 1]) if len(rec) > 1 else None,
        rows_below_threshold_recorded=float((rec <= threshold).mean()),
        rows_below_threshold_fd=float((fd <= threshold).mean()),
        row_threshold_agreement=float(((rec <= threshold) == (fd <= threshold)).mean()),
    )


def write_markdown(path, rows, table, provenance, feasible):
    scored = [r for r in rows if r["trace"] == "present"]
    matched = sum(r["legacy_repro_match"] for r in scored)
    lines = [
        "# XY re-score of the nav-8192 scene episodes",
        "",
        f"Generated {provenance['generated_utc']} by `scripts/navigation_distill/rescore_xy.py` "
        f"(sha256 `{provenance['script_sha256'][:16]}`, git `{provenance['git_head']}`).",
        f"Packet `{provenance['packet']}`: {provenance['unique_episodes']} unique episodes "
        f"({provenance['episode_paths']} paths including symlinked aliases); "
        f"{len(rows) - len(scored)} without a trace.",
        "",
        "Protocol: every row is one recorded single-env episode (50 Hz control rows). Legacy = the "
        "recorded `task-result.json` flag (3-D pelvis goal, reference-derived deadline). "
        "XY = XY pelvis distance <= 0.25 m and speed <= 0.10 m/s for 50 consecutive rows, "
        "completed before any contact > 1 N or fall on the same recorded rollout. The recorded "
        "speed is the 3-D pelvis speed, an upper bound on planar speed, so XY is a lower bound; "
        "XY-FD uses planar speed from XY finite differences instead. The 'feasible' columns "
        "restrict to the 19 task ids of the DAgger panels "
        f"({len(feasible)} ids).",
        "",
        f"**Legacy reproduction:** the 3-D port reproduces all recorded score fields for "
        f"{matched}/{len(scored)} episodes.",
        "",
        "## Evaluation panels (unassisted: teacher, full-command motor, navigation adapter)",
        "",
        "| panel | seed | episodes | legacy 3-D | XY | XY-FD | XY whole-trace | gained / lost "
        "| legacy 3-D /19 | XY /19 | XY: held / reached not held / never reached / contact "
        "| median final XY err (m) | median min XY err (m) | median path ratio | repro match |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    evals = [t for t in table if t["kind"] == "evaluation"]
    for t in evals:
        lines.append(
            f"| {t['panel']} | {','.join(t['seeds'])} | {t['episodes']} | {t['legacy_3d']} "
            f"| {t['xy']} | {t['xy_fd']} | {t['xy_whole_trace']} | {t['gained']} / {t['lost']} "
            f"| {t['legacy_3d_feasible']}/{t['feasible_episodes']} "
            f"| {t['xy_feasible']}/{t['feasible_episodes']} "
            f"| {' / '.join(str(v) for v in t['xy_outcomes'].values())} "
            f"| {_fmt(t['median_final_xy_err_m'])} | {_fmt(t['median_min_xy_err_m'])} "
            f"| {_fmt(t['median_path_ratio'])} | {t['legacy_repro_match']}/{t['episodes']} |"
        )
    lines += [
        "",
        "## Collection panels (assisted: motor demonstrations, learner-prefix recoveries, "
        "re-entry probes)",
        "",
        "Whole-episode flags mix a learner prefix with a motor or rewound suffix; they are not "
        "navigation results. The suffix columns are the receipts that admit DAgger rows.",
        "",
        "| panel | aliases | episodes | legacy 3-D | XY | suffix: legacy receipt "
        "| suffix: 3-D repro match | suffix: XY | gained / lost | repro match |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in table:
        if t["kind"] == "evaluation":
            continue
        suffix = (
            (f"{t['legacy_suffix']}/{t['suffix_episodes']}", f"{t['legacy_suffix_repro_match']}/"
             f"{t['suffix_episodes']}", f"{t['xy_suffix']}/{t['suffix_episodes']}")
            if t["suffix_episodes"] else ("–", "–", "–")
        )
        lines.append(
            f"| {t['panel']} | {', '.join(t['aliases']) or '–'} | {t['episodes']} "
            f"| {t['legacy_3d']} | {t['xy']} | {suffix[0]} | {suffix[1]} | {suffix[2]} "
            f"| {t['gained']} / {t['lost']} | {t['legacy_repro_match']}/{t['episodes']} |"
        )
    changed = [r for r in scored if r["xy_success"] != r["legacy_3d_success"]]
    lines += [
        "",
        f"## Episodes whose outcome changes ({len(changed)})",
        "",
        "| panel | episode | legacy 3-D | XY | legacy hold | XY hold row | final XY err | "
        "final 3-D err | final z - goal z | stop |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in changed:
        lines.append(
            f"| {r['panel']} | {r['episode']} | {r['legacy_3d_success']} | {r['xy_success']} "
            f"| {r['legacy_max_hold_ticks']} | {_fmt(r['xy_hold_row'])} "
            f"| {_fmt(r['final_xy_err_m'], 3)} | {_fmt(r['final_3d_err_m'], 3)} "
            f"| {_fmt(r['final_z_minus_goal_z_m'], 3)} | {r['stop_reason']} |"
        )
    mismatches = [r for r in scored if not r["legacy_repro_match"]]
    lines += ["", f"## Legacy reproduction mismatches ({len(mismatches)})", ""]
    lines += [f"- {r['panel']}/{r['episode']}: {r['legacy_repro_mismatch']}" for r in mismatches]
    suffix_bad = [
        r for r in scored
        if r.get("legacy_suffix_supported") is not None
        and r["legacy_suffix_repro"] != r["legacy_suffix_supported"]
    ]
    lines += ["", f"## Suffix receipt reproduction mismatches ({len(suffix_bad)})", ""]
    lines += [
        f"- {r['panel']}/{r['episode']}: receipt {r['legacy_suffix_supported']}, "
        f"repro {r['legacy_suffix_repro']}" for r in suffix_bad
    ]
    lines += [
        "",
        "## Speed-channel sensitivity",
        "",
        "Rows within 0.25 m XY of the goal (all episodes, row 0 excluded): "
        + ", ".join(f"{k} {_fmt(v, 4)}" for k, v in provenance["speed_channel"].items())
        + ". The 0.10 m/s hold threshold sits at the median in-tolerance pelvis speed, so "
        "a 50-row run is sensitive to the speed channel even when the channels agree closely.",
        "",
        f"Rows within 0.25 m XY but outside 0.25 m 3-D: "
        f"{sum(r.get('rows_xy_only_in_tol', 0) for r in scored)} over all episodes.",
        "",
        "## Columns (rescore.csv)",
        "",
        "- `xy_hold_row`: 0-based trace row on which the first XY hold completes "
        "(`xy_hold_s` = (row + 1) x 0.02 s).",
        "- `first_xy_reach_row` / `first_xy_reach_s`: first row within 0.25 m XY (time to goal).",
        "- `path_xy_m`: XY pelvis path from the task start through every row; `path_ratio` = "
        "path / straight start-goal XY distance (reference-derived goal).",
        "- `max_root_height_dev_m`: max |pelvis z - start z|; `final_z_minus_goal_z_m` explains "
        "most 3-D vs XY differences.",
        "- `trace`: `present` for every row scored here; `missing` rows keep only the legacy flag.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feasible-panel", default="eval/dag-approach-c2-91260",
                        help="panel whose task ids define the teacher-feasible subset")
    args = parser.parse_args()
    packet = args.packet.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    targets = [args.output / n for n in ("rescore.csv", "rescore_summary.md",
                                         "rescore_summary.json")]
    if any(t.exists() for t in targets):
        raise FileExistsError(f"Refusing to overwrite results in {args.output}")
    episodes = discover(packet)
    rows = [rescore_episode(Path(real), aliases, packet) for real, aliases in episodes.items()]
    rows.sort(key=lambda r: (r["panel"], r["episode"]))
    feasible = {
        r["task_id"] for r in rows if r["panel"] == args.feasible_panel
    }
    here = Path(__file__).resolve()
    try:
        head = subprocess.run(["git", "-C", str(here.parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    provenance = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        script_sha256=sha256(here),
        git_head=head,
        packet=str(packet),
        command=sys.argv,
        unique_episodes=len(rows),
        episode_paths=sum(len(a) for a in episodes.values()),
        feasible_panel=args.feasible_panel,
        feasible_task_ids=sorted(feasible),
    )
    with targets[0].open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in COLUMNS})
    table = summarize(rows, packet, feasible)
    provenance["speed_channel"] = speed_channel_stats(rows)
    write_markdown(targets[1], rows, table, provenance, feasible)
    with targets[2].open("x") as f:
        json.dump(dict(provenance=provenance, panels=table), f, indent=2)
    scored = [r for r in rows if r["trace"] == "present"]
    print(json.dumps(dict(
        episodes=len(rows),
        traces_missing=len(rows) - len(scored),
        legacy_repro_match=sum(r["legacy_repro_match"] for r in scored),
        legacy_3d=sum(r["legacy_3d_success"] for r in scored),
        xy=sum(r["xy_success"] for r in scored),
        output=str(args.output),
    )))


if __name__ == "__main__":
    main()
