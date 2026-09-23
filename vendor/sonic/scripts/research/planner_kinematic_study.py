"""Phase 0.6/0.7 (CPU): SONIC planner timing, kinematic goal study, posture table, clips.

Kinematic only: the reference is taken as the robot state (perfect tracking). No simulator,
no GPU. Run with the ONNX Runtime interpreter, for example:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=vendor/sonic \
    /home/robotixx/GR00T-WholeBodyControl/.venv_sim/bin/python \
        vendor/sonic/scripts/research/planner_kinematic_study.py all \
        --out /home/robotixx/motion2scene-training/workspace/phase0/planner
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    KIMODO_G1_JOINT_NAMES,
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)
from gear_sonic.research.planner.g1_geometry import (  # noqa: E402
    FLOOR_CONTACT_THRESHOLD_M,
    FOOT_GROUPS,
    HEAD_LINK,
    SENSOR_LINK,
    G1Geometry,
)
from gear_sonic.research.planner.goal_controllers import (  # noqa: E402
    DEFAULT_BANDS,
    DirectionSpeedStopController,
    WaypointController,
    goal_metrics,
    run_controller,
    stop_distance_trial,
)
from gear_sonic.research.planner.ort_session import (  # noqa: E402
    DEFAULT_PLANNER_ONNX,
    OrtPlannerSession,
)
from gear_sonic.research.planner.planner_runtime import (  # noqa: E402
    CONTROL_FPS,
    CRAWLING,
    DEPLOY_TOKEN_MASK,
    ELBOW_CRAWLING,
    IDEL_KNEEL_TWO_LEGS,
    IDEL_SQUAT,
    MODE_NAMES,
    SLOW_WALK,
    STEALTH_WALK,
    STEALTH_WALK_2,
    WALK,
    DeployPlannerRuntime,
    build_inputs,
    idle_command,
    locomotion_command,
    run_schedule,
    staged_crawl_schedule,
    standing_context,
    static_posture_command,
    waypoint_command,
    yaw_from_quat,
)

GOAL_SEED = 70601  # collection block 70000-79999; kinematic study only
GOAL_COUNT = 200
GOAL_DISTANCE_M = (1.0, 8.0)
GOAL_HORIZON_S = 30.0
REPLAN_SCALES = (1.0, 2.0)
STOP_PHASES_S = (3.0, 3.1, 3.2, 3.3, 3.4)
GATE_RADIUS_M = 0.10
GATE_RATE = 0.95
STEADY_WINDOW_S = 3.0


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def provenance(session, model_sha=None):
    return {
        "generated_at_utc": utc(),
        "code_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "planner_onnx": session.model_path,
        "planner_onnx_sha256": model_sha,
        "onnxruntime": session.ort_version,
        "intra_op_threads": session.intra_op_threads,
        "token_mask": list(DEPLOY_TOKEN_MASK),
        "random_seed": 1234,
        "kinematic_only": True,
        "physics_steps": 0,
    }


# --------------------------------------------------------------------------- timing
def timing_study(session, repeats=40):
    context = standing_context()
    commands = {
        "idle_0": idle_command(),
        "slow_walk_1": locomotion_command(SLOW_WALK, 0.0, speed=0.4),
        "walk_2": locomotion_command(WALK, 0.0),
        "squat_4_h0.4": static_posture_command(IDEL_SQUAT, 0.4),
        "crawl_8": locomotion_command(CRAWLING, 0.0),
        "elbow_crawl_14": locomotion_command(ELBOW_CRAWLING, 0.0),
        "stealth_18": locomotion_command(STEALTH_WALK, 0.0),
        "stealth2_22": locomotion_command(STEALTH_WALK_2, 0.0),
        "waypoint_1": waypoint_command(SLOW_WALK, (3.0, 0.0), 0.0),
    }
    for _ in range(3):  # warm-up
        session(build_inputs(context, commands["walk_2"]))
    session.call_seconds.clear()
    per_mode = {}
    for name, command in commands.items():
        start = len(session.call_seconds)
        counts = []
        for _ in range(repeats):
            counts.append(int(session(build_inputs(context, command))["num_pred_frames"][0]))
        values = np.asarray(session.call_seconds[start:]) * 1e3
        per_mode[name] = {
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "num_pred_frames": sorted(set(counts)),
        }
    return {"overall": session.timing_summary(), "per_command": per_mode, "repeats": repeats}


# --------------------------------------------------------------------------- goal study
def sample_goals(count=GOAL_COUNT, seed=GOAL_SEED):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(*GOAL_DISTANCE_M, size=count)
    bearing = rng.uniform(-math.pi, math.pi, size=count)
    return np.stack([distance * np.cos(bearing), distance * np.sin(bearing)], 1), distance, bearing


def calibrate_stop_distances(session, scale):
    trials = []
    for band in DEFAULT_BANDS:
        for phase in STOP_PHASES_S:
            runtime = DeployPlannerRuntime(session, replan_interval_scale=scale)
            trials.append(stop_distance_trial(runtime, band, phase))
    table = {}
    for band in DEFAULT_BANDS:
        rows = [t for t in trials if t["band"] == band.key]
        run_out = np.array([t["run_out_m"] for t in rows])
        table[band.key] = {
            "median_run_out_m": float(np.median(run_out)),
            "min_run_out_m": float(run_out.min()),
            "max_run_out_m": float(run_out.max()),
            "median_walk_speed_m_s": float(np.median([t["walk_speed_m_s"] for t in rows])),
        }
    return {"bands": table, "trials": trials}


def goal_study(session, out):
    goals, distance, bearing = sample_goals()
    calibration = {}
    episodes = []
    traces = {}
    for scale in REPLAN_SCALES:
        calibration[str(scale)] = calibrate_stop_distances(session, scale)
        stop = {k: v["median_run_out_m"] for k, v in calibration[str(scale)]["bands"].items()}
        for controller_name in ("P0", "P1"):
            key = f"{controller_name}_x{scale:g}"
            started = time.perf_counter()
            roots = []
            for index, goal in enumerate(goals):
                runtime = DeployPlannerRuntime(session, replan_interval_scale=scale)
                if controller_name == "P0":
                    controller = DirectionSpeedStopController(goal, stop)
                else:
                    controller = WaypointController(goal)
                frames = run_controller(runtime, controller, GOAL_HORIZON_S)
                row = goal_metrics(frames, goal)
                row.update(
                    {
                        "condition": key,
                        "controller": controller_name,
                        "replan_interval_scale": scale,
                        "goal_index": index,
                        "goal_xy": goal.tolist(),
                        "goal_distance_m": float(distance[index]),
                        "goal_bearing_deg": float(np.degrees(bearing[index])),
                        "planner_calls": len(runtime.calls),
                        "num_pred_frames": sorted({c.num_pred_frames for c in runtime.calls}),
                        "hold_ticks": runtime.hold_ticks,
                        "late_plans": runtime.late_plans,
                        "final_yaw_deg": float(np.degrees(yaw_from_quat(frames[-1, 3:7]))),
                    }
                )
                if controller_name == "P0":
                    row["stop_command_s"] = controller.stop_time_s
                episodes.append(row)
                roots.append(frames[:, :3].astype(np.float32))
                if controller_name in ("P0", "P1") and scale == 1.0 and index < 40:
                    traces[(key, index)] = frames
            np.save(out / f"goal_roots_{key}.npy", np.stack(roots))
            print(f"[goals] {key}: {time.perf_counter() - started:.0f}s", flush=True)
    return goals, calibration, episodes, traces


def summarize_goals(episodes):
    summary = {}
    for key in sorted({e["condition"] for e in episodes}):
        rows = [e for e in episodes if e["condition"] == key]
        final = np.array([r["final_error_m"] for r in rows])
        arrive = [r["time_to_arrive_010_s"] for r in rows if r["time_to_arrive_010_s"] is not None]
        ratio = np.array([r["path_length_ratio_smoothed"] for r in rows])
        raw_ratio = np.array([r["path_length_ratio_raw"] for r in rows])
        success = np.array([r["success_010"] for r in rows])
        n = len(rows)
        k = int(success.sum())
        summary[key] = {
            "episodes": n,
            "success_010": k,
            "success_010_rate": k / n,
            "success_010_wilson95": wilson(k, n),
            "success_025": int(sum(r["success_025"] for r in rows)),
            "final_error_median_m": float(np.median(final)),
            "final_error_p90_m": float(np.percentile(final, 90)),
            "final_error_p95_m": float(np.percentile(final, 95)),
            "final_error_max_m": float(final.max()),
            "arrived_010": len(arrive),
            "time_to_arrive_010_median_s": float(np.median(arrive)) if arrive else None,
            "path_ratio_smoothed_median": float(np.median(ratio)),
            "path_ratio_smoothed_p90": float(np.percentile(ratio, 90)),
            "path_ratio_raw_median": float(np.median(raw_ratio)),
            "stops": int(sum(r["stops"] for r in rows)),
            "planner_calls_median": float(np.median([r["planner_calls"] for r in rows])),
            "hold_ticks_median": float(np.median([r["hold_ticks"] for r in rows])),
            "num_pred_frames_seen": sorted({v for r in rows for v in r["num_pred_frames"]}),
            "success_by_distance_band": {
                band: int(sum(r["success_010"] for r in rows if lo <= r["goal_distance_m"] < hi))
                for band, (lo, hi) in {
                    "1-3m": (1, 3),
                    "3-5.5m": (3, 5.5),
                    "5.5-8m": (5.5, 8.01),
                }.items()
            },
            "episodes_by_distance_band": {
                band: int(sum(1 for r in rows if lo <= r["goal_distance_m"] < hi))
                for band, (lo, hi) in {
                    "1-3m": (1, 3),
                    "3-5.5m": (3, 5.5),
                    "5.5-8m": (5.5, 8.01),
                }.items()
            },
        }
    return summary


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


# --------------------------------------------------------------------------- postures
def posture_runs():
    """(name, mode, schedule, duration_s, kind, final_command_s, commanded body direction)."""
    runs = []
    for tenth in range(1, 9):
        height = tenth / 10
        command = static_posture_command(IDEL_SQUAT, height)
        runs.append(
            (f"squat_h{height:.1f}", IDEL_SQUAT, [(0.0, command)], 6.0, "static", 0.0, None)
        )
    kneel = static_posture_command(IDEL_KNEEL_TWO_LEGS, 0.4)
    runs.append(
        ("kneel_two_legs_h0.4", IDEL_KNEEL_TWO_LEGS, [(0.0, kneel)], 6.0, "static", 0.0, None)
    )
    forward, left, right = (1.0, 0.0), (0.0, 1.0), (0.0, -1.0)
    moving = [
        ("crawl_direct", CRAWLING, [(0.0, locomotion_command(CRAWLING, 0.0))], 10.0, 0.0, forward),
        ("crawl_staged", CRAWLING, staged_crawl_schedule(CRAWLING), 10.0, 2.0, forward),
        (
            "elbow_crawl_direct",
            ELBOW_CRAWLING,
            [(0.0, locomotion_command(ELBOW_CRAWLING, 0.0))],
            10.0,
            0.0,
            forward,
        ),
        (
            "elbow_crawl_staged",
            ELBOW_CRAWLING,
            staged_crawl_schedule(ELBOW_CRAWLING),
            12.0,
            4.0,
            forward,
        ),
        (
            "stealth_walk_18",
            STEALTH_WALK,
            [(0.0, locomotion_command(STEALTH_WALK, 0.0))],
            10.0,
            0.0,
            forward,
        ),
        (
            "stealth_walk2_22",
            STEALTH_WALK_2,
            [(0.0, locomotion_command(STEALTH_WALK_2, 0.0))],
            10.0,
            0.0,
            forward,
        ),
        (
            "strafe_left_mode1",
            SLOW_WALK,
            [(0.0, locomotion_command(SLOW_WALK, math.pi / 2, 0.0))],
            10.0,
            0.0,
            left,
        ),
        (
            "strafe_right_mode1",
            SLOW_WALK,
            [(0.0, locomotion_command(SLOW_WALK, -math.pi / 2, 0.0))],
            10.0,
            0.0,
            right,
        ),
        (
            "strafe_left_mode2",
            WALK,
            [(0.0, locomotion_command(WALK, math.pi / 2, 0.0))],
            10.0,
            0.0,
            left,
        ),
        (
            "walk_fwd_mode1",
            SLOW_WALK,
            [(0.0, locomotion_command(SLOW_WALK, 0.0))],
            10.0,
            0.0,
            forward,
        ),
        ("walk_fwd_mode2", WALK, [(0.0, locomotion_command(WALK, 0.0))], 10.0, 0.0, forward),
    ]
    runs += [(n, m, sch, d, "moving", e, direction) for n, m, sch, d, e, direction in moving]
    return runs


EXTRA_CLIPS = [
    ("walk_back_mode1", [(0.0, locomotion_command(SLOW_WALK, math.pi, 0.0))], 6.0),
    (
        "walk_turn90_mode1",
        [
            (0.0, locomotion_command(SLOW_WALK, 0.0)),
            (2.0, locomotion_command(SLOW_WALK, math.pi / 2)),
        ],
        6.0,
    ),
    ("walk_diag45_mode2", [(0.0, locomotion_command(WALK, math.pi / 4))], 6.0),
    ("waypoint_3m_ahead", [(0.0, waypoint_command(SLOW_WALK, (3.0, 0.0), 0.0))], 8.0),
    (
        "waypoint_turnaround",
        [(0.0, waypoint_command(SLOW_WALK, (-2.0, 2.0), 3 * math.pi / 4))],
        8.0,
    ),
]


def smoothed(values, seconds=0.3):
    width = max(1, int(round(seconds * CONTROL_FPS)))
    if len(values) <= width:
        return np.asarray(values, dtype=np.float64)
    pad = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(pad, np.ones(width) / width, mode="valid")


def settle_time(signal, steady_slice, tolerance_floor=0.02, fraction=0.1):
    series = smoothed(signal)
    steady = series[steady_slice]
    target = float(np.median(steady))
    # Gait oscillation that survives smoothing widens the band (half the steady p5-p95 range).
    oscillation = float(np.percentile(steady, 95) - np.percentile(steady, 5)) / 2
    tolerance = max(tolerance_floor, fraction * abs(series[0] - target), 1.5 * oscillation)
    outside = np.flatnonzero(np.abs(series - target) > tolerance)
    if len(outside) == 0:
        return 0.0, target, tolerance
    if outside[-1] >= len(series) - 1:
        return None, target, tolerance
    return float((outside[-1] + 1) / CONTROL_FPS), target, tolerance


def posture_metrics(name, mode, frames, geometry, kind, entry_s, direction, runtime):
    steady = slice(len(frames) - int(STEADY_WINDOW_S * CONTROL_FPS), len(frames))
    poses = geometry.link_poses(frames)
    groups = {proxy: geometry.group_z_extents(poses, proxy) for proxy in ("visual", "collision")}
    link_visual = geometry.link_z_extents(poses, "visual")
    link_collision = geometry.link_z_extents(poses, "collision")
    head_top = link_visual[HEAD_LINK][1]
    body_top = np.max([v[1] for v in link_visual.values()], axis=0)
    collision_top = np.max([v[1] for v in link_collision.values()], axis=0)
    pelvis_z = frames[:, 2]
    pelvis_settle, pelvis_target, pelvis_tol = settle_time(pelvis_z, steady)
    head_settle, head_target, head_tol = settle_time(head_top, steady)
    xy = frames[:, :2]
    net = xy[steady][-1] - xy[steady][0]
    window = (steady.stop - steady.start - 1) / CONTROL_FPS
    yaw0 = float(yaw_from_quat(frames[0, 3:7]))
    forward = np.array([math.cos(yaw0), math.sin(yaw0)])
    left = np.array([-forward[1], forward[0]])
    velocity = net / window
    path_speed = float(np.sum(np.linalg.norm(np.diff(xy[steady], axis=0), axis=1)) / window)
    contacts = {}
    for proxy, extents in groups.items():
        rows = {}
        for group, (low, _high) in extents.items():
            steady_low = low[steady]
            rows[group] = {
                "min_z_m": float(low.min()),
                "steady_min_z_m": float(steady_low.min()),
                "steady_fraction_below_threshold": float(
                    np.mean(steady_low < FLOOR_CONTACT_THRESHOLD_M)
                ),
            }
        contacts[proxy] = rows
    # Primary: the Isaac collision shapes (what the contact scorer sees). The visual shin mesh
    # (knee_link) reaches the ankle, so visual "knee" hits are reported but not used.
    candidates = {
        g: r["steady_fraction_below_threshold"]
        for g, r in contacts["collision"].items()
        if g not in FOOT_GROUPS and r["steady_fraction_below_threshold"] > 0
    }
    lowest = np.min([low for low, _ in groups["collision"].values()], axis=0)
    lowest_group = [
        min(groups["collision"], key=lambda g: groups["collision"][g][0][k])
        for k in range(len(frames))
    ]
    steady_lowest = [lowest_group[k] for k in range(steady.start, steady.stop)]
    foot_floor = float(min(groups["visual"][g][0][steady].min() for g in FOOT_GROUPS))
    excess = geometry.joint_limit_excess(frames)
    worst = np.argsort(-excess.max(axis=0))[:3]
    speed_reached = None
    if kind == "moving" and np.linalg.norm(velocity) > 0.02:
        run = smoothed(np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1) * CONTROL_FPS], 0.5)
        hits = np.flatnonzero(run >= 0.8 * np.linalg.norm(velocity))
        speed_reached = float(hits[0] / CONTROL_FPS) if len(hits) else None
    sensor_z = poses[SENSOR_LINK][1][:, 2]
    return {
        "name": name,
        "mode": mode,
        "mode_name": MODE_NAMES[mode],
        "kind": kind,
        "duration_s": len(frames) / CONTROL_FPS,
        "final_mode_command_s": entry_s,
        "commanded_direction_body": direction,
        "min_pelvis_z_m": float(pelvis_z.min()),
        "steady_pelvis_z_median_m": float(np.median(pelvis_z[steady])),
        "head_top_max_m": float(head_top.max()),
        "head_top_steady_max_m": float(head_top[steady].max()),
        "head_top_steady_median_m": float(np.median(head_top[steady])),
        "visual_body_top_steady_max_m": float(body_top[steady].max()),
        "collision_top_steady_max_m": float(collision_top[steady].max()),
        "mid360_z_steady_median_m": float(np.median(sensor_z[steady])),
        "foot_sole_steady_min_z_m": foot_floor,
        "floor_contact_candidates": sorted(candidates),
        "floor_contact_duty": candidates,
        "lowest_point_steady_min_z_m": float(lowest[steady].min()),
        "lowest_body_group_steady_share": {
            g: steady_lowest.count(g) / len(steady_lowest) for g in sorted(set(steady_lowest))
        },
        "contacts": contacts,
        "transit_velocity_steady_m_s": velocity.tolist(),
        "transit_speed_steady_m_s": float(np.linalg.norm(velocity)),
        "transit_forward_m_s": float(velocity @ forward),
        "transit_lateral_left_m_s": float(velocity @ left),
        "path_speed_steady_m_s": path_speed,
        "transition_pelvis_settle_s": pelvis_settle,
        "transition_head_top_settle_s": head_settle,
        "pelvis_settle_target_m": pelvis_target,
        "pelvis_settle_tolerance_m": pelvis_tol,
        "time_to_80pct_transit_speed_s": speed_reached,
        "joint_limit_violation_cells": int(np.count_nonzero(excess > 1e-6)),
        "joint_limit_violation_frames": int(np.count_nonzero(excess.max(axis=1) > 1e-6)),
        "joint_limit_max_excess_rad": float(excess.max()),
        "joint_limit_worst": [
            {"joint": KIMODO_G1_JOINT_NAMES[i], "max_excess_rad": float(excess[:, i].max())}
            for i in worst
            if excess[:, i].max() > 1e-6
        ],
        "planner_calls": len(runtime.calls),
        "num_pred_frames": sorted({c.num_pred_frames for c in runtime.calls}),
        "hold_ticks": runtime.hold_ticks,
        "replan_reasons": sorted({r for c in runtime.calls for r in c.reasons}),
    }


def save_clip(directory, name, frames, schedule, extra=None):
    entry = qpos_to_sonic_motion_entry(
        frames, source_fps=int(CONTROL_FPS), canonicalize_horizontal_origin=False
    )
    path = directory / f"planner_{name}.pkl"
    save_sonic_motion_file(path, motion_key=f"planner_{name}", motion_entry=entry)
    return {
        "name": f"planner_{name}",
        "file": path.name,
        "sha256": sha256(path),
        "frames": int(len(frames)),
        "fps": int(CONTROL_FPS),
        "duration_s": len(frames) / CONTROL_FPS,
        "schedule": [
            {
                "t_s": t,
                "mode": c.mode,
                "mode_name": MODE_NAMES[c.mode],
                "speed": c.speed,
                "height": c.height,
                "movement_direction": list(c.movement_direction),
                "facing_direction": list(c.facing_direction),
                "has_specific_target": c.has_specific_target,
                "target_xy": list(c.target_positions[-1][:2]) if c.has_specific_target else None,
            }
            for t, c in schedule
        ],
        **(extra or {}),
    }


def posture_study(session, geometry, out):
    clip_dir = out / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    table, clips = [], []
    for name, mode, schedule, duration, kind, entry_s, direction in posture_runs():
        runtime = DeployPlannerRuntime(session)
        frames = run_schedule(runtime, schedule, duration)
        table.append(
            posture_metrics(name, mode, frames, geometry, kind, entry_s, direction, runtime)
        )
        clips.append(save_clip(clip_dir, name, frames, schedule, {"source": "posture_table"}))
        print(f"[posture] {name}", flush=True)
    for name, schedule, duration in EXTRA_CLIPS:
        runtime = DeployPlannerRuntime(session)
        frames = run_schedule(runtime, schedule, duration)
        clips.append(save_clip(clip_dir, name, frames, schedule, {"source": "extra_walk"}))
    return table, clips


def export_goal_clips(out, goals, episodes, traces):
    clip_dir = out / "goal_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    by_key = {(e["condition"], e["goal_index"]): e for e in episodes}
    for (key, index), frames in sorted(traces.items()):
        row = by_key[(key, index)]
        settle = row["time_to_settle_010_s"]
        end_s = GOAL_HORIZON_S if settle is None else min(GOAL_HORIZON_S, settle + 3.0)
        trimmed = frames[: int(round(end_s * CONTROL_FPS))]
        name = f"{key.split('_')[0]}_goal{index:03d}"
        entry = save_clip(
            clip_dir,
            name,
            trimmed,
            [],
            {
                "source": "goal_study",
                "condition": key,
                "goal_index": index,
                "goal_xy": goals[index].tolist(),
                "kinematic_final_error_m": row["final_error_m"],
                "trimmed_at_s": end_s,
            },
        )
        manifest.append(entry)
    write_json(clip_dir / "manifest.json", {"clips": manifest, "fps": int(CONTROL_FPS)})
    return len(manifest)


# --------------------------------------------------------------------------- reports
def fmt(value, digits=2):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


GOAL_PROTOCOL = [
    f"{GOAL_COUNT} goals, seed {GOAL_SEED}: distance U(1, 8) m, bearing U(−180°, 180°), from the "
    "deploy standing start (origin, facing +x, pelvis 0.78874 m, default joint angles).",
    f"Horizon {GOAL_HORIZON_S:g} s at 50 Hz. The controller acts at the planner-thread rate (10 Hz) "
    "and reads the reference frame the tracker would observe (perfect tracking).",
    "Runtime `gear_sonic/research/planner/planner_runtime.py`: deploy replan triggers, context at "
    "cursor+2 with 30 Hz spacing, 30→50 Hz resampling, 8-frame cross-fade, token mask "
    "`[0,0,0,1,1,1,0,0,0,0,0]`, seed 1234, plans blended at the tick of the call (latency 0).",
    "**P0** direction/speed/stop: face and move toward the goal. WALK (mode 2, default speed) "
    "beyond 2.5 m, SLOW_WALK 0.4 m/s beyond 1.0 m, SLOW_WALK 0.2 m/s inside. Heading re-commanded "
    "when off by >5° (frozen inside 0.3 m). IDLE (latched) once the goal is within the calibrated "
    "run-out of the current band, or when it falls behind inside 0.3 m. Deadband and freeze radius "
    "were chosen on 20 development goals (seed 70602), never on the 200-goal set.",
    "**P1** waypoint (extension E1): SLOW_WALK with `has_specific_target=1`, the goal in all four "
    "slots and final heading = start bearing. The command never changes, so only the planner's "
    "timer replans.",
    "Replan interval ×1 = deploy (1.0 s in modes 1/2) and ×2 (2.0 s). P0's run-outs are "
    "calibrated per interval on straight walks (5 gait phases per band), not on the goal set.",
    f"Gate (roadmap §8, 0.6): P0 or P1 ends within {GATE_RADIUS_M:.2f} m of the goal in "
    f"≥{GATE_RATE:.0%} of goals.",
    "Metrics: final root-XY error; time to first come within 0.10 m; path-length ratio = root "
    "path / straight line (*smoothed* uses a 0.5 s moving average to remove gait sway, *raw* keeps "
    "it); stops = net root speed over the last 0.5 s < 0.1 m/s.",
]


def goal_report(result, notes):
    summary = result["summary"]
    lines = [
        "# SONIC planner kinematic goal study (Phase 0.6)",
        "",
        f"Generated {result['provenance']['generated_at_utc']} from code commit "
        f"`{result['provenance']['code_commit']}`. Kinematic only: no physics, no tracking. "
        "Every number is measured on this host **[M]** unless marked otherwise.",
        "",
        "## Protocol",
        "",
    ]
    lines += [f"- {item}" for item in GOAL_PROTOCOL]
    lines += ["", "## Results", ""]
    header = [
        "Condition",
        "≤0.10 m",
        "Wilson 95%",
        "≤0.25 m",
        "Final err median / p95 / max (m)",
        "Arrive ≤0.10 m, median s (n)",
        "Path ratio smoothed median / p90",
        "Raw ratio median",
        "Stops",
        "Calls (median)",
        "Hold ticks (median)",
    ]
    rows = []
    for key, v in summary.items():
        lo, hi = v["success_010_wilson95"]
        rows.append(
            [
                key,
                f"{v['success_010']}/{v['episodes']} ({v['success_010_rate']:.1%})",
                f"{lo:.3f}–{hi:.3f}",
                f"{v['success_025']}/{v['episodes']}",
                f"{v['final_error_median_m']:.3f} / {v['final_error_p95_m']:.3f} / "
                f"{v['final_error_max_m']:.3f}",
                f"{fmt(v['time_to_arrive_010_median_s'], 1)} ({v['arrived_010']})",
                f"{v['path_ratio_smoothed_median']:.3f} / {v['path_ratio_smoothed_p90']:.3f}",
                f"{v['path_ratio_raw_median']:.3f}",
                f"{v['stops']}/{v['episodes']}",
                f"{v['planner_calls_median']:.0f}",
                f"{v['hold_ticks_median']:.0f}",
            ]
        )
    lines += md_table(header, rows)
    lines += ["", "Success ≤0.10 m by goal distance:", ""]
    bands = ("1-3m", "3-5.5m", "5.5-8m")
    lines += md_table(
        ["Condition", "1–3 m", "3–5.5 m", "5.5–8 m"],
        [
            [key]
            + [
                f"{v['success_by_distance_band'][b]}/{v['episodes_by_distance_band'][b]}"
                for b in bands
            ]
            for key, v in summary.items()
        ],
    )
    lines += ["", "## Gate", "", result["gate"]["statement"], ""]
    lines += ["## P0 stop calibration (run-out after the IDLE command)", ""]
    rows = []
    for scale, calibration in result["calibration"].items():
        for band, row in calibration["bands"].items():
            rows.append(
                [
                    f"×{float(scale):g}",
                    band,
                    f"{row['median_walk_speed_m_s']:.2f}",
                    f"{row['median_run_out_m']:.3f} [{row['min_run_out_m']:.3f}, "
                    f"{row['max_run_out_m']:.3f}]",
                ]
            )
    lines += md_table(
        ["Interval", "Band", "Walk speed (m/s)", "Run-out median [min, max] (m)"], rows
    )
    timing = notes.get("timing")
    if timing:
        lines += ["", "## Planner timing on this host", ""] + [f"- {t}" for t in timing]
    lines += ["", "## Findings", ""] + [f"- {f}" for f in notes.get("goal_findings", [])]
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in notes.get("goal_caveats", [])] + [""]
    return "\n".join(lines)


POSTURE_PROTOCOL = [
    "Deploy standing start, then the command schedule through the deploy-faithful runtime "
    "(seed 1234, deploy token mask, replan interval ×1). Static modes (4, 5) send zero movement "
    "and speed 0, as the deploy gamepad does. Moving modes use the gamepad defaults (crawl 0.7 m/s "
    "at height 0.4, elbow crawl 0.7 m/s at 0.3, SLOW_WALK 0.4 m/s, others −1 = mode default) and "
    "move +x. Strafes move ±y while facing +x.",
    "*direct* = target mode commanded at t=0 from IDLE. *staged* = the deploy gamepad path: "
    "IDEL_KNEEL_TWO_LEGS (mode 5, h 0.4) at t=0, CRAWLING at 2 s, ELBOW_CRAWLING at 4 s. Run "
    "lengths: 6 s for static modes, 10 s for moving modes, 12 s for the staged elbow crawl, so "
    "the steady window starts after every transition has settled.",
    "FK: the Isaac Lab URDF (`robot_description/urdf/g1/main.urdf`). Its revolute joints equal the "
    "motion-lib MJCF's; the FK matches MuJoCo to 1.5e-6 m. The floor is z 0 of the planner world.",
    f"Floor-contact candidate: a body group whose lowest point is < {FLOOR_CONTACT_THRESHOLD_M} m "
    f"during the steady window (last {STEADY_WINDOW_S:g} s), using the *Isaac collision* shapes "
    "that trigger contact sensors (hand meshes, knee/elbow/hip cylinders, pelvis sphere). Feet are "
    "excluded; duty = share of steady frames below the threshold. Visual-mesh minima are in the "
    "JSON (the visual shin mesh reaches the ankle, so it is not used for the knee).",
    f"Head top: max world z of the `{HEAD_LINK}` visual mesh (head_link.STL, fixed to torso_link at "
    "xyz (0.0039635, 0, −0.044), rpy 0). The Isaac collision model has no head shape. Body top = "
    "max over all visual meshes. MID-360 = origin of `mid360_link` (torso_link + (0.000284, "
    "0.00003, 0.41618)).",
    "Transition time: from the t=0 command until the 0.3 s-smoothed pelvis height (or head top) "
    "enters and stays within max(0.02 m, 10% of the change, 1.5 × half the steady p5–p95 range) "
    "of its steady-window median; 80% speed = first time the 0.5 s-smoothed root speed reaches "
    "80% of the steady transit speed. Transit "
    "speed: net root displacement over the steady window, split into forward/left of the start "
    "heading.",
    "Joint limits: URDF ranges (identical to the MJCF). A cell is one (frame, joint) beyond range "
    "by >1e-6 rad.",
]


def posture_report(result, notes):
    standing = result["standing"]
    lines = [
        "# SONIC planner posture capability table (Phase 0.7, CPU part)",
        "",
        f"Generated {result['provenance']['generated_at_utc']} from code commit "
        f"`{result['provenance']['code_commit']}`. Kinematic FK of the planner's 50 Hz reference "
        "only: no physics, no tracking. **[M]** on this host.",
        "",
        "## Protocol",
        "",
    ]
    lines += [f"- {item}" for item in POSTURE_PROTOCOL]
    lines += [
        f"- Standing calibration: at the deploy default height the foot soles are "
        f"{standing['foot_sole_min_z_m']:.3f} m above z 0, the head top is at "
        f"{standing['head_top_m']:.3f} m and the MID-360 origin at {standing['mid360_z_m']:.3f} m.",
        "",
        "## Table",
        "",
    ]
    header = [
        "Run",
        "Mode",
        "Min pelvis z (m)",
        "Steady pelvis z (m)",
        "Head top steady max (m)",
        "Body top steady max (m)",
        "MID-360 z (m)",
        "Transit fwd / left (m/s)",
        "Pelvis settle (s)",
        "Head settle (s)",
        "80% speed (s)",
        "Floor-contact candidates (steady duty)",
        "Lowest point (m)",
        "Foot sole min z (m)",
        "Joint-limit cells (max rad)",
        "Hold ticks",
    ]
    rows = []
    for r in result["table"]:
        rows.append(
            [
                r["name"],
                f"{r['mode']} {r['mode_name']}",
                f"{r['min_pelvis_z_m']:.3f}",
                f"{r['steady_pelvis_z_median_m']:.3f}",
                f"{r['head_top_steady_max_m']:.3f}",
                f"{r['visual_body_top_steady_max_m']:.3f}",
                f"{r['mid360_z_steady_median_m']:.3f}",
                f"{r['transit_forward_m_s']:+.2f} / {r['transit_lateral_left_m_s']:+.2f}",
                fmt(r["transition_pelvis_settle_s"]),
                fmt(r["transition_head_top_settle_s"]),
                fmt(r["time_to_80pct_transit_speed_s"]),
                ", ".join(f"{g} {d:.0%}" for g, d in sorted(r["floor_contact_duty"].items()))
                or "none",
                f"{r['lowest_point_steady_min_z_m']:.3f}",
                f"{r['foot_sole_steady_min_z_m']:.3f}",
                f"{r['joint_limit_violation_cells']} ({r['joint_limit_max_excess_rad']:.3f})",
                str(r["hold_ticks"]),
            ]
        )
    lines += md_table(header, rows)
    lines += ["", "## Findings", ""] + [f"- {f}" for f in notes.get("posture_findings", [])]
    lines += [
        "",
        "## Clips for the GPU tracking run (not run here)",
        "",
        f"- {len(result['clips'])} planner clips, 50 fps motion-lib pickles, in `clips/` "
        "(manifest `clips/manifest.json` lists each command schedule and sha256).",
        f"- {result['goal_clips']} goal-study references (B1-OL candidates: the first 40 goals of "
        "P0_x1 and P1_x1, trimmed 3 s after settling) in `goal_clips/`.",
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {c}" for c in notes.get("posture_caveats", [])] + [""]
    return "\n".join(lines)


def render_reports(out, notes_path=None):
    notes = json.loads(Path(notes_path).read_text()) if notes_path else {}
    if (out / "kinematic_goal_study.json").exists():
        result = json.loads((out / "kinematic_goal_study.json").read_text())
        (out / "kinematic_goal_study.md").write_text(goal_report(result, notes))
    if (out / "posture_table.json").exists():
        result = json.loads((out / "posture_table.json").read_text())
        (out / "posture_table.md").write_text(posture_report(result, notes))


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["timing", "goals", "postures", "all", "report"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_PLANNER_ONNX)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--robot-description", default=None)
    parser.add_argument("--notes", default=None, help="JSON with findings/caveats for the reports")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        render_reports(out, args.notes)
        return
    session = OrtPlannerSession(args.model, intra_op_threads=args.threads)
    model_sha = sha256(args.model)
    started = time.perf_counter()
    if args.command in ("timing", "all"):
        timing = timing_study(session)
        timing["provenance"] = provenance(session, model_sha)
        write_json(out / "timing.json", timing)
        print(json.dumps(timing["overall"], indent=1), flush=True)
    if args.command in ("postures", "all"):
        geometry = G1Geometry(args.robot_description)
        session.call_seconds.clear()
        table, clips = posture_study(session, geometry, out)
        standing = standing_context()[:1]
        poses = geometry.link_poses(standing)
        visual = geometry.link_z_extents(poses, "visual")
        feet = geometry.group_z_extents(poses, "visual")
        write_json(out / "clips" / "manifest.json", {"clips": clips, "fps": int(CONTROL_FPS)})
        result = {
            "provenance": provenance(session, model_sha),
            "geometry": {
                "urdf": str(geometry.urdf_path),
                "urdf_sha256": sha256(geometry.urdf_path),
                "head_link": HEAD_LINK,
                "sensor_link": SENSOR_LINK,
                "floor_contact_threshold_m": FLOOR_CONTACT_THRESHOLD_M,
            },
            "standing": {
                "foot_sole_min_z_m": float(min(feet[g][0][0] for g in FOOT_GROUPS)),
                "head_top_m": float(visual[HEAD_LINK][1][0]),
                "mid360_z_m": float(poses[SENSOR_LINK][1][0, 2]),
            },
            "table": table,
            "clips": clips,
            "goal_clips": 0,
            "timing": session.timing_summary(),
        }
        write_json(out / "posture_table.json", result)
    if args.command in ("goals", "all"):
        session.call_seconds.clear()
        goals, calibration, episodes, traces = goal_study(session, out)
        summary = summarize_goals(episodes)
        passing = [k for k, v in summary.items() if v["success_010_rate"] >= GATE_RATE]
        default_passing = [k for k in passing if k.endswith("x1")]
        if default_passing:
            statement = f"**PASS** at the deploy replan interval via {', '.join(default_passing)}."
        else:
            statement = "**FAIL** at the deploy replan interval."
        statement += f" Conditions at or above {GATE_RATE:.0%}: {', '.join(passing) or 'none'}."
        gate = {
            "rule": f"P0 or P1 final root-XY error <= {GATE_RADIUS_M} m in >= {GATE_RATE:.0%} "
            f"of {GOAL_COUNT} goals",
            "passing_conditions": passing,
            "passes_at_deploy_interval": bool(default_passing),
            "statement": statement,
        }
        result = {
            "provenance": provenance(session, model_sha),
            "protocol": {
                "goal_seed": GOAL_SEED,
                "goal_count": GOAL_COUNT,
                "distance_m": GOAL_DISTANCE_M,
                "horizon_s": GOAL_HORIZON_S,
                "replan_interval_scales": REPLAN_SCALES,
                "stop_phases_s": STOP_PHASES_S,
                "bands": [b.__dict__ for b in DEFAULT_BANDS],
                "p0_heading_deadband_deg": 5.0,
                "p0_freeze_radius_m": 0.3,
                "p0_tuning": "20 development goals, seed 70602",
            },
            "summary": summary,
            "gate": gate,
            "calibration": calibration,
            "episodes": episodes,
            "timing": session.timing_summary(),
        }
        write_json(out / "kinematic_goal_study.json", result)
        count = export_goal_clips(out, goals, episodes, traces)
        if (out / "posture_table.json").exists():
            posture = json.loads((out / "posture_table.json").read_text())
            posture["goal_clips"] = count
            write_json(out / "posture_table.json", posture)
    render_reports(out, args.notes)
    print(f"[done] {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
