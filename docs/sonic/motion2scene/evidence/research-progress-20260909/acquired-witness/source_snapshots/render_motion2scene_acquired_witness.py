#!/usr/bin/env python3
"""Render audited acquired response witnesses from recorded states, without stepping physics.

The pair is identified after inspecting the recorded outcomes. All seven bank
outcomes are retained; the two displayed schedules illustrate the disjoint pair.
Frames share a control index within each scene, chosen at the failed displayed
schedule's maximum measured beam force. Visual meshes are not collision bodies.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from render_motion2scene_execution_demo import model_for  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.trajectory_export import (  # noqa: E402
    convert_trajectory_joint_order_to_mujoco,
)

OPTIONS = ("sustained_e070_r255", "prior_splice_e050_r265")
LABELS = ("Sustained, enter 1.40 s", "Prior splice, enter 1.00 s")


def teacher_group(task, result):
    previous = read_checked(result["previous_audit"])
    corpus = next((c for c in previous["corpora"] if c["run_id"] == task["run_id"]), None)
    if task["round"] <= 2 and corpus is not None:
        group = read_checked(read_checked(corpus["checkpoint"])["teachers"])[task["round"]]
    else:
        registration = read_checked(result["registration"])
        row = next(r for r in registration["new_collections"] if r["run_id"] == task["run_id"])
        group = read_checked(read_checked(row["trained"])["teachers"])[task["round"]]
    if group["collection"] != task["collection"]:
        raise ValueError("recorded feature provenance differs from the audited collection")
    return group


def recorded_branch(row):
    for name in ("trajectory", "sensor", "physics_beam_contacts"):
        checked(Path(row[name]["path"]), row[name]["sha256"])
    if not row["measurement_admitted"] or row["outcome"]["task_outcome"] not in ("pass", "failure"):
        raise ValueError("illustration requires complete independently audited physical records")
    payload = load_reset_capture(row["trajectory"]["path"])
    converted, mapping = convert_trajectory_joint_order_to_mujoco(payload)
    q = np.concatenate([converted[k] for k in ("root_pos_w", "root_quat_w", "dof_pos")], axis=1)
    with np.load(row["physics_beam_contacts"]["path"], allow_pickle=False) as data:
        force = np.linalg.norm(data["force_w"], axis=-1).max(axis=1)
        if len(force) != 4 * len(q) or not np.array_equal(
            data["control_steps"], data["physics_steps"][3::4]
        ):
            raise ValueError("four physics samples per recorded control state required")
    times = np.asarray(payload["motion_time_s"]).reshape(-1)
    if len(q) != 298 or len(times) != len(q) or not np.all(np.isfinite(q)):
        raise ValueError("complete finite six-second capture required")
    return q, times, force, mapping


def image_at(beam, q, index, lookat):
    model, xml_hash = model_for(beam)
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.8
    robot_geoms = model.geom_bodyid > 0
    model.geom_matid[robot_geoms] = -1
    model.geom_rgba[robot_geoms] = [0.65, 0.70, 0.78, 1]
    data = mujoco.MjData(model)
    data.qpos[:] = q[index]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 90, -8, 3.1
    camera.lookat[:] = lookat
    renderer = mujoco.Renderer(model, 380, 560)
    try:
        renderer.update_scene(data, camera=camera)
        return renderer.render(), xml_hash
    finally:
        renderer.close()


def render(result_path, out):
    result_ref = artifact(result_path)
    result = read_checked(result_ref)
    pairs = result["same_seed_information_pairs"]
    if len(pairs) != 1:
        raise ValueError("this bounded illustration requires exactly one audited same-seed pair")
    pair = pairs[0]
    tasks = [
        next(
            t
            for t in result["unique_conditions"]
            if t["candidate_id"] == pair[side] and t["seed"] == pair[side + "_seed"]
        )
        for side in ("left", "right")
    ]
    if set(pair["left_passing"]) & set(pair["right_passing"]):
        raise ValueError("the displayed pair must have disjoint measured passing schedules")
    out.mkdir(parents=True, exist_ok=False)
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.7), width_ratios=(0.94, 1, 1))
    records, feature_rows, bank_rows = [], [], []
    schedule_ids = list(teacher_group(tasks[0], result)["option_ids"])
    for i, task in enumerate(tasks):
        group = teacher_group(task, result)
        target = next(t for t in group["targets"] if t["phase_tick"] == 15)
        if not target.get("available", True) or len(target["features"]) != 114:
            raise ValueError("the original first-phase observation must be available")
        collection = read_checked(task["collection"])
        manifest = read_checked(collection["manifest"])
        scene = read_checked(manifest["scene_definition"])
        if scene["split"] != "development" or len(scene["beams"]) != 1:
            raise ValueError("only the acquired single-beam development pair is illustrated")
        rows = [next(r for r in collection["rows"] if r["forced_option_id"] == s) for s in OPTIONS]
        captures = [recorded_branch(r) for r in rows]
        failures = [j for j, row in enumerate(rows) if row["outcome"]["task_outcome"] == "failure"]
        if len(failures) != 1 or any(
            task["outcomes"][row["forced_option_id"]] != row["outcome"]["task_outcome"]
            for row in rows
        ):
            raise ValueError(
                "the illustrated schedules must reproduce the audited pass/fail reversal"
            )
        frame = int(captures[failures[0]][2].argmax() // 4)
        beam = scene["beams"][0]
        lookat = np.array([*beam["center_xy_m"], 0.8])
        lookat[:2] = (captures[0][0][frame, :2] + lookat[:2]) / 2
        label = "A" if i == 0 else "B"
        ax = fig.add_subplot(grid[i, 0])
        indices = [[4 + 7 * j for j in range(4)], [1 + 7 * j for j in range(4)]]
        values = np.asarray(target["features"])[indices]
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect=0.8)
        ax.set_xticks(range(4), ["0–0.75", "0.75–1.5", "1.5–2.5", "2.5–4"], rotation=25)
        ax.set_yticks([0, 1], ["Upper hit", "Ceiling\nfraction"])
        ax.set_xlabel("Observed corridor distance band (m)")
        ax.set_title(
            f"Scene {label}: observation at 0.30 s\n{task['candidate_id'].removeprefix('primary_candidate_')}"
        )
        for v in range(2):
            for j in range(4):
                ax.text(
                    j,
                    v,
                    f"{values[v, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if values[v, j] > 0.5 else "black",
                )
        for k, value in enumerate(target["features"]):
            feature_rows.append(
                dict(
                    scene=label,
                    candidate_id=task["candidate_id"],
                    phase_tick=15,
                    feature=target["feature_names"][k],
                    value=value,
                )
            )
        for j, (row, (q, times, force, mapping)) in enumerate(zip(rows, captures, strict=True)):
            image, xml_hash = image_at(beam, q, frame, lookat)
            ax = fig.add_subplot(grid[i, j + 1])
            ax.imshow(image)
            ax.set_axis_off()
            passed = row["outcome"]["task_outcome"] == "pass"
            ax.set_title(
                f"{LABELS[j]}\nFull trial: {'PASS' if passed else 'FAIL'}",
                color="#14765c" if passed else "#a23731",
            )
            ax.text(
                0.5,
                -0.03,
                f"Recorded motion time {times[frame]:.2f} s; peak beam force {force.max():.0f} N",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )
            records.append(
                dict(
                    scene=label,
                    candidate_id=task["candidate_id"],
                    collection=task["collection"],
                    scene_definition=manifest["scene_definition"],
                    option_id=row["forced_option_id"],
                    trajectory=row["trajectory"],
                    sensor=row["sensor"],
                    physics_beam_contacts=row["physics_beam_contacts"],
                    visual_xml_sha256=xml_hash,
                    joint_order=mapping,
                    displayed_control_index=frame,
                    displayed_motion_time_s=float(times[frame]),
                    maximum_measured_beam_force_n=float(force.max()),
                    full_trial_outcome=row["outcome"]["task_outcome"],
                )
            )
        for schedule in schedule_ids:
            bank_rows.append(
                dict(
                    scene=label,
                    candidate_id=task["candidate_id"],
                    seed=task["seed"],
                    schedule_id=schedule,
                    outcome=task["outcomes"][schedule],
                )
            )
    ax = fig.add_subplot(grid[2, :])
    ax.set_axis_off()
    short = [
        s.replace("_e", "\nenter ")
        .replace("_r", " / recover ")
        .replace("prior_splice", "prior splice")
        for s in schedule_ids
    ]
    table = ax.table(
        cellText=[
            ["PASS" if t["outcomes"][s] == "pass" else "FAIL" for s in schedule_ids] for t in tasks
        ],
        rowLabels=["Scene A", "Scene B"],
        colLabels=short,
        cellLoc="center",
        bbox=[0.025, 0.27, 0.97, 0.63],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row > 0 and col >= 0:
            cell.set_facecolor(
                "#dceddf" if tasks[row - 1]["outcomes"][schedule_ids[col]] == "pass" else "#f7ded9"
            )
    ax.text(
        0.5,
        0.13,
        "Complete measured bank outcomes; entry/recovery labels above use control ticks (50 Hz).",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.text(
        0.5,
        0.01,
        "Recorded IsaacLab states rendered with MuJoCo forward kinematics only. "
        "Visual meshes differ from native collision bodies.",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.suptitle("An acquired pair requires different complete schedules", fontsize=15)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"acquired_response_witness.{suffix}", dpi=180)
    plt.close(fig)
    for name, rows in (
        ("first_phase_features", feature_rows),
        ("complete_schedule_outcomes", bank_rows),
    ):
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return write_new(
        out / "receipt.json",
        dict(
            source=result_ref,
            implementation=artifact(Path(__file__)),
            visual_model_implementation=artifact(
                ROOT / "scripts/research/render_motion2scene_execution_demo.py"
            ),
            records=records,
            files=[
                artifact(p) for p in sorted(out.iterdir()) if p.suffix in (".pdf", ".png", ".csv")
            ],
            scope=__doc__,
            new_physics_steps=0,
            interpretation=(
                "Post-hoc acquired-pool illustration, not a balanced arm comparison or evidence of "
                "robust observational discrimination or policy generalization."
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.audit, args.out)))
