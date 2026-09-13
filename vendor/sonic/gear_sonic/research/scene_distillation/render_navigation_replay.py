"""Matched MuJoCo mesh replay of measured Isaac student poses and task outcomes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def scene_model(robot_xml, task):
    tree = ET.parse(robot_xml).getroot()
    tree.find("compiler").set("meshdir", str(robot_xml.parent / "meshes"))
    visual = tree.find("visual")
    if visual is None:
        visual = ET.SubElement(tree, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="640")
    ET.SubElement(visual, "headlight", ambient=".5 .5 .5", diffuse=".7 .7 .7", specular=".1 .1 .1")
    asset = tree.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="replay_floor",
        type="2d",
        builtin="checker",
        rgb1=".84 .86 .88",
        rgb2=".73 .77 .80",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="replay_floor",
        texture="replay_floor",
        texrepeat="12 12",
        texuniform="true",
        reflectance=".05",
    )
    world = tree.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="replay_floor",
        type="plane",
        size="20 20 .01",
        material="replay_floor",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", diffuse=".8 .8 .8")
    for i, obstacle in enumerate(task["obstacles"]):
        if obstacle["shape"] not in ("box", "beam"):
            raise ValueError("This replay supports the recorded box/corridor tasks")
        ET.SubElement(
            world,
            "geom",
            name=f"replay_obstacle_{i}",
            type="box",
            pos=" ".join(map(str, obstacle["center_xyz"])),
            quat=" ".join(map(str, obstacle["quaternion_wxyz"])),
            size=" ".join(str(x / 2) for x in obstacle["full_dimensions_xyz"]),
            rgba=".26 .42 .55 1",
            contype="0",
            conaffinity="0",
        )
    goal = np.array(task["goal_xyz"])
    for i in range(48):
        angles = np.array([i, i + 1]) * 2 * np.pi / 48
        xy = goal[:2] + task["goal_tolerance_m"] * np.stack([np.cos(angles), np.sin(angles)], -1)
        ends = np.column_stack([xy, [0.018, 0.018]])
        ET.SubElement(
            world,
            "geom",
            name=f"goal_ring_{i}",
            type="capsule",
            size=".012",
            fromto=" ".join(map(str, ends.flatten())),
            rgba=".08 .72 .28 1",
            contype="0",
            conaffinity="0",
        )
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))


def load_recording(packet, name):
    folder = packet / name / "task"
    config = json.loads((packet / (name + "-config.json")).read_text())
    score = json.loads((folder / "task-result.json").read_text())
    task = json.loads(Path(config["task_path"]).read_text())
    with np.load(folder / "robot-poses.npz") as f:
        poses = {k: f[k].copy() for k in f.files}
    with np.load(folder / "trace.npz") as f:
        trace = {k: f[k].copy() for k in f.files}
    n = score["control_steps"]
    assert len(poses["root_xyz"]) == n == len(trace["root_xyz"])
    np.testing.assert_allclose(poses["root_xyz"], trace["root_xyz"], atol=1e-5)
    distance = np.linalg.norm(trace["root_xyz"] - task["goal_xyz"], axis=1)
    hold, run = [], 0
    for ok in (distance <= 0.25) & (trace["speed"] <= 0.1):
        run = run + 1 if ok else 0
        hold.append(run)
    assert max(hold) == score["max_hold_ticks"]
    return dict(
        config=config,
        score=score,
        task=task,
        poses=poses,
        trace=trace,
        distance=distance,
        hold=hold,
        n=n,
        folder=folder,
    )


def replay_frame(model, data, renderer, camera, record, index):
    poses = record["poses"]
    data.qpos[:3] = poses["root_xyz"][index]
    data.qpos[3:7] = poses["root_wxyz"][index]
    for name, value in zip(poses["joint_names"], poses["joint_pos"][index]):
        data.qpos[model.jnt_qposadr[model.joint(str(name)).id]] = value
    mujoco.mj_forward(model, data)
    # Set each visual geometry from the recorded body transform. This avoids
    # small fixed-link offsets between the Isaac and MuJoCo asset variants.
    lookup = {str(n): i for i, n in enumerate(poses["body_names"])}
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if b == 0:
            continue
        name = model.body(b).name
        if name not in lookup:
            raise ValueError(f"Unrecorded robot body: {name}")
        k = lookup[name]
        body_rot, local_rot = np.empty(9), np.empty(9)
        mujoco.mju_quat2Mat(body_rot, poses["body_wxyz"][index, k].astype(float))
        mujoco.mju_quat2Mat(local_rot, model.geom_quat[g])
        body_rot = body_rot.reshape(3, 3)
        data.geom_xpos[g] = poses["body_xyz"][index, k] + body_rot @ model.geom_pos[g]
        data.geom_xmat[g] = (body_rot @ local_rot.reshape(3, 3)).ravel()
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def render_pair(packet, output, key, robot_xml):
    records = [load_recording(packet, p + "-" + key) for p in ["full", "nav"]]
    assert records[0]["score"]["task_sha256"] == records[1]["score"]["task_sha256"]
    task = records[0]["task"]
    model = scene_model(robot_xml, task)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=640, width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    start, goal = np.array(task["start_xyz"]), np.array(task["goal_xyz"])
    direction = goal - start
    direction[2] = 0
    camera.lookat[:] = (start + goal) / 2
    camera.lookat[2] = 0.65
    camera.distance = max(4.3, np.linalg.norm(direction) * 1.35 + 1.4)
    camera.elevation = -35
    camera.azimuth = np.degrees(np.arctan2(direction[1], direction[0])) + 12
    fonts = {
        s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
        for s in [20, 23, 27, 32]
    }
    color = ["#62b5ff", "#ffbe66"]
    titles = ["FULL-COMMAND STUDENT", "GOAL + SCENE STUDENT"]
    inputs = ["Current motion targets + measured history", "Goal + obstacle map + measured history"]
    nframes = int(np.ceil(max(r["n"] for r in records) / 2)) + 30
    path = output / (key + ".mp4")
    cached = [None, None]
    last = [None, None]
    writer = imageio.get_writer(
        str(path),
        fps=25,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", "19", "-preset", "fast"],
    )
    try:
        for frame in range(nframes):
            canvas = Image.new("RGB", (1920, 864), "#101a28")
            draw = ImageDraw.Draw(canvas)
            tick = (frame + 1) * 2
            for panel, r in enumerate(records):
                k = min(tick - 1, r["n"] - 1)
                x = panel * 960
                if last[panel] != k:
                    cached[panel] = replay_frame(model, data, renderer, camera, r, k)
                    last[panel] = k
                canvas.paste(Image.fromarray(cached[panel]), (x, 96))
                draw.text((x + 25, 12), titles[panel], font=fonts[32], fill=color[panel])
                draw.text((x + 25, 55), inputs[panel], font=fonts[23], fill="#d1dcea")
                ended = tick >= r["n"]
                state = (
                    (
                        "PASS: goal held"
                        if r["score"]["navigation_success"]
                        else "NOT COMPLETED: deadline"
                    )
                    if ended
                    else "Running"
                )
                draw.rectangle((x + 15, 107, x + 600, 144), fill="#162232")
                draw.text(
                    (x + 25, 111),
                    f"{key}  |  t={min(tick,r['n'])*.02:.2f}s  |  {state}",
                    font=fonts[20],
                    fill="white",
                )
                draw.text(
                    (x + 25, 750),
                    f"Goal distance {r['distance'][k]:.2f} m / 0.25 m   |   "
                    f"Speed {r['trace']['speed'][k]:.2f} / 0.10 m/s",
                    font=fonts[23],
                    fill="white",
                )
                draw.text(
                    (x + 25, 786),
                    f"Consecutive hold: {r['hold'][k]} / 50 ticks"
                    + ("   |   Final frame held" if ended else ""),
                    font=fonts[23],
                    fill=color[panel],
                )
            draw.line((960, 0, 960, 864), fill="#101a28", width=6)
            draw.text(
                (28, 831),
                "Measured Isaac physics shown as MuJoCo mesh replay | 1x speed | "
                "Green ring: goal region | Same task, fixed camera",
                font=fonts[20],
                fill="#bccad9",
            )
            writer.append_data(np.asarray(canvas))
            if frame == min(100, nframes - 1):
                canvas.save(output / (key + ".png"))
    finally:
        writer.close()
        renderer.close()
    result = dict(
        task=key,
        video=str(path),
        frames=nframes,
        fps=25,
        robot_xml=str(robot_xml),
        provenance="measured Isaac link transforms rendered in MuJoCo; no resimulation or reference animation",
        full=records[0]["score"],
        navigation=records[1]["score"],
        sources={
            str(r["folder"] / "robot-poses.npz"): sha(r["folder"] / "robot-poses.npz")
            for r in records
        },
    )
    write_new(output / (key + "-receipt.json"), result)
    print(
        key,
        records[0]["score"]["navigation_success"],
        records[1]["score"]["navigation_success"],
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "gear_sonic_deploy/g1/g1_29dof_old.xml",
    )
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    keys = ["00908-stop-corridor", "00413-stop-clear", "00265-stop-corridor"]
    results = [render_pair(a.packet, a.output, k, a.robot_xml) for k in keys]
    concat = a.output / "concat.txt"
    concat.write_text("".join("file '" + k + ".mp4'\n" for k in keys))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(a.output / "student-comparison.mp4"),
        ],
        check=True,
    )
    write_new(a.output / "results.json", results)
    write_new(a.output / "video-sha256.json", {f.name: sha(f) for f in a.output.glob("*.mp4")})


if __name__ == "__main__":
    main()
