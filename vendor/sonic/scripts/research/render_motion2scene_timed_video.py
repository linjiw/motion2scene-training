#!/usr/bin/env python3
"""Visualize recorded six-second physical alternatives without dynamics integration."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from motion2scene_timing_diagnostic import artifact, checked, write_new
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_motion2scene_execution_demo import model_for

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.trajectory_export import convert_trajectory_joint_order_to_mujoco


def render(result_path, out, preview=False):
    result = json.loads(result_path.read_text())
    ref = result["manifest"]
    manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    ref = manifest["scene_definition"]
    scene = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    if scene["split"] != "development" or len(result["rows"]) != 3:
        raise ValueError("three matched development alternatives required")
    rows, models, traces, sources = result["rows"], [], [], []
    beam = scene["beam"]
    labels = ("Walk", "Short adaptation", "Sustained adaptation")
    for row in rows:
        for key in ("trajectory", "sensor", "physics_beam_contacts"):
            checked(Path(row[key]["path"]), row[key]["sha256"])
        payload = load_reset_capture(row["trajectory"]["path"])
        converted, mapping = convert_trajectory_joint_order_to_mujoco(payload)
        q = np.concatenate([converted[key] for key in ("root_pos_w", "root_quat_w", "dof_pos")], 1)
        with np.load(row["physics_beam_contacts"]["path"], allow_pickle=False) as data:
            force = np.linalg.norm(data["force_w"], axis=-1).max(axis=1).reshape(-1, 4).max(axis=1)
        if len(q) != 298 or len(force) != 298 or not row["measurement_admitted"]:
            raise ValueError("complete synchronized six-second physical records required")
        model, xml_hash = model_for(beam)
        model.vis.headlight.ambient[:] = 0.55
        model.vis.headlight.diffuse[:] = 0.8
        robot_geoms = model.geom_bodyid > 0
        model.geom_matid[robot_geoms] = -1
        model.geom_rgba[robot_geoms] = [0.65, 0.70, 0.78, 1.0]
        models.append((model, mujoco.MjData(model), mujoco.Renderer(model, 320, 400)))
        traces.append((q, force))
        sources.append(dict(row=row, joint_order=mapping, visual_xml_sha256=xml_hash))
    out.mkdir(parents=True, exist_ok=False)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fonts = {size: ImageFont.truetype(font_path, size) for size in (16, 20, 24, 29)}
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 90, -6, 3.4
    writer = None
    try:
        if not preview:
            writer = imageio.get_writer(
                out / "duration_execution.mp4",
                fps=25,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                ffmpeg_params=[
                    "-movflags",
                    "+faststart",
                    "-map_metadata",
                    "-1",
                    "-pix_fmt",
                    "yuv420p",
                ],
            )
        for k in ([136] if preview else list(range(0, 298, 2))):
            canvas = Image.new("RGB", (1200, 520), "#142330")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (20, 14),
                "Motion2Scene | Executed duration alternatives",
                font=fonts[29],
                fill="#edf4f7",
            )
            draw.text(
                (20, 52),
                f"Development passage | length {beam['length_m']:.2f} m | underside {beam['underside_m']:.3f} m",
                font=fonts[20],
                fill="#91d3d9",
            )
            camera.lookat[:] = [*beam["center_xy_m"], 0.8]
            camera.lookat[:2] = (traces[0][0][k, :2] + camera.lookat[:2]) / 2
            for j, ((model, data, renderer), (q, force), row, label) in enumerate(
                zip(models, traces, rows, labels, strict=True)
            ):
                data.qpos[:] = q[k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (j * 400, 130))
                draw.text((j * 400 + 14, 88), label, font=fonts[24], fill="#edf4f7")
                status = "PASS" if row["pass"] else "FAIL"
                draw.text(
                    (j * 400 + 14, 115),
                    f"Full trial: {status} | beam force now {force[k]:.1f} N",
                    font=fonts[16],
                    fill="#91d3d9" if row["pass"] else "#f0b18d",
                )
            draw.text(
                (20, 458),
                f"Recorded time {k / 50:.2f} s | real-time replay | complete physical outcomes shown above",
                font=fonts[20],
                fill="#edf4f7",
            )
            draw.text(
                (20, 490),
                "IsaacLab states and measured contacts; MuJoCo visual replay only. "
                "Visual meshes differ from native collision bodies.",
                font=fonts[16],
                fill="#bac6cf",
            )
            if preview or k == 136:
                canvas.save(out / "preview.png")
            if writer is not None:
                writer.append_data(np.asarray(canvas))
    finally:
        if writer is not None:
            writer.close()
        for _, _, renderer in models:
            renderer.close()
    write_new(
        out / "receipt.json",
        dict(
            source=artifact(result_path),
            scene=scene,
            records=sources,
            video=artifact(out / "duration_execution.mp4") if not preview else None,
            preview=artifact(out / "preview.png"),
            scope=(
                "recorded development PhysX outcomes; visual forward kinematics only, "
                "no MuJoCo physics or new rollouts"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    render(args.result, args.out, args.preview)
