#!/usr/bin/env python3
"""Anonymous video from pinned records; mj_forward visualization, never mj_step."""

from pathlib import Path

from audit_motion2scene_submission import DOC, Audit, dump
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from render_motion2scene_execution_demo import model_for

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.trajectory_export import convert_trajectory_joint_order_to_mujoco

OUT = DOC / "submission"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WHITE, BLUE, GREY = "#edf2f7", "#8bd5dc", "#b2bac5"


def text(draw, xy, value, size=24, fill=WHITE):
    draw.text(xy, value, font=ImageFont.truetype(FONT, size), fill=fill)


def card(title, lines):
    frame = Image.new("RGB", (1024, 576), "#172330")
    draw = ImageDraw.Draw(frame)
    text(draw, (34, 32), title, 32)
    for i, line in enumerate(lines):
        text(draw, (34, 118 + i * 50), line, 23, BLUE if i == 0 else WHITE)
    text(draw, (34, 535), "Motion2Scene | anonymous research submission", 17, GREY)
    return frame


def hold(writer, frame, seconds):
    for _ in range(25 * seconds):
        writer.append_data(np.asarray(frame))


def replay(audit, writer, records, titles, heading, subtitle, poster):
    models, traces, provenance = [], [], []
    for row in records:
        for name in ("trajectory", "physics", "decision"):
            audit.ref(row[name])
        payload = load_reset_capture(row["trajectory"]["path"])
        payload, joint_order = convert_trajectory_joint_order_to_mujoco(payload)
        q = np.concatenate([payload[k] for k in ("root_pos_w", "root_quat_w", "dof_pos")], 1)
        with np.load(row["physics"]["path"]) as archive:
            force = np.linalg.norm(archive["force_w"], axis=-1).max(1).reshape(-1, 4).max(1)
        assert len(q) == len(force) == 199 and row["reset_count"] == 0
        model, xml_hash = model_for(row["beam"])
        models.append((model, mujoco.MjData(model), mujoco.Renderer(model, 320, 512)))
        traces.append((q, force))
        provenance.append(
            {
                "cell_id": row["cell_id"],
                "trajectory": row["trajectory"],
                "physics": row["physics"],
                "decision": row["decision"],
                "beam": row["beam"],
                "joint_order": joint_order,
                "visual_xml_sha256": xml_hash,
                "pass": row["pass"],
            }
        )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 130, -12, 3.8
    try:
        for i in range(200):  # 4 recorded seconds replayed at half speed, 25 fps.
            k = min(i, 198)
            frame = Image.new("RGB", (1024, 576), "#172330")
            draw = ImageDraw.Draw(frame)
            text(draw, (22, 12), heading, 27)
            text(draw, (22, 50), subtitle, 18, BLUE)
            camera.lookat[:] = [*records[0]["beam"]["center_xy_m"], 0.9]
            camera.lookat[:2] = (traces[0][0][k, :2] + camera.lookat[:2]) / 2
            for j, ((model, data, renderer), (q, force), row, label) in enumerate(
                zip(models, traces, records, titles)
            ):
                data.qpos[:] = q[k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame.paste(Image.fromarray(renderer.render()), (j * 512, 155))
                text(draw, (j * 512 + 20, 91), label, 24)
                value = "PASS" if row["pass"] else "FAIL"
                text(
                    draw, (j * 512 + 20, 125), f"Passage: {value} | force now {force[k]:.1f} N", 19
                )
            text(
                draw, (22, 486), f"t = {k / 50:.2f} s | decision at 0.30 s | replay speed 0.5x", 21
            )
            text(
                draw,
                (22, 518),
                "Recorded Isaac states and contacts; MuJoCo visual replay only.",
                18,
                GREY,
            )
            text(
                draw,
                (22, 545),
                "No dynamics integration; visual mesh differs from collision bodies.",
                17,
                GREY,
            )
            writer.append_data(np.asarray(frame))
            if i == 75:
                frame.save(poster)
    finally:
        for _, _, renderer in models:
            renderer.close()
    return provenance


def main():
    audit = Audit()
    analysis = audit.read(OUT / "evidence/analysis.json")

    def row(ident):
        reduced = next(r for r in analysis["seed_level_rows"] if r["cell_id"] == ident)
        result = audit.read(reduced["result"]["path"], reduced["result"]["sha256"])
        return next(r for r in result["rows"] if r["cell_id"] == ident)

    pair = next(
        p
        for p in analysis["studies"]["nominal"]["pairs"]
        if p["group_id"] == "nom_41001_analytic_00"
    )
    training = [row(ident) for ident in pair["row_ids"]]
    failure = [
        row("nom_eval_41001_layout_01_p8511_analytic"),
        row("nom_eval_41001_layout_01_p8511_motion2scene"),
    ]
    assert [r["pass"] for r in training] == [False, True]
    assert [r["pass"] for r in failure] == [True, False]
    path = OUT / "motion2scene-anonymous.mp4"
    scenes = []
    with imageio.get_writer(
        path,
        fps=25,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart", "-map_metadata", "-1", "-pix_fmt", "yuv420p"],
    ) as writer:
        hold(
            writer,
            card(
                "Motion2Scene",
                [
                    "Constructing scenes from executed humanoid contrasts",
                    "Question: which scenes teach when to adapt?",
                    "Frozen controller: walk or request d040 once at 0.30 s.",
                    "Paired physical labels; one overhead-beam family.",
                    "Simulation study on three development motion carriers.",
                ],
            ),
            8,
        )
        scenes += replay(
            audit,
            writer,
            training,
            ["Walk command", "Request d040"],
            "Executed alternatives produce a useful training contrast",
            "Nominal analytic training scene | C1 | matched label seed 8722",
            OUT / "figures/video-training-preview.png",
        )
        observation = card("Decision-time information can be ambiguous", [])
        draw = ImageDraw.Draw(observation)
        text(draw, (34, 98), "Same 214 inputs at 0.30 s; conflicting physical labels", 24, BLUE)
        figure = OUT / "figures/full-input-aliasing.png"
        audit.pin(figure)
        panel = ImageOps.contain(Image.open(figure).convert("RGB"), (960, 290))
        observation.paste(panel, ((1024 - panel.width) // 2, 155))
        text(draw, (34, 463), "Post-hoc diagnostic: all original training groups retained.", 21)
        text(draw, (34, 500), "Fitting ambiguity does not imply an unavoidable passage error.", 20)
        hold(writer, observation, 10)
        scenes += replay(
            audit,
            writer,
            failure,
            ["Analytic selector: requests d040", "Learned-corpus selector: walks"],
            "A missed adaptation in the nominal evaluation bank",
            "C1 | station 0.35 | underside 1.27 m | matched physics seed 8511",
            OUT / "figures/video-failure-preview.png",
        )
        hold(
            writer,
            card(
                "The same 36 nominal conditions",
                [
                    "Passage: uniform / target-only 14; analytic 24; learned 22.",
                    "Scripted rays: 24. Always d040: 24 (post-hoc matched audit).",
                    "Unnecessary requests on 14 both-pass conditions:",
                    "always d040 14; analytic 8; learned 6.",
                    "Learned construction misses 2 of the 10 available rescues.",
                    "15 groups per arm: equal labels, unequal acquisition compute.",
                ],
            ),
            12,
        )
        hold(
            writer,
            card(
                "Limits determine the claim",
                [
                    "Retained perturbation contract: contrast eligibility is scarce.",
                    "The successful nominal study uses a narrower pose contract.",
                    "31/36 conditions had earlier outcomes before registration.",
                    "No source-held-out, hardware, or robustness-transfer claim.",
                    "Contrast targeting helps here; learned superiority is unproved.",
                    "Refusal continues walking. It is not a protective stop.",
                ],
            ),
            10,
        )
    dump(
        OUT / "evidence/video-provenance.json",
        {
            "post_hoc_illustration": True,
            "video": audit.pin(path),
            "script": audit.pin(Path(__file__)),
            "inputs": list(audit.files.values()),
            "scenes": scenes,
            "duration_seconds": 56,
            "fps": 25,
            "size": [1024, 576],
            "selection": (
                "One named analytic training contrast, and the first carrier/layout where the "
                "nominal analytic selector passes and learned-corpus selector fails; illustrative, "
                "not a new evaluation sample. All aggregate results use the complete bank."
            ),
            "scope": "Recorded-state visual replay via mj_forward only; no new physical simulation.",
        },
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
