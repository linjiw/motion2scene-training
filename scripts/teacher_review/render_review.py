#!/usr/bin/env python3
"""Render review MP4s: release | 8192x500 | previous-8000 G1 meshes, each with the reference ghost.

MuJoCo EGL offscreen replay of captured Isaac Lab states (pose_capture.py). Errors, completion and
failure times come from the native Isaac trajectories / metrics, never from the mesh replay.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_scene import G1Scene  # noqa: E402
from review_common import (  # noqa: E402
    ARMS, DEFAULT_REVIEW, Run, ledger_by_key, metrics_dir, quat_yaw_wxyz,
)

PANEL_W, PANEL_H = 640, 480
HEADER_H, TIMELINE_H = 60, 260
WIDTH, HEIGHT = PANEL_W * 3, HEADER_H + PANEL_H + TIMELINE_H  # 1920 x 800
FRAME_STRIDE = 2  # 50 Hz control steps -> 25 fps real time
OUT_FPS = 25
TAIL_S = 1.5
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"
FONT = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 17)
FONT_SMALL = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 14)
FONT_BOLD = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 20)


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def smooth(values, tau_frames):
    """Zero-phase exponential smoothing (forward + backward pass)."""
    alpha = 1.0 - np.exp(-1.0 / max(tau_frames, 1e-6))
    out = np.asarray(values, dtype=np.float64)
    forward = out.copy()
    for index in range(1, len(out)):
        forward[index] = forward[index - 1] + alpha * (out[index] - forward[index - 1])
    backward = forward.copy()
    for index in range(len(out) - 2, -1, -1):
        backward[index] = backward[index + 1] + alpha * (forward[index] - backward[index + 1])
    return backward


def camera_track(anchor, T):
    ghost = np.stack([anchor.track_q[anchor.ghost_index(k)] for k in range(T)])
    look_x = smooth(ghost[:, 0], 0.35 / anchor.dt)
    look_y = smooth(ghost[:, 1], 0.35 / anchor.dt)
    look_z = smooth(ghost[:, 2], 0.5 / anchor.dt) - 0.05  # follows crouches / raised references
    yaw = smooth(np.unwrap(quat_yaw_wxyz(ghost[:, 3:7])), 1.5 / anchor.dt)
    return look_x, look_y, look_z, np.degrees(yaw)


def timeline_background(runs, T, dt):
    fig = plt.figure(figsize=(WIDTH / 100, TIMELINE_H / 100), dpi=100, facecolor="#f5f7fb")
    ax_err = fig.add_axes([0.045, 0.56, 0.80, 0.38])
    ax_xy = fig.add_axes([0.045, 0.14, 0.80, 0.38], sharex=ax_err)
    t = np.arange(T) * dt
    err_top = 150.0
    xy_top = 20.0
    for run in runs:
        spec = ARMS[run.arm]
        v = run.valid
        status = "complete" if run.completed else f"failed {v * dt:.2f}s"
        ax_err.plot(t[:v], run.err_global_mm[:v], color=spec["color"], lw=1.8,
                    label=f"{spec['short']}: {status}")
        ax_xy.plot(t[:v], 100 * run.root_xy_m[:v], color=spec["color"], lw=1.6)
        if v:
            err_top = max(err_top, float(np.percentile(run.err_global_mm[:v], 98)) * 1.15)
            xy_top = max(xy_top, float(100 * run.root_xy_m[:v].max()) * 1.15)
        if not run.completed:
            for axis in (ax_err, ax_xy):
                axis.axvline(v * dt, color=spec["color"], ls=":", lw=1.6)
    ax_err.axhspan(0, 100, color="#2ca02c", alpha=0.06)
    ax_err.set_ylim(0, err_top)
    ax_xy.set_ylim(0, xy_top)
    ax_err.set_xlim(0, T * dt)
    ax_err.set_ylabel("body err (mm)", fontsize=9)
    ax_xy.set_ylabel("root XY (cm)", fontsize=9)
    ax_xy.set_xlabel("time (s)", fontsize=9)
    for axis in (ax_err, ax_xy):
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=8)
    plt.setp(ax_err.get_xticklabels(), visible=False)
    ax_err.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=9, frameon=False)
    fig.text(0.855, 0.10, "gray robot = simulated policy\nblue ghost = reference motion\n"
             "red robot = frozen at failure step\ncurves: native Isaac 14-body positions,\n"
             "stopped at failure (dotted)", fontsize=8, color="#555555")
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0 = ax_err.transData.transform((0, 0))[0]
    x1 = ax_err.transData.transform((T * dt, 0))[0]
    y_top = TIMELINE_H - ax_err.bbox.y1
    y_bottom = TIMELINE_H - ax_xy.bbox.y0
    plt.close(fig)
    return image, (x0, x1, int(y_top), int(y_bottom))


def status_for(run, k, dt):
    v = run.valid
    if k < v:
        if run.completed and k >= run.T - 1:
            return ("COMPLETE  mean body err {:.0f} mm  final root XY {:.0f} cm".format(
                run.err_global_mm[:v].mean(), 100 * run.root_xy_m[v - 1]), (40, 140, 60), (255, 255, 255))
        return ("TRACKING  body err {:4.0f} mm  root XY {:3.0f} cm".format(
            run.err_global_mm[k], 100 * run.root_xy_m[k]), (255, 255, 255), (25, 25, 25))
    terms = "+".join(run.failure_terms) or "termination"
    return (f"FAILED at {v * dt:.2f} s ({run.progress:.0%})  {terms}", (205, 40, 40), (255, 255, 255))


def render_motion(scene, renderer, review, key, split, arms, ledger, featured, out_dir, poster_dir):
    runs = [Run(review, arm, split, key) for arm in arms]
    # Recorded length can differ between arms for FAILED motions (the batch loop stops when the
    # longest surviving motion ends), so the video spans the full reference: n - 1 frames.
    if len({run.n for run in runs}) != 1:
        raise RuntimeError(f"{key}: arms disagree on motion_num_steps {[r.n for r in runs]}")
    T, dt = runs[0].n - 1, runs[0].dt
    anchor = max(runs, key=lambda r: r.F)
    look_x, look_y, look_z, yaw_deg = camera_track(anchor, T)
    timeline, (x0, x1, y_top, y_bottom) = timeline_background(runs, T, dt)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.elevation, camera.distance = -14.0, 3.3
    row = ledger.get(key, {})
    prompts = " -> ".join(row.get("prompts", []))
    if len(prompts) > 190:
        prompts = prompts[:187] + "..."
    tag = featured.get(key)
    header1 = (f"{key}  |  {'HELD-OUT DEV' if split == 'development' else 'TRAIN'}  |  "
               f"{row.get('category', '?')}  |  {T * dt:.1f} s  |  route {row.get('route', '?')}")
    header2 = (f"[{tag}]  " if tag else "") + prompts
    frames = list(range(0, T, FRAME_STRIDE))
    frames += [T - 1] * int(TAIL_S * OUT_FPS)
    tmp_path = out_dir / f".{key}.tmp.mp4"
    writer = imageio.get_writer(tmp_path, fps=OUT_FPS, codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p", ffmpeg_log_level="error")
    poster_frame = frames[len(frames) // 2 - int(TAIL_S * OUT_FPS) // 2]
    started = time.perf_counter()
    for out_index, k in enumerate(frames):
        canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for panel, run in enumerate(runs):
            alive = k < run.valid
            # After a failure the panel freezes robot, ghost and camera at the last valid frame,
            # so the failure pose stays visible next to the reference it was tracking.
            shown = k if alive else max(run.valid - 1, 0)
            camera.lookat[:] = (look_x[shown], look_y[shown], look_z[shown])
            camera.azimuth = yaw_deg[shown] + 150.0
            scene.pose(run.robot_q[shown], anchor.track_q[anchor.ghost_index(shown)])
            scene.tint_robot(not alive)
            renderer.update_scene(scene.data, camera=camera, scene_option=scene.option)
            canvas[HEADER_H : HEADER_H + PANEL_H, panel * PANEL_W : (panel + 1) * PANEL_W] = renderer.render()
        scene.tint_robot(False)
        strip = timeline.copy()
        cursor = int(round(x0 + (x1 - x0) * k / max(T, 1)))
        strip[y_top:y_bottom, max(cursor - 1, 0) : cursor + 1] = (30, 30, 30)
        canvas[HEADER_H + PANEL_H :] = strip
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, WIDTH, HEADER_H], fill=(24, 35, 52))
        draw.text((12, 6), header1 + f"  |  t = {k * dt:5.2f} s", font=FONT_BOLD, fill=(255, 255, 255))
        draw.text((12, 36), header2, font=FONT_SMALL, fill=(200, 210, 225))
        for panel, run in enumerate(runs):
            left = panel * PANEL_W
            spec = ARMS[run.arm]
            draw.rectangle([left + 8, HEADER_H + 8, left + 300, HEADER_H + 34], fill=hex_rgb(spec["color"]))
            draw.text((left + 14, HEADER_H + 11), spec["label"], font=FONT, fill=(255, 255, 255))
            text, background, foreground = status_for(run, k, dt)
            top = HEADER_H + PANEL_H - 30
            draw.rectangle([left, top, left + PANEL_W, top + 30], fill=background)
            draw.text((left + 10, top + 5), text, font=FONT, fill=foreground)
            if panel:
                draw.line([left, HEADER_H, left, HEADER_H + PANEL_H], fill=(120, 120, 120), width=2)
        frame = np.asarray(image)
        writer.append_data(frame)
        if k == poster_frame and out_index < len(frames) - int(TAIL_S * OUT_FPS):
            Image.fromarray(frame).resize((960, 400)).save(poster_dir / f"{key}.jpg", quality=85)
    writer.close()
    final = out_dir / f"{key}.mp4"
    os.replace(tmp_path, final)
    elapsed = time.perf_counter() - started
    return {
        "video": str(final.relative_to(out_dir.parent)),
        "bytes": final.stat().st_size,
        "output_frames": len(frames),
        "render_seconds": round(elapsed, 2),
        "arms": {run.arm: run.summary() for run in runs},
        "ghost_source_arm": anchor.arm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--motions", default="featured",
                        help="'featured', 'all', or comma-separated motion keys")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    review = args.review
    arms = args.arms.split(",")
    lock = json.loads((review / "evaluation-lock.json").read_text())
    featured = {f["motion_key"]: f["stratum"] for f in lock["featured_videos"]}
    split_of = {k: s for s, spec in lock["splits"].items() for k in spec["motion_keys"]}
    if args.motions == "featured":
        keys = list(featured)
    elif args.motions == "all":
        keys = list(featured) + sorted(k for k in split_of if k not in featured)
    else:
        keys = args.motions.split(",")
    available = [k for k in keys if all(
        (metrics_dir(review, arm, split_of[k]) / f"{k}.pose.npz").exists() for arm in arms)]
    missing = sorted(set(keys) - set(available))
    out_dir, poster_dir = review / "videos", review / "posters"
    out_dir.mkdir(exist_ok=True)
    poster_dir.mkdir(exist_ok=True)
    receipt_path = review / "render-receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"motions": {}}
    ledger = ledger_by_key()
    scene = G1Scene()
    renderer = mujoco.Renderer(scene.model, height=PANEL_H, width=PANEL_W)
    for number, key in enumerate(available, 1):
        if args.skip_existing and (out_dir / f"{key}.mp4").exists():
            continue
        result = render_motion(scene, renderer, review, key, split_of[key], arms, ledger,
                               featured, out_dir, poster_dir)
        receipt["motions"][key] = result
        receipt_path.write_text(json.dumps(receipt, indent=2))
        print(f"[{number}/{len(available)}] {key} {result['output_frames']} frames "
              f"{result['render_seconds']} s", flush=True)
    renderer.close()
    receipt.update(missing_inputs=missing, width=WIDTH, height=HEIGHT, fps=OUT_FPS,
                   renderer=f"mujoco {mujoco.__version__} {os.environ.get('MUJOCO_GL')}",
                   model="gear_sonic/data/assets/robot_description/urdf/g1/main.urdf")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    if missing:
        print("missing inputs (not rendered):", missing)


if __name__ == "__main__":
    main()
