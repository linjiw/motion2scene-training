"""Summarize matched native rollouts and render five preselected motion overlays."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np

CHAINS = ([0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9, 10], [7, 11, 12, 13])


def valid_steps(progress, expected_steps, available_steps):
    """Exclude the terminating/reset sample; native progress counts surviving steps."""
    return min(int(round(progress * expected_steps)), available_steps)


def main(packet):
    lock = json.loads((packet / "evaluation-lock.json").read_text())
    metadata = joblib.load(packet / "motions/metadata.pkl")
    records, trajectories = [], {}
    for arm in ("release", "trained"):
        assert json.loads((packet / arm / "exit.json").read_text())["exit_code"] == 0
        metrics = json.loads((packet / arm / "metrics/metrics_eval.json").read_text())[
            "eval/all_metrics_dict"
        ]
        assert len(metrics["motion_keys"]) == 120
        for i, key in enumerate(metrics["motion_keys"]):
            data = np.load(packet / arm / "metrics" / f"{key}.npz")
            ref, pred = data["reference"], data["tracked"]
            expected = int(np.floor(metadata[key]["length"] * 50 / metadata[key]["fps"]))
            count = valid_steps(metrics["progress"][i], expected, len(pred))
            error = np.linalg.norm(pred[:count] - ref[:count], axis=-1) * 1000
            root = np.linalg.norm(pred[:count, 0, :2] - ref[:count, 0, :2], axis=-1)
            local = (
                np.linalg.norm(
                    (pred[:count] - pred[:count, :1]) - (ref[:count] - ref[:count, :1]),
                    axis=-1,
                )
                * 1000
            )
            row = {
                "arm": arm,
                "motion_id": key.split("_")[-1],
                "split": (
                    "development" if key.split("_")[-1] in lock["development_ids"] else "train"
                ),
                "completed": not metrics["terminated"][i],
                "progress": metrics["progress"][i],
                "expected_steps": expected,
                "retained_steps": count,
                "prefix_global_mpjpe_mm": float(error.mean()) if count else None,
                "prefix_local_mpjpe_mm": float(local.mean()) if count else None,
                "prefix_root_xy_mean_m": float(root.mean()) if count else None,
                "native_global_mpjpe_mm_including_resets": metrics["mpjpe_g"][i],
            }
            records.append(row)
            trajectories[arm, row["motion_id"]] = (ref, pred, error.mean(-1), row)
    with (packet / "per-motion-results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for split in ("train", "development"):
        summary[split] = {}
        for arm in ("release", "trained"):
            rows = [r for r in records if r["split"] == split and r["arm"] == arm]
            summary[split][arm] = {
                "motions": len(rows),
                "completed": sum(r["completed"] for r in rows),
                "mean_progress": float(np.mean([r["progress"] for r in rows])),
                "mean_prefix_global_mpjpe_mm": float(
                    np.mean([r["prefix_global_mpjpe_mm"] for r in rows if r["retained_steps"]])
                ),
                "mean_prefix_local_mpjpe_mm": float(
                    np.mean([r["prefix_local_mpjpe_mm"] for r in rows if r["retained_steps"]])
                ),
            }
        ids = lock[split + "_ids"]
        paired = [
            (trajectories["release", k][3]["completed"], trajectories["trained", k][3]["completed"])
            for k in ids
        ]
        summary[split]["paired"] = {
            "improved": sum(not a and b for a, b in paired),
            "regressed": sum(a and not b for a, b in paired),
        }
    common_prefix = {}
    for split in ("train", "development"):
        paired_errors = []
        for key in lock[split + "_ids"]:
            count = min(
                trajectories[arm, key][3]["retained_steps"] for arm in ("release", "trained")
            )
            row = {"motion_id": key, "common_steps": count}
            for arm in ("release", "trained"):
                error = trajectories[arm, key][2][:count]
                row[arm] = float(error.mean()) if count else None
            paired_errors.append(row)
        common_prefix[split] = {
            "per_motion": paired_errors,
            "common_prefix_mean_mpjpe_mm": {
                arm: float(np.mean([r[arm] for r in paired_errors if r[arm] is not None]))
                for arm in ("release", "trained")
            },
        }
    (packet / "common-prefix-errors.json").write_text(json.dumps(common_prefix, indent=2))
    (packet / "summary.json").write_text(json.dumps(summary, indent=2))
    videos = packet / "videos"
    videos.mkdir(exist_ok=True)
    for key in lock["video_ids"]:
        fig = plt.figure(figsize=(12.8, 7.2), facecolor="#f5f7fb")
        grid = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        axes = [fig.add_subplot(grid[0, j], projection="3d") for j in range(2)]
        chart = fig.add_subplot(grid[1, :])
        series = [trajectories[arm, key] for arm in ("release", "trained")]
        length = min(
            max(len(x[0]) for x in series),
            max(x[3]["retained_steps"] for x in series) + 50,
        )
        lines = []
        labels = []
        colors = ["#d97823", "#147dcc"]
        for ax, (ref, pred, error, row), color in zip(axes, series, colors):
            origin = ref[0, 0].copy()
            origin[2] = 0
            ref -= origin
            pred -= origin
            center = ref.reshape(-1, 3).mean(0)
            radius = max(0.9, np.ptp(ref[..., :2].reshape(-1, 2), axis=0).max() / 2 + 0.3)
            ax.set(
                xlim=(center[0] - radius, center[0] + radius),
                ylim=(center[1] - radius, center[1] + radius),
                zlim=(0, 1.6),
                xlabel="x (m)",
                ylabel="y (m)",
            )
            ax.set_box_aspect((2 * radius, 2 * radius, 1.6))
            ax.view_init(elev=22, azim=-58)
            ax.set_title(
                f"{row['arm'].upper()}  |  {'COMPLETE' if row['completed'] else 'FAILED'}"
                f"  |  {row['progress']:.1%} progress",
                fontweight="bold",
            )
            valid_ref = ref[: row["retained_steps"]]
            ax.plot(
                valid_ref[:, 0, 0],
                valid_ref[:, 0, 1],
                np.zeros(len(valid_ref)),
                color="#aeb6c5",
                lw=1,
            )
            pairs = []
            for _ in CHAINS:
                pairs.append(
                    (
                        ax.plot([], [], [], color="#888e98", lw=2, linestyle="--", alpha=0.7)[0],
                        ax.plot([], [], [], color=color, lw=3, marker="o", markersize=3)[0],
                    )
                )
            lines.append(pairs)
            labels.append(ax.text2D(0.02, 0.02, "", transform=ax.transAxes, color=color))
            chart.plot(np.arange(len(error)) * 0.02, error, color=color, label=row["arm"])
            if not row["completed"]:
                chart.axvline(row["retained_steps"] * 0.02, color=color, linestyle=":")
        chart.set(xlim=(0, length * 0.02), xlabel="Time (s)", ylabel="Body position error (mm)")
        chart.legend(loc="upper right")
        chart.grid(alpha=0.2)
        title = fig.suptitle(
            f"Motion {key} | Recorded Isaac Lab tracking | dashed gray = reference", fontsize=15
        )
        fig.text(
            0.5,
            0.01,
            "50 Hz rollout shown at 25 fps • measured skeleton • after failure: reference frozen, robot hidden",
            ha="center",
            fontsize=9,
        )
        fig.subplots_adjust(top=0.88, bottom=0.11, hspace=0.22)
        writer = FFMpegWriter(
            fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]
        )
        with writer.saving(fig, str(videos / f"motion-{key}.mp4"), dpi=100):
            for frame in range(0, length, 2):
                title.set_text(f"Motion {key} | t = {frame*.02:.2f} s | dashed gray = reference")
                for pairs, label, (ref, pred, error, row) in zip(lines, labels, series):
                    index = min(frame, max(0, row["retained_steps"] - 1), len(ref) - 1)
                    alive = frame < row["retained_steps"]
                    for chain, (reference_line, tracked_line) in zip(CHAINS, pairs):
                        xyz = ref[index, chain]
                        reference_line.set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                        xyz = pred[index, chain] if alive else np.empty((0, 3))
                        tracked_line.set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                    label.set_text(
                        f"Error: {error[frame]:.0f} mm"
                        if alive
                        else (
                            "Motion complete"
                            if row["completed"]
                            else "Tracking failed — subsequent resets excluded"
                        )
                    )
                writer.grab_frame()
        fig.savefig(videos / f"motion-{key}.png", dpi=100)
        plt.close(fig)
        print("Rendered", key, flush=True)
    html = (
        '<!doctype html><meta charset="utf-8"><title>SONIC matched tracking evaluation</title>'
        "<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;"
        "background:#f5f7fb;color:#182334}video{width:100%}pre{white-space:pre-wrap}</style>"
        "<h1>SONIC tracking: release vs 8,000-step fine-tune</h1>"
        "<p>Five development motions selected before evaluation. Dashed gray: reference; "
        "orange: release; blue: trained. These are measured skeleton replays, not camera footage. "
        "After the first failure, the reference freezes and the robot disappears. "
        "Videos end one second after the longer valid rollout.</p>"
    )
    html += "<pre>" + json.dumps(summary, indent=2) + "</pre>"
    for key in lock["video_ids"]:
        html += f'<h2>Motion {key}</h2><video controls preload="metadata" src="videos/motion-{key}.mp4"></video>'
    (packet / "index.html").write_text(html)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    main(parser.parse_args().packet)
