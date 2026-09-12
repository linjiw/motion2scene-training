"""Matched, failure-censored replay of the best full/navigation-command student motion."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np

from gear_sonic.research.hindsight_training.render_qualification import CHAINS, valid_steps
from gear_sonic.research.hindsight_training.runtime import sha, write_new


def render(packet, output):
    output.mkdir(parents=True, exist_ok=False)
    # Native scorer keeps expected_steps - 1 samples for this completed full-command motion.
    with np.load(packet / "corrected-trajectories/final-eval-full/hindsight_00490.npz") as complete:
        expected_steps = len(complete["tracked"]) + 1
    data = {}
    for profile in ("full", "navigation"):
        metrics = json.loads((packet / f"final-eval-{profile}/metrics_eval.json").read_text())[
            "eval/all_metrics_dict"
        ]
        i = metrics["motion_keys"].index("hindsight_00490")
        path = packet / f"corrected-trajectories/final-eval-{profile}/hindsight_00490.npz"
        with np.load(path) as f:
            ref, pred = f["reference"].copy(), f["tracked"].copy()
        count = valid_steps(metrics["progress"][i], expected_steps, len(pred))
        origin = ref[0, 0].copy()
        origin[2] = 0
        ref -= origin
        pred -= origin
        data[profile] = (ref, pred, count, not metrics["terminated"][i])
    receipt = {
        p: {"completed": v[3], "retained_steps": v[2], "retained_seconds": 0.02 * v[2]}
        for p, v in data.items()
    }
    receipt.update(
        motion_id="00490",
        selection="Only full-command completion and maximum navigation progress among all 100",
        student_checkpoint=json.loads((packet / "complete.json").read_text()),
        visualization=(
            "Measured skeleton replay, corrected native reference/robot roles; "
            "post-failure/reset samples excluded"
        ),
    )
    write_new(output / "comparison.json", receipt)
    for profiles, name in [
        (("full", "navigation"), "comparison"),
        (("full",), "full-command"),
        (("navigation",), "navigation-command"),
    ]:
        fig = plt.figure(figsize=(6.4 * len(profiles), 6.4), facecolor="#f6f8fb")
        axes = []
        lines = []
        xyz = np.concatenate(
            [x[: data[p][2]].reshape(-1, 3) for p in profiles for x in data[p][:2]]
        )
        low, high = xyz.min(0), xyz.max(0)
        middle = (low + high) / 2
        radius = max(high[0] - low[0], high[1] - low[1], 1.5) / 2 + 0.3
        for i, p in enumerate(profiles):
            ax = fig.add_subplot(1, len(profiles), i + 1, projection="3d")
            axes.append(ax)
            ax.set(
                xlim=(middle[0] - radius, middle[0] + radius),
                ylim=(middle[1] - radius, middle[1] + radius),
                zlim=(0, 1.8),
                xlabel="x (m)",
                ylabel="y (m)",
            )
            ax.set_box_aspect((2 * radius, 2 * radius, 1.8))
            ax.view_init(elev=22, azim=-60)
            color = "#1676bd" if p == "full" else "#df7c24"
            lines.append(
                [
                    (
                        ax.plot([], [], [], "--", color="gray", lw=1.5)[0],
                        ax.plot([], [], [], color=color, lw=3)[0],
                    )
                    for _ in CHAINS
                ]
            )
        fig.subplots_adjust(top=0.85, bottom=0.12, left=0.02, right=0.98)
        fig.text(
            0.5,
            0.04,
            "Dashed gray: reference | Color: simulated student | Skeleton replay, not camera footage",
            ha="center",
            fontsize=10,
        )
        length = max(data[p][2] for p in profiles) + 50
        with FFMpegWriter(
            fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"]
        ).saving(fig, str(output / (name + ".mp4")), 100) as writer:
            for t in range(0, length, 2):
                for p, ax, pairs in zip(profiles, axes, lines):
                    ref, pred, count, complete = data[p]
                    k = min(t, count - 1)
                    for chain, (a, b) in zip(CHAINS, pairs):
                        r = ref[k, chain]
                        a.set_data_3d(r[:, 0], r[:, 1], r[:, 2])
                        q = pred[k, chain] if t < count or complete else np.empty((0, 3))
                        b.set_data_3d(q[:, 0], q[:, 1], q[:, 2])
                    state = (
                        (
                            "COMPLETE (final pose frozen)"
                            if complete
                            else "FAILED (subsequent states excluded)"
                        )
                        if t >= count
                        else f"Tracking | body error {np.linalg.norm(pred[k]-ref[k],axis=-1).mean()*1000:.0f} mm"
                    )
                    ax.set_title(f"{p.capitalize()} commands | 00490\n{state}", fontsize=11)
                fig.suptitle(f"Same student, same motion | t = {t*.02:.2f} s", fontsize=15)
                writer.grab_frame()
        plt.close(fig)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Student command comparison</title><style>body{'
        "font:18px system-ui;max-width:1200px;margin:40px auto;background:#f6f8fb}video{width:100%}"
        "</style><h1>Student: full vs navigation commands</h1><p>Motion 00490: full completes; navi"
        "gation fails after 78.6% progress. This is the best example, not representative of all 100"
        " motions. Navigation here means four reference-derived controls, not autonomous scene/goal"
        ' navigation.</p><video controls src="comparison.mp4"></video><p><a href="full-command.mp4"'
        '>Full commands</a> · <a href="navigation-command.mp4">Navigation commands</a></p>'
    )
    write_new(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.packet, args.output)
