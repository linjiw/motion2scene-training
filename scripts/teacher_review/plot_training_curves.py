#!/usr/bin/env python3
"""Plot a teacher run's PPO training curves from tracking-run-1/metrics.jsonl (one row per iteration)."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_common import DEFAULT_REVIEW, PACKET  # noqa: E402

# (metrics key, panel title, display scale)
PANELS = [
    ("objective/rewards", "Mean episode reward", 1),
    ("objective/length", "Mean episode length (control steps)", 1),
    ("Env/Metrics/motion/error_body_pos", "Body position error (mm)", 1000),
    ("Env/Metrics/motion/error_joint_pos", "Joint position error (rad)", 1),
    ("Env/Episode_Termination/time_out", "Episodes ending by time-out (%)", 100),
    ("Policy/mean_noise_std", "Policy action noise std", 1),
]
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"


def ema(values, alpha=0.1):
    out = np.empty_like(values)
    out[0] = values[0]
    for index in range(1, len(values)):
        out[index] = out[index - 1] + alpha * (values[index] - out[index - 1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=PACKET / "tracking-run-1")
    parser.add_argument("--out", type=Path, default=DEFAULT_REVIEW / "training-curves.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").open() if line.strip()]
    iterations = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), dpi=110, facecolor=SURFACE)
    for ax, (key, title, scale) in zip(axes.flat, PANELS):
        values = np.array([row[key] for row in rows], dtype=float) * scale
        ax.set_facecolor(SURFACE)
        ax.plot(iterations, values, color=SERIES, alpha=0.22, lw=1)
        ax.plot(iterations, ema(values), color=SERIES, lw=2)
        ax.set_title(title, fontsize=11, color="#0b0b0b", loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors="#52514e", labelsize=9)
        ax.grid(color="#ecebe6", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, len(rows))
        head, tail = values[: min(10, len(values))].mean(), values[-min(50, len(values)):].mean()
        print(f"{key}: first10={head:.4g} last50={tail:.4g}")
    for ax in axes[1]:
        ax.set_xlabel("PPO iteration", fontsize=9, color="#52514e")
    title = args.title or f"{args.run_dir.parent.name}, {len(rows)} iterations"
    fig.suptitle(f"{title} (thin line: raw, thick: EMA 0.1)", x=0.01, ha="left", fontsize=12,
                 color="#0b0b0b")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
