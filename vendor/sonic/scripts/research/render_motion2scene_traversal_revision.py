#!/usr/bin/env python3
"""Render measured option capability and construction results with bounded captions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)


def render(result_path, construction_path, out):
    result = json.loads(result_path.read_text())
    construction = json.loads(construction_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), constrained_layout=True)
    colors = ["#737d8c", "#db704b", "#2a9986", "#4678ac", "#6a57a5"]
    for row, color in zip(rows, colors):
        payload = load_reset_capture(
            checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
        )
        n = row["first_episode_frames"]
        label = row["option"]["option_id"].replace("_t030", "")
        axes[0].plot(
            payload["motion_time_s"][:n],
            np.asarray(payload["root_pos_w"])[:n, 2],
            color=color,
            label=label,
            lw=1.5,
        )
    axes[0].axvline(0.3, ls=":", color="black", alpha=0.5)
    axes[0].set(xlabel="Time (s)", ylabel="Recorded root height (m)", title="Executed alternatives")
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    names = [r["option"]["option_id"].replace("_t030", "") for r in rows]
    forces = [r["maximum_beam_normal_force_n_through_passage"] for r in rows]
    axes[1].bar(names, forces, color=colors, width=0.7)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set(ylabel="Peak measured beam force (N)", title="Same constructed scene")
    axes[1].set_ylim(0, max(forces) * 8)
    for i, row in enumerate(rows):
        axes[1].text(
            i,
            max(forces[i] * 1.3, 0.15),
            "pass" if row["pass"] else "fail",
            ha="center",
            fontsize=9,
        )
    arms = ("uniform", "learned", "analytic_resample")
    counts = [
        sum(r["eligible"] for r in construction["sampling"] if r["arm"] == arm) for arm in arms
    ]
    axes[2].bar(
        ["Uniform", "Learned", "Analytic\nresampling"],
        counts,
        color=["#aeb5bd", "#6a57a5", "#2a9986"],
    )
    axes[2].set(
        ylabel="Geometrically eligible / 192",
        ylim=(0, 192),
        title="Proposal ablation (development)",
    )
    for i, count in enumerate(counts):
        axes[2].text(i, count + 4, str(count), ha="center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    fig.savefig(out / "traversal_revision.pdf")
    fig.savefig(out / "traversal_revision.png", dpi=180)
    plt.close(fig)
    caption = (
        "Development evidence, not held-out performance. Left: actual first-episode root heights "
        "under five reference choices; root height is illustrative, not a clearance certificate. "
        "Middle: source 41002, physics seed 8731, route progress 0.55, underside 1.249 m; "
        "walk and d040 fail, while d055/d070/d085 pass with zero recorded beam force and legal "
        "entry/return. The scene was selected from empty-scene executed geometry and registered "
        "before these physical labels. Right: the same-context proposal ablation uses 64 draws "
        "per constructor per each of three development carriers; analytic resampling is stronger "
        "than the learned proposal here. Its denominator counts geometric proposals, not physical "
        "rollouts. Analytic search and model-fit costs are separately reported in the source result."
    )
    (out / "figure.json").write_text(
        json.dumps(
            {
                "physical_result": artifact(result_path),
                "construction_result": artifact(construction_path),
                "caption": caption,
            },
            indent=2,
        )
        + "\n"
    )
    print(caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-result", type=Path, required=True)
    parser.add_argument("--construction-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.physical_result, args.construction_result, args.out)
