#!/usr/bin/env python3
"""Plot recorded script inputs and neutral movement; no simulated robot imagery."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def draw(result_path, out):
    data = json.loads(result_path.read_text())
    if len(data["traces"]) != 6 or data["new_physical_executions"] != 0:
        raise ValueError("six recorded traces and explicit offline scope required")
    indexed = {trace["scene_id"]: trace for trace in data["traces"]}
    examples = [indexed[name] for name in data["displayed_examples"]]
    if len(examples) != 2 or any(t["commitment"] is None for t in examples):
        raise ValueError("the two previously inspected commitment examples are required")
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 7.7), sharex=True, layout="constrained")
    bands = ["0–0.75", "0.75–1.5", "1.5–2.5", "2.5–4"]
    for column, trace in enumerate(examples):
        series = trace["precommit_series"]
        times = np.array([row["registered_phase_s"] for row in series])
        upper = np.array([row["upper_hit_by_band"] for row in series]).T
        unknown = np.array([row["unknown_fraction_by_band"] for row in series]).T
        if upper.shape != unknown.shape or upper.shape[0] != 4:
            raise ValueError("four measured corridor bands required")
        extent = (times[0] - 0.01, times[-1] + 0.01, -0.5, 3.5)
        image = axes[0, column].imshow(
            upper,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=extent,
            cmap=ListedColormap(["#edf0f2", "#126d75"]),
            vmin=0,
            vmax=1,
        )
        missing = axes[1, column].imshow(
            unknown,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        first = trace["decisions"][0]
        origin = np.array(first["root_pos_w"][:2])
        moving = [row for row in series if row["tick"] >= first["tick"]]
        displacement = [np.linalg.norm(np.array(row["root_xy_w"]) - origin) for row in moving]
        axes[2, column].plot(
            [row["registered_phase_s"] for row in moving],
            displacement,
            color="#126d75",
            linewidth=2,
        )
        axes[2, column].scatter(
            trace["commitment"]["registered_phase_s"],
            displacement[-1],
            color="#a44435",
            zorder=5,
        )
        axes[2, column].annotate(
            f"{displacement[-1]:.3f} m",
            (trace["commitment"]["registered_phase_s"], displacement[-1]),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            fontsize=10,
        )
        for decision in trace["decisions"]:
            time = decision["registered_phase_s"]
            commit = decision["decision"] == "COMMIT"
            for axis in axes[:, column]:
                axis.axvline(
                    time,
                    color="#a44435" if commit else "#65727c",
                    linestyle="-" if commit else "--",
                    linewidth=1,
                )
            axes[0, column].text(
                time,
                3.6,
                "COMMIT" if commit else "WAIT",
                color="#a44435" if commit else "#46525d",
                ha="right" if commit else "center",
                va="bottom",
                fontsize=9,
            )
        for axis in axes[:2, column]:
            axis.set_yticks(range(4), bands)
            axis.set_ylim(-0.5, 4.05)
        for axis in axes[:, column]:
            axis.set_xlim(0, 1.5)
            axis.set_xticks([0, 0.3, 0.6, 1.0, 1.4])
            axis.spines[["top", "right"]].set_visible(False)
        axes[2, column].set_ylim(-0.015, 0.5)
        axes[2, column].grid(axis="y", alpha=0.25)
        axes[2, column].set_xlabel("Registered control phase (s)")
        schedule = trace["commitment"]["selected_option_id"]
        title = trace["scene_id"].replace("_development_0284", "").replace("_development_0100", "")
        axes[0, column].set_title(f"{title.replace('_', ' ')}\n{schedule}", fontsize=11, pad=21)
    axes[0, 0].set_ylabel("Upper-hit indicator\nCorridor band ahead (m)")
    axes[1, 0].set_ylabel("Unknown fraction\nCorridor band ahead (m)")
    axes[2, 0].set_ylabel("Net planar displacement\nsince first decision (m)")
    fig.colorbar(image, ax=axes[0, :], ticks=[0, 1], shrink=0.8, label="Recorded upper hit")
    fig.colorbar(missing, ax=axes[1, :], ticks=[0, 0.5, 1], shrink=0.8, label="Unknown fraction")
    fig.suptitle("Strong-script examples from actual executions · seed 8732", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"perception_to_decision.{suffix}", dpi=220)
    plt.close(fig)
    rows = [
        "# Recorded perception-to-decision examples",
        "",
        "![Recorded script inputs and neutral execution](perception_to_decision.png)",
        "",
        "These are the two previously inspected development contexts requiring disjoint bank responses. "
        "All six script traces are retained in the source, and all 16 recorded neutral-phase readouts "
        "match the frozen script. The figure uses recorded features and robot states, "
        "not generated robot imagery.",
        "",
        "| Context | Phase (s) | Delivered packet capture (s) | Decision | Hazard bands |",
        "|---|---:|---:|---|---|",
    ]
    for trace in examples:
        for decision in trace["decisions"]:
            hazard = ", ".join(
                bands[i]
                for i, value in enumerate(decision["script_readout"]["hazard_bands"])
                if value
            )
            label = decision["decision"]
            if label == "COMMIT":
                label += " " + decision["selected_option_id"]
            rows.append(
                f"| {trace['scene_id']} | {decision['registered_phase_s']:.2f} | "
                f"{decision['delivered_packet_capture_elapsed_s']:.2f} | "
                f"{label} | {hazard or 'none'} |"
            )
    rows += [
        "",
        "Both examples already have different recorded upper-hit bands at the first decision. "
        "The plots therefore do not demonstrate an additional-information advantage for WAIT. "
        "WAIT preserves a later supported entry while neutral motion continues; final-phase neutral "
        "selection has no later entry and is labeled CONTINUE_NEUTRAL in the complete extraction.",
        "",
        "Upper-hit zero means no recorded upper hit, not certified free space. Unknown fractions "
        "remain explicit. No postcommit features are plotted. States and features were captured "
        "before their registered command phase; the separate packet capture timestamps are in "
        "the table. Blank time after commitment contains no plotted measurement.",
        "",
        data["limitations"],
        "",
        f"Source: `{result_path.resolve()}`, SHA256 `{hashlib.sha256(result_path.read_bytes()).hexdigest()}`.",
        "",
        "Reproduce with `scripts/research/motion2scene_plot_script_perception.py "
        "--result <result.json> --out <new-directory>`.",
        "",
    ]
    (out / "README.md").write_text("\n".join(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    draw(args.result, args.out)
