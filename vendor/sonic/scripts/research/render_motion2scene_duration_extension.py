#!/usr/bin/env python3
"""Render executed duration alternatives and independently labeled scene witnesses."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)

OPTIONS = ("neutral", "short_e015_r265", "sustained_e015_r255")
NAMES = dict(zip(OPTIONS, ("Neutral", "Short", "Sustained")))
COLORS = dict(zip(OPTIONS, ("#747d8a", "#397eb4", "#22876f")))


def read_bound(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def render(data, out):
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "duration_extension"
    if prefix.with_suffix(".json").exists():
        raise FileExistsError("Use a new output directory for an existing figure receipt")
    qualification_ref = artifact(
        data / "m2s-environment-contact-qualification-v1/environment_result.json"
    )
    screen_ref = artifact(data / "m2s-timed-scene-screen-v1/result.json")
    physical_ref = artifact(data / "m2s-timed-duration-teachers-v1/result.json")
    qualification, screen, physical = map(read_bound, (qualification_ref, screen_ref, physical_ref))
    registration = read_bound(screen["registration"])
    physical_registration = read_bound(physical["registration"])
    if physical_registration["selected_grid_indices"] != [187, 247]:
        raise ValueError("Unexpected physically registered scene indices")
    if not physical["measurement_admitted"]:
        raise ValueError("Physical suite is not admitted")
    qualification_rows = {r["cell_id"]: r for r in qualification["rows"]}
    plot_data = {}
    sources = []
    curves = []
    for option in OPTIONS:
        row = qualification_rows[option]
        evidence = read_bound(row["evidence"])
        trajectory = evidence["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(trajectory["path"]), trajectory["sha256"]))
        elapsed = np.asarray(payload["motion_time_s"])
        height = np.asarray(evidence["executed_body_height_m"])
        if evidence["height_measurement"] != "executed_outer_collision_height_above_support_plane":
            raise ValueError("Expected executed conservative outer geometry")
        if not row["qualified"] or not all(row["checks"].values()):
            raise ValueError("Empty-scene option is not qualified under the declared criterion")
        np.testing.assert_allclose(
            elapsed, (np.asarray(evidence["command_ticks"]) - 1) / 50, atol=1e-12
        )
        if elapsed.shape != height.shape or height.shape != (298,):
            raise ValueError("Measured height/clock correspondence is incomplete")
        plot_data[f"{option}_elapsed_s"] = elapsed
        plot_data[f"{option}_height_m"] = height
        sources.extend((row["evidence"], trajectory))
        curves.append(
            {
                "option_id": option,
                "samples": len(height),
                "min_m": float(height.min()),
                "max_m": float(height.max()),
            }
        )

    # The screen used earlier captures: preserve that provenance instead of
    # silently substituting the later environment-contact qualification.
    for ref in registration["references"]:
        checked(Path(ref["path"]), ref["sha256"])
    child_results = {}
    for child in physical["children"]:
        ref = child["result"]
        child_results[Path(ref["path"]).parent.name] = read_bound(ref)
        sources.append(ref)
    scenes = []
    for index, child_name in ((187, "short_beam"), (247, "long_passage")):
        scene_ref = next(
            r for r in screen["scenes"] if Path(r["path"]).stem == f"timed_development_{index:04d}"
        )
        scene = read_bound(scene_ref)
        witness = next(r for r in screen["selected"] if r["grid_index"] == index)
        rows = {r["forced_option_id"]: r for r in child_results[child_name]["rows"]}
        if set(rows) != set(OPTIONS) or any(not r["measurement_admitted"] for r in rows.values()):
            raise ValueError("Incomplete physical option table")
        outcomes = [
            {
                "option_id": o,
                "pass": rows[o]["pass"],
                "passage_time_s": rows[o]["costs"]["passage_time_s"],
            }
            for o in OPTIONS
        ]
        for r in outcomes:
            if (r["passage_time_s"] is not None) != r["pass"]:
                raise ValueError("Unsuccessful passage must not receive a success time")
        scenes.append(
            {"grid_index": index, "beam": scene["beam"], "witness": witness, "outcomes": outcomes}
        )
        sources.append(scene_ref)

    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}
    )
    fig = plt.figure(figsize=(12.4, 4.2), layout="constrained")
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.9, 1, 1), height_ratios=(1, 1.32), hspace=0.16, wspace=0.12
    )
    ax = fig.add_subplot(grid[:, 0])
    for option in OPTIONS:
        ax.plot(
            plot_data[f"{option}_elapsed_s"],
            plot_data[f"{option}_height_m"],
            color=COLORS[option],
            label=NAMES[option],
            linewidth=1.7,
        )
    ax.set(
        xlim=(0, 5.94),
        ylim=(1.175, 1.327),
        xlabel="Recorded elapsed time (s)",
        ylabel="Executed outer-envelope height (m)",
        title="A  Measured empty-scene alternatives",
    )
    ax.legend(frameon=False, loc="lower left", ncol=3, columnspacing=1.0, handlelength=1.7)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    for col, (scene, title) in enumerate(zip(scenes, ("B  Short beam", "C  Long passage")), 1):
        beam = scene["beam"]
        top = fig.add_subplot(grid[0, col])
        top.add_patch(
            Rectangle(
                (-beam["length_m"] / 2, beam["underside_m"]),
                beam["length_m"],
                beam["thickness_m"],
                facecolor="#9fa6b0",
                edgecolor="#45515e",
                linewidth=1,
            )
        )
        top.annotate(
            "",
            (-beam["length_m"] / 2, 1.22),
            (beam["length_m"] / 2, 1.22),
            arrowprops={"arrowstyle": "|-|", "lw": 0.8, "color": "#45515e"},
        )
        top.text(0, 1.208, f"{beam['length_m']:.2f} m span", ha="center", va="top")
        top.text(
            0, 1.39, f"Underside {beam['underside_m']:.3f} m", ha="center", va="bottom", fontsize=8
        )
        top.set(
            xlim=(-0.65, 0.65),
            ylim=(1.18, 1.42),
            xticks=(-0.5, 0, 0.5),
            yticks=(1.2, 1.3, 1.4),
            xlabel="Position along beam (m)",
            title=title,
        )
        top.tick_params(labelsize=8)
        top.spines[["top", "right"]].set_visible(False)
        bottom = fig.add_subplot(grid[1, col])
        bottom.set_axis_off()
        w = scene["witness"]
        bottom.text(0, 0.97, "Geometric screen · 81 offsets", fontsize=9, weight="bold", va="top")
        bottom.text(
            0,
            0.79,
            f"{NAMES[w['positive']]} worst: +{1000*w['positive_min_m']:.1f} mm",
            color=COLORS[w["positive"]],
        )
        bottom.text(
            0,
            0.66,
            f"{NAMES[w['negative']]} nominal: {1000*w['negative_nominal_m']:.1f} mm",
            color=COLORS[w["negative"]],
        )
        bottom.text(0, 0.43, "Physical episodes · nominal", fontsize=9, weight="bold")
        for y, outcome in zip((0.28, 0.15, 0.02), scene["outcomes"]):
            o = outcome["option_id"]
            value = f"PASS  {outcome['passage_time_s']:.2f} s" if outcome["pass"] else "FAIL"
            bottom.text(0, y, NAMES[o], color=COLORS[o])
            bottom.text(
                0.99, y, value, ha="right", color="#204b3c" if outcome["pass"] else "#a64336"
            )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)
    np.savez_compressed(prefix.with_suffix(".npz"), **plot_data)
    caption = (
        "Six-second development extension with a frozen tracker. A: conservative native outer-body "
        "envelope heights reconstructed from three actual empty-scene qualification trajectories, "
        "plotted against recorded physical elapsed time (298 samples, 0–5.94 s). These are executed "
        "body envelopes, not target-reference or root heights. B–C: the two development beam spans "
        "and signed geometric witnesses, selected before their physical outcomes. The positive "
        "margin is the minimum over 81 finite world-position/yaw offsets; the negative margin is "
        "nominal only. The screen used earlier empty-scene captures whose original strict net-force "
        "qualification failed; those failures remain recorded. Separate new captures subsequently "
        "qualified the same options under a declared environment-contact criterion that resolves "
        "self-contact. Physical tables are the independently recorded nominal passage outcomes: "
        "both adaptations pass the short beam, while only the sustained option passes the long "
        "passage. Costs are measured passage times within each scene; finish planes differ between "
        "scenes. A positive finite-offset geometric margin does not establish physical robustness. "
        "This figure establishes a development duration capability distinction, not learned-policy, "
        "curriculum, held-out, or navigation gains."
    )
    receipt = {
        "schema": "motion2scene_duration_extension_figure_v1",
        "scope": "development_only",
        "driver": artifact(Path(__file__).resolve()),
        "qualification": qualification_ref,
        "screen": screen_ref,
        "screen_registration": screen["registration"],
        "screen_source_references": registration["references"],
        "screen_source_qualification_status": registration["source_qualification"],
        "physical_result": physical_ref,
        "physical_registration": physical["registration"],
        "analysis_correction": physical["analysis_correction"],
        "sources": sources,
        "curves": curves,
        "scenes": scenes,
        "caption": caption,
        "outputs": [artifact(prefix.with_suffix(f".{s}")) for s in ("pdf", "svg", "png", "npz")],
    }
    prefix.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": artifact(prefix.with_suffix(".json")), "curves": curves}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/home/linjiw/research-data/groot-wbc"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.data, args.out)
