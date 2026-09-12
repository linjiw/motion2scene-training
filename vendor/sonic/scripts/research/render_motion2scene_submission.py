#!/usr/bin/env python3
"""Render four scientific figures and exact-count LaTeX tables from audited records."""

from pathlib import Path

from audit_motion2scene_submission import ARMS, DOC, Audit, condition, dump
import matplotlib
from motion2scene_reactive_interface import physics_windows

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)

OUT = DOC / "submission"
LABELS = {
    "uniform": "Uniform",
    "analytic": "Analytic",
    "no_contrast": "Target-only",
    "motion2scene": "Motion2Scene",
    "scripted_rays": "Scripted rays",
    "privileged_geometry": "Privileged geometry",
    "always_walk": "Always walk",
    "always_d040": "Always d040",
    "hindsight_command_oracle": "Outcome oracle",
}
ORDER = (
    *ARMS,
    "scripted_rays",
    "privileged_geometry",
    "always_walk",
    "always_d040",
    "hindsight_command_oracle",
)
COLORS = ["#b8d7c8", "#146884", "#e4e7ea", "#d68a49"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name):
    fig.savefig(
        OUT / "figures" / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "Matplotlib", "Author": ""},
    )
    fig.savefig(OUT / "figures" / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def executed(audit, analysis):
    pair = next(
        p
        for p in analysis["studies"]["nominal"]["pairs"]
        if p["group_id"] == "nom_41001_analytic_00"
    )
    records = [
        next(r for r in analysis["seed_level_rows"] if r["cell_id"] == ident)
        for ident in pair["row_ids"]
    ]
    result = audit.read(records[0]["result"]["path"], records[0]["result"]["sha256"])
    rows = [next(r for r in result["rows"] if r["cell_id"] == p["cell_id"]) for p in records]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.15), constrained_layout=True)
    for r, color, label in zip(rows, ("#9a4934", "#146884"), ("Walk", "Request d040")):
        audit.ref(r["trajectory"])
        payload = load_reset_capture(r["trajectory"]["path"])
        t = np.arange(len(payload["root_pos_w"])) / 50
        axes[0].plot(
            t, np.asarray(payload["root_pos_w"])[:, 2], color=color, label=label + " executed"
        )
        audit.ref(r["bank"])
        with np.load(r["bank"]["path"]) as bank:
            # The commanded bank switches at 0.30 s and returns at the recorded legal switch.
            index = np.zeros(len(t), int)
            if r["action"]:
                sensor = audit.read(r["sensor"]["path"], r["sensor"]["sha256"])
                for switch in sensor["switches"]:
                    index[t >= switch["time_s"]] = switch["to"]
            reference = bank["root_xyz"][index, np.arange(len(t)), 2]
        axes[0].plot(
            t, reference, color=color, linestyle="--", linewidth=0.9, label=label + " reference"
        )
        audit.ref(r["contacts"])
        audit.ref(r["physics"])
        with np.load(r["contacts"]["path"]) as sample, np.load(r["physics"]["path"]) as physics:
            _, force, _ = physics_windows(physics, sample["force_w"])
        end = r["passage_finish_frame_exclusive"] or r["first_episode_frames"]
        axes[1].plot(t[:end], np.linalg.norm(force[:end], axis=-1).max(1), color=color, label=label)
    axes[0].set(xlabel="Time from reset (s)", ylabel="Root height (m)")
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2)
    axes[1].set(xlabel="Time from reset (s)", ylabel="Peak beam force (N)", yscale="symlog")
    axes[1].axhline(1, color="#666666", linestyle=":", linewidth=0.8)
    axes[1].text(3.8, 1.4, "1 N", ha="right", fontsize=7)
    axes[1].legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.axvline(0.3, color="#777777", linestyle=":", linewidth=0.7)
        ax.set_xlim(0, 4)
    save(fig, "executed-contrast")
    return [r["cell_id"] for r in records]


def envelope(analysis):
    fig, ax = plt.subplots(figsize=(3.25, 1.9), constrained_layout=True)
    for source, color in zip((41001, 41002, 41003), ("#146884", "#b86436", "#555555")):
        rr = [r for r in analysis["envelope"]["rows"] if r["source"] == source]
        ax.plot(
            [r["envelope_xy_mm"] for r in rr],
            [r["witnesses"] for r in rr],
            marker="o",
            markersize=3,
            linewidth=1.1,
            color=color,
            label=f"C{source-41000}",
        )
    ax.set(
        xlabel="Horizontal envelope half-width (mm)",
        ylabel="Geometric witnesses / 11,421",
        xlim=(-0.3, 20.5),
    )
    ax.legend(frameon=False, fontsize=7)
    ax.text(20, 45, "2 / 0 / 32", ha="right", fontsize=7)
    save(fig, "acceptance-envelope")


def heatmap(analysis):
    commands = sorted(analysis["command_lookup"]["conditions"], key=condition)
    values, refusals = [], []
    for arm in ORDER:
        row, refused = [], []
        for c in commands:
            if arm in ("always_walk", "always_d040", "hindsight_command_oracle"):
                action = int(
                    arm == "always_d040"
                    or (arm == "hindsight_command_oracle" and c["outcomes"] == [False, True])
                )
                refusal = arm == "hindsight_command_oracle" and not any(c["outcomes"])
                passed = c["outcomes"][action]
            else:
                actual = next(
                    r
                    for r in analysis["seed_level_rows"]
                    if condition(r) == condition(c)
                    and r["arm"] == arm
                    and r["suite"] == "traversal"
                    and (r["cell_id"].startswith("nom_") if arm in ARMS else True)
                )
                action, passed = actual["action"], actual["pass"]
                # Refusals are in full outcome records; the reduced seed rows omit readout.
                refusal = False
            row.append((0 if passed else 2) + action)
            refused.append(refusal)
        values.append(row)
        refusals.append(refused)
    fig, ax = plt.subplots(figsize=(6.7, 2.35), constrained_layout=True)
    ax.imshow(
        values, cmap=ListedColormap(COLORS), vmin=0, vmax=3, aspect="auto", interpolation="nearest"
    )
    ax.set_yticks(range(len(ORDER)), [LABELS[a] for a in ORDER])
    ax.set_xticks(
        range(36), [str(int(c["layout"].split("_")[1]) + 1) for c in commands], fontsize=6
    )
    ax.set_xlabel("Layout (station-major, then increasing height), physics seed 8511")
    for x in (11.5, 23.5):
        ax.axvline(x, color="white", linewidth=2)
    for x, label in ((5.5, "C1"), (17.5, "C2"), (29.5, "C3")):
        ax.text(x, -0.85, label, ha="center", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(
        [Patch(color=c) for c in COLORS],
        ["Walk: pass", "d040: pass", "Walk: fail", "d040: fail"],
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    save(fig, "paired-outcomes")


def aliasing(analysis):
    diag = analysis["studies"]["nominal"]["fit_diagnostics"]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.7, 1.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    classes = diag["motion2scene"]["exact_aliasing"]["conflicting_classes"]
    ax = axes[0]
    for i, c in enumerate(classes):
        n = len(c["ids"])
        labels = np.array(c["labels"])
        bp, contrast = int((labels[:, 0] == 1).sum()), int((labels[:, 0] == 0).sum())
        ax.barh(i, bp, color="#b8d7c8", height=0.5, label="Both pass" if i == 0 else None)
        ax.barh(
            i, contrast, left=bp, color="#146884", height=0.5, label="d040 only" if i == 0 else None
        )
        ax.text(bp / 2, i, str(bp), va="center", ha="center", fontsize=8)
        ax.text(
            bp + contrast / 2, i, str(contrast), va="center", ha="center", color="white", fontsize=8
        )
        ax.text(n + 0.13, i, "same 214 inputs", va="center", fontsize=7)
    ax.set(
        yticks=[0, 1],
        yticklabels=["C1 alias class", "C3 alias class"],
        xlabel="Training groups with identical full observations",
        xlim=(0, 6.2),
    )
    ax.set_xticks(range(5))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=7, frameon=False)
    axes[1].bar(
        [0, 1],
        [diag[a]["bce"] for a in ("analytic", "motion2scene")],
        color=["#b86436", "#146884"],
        width=0.5,
    )
    floor = diag["motion2scene"]["exact_aliasing"]["bce_infimum_for_identical_features"]
    axes[1].hlines(floor, 0.65, 1.35, colors="black", linestyles=":", linewidth=1.5)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Analytic", "Motion2Scene"],
        yscale="log",
        ylim=(1e-4, 0.5),
        ylabel="Mean binary cross entropy",
    )
    axes[1].text(0, 0.0008, "0.000542", ha="center", fontsize=7)
    axes[1].text(1, 0.21, "0.139004", ha="center", fontsize=7)
    axes[1].text(0.98, 0.03, "Exact-input floor\n0.138629", ha="center", fontsize=7)
    save(fig, "full-input-aliasing")


def tabular(headers, rows, spec):
    return (
        "\\begin{tabular}{"
        + spec
        + "}\n\\toprule\n"
        + " & ".join(headers)
        + " \\\\\n\\midrule\n"
        + "\n".join(" & ".join(map(str, row)) + " \\\\" for row in rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def tables(analysis, diagnostics):
    rows = []
    for arm in ARMS:
        s = analysis["studies"]["nominal"]
        chosen = s["composition"][arm]
        c = chosen["outcomes"]
        acquisition = [r for r in s["acquisition"] if r["arm"] == arm]
        rows.append(
            [
                LABELS[arm],
                "48",
                sum(r["eligible_outputs"] for r in acquisition),
                "9 + 6",
                c["11"],
                c["01"],
                c["10"],
                c["00"],
                f"{chosen['reported_bce']:.6f}",
            ]
        )
    (OUT / "tables" / "acquisition.tex").write_text(
        tabular(
            [
                "Construction",
                "Outputs",
                "Eligible",
                "Groups",
                "Both pass",
                "d040 only",
                "Walk only",
                "Neither",
                "BCE",
            ],
            rows,
            "lrrrrrrrr",
        )
    )
    rows = []
    for arm in ORDER:
        v = analysis["comparators"][arm]
        rows.append(
            [
                LABELS[arm]
                + ("$^{*}$" if arm.startswith("always_") or arm.startswith("hindsight") else ""),
                f"{v['pass']}/36",
                f"{v['d040_requests']}/36",
                f"{v.get('both_pass',{}).get('d040_requests',0)}/14",
                f"{v.get('both_fail',{}).get('d040_requests',0)}/12",
                f"{v['refusals']}/36",
            ]
        )
    (OUT / "tables" / "comparators.tex").write_text(
        tabular(
            ["Policy", "Passage", "Requests", "Both-pass req.", "Both-fail req.", "Refusals"],
            rows,
            "lrrrrr",
        )
    )
    rows = []
    short = {"motion2scene": "M2S", "analytic": "An.", "uniform": "Un.", "no_contrast": "Tgt."}
    for c in analysis["studies"]["nominal"]["primary"]["registered_pairs"]:
        cells = c["counts"]
        rows.append(
            [
                short[c["first"]] + " / " + short[c["second"]],
                cells["both_pass"],
                cells["only_first"],
                cells["only_second"],
                cells["both_fail"],
                f"{c['descriptive_discordance']['p_value']:.12g}",
            ]
        )
    (OUT / "tables" / "paired.tex").write_text(
        tabular(["A / B", "Both", "A only", "B only", "Neither", "$p$"], rows, "lrrrrr")
    )
    rows = []
    for arm in ARMS:
        s = analysis["studies"]["robust"]
        c = s["composition"][arm]
        rows.append(
            [
                LABELS[arm],
                f"{c['groups']}/24",
                c["outcomes"]["01"],
                f"{s['primary']['per_arm'][arm]['pass']}/72",
            ]
        )
    (OUT / "tables" / "robust.tex").write_text(
        tabular(["Construction", "Groups", "Useful", "Passage"], rows, "lrrr")
    )
    rows = []
    for study in ("robust", "nominal", "post_hoc_command_audit"):
        for stage in ("labels", "evaluation"):
            records = [r for r in analysis["costs"] if r["study"] == study and r["stage"] == stage]
            if records:
                rows.append(
                    [
                        study.replace("post_hoc_command_audit", "Post-hoc audit").capitalize(),
                        stage,
                        sum(r["cells"] for r in records),
                        f"{sum(r['hours'] for r in records):.6f}",
                    ]
                )
    (OUT / "tables" / "cost.tex").write_text(
        tabular(["Study", "Stage", "Runs", "Process hours"], rows, "llrr")
    )
    dump(
        OUT / "evidence" / "figure-sources.json",
        {
            "analysis": "analysis.json",
            "diagnostics": "diagnostics.json",
            "figures": {
                "executed-contrast": "nom_41001_analytic_00, both commands, actual bank and force records",
                "acceptance-envelope": (
                    "analysis.envelope.rows; finite geometric diagnostic, not physical outcomes"
                ),
                "paired-outcomes": (
                    "analysis.seed_level_rows plus validated command_lookup; "
                    "nominal four arms, seed8511 reused comparators"
                ),
                "full-input-aliasing": (
                    "analysis.studies.nominal.fit_diagnostics; all original groups retained"
                ),
            },
            "post_hoc_figures": True,
        },
    )


def main():
    audit = Audit()
    analysis = audit.read(OUT / "evidence/analysis.json")
    diagnostics = audit.read(OUT / "evidence/diagnostics.json")
    for name in ("figures", "tables"):
        (OUT / name).mkdir(exist_ok=True)
    examples = executed(audit, analysis)
    envelope(analysis)
    heatmap(analysis)
    aliasing(analysis)
    tables(analysis, diagnostics)
    dump(
        OUT / "evidence" / "render-inputs.json",
        {
            "inputs": list(audit.files.values()),
            "example_cells": examples,
            "script": audit.pin(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
