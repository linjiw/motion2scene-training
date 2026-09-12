#!/usr/bin/env python3
"""Generate acquisition response curves and matched-repertoire WAIT figures."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

LABELS = {
    "uniform": "Screened uniform",
    "target_only": "Target-only",
    "analytic_contrast": "Executed contrast",
    "observation_curriculum": "Executed + replay",
    "reference_contrast": "Reference contrast",
}
COLORS = dict(zip(LABELS, ["#64748b", "#d97706", "#0f766e", "#7c3aed", "#be123c"], strict=True))


def render(acquisition, wait, out):
    refs = [artifact(path) for path in acquisition]
    reports = sorted([read_checked(ref) for ref in refs], key=lambda x: x["budget"])
    if not reports or len({r["budget"] for r in reports}) != len(reports):
        raise ValueError("distinct completed acquisition checkpoints required")
    if any(r["plan"] != reports[0]["plan"] for r in reports):
        raise ValueError("curves must extend the identical acquisition trajectories")
    rows = []
    for report in reports:
        for corpus in report["corpora"]:
            response = corpus["responses"]
            if not response["complete"] or response["assigned_tasks"] != report["budget"]:
                raise ValueError("complete assigned task denominators required")
            rows.append(
                dict(
                    budget=report["budget"],
                    arm=corpus["arm"],
                    seed=corpus["seed"],
                    actual_steps=corpus["physical_cost"]["total_recorded_steps"],
                    tasks=response["assigned_tasks"],
                    solvable=response["bank_solvable_lower"],
                    adaptation_required=response["adaptation_required_lower"],
                    best_fixed=response["best_fixed_passages_lower"],
                    capability_minus_best_fixed=response["capability_minus_best_fixed_lower"],
                )
            )
    wait_ref = artifact(wait)
    control = read_checked(wait_ref)
    if (
        control["new_physics_steps"] != 0
        or control["complete_schedule_choices_in_both_controls"] != 7
    ):
        raise ValueError("seven-schedule finite information control required")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "acquisition_by_corpus.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Budget & Constructor & Solvable & Adaptation & Best fixed & Steps/arm \\",
        r"\midrule",
    ]
    for report in reports:
        for arm, label in LABELS.items():
            if arm not in report["arms"]:
                continue
            entry = report["arms"][arm]
            response = entry["pooled_responses"]
            n = response["assigned_tasks"]
            cells = [
                f'M{report["budget"]}',
                label,
                f'{response["bank_solvable_lower"]}/{n}',
                f'{response["adaptation_required_lower"]}/{n}',
                f'{response["best_fixed_passages_lower"]}/{n}',
                f'{entry["actual_recorded_steps"]:,}',
            ]
            table.append(" & ".join(cells) + r" \\")
    table += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Measured acquired-task responses across three corpora per arm. "
        r"Adaptation requires a measured neutral failure and a passing alternative. "
        r"Best fixed is the strongest one schedule over that arm's assigned tasks, "
        r"selected here for a coverage diagnostic. Recorded physics steps include each "
        r"corpus's own bootstrap, students and teachers. These are construction-specific "
        r"training tasks, not common-set policy performance.}",
        r"\label{tab:acquisitionyield}",
        r"\end{table}",
    ]
    (out / "acquisition_yield.tex").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for arm, label in LABELS.items():
        selected = [r for r in rows if r["arm"] == arm]
        if not selected:
            continue
        for seed in sorted({r["seed"] for r in selected}):
            local = [r for r in selected if r["seed"] == seed]
            x = [r["actual_steps"] / 1000 for r in local]
            for ax, key in zip(
                axes, ("adaptation_required", "capability_minus_best_fixed"), strict=True
            ):
                ax.plot(
                    x, [r[key] / r["tasks"] for r in local], color=COLORS[arm], alpha=0.35, lw=1
                )
        budgets = sorted({r["budget"] for r in selected})
        x = [
            np.mean([r["actual_steps"] for r in selected if r["budget"] == b]) / 1000
            for b in budgets
        ]
        for ax, key in zip(
            axes, ("adaptation_required", "capability_minus_best_fixed"), strict=True
        ):
            y = [
                np.mean([r[key] / r["tasks"] for r in selected if r["budget"] == b])
                for b in budgets
            ]
            marker_size = 5 + 2 * list(LABELS).index(arm)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=marker_size,
                markerfacecolor="none",
                label=label,
                color=COLORS[arm],
                lw=1.5,
                alpha=0.8,
            )
    axes[0].set_title("Measured adaptation-required task fraction")
    axes[1].set_title("Bank minus best fixed schedule / assigned tasks")
    for ax in axes:
        ax.set_xlabel("Recorded acquisition physics steps per corpus (thousands)")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(fontsize=8)
    if all(r["capability_minus_best_fixed"] == 0 for r in rows):
        axes[1].text(
            0.5,
            0.55,
            "Every measured corpus has a fixed schedule\ncovering all of its solvable tasks.",
            ha="center",
            transform=axes[1].transAxes,
            fontsize=10,
        )
    fig.savefig(out / "acquisition_responses.pdf")
    fig.savefig(out / "acquisition_responses.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    conditions = control["conditions"]
    x = np.arange(len(conditions))
    ax.bar(
        x - 0.18,
        [c["initial_choice"]["maximum_passages"] for c in conditions],
        0.36,
        color="#64748b",
        label="Choose complete schedule once",
    )
    ax.bar(
        x + 0.18,
        [c["sequential_wait_maximum_passages"] for c in conditions],
        0.36,
        color="#0f766e",
        label="Sequential decisions with WAIT",
    )
    ax.set_xticks(
        x,
        [
            "Never" if c["reveal_tick"] is None else f'{c["reveal_tick"] * .02:.2f} s'
            for c in conditions
        ],
    )
    ax.set_xlabel("Decision when corridor information becomes available")
    ax.set_ylabel("Maximum passages (6 recorded contexts)")
    ax.set_ylim(0, 7.4)
    ax.set_yticks(range(7))
    ax.legend(fontsize=8, loc="upper center")
    ax.set_title("Same seven schedules; finite information limits")
    fig.savefig(out / "wait_same_repertoire.pdf")
    fig.savefig(out / "wait_same_repertoire.png", dpi=180)
    plt.close(fig)
    return write_new(
        out / "result.json",
        dict(
            acquisition=refs,
            wait=wait_ref,
            implementation=artifact(Path(__file__)),
            outputs=[
                artifact(out / name)
                for name in (
                    "acquisition_by_corpus.csv",
                    "acquisition_yield.tex",
                    "acquisition_responses.pdf",
                    "acquisition_responses.png",
                    "wait_same_repertoire.pdf",
                    "wait_same_repertoire.png",
                )
            ],
            interpretation=(
                "Acquisition lines are corpus means with three individual trajectories, no confidence intervals. "
                "Constructor-specific training tasks are not a common evaluation set. "
                "WAIT bars are exact finite branch-table optima, not newly executed policies."
            ),
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, nargs="+", required=True)
    parser.add_argument("--wait", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.acquisition, args.wait, args.out)))
