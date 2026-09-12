#!/usr/bin/env python3
"""Plot completed development policy outcomes with measured within-scene costs."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.motion2scene_export_traversal_dataset import (  # noqa: E402
    checked,
    digest,
    write_json,
)

MODES = ("learned", "scripted_sustained", "constant_short", "constant_sustained", "always_walk")
LABELS = ("Learned", "Scripted", "Constant short", "Constant sustained", "Always walk")


def render(result_path, out):
    prefix = out / "duration_policy_comparison"
    if prefix.with_suffix(".json").exists():
        raise FileExistsError("Preserve existing figure and receipt")
    suite = json.loads(result_path.read_text())
    checked(suite["registration"])
    rows, sources = {}, []
    for group in suite["groups"]:
        child = json.loads(checked(group["result"]).read_text())
        assert child["rows"] == group["rows"] and len(child["rows"]) == 1
        row = child["rows"][0]
        assert row["measurement_admitted"] and row["physics_steps"] == 1192
        manifest = json.loads(checked(group["manifest"]).read_text())
        assert manifest["cells"][0]["runtime_seed"] == 8732
        interface = json.loads(checked(row["sensor"]).read_text())
        entered = [s["to"] for s in interface["switches"] if s["to"] != "neutral"]
        key = (group["mode"], group["scene_index"])
        assert key not in rows
        rows[key] = {
            "mode": key[0],
            "scene_index": key[1],
            "pass": row["pass"],
            "passage_time_s": row["costs"]["passage_time_s"],
            "actual_entered_option_ids": entered,
            "terminal_hold_failure": bool(row["passage"]["stabilization_failed"]),
        }
        sources.extend((group["result"], group["manifest"], row["sensor"]))
    assert set(rows) == {(m, s) for m in MODES for s in range(3)}
    assert suite["episodes"] == 15 and suite["physics_steps"] == 17880
    counts = [sum(rows[(m, s)]["pass"] for s in range(3)) for m in MODES]
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.65), gridspec_kw={"width_ratios": [1, 1.45]})
    fig.subplots_adjust(left=0.15, right=0.995, top=0.81, bottom=0.28, wspace=0.34)
    colors = ["#267b70", "#487fa1", "#85919e", "#85919e", "#85919e"]
    axes[0].barh(range(1, 6), counts, color=colors, height=0.62)
    axes[0].set_yticks(range(1, 6), LABELS)
    axes[0].set_ylim(5.5, -0.5)
    axes[0].set_xlim(0, 3.4)
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_xlabel("Completed development scenes (out of 3)", fontsize=9)
    for i, count in enumerate(counts, start=1):
        axes[0].text(count + 0.06, i, f"{count}/3", va="center", fontsize=10)
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)
    axes[1].axis("off")
    table_rows, cell_colors = [], []
    for mode in MODES:
        text_row, color_row = [], []
        for scene in range(3):
            row = rows[(mode, scene)]
            text_row.append(
                f"{row['passage_time_s']:.2f} s"
                if row["pass"]
                else ("Hold failure*" if mode == "constant_sustained" and scene == 0 else "Fail")
            )
            color_row.append("#eaf3ef" if row["pass"] else "#f8e8e5")
        table_rows.append(text_row)
        cell_colors.append(color_row)
    table = axes[1].table(
        cellText=table_rows,
        cellColours=cell_colors,
        colLabels=["Empty", "Short beam", "Long passage"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for cell in table.get_celld().values():
        cell.set_edgecolor("white")
    axes[1].set_title("Measured passage time (compare within each column)", fontsize=10, pad=13)
    fig.suptitle(
        "Perceptive duration selection: 15 actual development episodes", fontsize=13, y=0.985
    )
    fig.text(
        0.15,
        0.07,
        "*Crossed without contact/fall; the 0.30 s hold did not finish by the 5.94 s horizon.",
        fontsize=9,
    )
    fig.text(
        0.15,
        0.005,
        (
            "Same three training layouts, new physics seed 8732. "
            "Learned matches script and misses 0.04 s on the short beam."
        ),
        fontsize=9,
    )
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(prefix.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_json(
        prefix.with_suffix(".json"),
        {
            "schema": "motion2scene_duration_policy_comparison_figure_v1",
            "source_result": {"path": str(result_path), "sha256": digest(result_path)},
            "sources": sources,
            "rows": list(rows.values()),
            "counts": dict(zip(MODES, counts)),
            "renderer": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
            "artifacts": [
                {
                    "path": str(prefix.with_suffix("." + s)),
                    "sha256": digest(prefix.with_suffix("." + s)),
                }
                for s in ("pdf", "svg", "png")
            ],
            "caption": (
                "Fifteen actual six-second development episodes on the same three training layouts, "
                "physics seed 8732. Learned and scripted policies pass 3/3 with identical choices/times. "
                "Constant short passes 2/3 and is 0.04 s faster on the short beam. Constant sustained "
                "passes 2/3; empty crossing has insufficient remaining stabilization time under the "
                "registered finite horizon, without fall or contact. Always walk passes 1/3. No held-out, "
                "curriculum or learned duration-selection improvement is established."
            ),
            "new_physics_steps": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.result, args.out)
