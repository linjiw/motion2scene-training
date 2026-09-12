#!/usr/bin/env python3
"""Export static completion figures from a validated V4 statistics receipt."""

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import motion2scene_evaluation_statistics as statistics
import numpy as np

SCHEMA = "motion2scene_evaluation_figures_v1"
ARM_NAMES = {
    "uniform": "Uniform",
    "target_only": "Target only",
    "analytic_contrast": "Analytic contrast",
    "analytic_observation_curriculum": "Observation curriculum",
}
COLORS = dict(zip(statistics.ARMS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), strict=True))
BASELINE_NAMES = {
    "always_walk": "Always walk",
    "constant_development_selected": "Constant adaptation",
    "scripted_multi_option": "Scripted controller",
}
BASELINE_STYLES = (":", "--", "-.")


def load_statistics(path, expected_sha256=None):
    """Validate displayed quantities against retained assignments; do not refit CIs."""
    ref = statistics.file_ref(path)
    if expected_sha256 and ref["sha256"].removeprefix("sha256:") != expected_sha256.removeprefix(
        "sha256:"
    ):
        raise ValueError("statistics input hash differs")
    value = json.loads(Path(path).read_text())
    if value.get("schema") != statistics.OUTPUT_SCHEMA:
        raise ValueError("a motion2scene_reserved_statistics_v1 result is required")
    receipt = {
        "schema": statistics.SCHEMA,
        "status": value["input_status"],
        "report": {
            "assigned_episodes": 972,
            "complete": value["complete"],
            "rows": value["assigned_rows"],
        },
    }
    # One draw reconstructs identities, bounds and descriptive means only.
    # Displayed intervals remain exactly those in the hashed statistics input.
    reconstructed = statistics.summarize(receipt, draws=1)
    for key in (
        "assigned_outcomes",
        "policy_summaries",
        "per_corpus_learning_curves",
        "mean_learning_curves",
    ):
        if value[key] != reconstructed[key]:
            raise ValueError(f"statistics {key} disagrees with retained assignments")
    actual = value["paired_comparisons"]
    expected = reconstructed["paired_comparisons"]
    if len(actual) != len(expected):
        raise ValueError("all34 declared paired comparisons are required")
    for row, checked in zip(actual, expected, strict=True):
        for key in (
            "comparison_id",
            "left",
            "right",
            "left_checkpoint",
            "right_checkpoint",
            "kind",
            "contrast_direction",
            "assigned_matched_pairs",
            "complete_matched_pairs",
            "unknown_matched_pairs",
            "unique_left_episodes",
            "unique_right_episodes",
            "shared_right_baseline",
        ):
            if row[key] != checked[key]:
                raise ValueError(f"comparison {key} disagrees with retained assignments")
        completion = row["completion"]
        for key in ("difference_lower", "difference_upper", "mean_difference"):
            if completion[key] != checked["completion"][key]:
                raise ValueError("paired completion bounds or mean differ")
        interval = completion["crossed_95_percentile"]
        if row["unknown_matched_pairs"]:
            if interval is not None:
                raise ValueError("unknown comparisons cannot carry a completion interval")
        elif interval is not None:
            limits = np.asarray(interval, dtype=float)
            if (
                limits.shape != (2,)
                or not np.isfinite(limits).all()
                or not (-1 <= limits[0] <= limits[1] <= 1)
            ):
                raise ValueError("invalid crossed completion interval")
    bootstrap = value["bootstrap"]
    if (
        bootstrap["draws"] != statistics.DRAWS
        or bootstrap["seed"] != statistics.RANDOM_SEED
        or bootstrap["corpus_order"] != list(statistics.CORPORA)
        or bootstrap["evaluation_seed_order"] != list(statistics.EVALUATION_SEEDS)
    ):
        raise ValueError("publication plots require the declared bootstrap specification")
    return value, ref


def make_plot_data(value):
    """Expose exactly plotted values, including bounds and unique denominators."""
    panels = []
    for corpus in (*statistics.CORPORA, None):
        series = []
        for arm in statistics.ARMS:
            points = []
            for checkpoint in statistics.CHECKPOINTS:
                rows = [
                    row
                    for row in value["per_corpus_learning_curves"]
                    if row["training_arm"] == arm
                    and row["checkpoint"] == checkpoint
                    and (corpus is None or row["acquisition_seed"] == corpus)
                ]
                lower = float(np.mean([row["completion_lower"] for row in rows]))
                upper = float(np.mean([row["completion_upper"] for row in rows]))
                points.append(
                    dict(
                        checkpoint=checkpoint,
                        actual_steps=float(
                            np.mean([r["acquisition_actual_physics_steps"] for r in rows])
                        ),
                        assigned_maximum_steps=statistics.CHECKPOINTS[checkpoint],
                        lower=lower,
                        upper=upper,
                        mean=(
                            lower
                            if all(r["complete_outcome_mean"] is not None for r in rows)
                            else None
                        ),
                        assigned=sum(row["assigned"] for row in rows),
                        measured=sum(row["measured_outcome_denominator"] for row in rows),
                        corpus_count=len(rows),
                    )
                )
            series.append(dict(arm=arm, points=points))
        panels.append(dict(acquisition_seed=corpus, series=series))
    return dict(
        panels=panels,
        baselines={key: value["policy_summaries"][key] for key in statistics.BASELINES},
        comparisons=[
            {
                key: row[key]
                for key in (
                    "comparison_id",
                    "left",
                    "right",
                    "left_checkpoint",
                    "right_checkpoint",
                    "completion",
                    "assigned_matched_pairs",
                    "complete_matched_pairs",
                    "unknown_matched_pairs",
                    "unique_left_episodes",
                    "unique_right_episodes",
                    "shared_right_baseline",
                )
            }
            for row in value["paired_comparisons"]
        ],
    )


def decorate(figure, title, caption, synthetic, provenance):
    prefix = "SYNTHETIC — NOT PAPER EVIDENCE\n" if synthetic else ""
    figure.suptitle(prefix + title, fontsize=13, fontweight="bold")
    figure.text(0.04, 0.018, textwrap.fill(caption, 150), ha="left", va="bottom", fontsize=8)
    figure.text(0.98, 0.005, provenance, ha="right", va="bottom", fontsize=5.5, color="0.4")


def learning_figure(data, synthetic, provenance):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), sharex=True, sharey=True)
    all_x = [
        point["actual_steps"] / 1000
        for panel in data["panels"]
        for series in panel["series"]
        for point in series["points"]
    ]
    padding = max(1.0, (max(all_x) - min(all_x)) * 0.12)
    for axis, panel in zip(axes.flat, data["panels"], strict=True):
        for key, style in zip(statistics.BASELINES, BASELINE_STYLES, strict=True):
            row = data["baselines"][key]
            lo, hi = 100 * row["completion_lower"], 100 * row["completion_upper"]
            if row["complete_outcome_mean"] is None:
                axis.axhspan(lo, hi, facecolor="0.7", alpha=0.10, hatch="//", zorder=0)
                axis.axhline(hi, color="0.5", ls=style, lw=0.9, zorder=1)
            axis.axhline(lo, color="0.4", ls=style, lw=0.9, zorder=1)
        for series in panel["series"]:
            color, points = COLORS[series["arm"]], series["points"]
            x = np.array([point["actual_steps"] for point in points]) / 1000
            lower = np.array([point["lower"] for point in points]) * 100
            upper = np.array([point["upper"] for point in points]) * 100
            complete = all(point["mean"] is not None for point in points)
            axis.plot(x, lower, color=color, lw=1.8, ls="-" if complete else "--")
            if not complete:
                axis.plot(x, upper, color=color, lw=1.2, ls="--")
                axis.fill_between(x, lower, upper, color=color, alpha=0.06)
            for index, point in enumerate(points):
                marker = "o" if point["checkpoint"] == "M2" else "s"
                if point["mean"] is not None:
                    axis.scatter(
                        x[index], 100 * point["mean"], marker=marker, s=32, color=color, zorder=4
                    )
                else:
                    axis.vlines(x[index], lower[index], upper[index], color=color, lw=1.4)
                    axis.scatter([x[index]], [lower[index]], marker="v", color=color, s=32)
                    axis.scatter([x[index]], [upper[index]], marker="^", color=color, s=32)
        corpus = panel["acquisition_seed"]
        axis.set_title(
            (
                f"Acquired corpus {corpus} · 36 assigned episodes / point"
                if corpus
                else "Equal-weight mean of 3 corpora · 108 assignments / point"
            ),
            fontsize=9,
        )
        axis.set_ylim(-3, 103)
        axis.set_xlim(max(0, min(all_x) - padding), max(all_x) + padding)
        axis.grid(axis="y", color="0.9", lw=0.7)
        axis.set_xlabel("Actual acquisition physics steps (thousands)")
        axis.set_ylabel("Completion / assigned episodes (%)")
        axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], color=COLORS[key], lw=2, label=ARM_NAMES[key]) for key in statistics.ARMS
    ]
    handles += [
        Line2D([], [], color="0.4", ls=style, label=BASELINE_NAMES[key] + " (36 unique)")
        for key, style in zip(statistics.BASELINES, BASELINE_STYLES, strict=True)
    ]
    handles += [
        Line2D([], [], color="0.2", marker="o", ls="none", label="M2"),
        Line2D([], [], color="0.2", marker="s", ls="none", label="M4"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    caption = (
        "M2/M4 share each acquisition trajectory; lines connect only these two measured checkpoints. "
        "Individual x coordinates use actual steps; aggregate x and completion use equal-weight means "
        "of three corpora. "
        "Assigned maxima: M2=27,416; M4=46,488. Each baseline has 36 unique episodes (18 layouts × 2 seeds), "
        "repeated visually across panels. Triangles, dashed edges and shading show retained lower/upper"
        " bounds for unknown outcomes, "
        "not confidence intervals; no unknown midpoint is plotted. No learning-curve confidence "
        "interval is inferred."
    )
    decorate(figure, "Acquisition learning curves", caption, synthetic, provenance)
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.25,
        top=0.89 if synthetic else 0.92,
        hspace=0.34,
        wspace=0.22,
    )
    return figure, caption


def comparison_label(row):
    names = {**ARM_NAMES, **BASELINE_NAMES}
    left = names[row["left"]] + " " + row["left_checkpoint"]
    right = names[row["right"]] + (" " + row["right_checkpoint"] if row["right_checkpoint"] else "")
    return left + " − " + right


def comparison_figure(data, synthetic, provenance):
    figure, axis = plt.subplots(figsize=(12, 13.5))
    rows = data["comparisons"]
    extents = [0.0]
    for index, row in enumerate(rows):
        y, item = len(rows) - index - 1, row["completion"]
        color = COLORS[row["left"]]
        if item["mean_difference"] is None:
            lo, hi = 100 * item["difference_lower"], 100 * item["difference_upper"]
            axis.hlines(y, lo, hi, color=color, ls="--", lw=1.5)
            axis.scatter(lo, y, color=color, marker="<", s=35)
            axis.scatter(hi, y, color=color, marker=">", s=35)
            extents += [lo, hi]
        else:
            point = 100 * item["mean_difference"]
            interval = item["crossed_95_percentile"]
            if interval is not None:
                lo, hi = np.asarray(interval) * 100
                axis.hlines(y, lo, hi, color=color, lw=1.5)
                axis.vlines([lo, hi], y - 0.1, y + 0.1, color=color, lw=1)
                extents += [lo, hi]
            axis.scatter(point, y, color=color, s=28, zorder=3)
            extents.append(point)
        suffix = " · bounds" if row["unknown_matched_pairs"] else ""
        axis.text(
            1.02,
            y,
            f"{row['complete_matched_pairs']}/{row['assigned_matched_pairs']}" + suffix,
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=8,
        )
    limit = min(105, max(10, np.ceil(max(abs(v) for v in extents) / 10) * 10 + 5))
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-1, len(rows))
    axis.axvline(0, color="0.5", lw=0.8, ls=":")
    axis.set_yticks(range(len(rows)), [comparison_label(row) for row in reversed(rows)], fontsize=8)
    groups = [
        (
            row["left_checkpoint"],
            (
                "baseline"
                if row["shared_right_baseline"]
                else (
                    "update" if row["left_checkpoint"] != row["right_checkpoint"] else "constructor"
                )
            ),
        )
        for row in rows
    ]
    for index in range(1, len(rows)):
        if groups[index] != groups[index - 1]:
            axis.axhline(len(rows) - index - 0.5, color="0.85", lw=0.7)
    axis.text(
        1.02,
        len(rows) - 0.05,
        "Known / assigned\nmatched pairs",
        transform=axis.get_yaxis_transform(),
        fontsize=8,
        va="bottom",
    )
    axis.set_xlabel("Completion difference (percentage points; positive favors left)")
    axis.grid(axis="x", color="0.93", lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    caption = (
        "All 34 predeclared left-minus-right completion contrasts. Points show complete-outcome "
        "differences; solid intervals are "
        "the supplied 95% paired crossed corpus/layout bootstrap percentiles (20,000 draws, seed 202609081822). "
        "The two evaluation seeds stay within each sampled layout. Dashed lines with arrow endpoints "
        "are unknown-outcome "
        "identification bounds, with no point estimate or confidence interval. Each contrast has 108 "
        "assigned pairs; "
        "shared baselines contribute only 36 unique right-hand episodes, without independent corpus copies. "
        "Only three acquired corpora; intervals are descriptive, without multiplicity correction. No "
        "time/cost effect is plotted."
    )
    decorate(figure, "Paired completion comparisons", caption, synthetic, provenance)
    figure.subplots_adjust(left=0.39, right=0.84, bottom=0.13, top=0.90 if synthetic else 0.93)
    return figure, caption


def run(args):
    value, input_ref = load_statistics(args.input, args.expected_sha256)
    if args.data_kind != "synthetic" and "synthetic" in str(args.input).lower():
        raise ValueError("synthetic inputs must retain their visible synthetic label")
    out = args.out.resolve()
    if out.is_relative_to(args.input.resolve().parent):
        raise ValueError(
            "figures require a separate output directory outside the statistics folder"
        )
    source = statistics.file_ref(__file__)
    data = make_plot_data(value)
    registration = dict(
        schema=SCHEMA + "_registration",
        statistics=input_ref,
        plot_source=source,
        validation_source=statistics.file_ref(statistics.__file__),
        statistics_registration=value.get("registration"),
        data_kind=args.data_kind,
        bootstrap=value["bootstrap"],
        matplotlib_version=matplotlib.__version__,
        numpy_version=np.__version__,
        scope="Presentation of supplied statistics only; displayed intervals are copied, with no new evaluation",
        validation_reconstruction_draws=1,
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    provenance = (
        "statistics SHA256 "
        + input_ref["sha256"][7:19]
        + " · plot source "
        + source["sha256"][7:19]
    )
    artifacts, captions = {}, {}
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 9}):
        for name, factory in (
            ("learning_curves", learning_figure),
            ("paired_completion", comparison_figure),
        ):
            figure, caption = factory(data, args.data_kind == "synthetic", provenance)
            for suffix in ("pdf", "png"):
                path = out / (name + "." + suffix)
                metadata = (
                    {
                        "Title": name,
                        "Subject": caption,
                        "Keywords": json.dumps(registration, sort_keys=True),
                    }
                    if suffix == "pdf"
                    else {
                        "Description": json.dumps(registration, sort_keys=True),
                        "Caption": caption,
                    }
                )
                figure.savefig(path, dpi=250, metadata=metadata)
                artifacts[path.name] = statistics.file_ref(path)
            captions[name] = caption
            plt.close(figure)
    result = dict(
        schema=SCHEMA,
        status="complete",
        registration=statistics.file_ref(out / "registration.json"),
        data_kind=args.data_kind,
        statistics=input_ref,
        figures=artifacts,
        captions=captions,
        plotted_data=data,
        assigned_outcomes=value["assigned_outcomes"],
        new_physics_steps=0,
        displayed_intervals_recomputed=False,
        cost_or_time_plotted=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "result": statistics.file_ref(out / "result.json"),
                "data_kind": args.data_kind,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--data-kind", choices=("synthetic", "reported"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
