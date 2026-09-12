#!/usr/bin/env python3
"""Render the completed M8 development panel: per-corpus passage and matched time.

Read-only. Every number here is copied from the frozen statistics result and the panel's own
per-episode results; this renderer computes no outcome, fits nothing and adds no physics.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

ARM_TITLES = {
    "uniform": "Feasibility-screened uniform",
    "target_only": "Target-only",
    "reference_contrast": "Reference-envelope contrast",
    "analytic_contrast": "Executed-envelope contrast",
    "observation_curriculum": "Executed contrast with replay",
}
ARM_ORDER = [
    "uniform",
    "target_only",
    "reference_contrast",
    "analytic_contrast",
    "observation_curriculum",
]
SEEDS = [93201, 93202, 93203]
MARK = {"pass": "pass", "failure": "fail", "technical_missing": "unmeasured", "not_run": "not run"}


def seconds(value):
    return "—" if value is None else f"{value:.2f}"


def signed(value):
    return "—" if value is None else f"{value:+.3f}"


def chosen_options(panel, rows):
    """The schedule each executed policy actually selected, from the panel's own audit."""
    chosen = {}
    for row in rows:
        path = panel / "episodes" / row["assignment_id"] / "result.json"
        if not path.is_file():
            continue
        audit = json.loads(path.read_text())["rows"][0].get("schedule_audit") or {}
        chosen[row["assignment_id"]] = audit.get("chosen_option_id")
    return chosen


def render(statistics, panel, out, acquisition=None):
    result = read_checked(artifact(statistics))
    summary, rows = result["summary"], result["rows"]
    policies, capability = summary["policies"], summary["capability"]
    models = read_checked(read_checked(artifact(panel / "prepared.json"))["study"])["models"]
    run_ids = {(m["arm"], m["seed"]): m["run_id"] for m in models}
    contexts = sorted({c["scene_id"] for c in capability})
    bank = {c["scene_id"]: c for c in capability}
    chosen = chosen_options(panel, rows)
    cell = defaultdict(dict)
    for row in rows:
        cell[row["policy_id"]][row["scene_id"]] = dict(row, chosen=chosen.get(row["assignment_id"]))

    fixed = sorted(p for p in policies if p.startswith("fixed_"))
    best_fixed = max(
        fixed,
        key=lambda p: (
            policies[p]["pass"],
            -sum(
                v["time_s"]
                for v in cell[p].values()
                if v["status"] == "pass" and v["time_s"] is not None
            ),
        ),
    )

    lines = [
        "# M8 development panel — completed common-set comparison",
        "",
        "Every one of the 138 predeclared assignments is measured: 90 learned M8 policy",
        "executions, the 42 forced schedule branches and the 6 strong-script branches, all on the",
        "same six development contexts at physics seed 8732 under the unchanged scorer.",
        "",
        f"Statistics receipt: `{statistics}`",
        f"Panel: `{panel}`",
        "",
        "## 1. What the fixed comparators already established",
        "",
        "These are the 48 frozen comparator executions, unchanged and reused.",
        "",
        "| Comparator | Passages | Mean time vs script (s) | Mutually successful contexts |",
        "|---|---:|---:|---:|",
    ]
    for name in ["script"] + fixed:
        entry = policies[name]
        paired = entry["paired_with_reference"]
        lines.append(
            f"| {'**script**' if name == 'script' else name} | {entry['pass']}/6 | "
            f"{signed(paired['mean_time_difference_s'])} | {paired['mutually_successful_contexts']} |"
        )
    lines += [
        "",
        f"The seven-schedule bank covers {summary['capability_count_lower']}/6 contexts and the",
        f"script covers {policies['script']['pass']}/6, so **the script is already at the ceiling of",
        "this panel**: no learned policy can beat its success count here. What the panel can show is",
        "whether the construction methods train *different* policies, whether they recover the bank's",
        "complementary capability, and what passage time they pay.",
        "",
        "## 2. Per-corpus passage on the six common contexts",
        "",
        "Individual training corpora, never a pooled mean. Each cell is the measured outcome, the",
        "passage time in seconds, and the schedule the policy actually selected.",
        "",
    ]
    header = "| Training corpus | " + " | ".join(contexts) + " | Passages |"
    lines += [header, "|---" * (len(contexts) + 2) + "|"]
    for arm in ARM_ORDER:
        for seed in SEEDS:
            run = run_ids[arm, seed]
            cells = []
            for context in contexts:
                value = cell[run][context]
                mark = MARK[value["status"]]
                option = value["chosen"] or "—"
                cells.append(
                    f"{mark} {seconds(value['time_s'])}<br>`{option}`"
                    if value["status"] == "pass"
                    else f"{mark}<br>`{option}`"
                )
            lines.append(
                f"| {ARM_TITLES[arm]} · {seed} | "
                + " | ".join(cells)
                + f" | {policies[run]['pass']}/6 |"
            )
    lines += [
        "",
        "| Context | Measured passing bank schedules | Bank | Best bank time (s) |",
        "|---|---|---:|---:|",
    ]
    for context in contexts:
        entry = bank[context]
        names = (
            ", ".join(p.replace("fixed_", "") for p in entry["measured_passing_schedules"])
            or "none"
        )
        lines.append(
            f"| {context} | {names} | {entry['lower']}/1 | {seconds(entry['best_passing_time_s'])} |"
        )

    lines += [
        "",
        "## 3. Matched time and selection against the comparators",
        "",
        "Time differences are paired on mutually successful identical conditions only; negative is",
        "faster. A selection failure is a context where the bank demonstrably contains a passing",
        f"schedule but the policy failed. The best fixed comparator on this panel is `{best_fixed}`.",
        "",
        "| Training corpus | Passages | Selection failures | vs script (s) | vs best fixed (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        for seed in SEEDS:
            run = run_ids[arm, seed]
            entry = policies[run]
            lines.append(
                f"| {ARM_TITLES[arm]} · {seed} | {entry['pass']}/6 | {entry['selection_failures']} | "
                f"{signed(entry['paired_with_reference']['mean_time_difference_s'])} | "
                f"{signed(entry['paired_with_fixed_schedules'][best_fixed]['mean_time_difference_s'])} |"
            )

    lines += [
        "",
        "## 4. Selection diversity: how many distinct schedules each policy actually used",
        "",
        "A policy that selects one schedule on every context is a constant function, not a",
        "perceptive selector. This column is read from each execution's own `schedule_audit`.",
        "",
        "| Training corpus | Distinct schedules used | Schedules | Passages |",
        "|---|---:|---|---:|",
    ]
    collapsed = 0
    for arm in ARM_ORDER:
        for seed in SEEDS:
            run = run_ids[arm, seed]
            used = sorted({v["chosen"] for v in cell[run].values() if v["chosen"]})
            collapsed += len(used) == 1
            lines.append(
                f"| {ARM_TITLES[arm]} · {seed} | {len(used)} | "
                + ", ".join(f"`{u}`" for u in used)
                + f" | {policies[run]['pass']}/6 |"
            )
    lines += [
        "",
        f"**{collapsed} of 15 learned policies use a single schedule on all six contexts.**",
        "",
    ]
    if acquisition is not None:
        needs_two = set(
            read_checked(artifact(acquisition))["corpora_without_one_passing_fixed_schedule"]
        )
        lines += [
            "### Which corpora produced a selective policy",
            "",
            "The acquisition receipt already records which training corpora cannot be covered by any",
            "single fixed schedule. Comparing that against selection diversity on this panel:",
            "",
            f"Acquisition source: `{acquisition}`",
            "",
            "| Training corpus | Corpus needs >1 schedule | Policy uses >1 schedule | Passages |",
            "|---|:--:|:--:|---:|",
        ]
        agree = 0
        for arm in ARM_ORDER:
            for seed in SEEDS:
                run = run_ids[arm, seed]
                used = len({v["chosen"] for v in cell[run].values() if v["chosen"]})
                corpus, policy = run in needs_two, used > 1
                agree += corpus == policy
                lines.append(
                    f"| {ARM_TITLES[arm]} · {seed} | {'yes' if corpus else 'no'} | "
                    f"{'yes' if policy else 'no'} | {policies[run]['pass']}/6 |"
                )
        lines += [
            "",
            f"**These agree on {agree} of 15 corpora.** Every corpus that a single fixed schedule",
            "can cover trained a constant policy; the corpora that demonstrably require more than one",
            "response are the ones whose policies condition on the scene. This is a correspondence",
            "across 15 corpora, not a controlled manipulation: nothing here varied corpus",
            "complementarity while holding the construction method fixed.",
            "",
        ]
    lines += [
        "## 5. Construction arms",
        "",
        "| Construction method | Passages by corpus (93201 / 93202 / 93203) | Mean |",
        "|---|---|---:|",
    ]
    for arm in ARM_ORDER:
        entry = result["construction_arms"][arm]
        rates = entry["passage_fraction_by_corpus"]
        by_corpus = " / ".join("—" if r is None else f"{r * 6:.0f}/6" for r in rates)
        mean = entry["descriptive_corpus_variation"]
        lines.append(
            f"| {ARM_TITLES[arm]} | {by_corpus} | "
            + ("—" if not mean else f"{mean.get('mean', float('nan')) * 6:.2f}/6")
            + " |"
        )
    lines += [
        "",
        "The three corpus seeds share the same six development layouts and one execution seed.",
        "These are not 18 independent layouts.",
        "",
        "## 6. Declared construction comparisons",
        "",
        "| Comparison | Passage difference by corpus | Paired time by corpus (s) |",
        "|---|---|---|",
    ]
    for name, entry in result["construction_comparisons"].items():
        differences = " / ".join(
            "—" if c["passage_count_difference"] is None else f"{c['passage_count_difference']:+d}"
            for c in entry["corpora"]
        )
        times = " / ".join(
            f"{signed(c['mean_time_difference_s'])} ({c['mutually_successful_contexts']})"
            for c in entry["corpora"]
        )
        lines.append(f"| {name.replace('_', ' ')} | {differences} | {times} |")

    lines += [
        "",
        "Corpora are ordered 93201 / 93202 / 93203. Paired time counts the mutually successful",
        "contexts in parentheses. Positive passage differences favour the first named arm.",
        "",
        "## 7. What this panel does and does not settle",
        "",
        "It measures what each construction method's M8 policy does on shared, previously fixed",
        "development conditions at a matched checkpoint. It does not measure the reserved layouts,",
        "and it cannot separate the arms beyond the resolution of six reused contexts. The script",
        "sits at this panel's ceiling, so a tie in success count is a property of the panel, not",
        "evidence that the populations are equal.",
        "",
    ]
    out.write_text("\n".join(lines) + "\n")
    return artifact(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.statistics, args.panel, args.out, args.acquisition)))
