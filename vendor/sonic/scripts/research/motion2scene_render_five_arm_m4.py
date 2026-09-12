#!/usr/bin/env python3
"""Render the completed five-arm M4 audit without changing measured results."""

import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

DATA = ROOT.parent / "research-data/groot-wbc"
EVIDENCE = ROOT / "docs/motion2scene/evidence/research-progress-20260909/five_arm_M4"
LABELS = {
    "uniform": "Screened uniform",
    "target_only": "Target-only",
    "reference_contrast": "Reference contrast",
    "analytic_contrast": "Executed contrast",
    "observation_curriculum": "Executed + replay",
}


def save(path, value):
    """Regenerate equal artifacts, refusing to replace different existing bytes."""
    data = value.encode() if isinstance(value, str) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"retain different existing artifact: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return artifact(path)


def save_json(path, value):
    return save(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    watch = DATA / "m2s-five-arm-M4-audit-watch-20260910-v1"
    completion_ref = artifact(watch / "complete.json")
    completion = read_checked(completion_ref)
    audit = read_checked(completion["result"])
    previous_ref = artifact(DATA / "m2s-response-diversity-M4-20260909-v1/result.json")
    previous = read_checked(previous_ref)
    boundaries_ref = artifact(watch / "M4_boundaries_complete.json")
    boundaries = read_checked(boundaries_ref)
    indexed = {r["run_id"]: r for r in audit["corpora"]}
    if audit["budget"] != 4 or len(indexed) != 15 or set(audit["arms"]) != set(LABELS):
        raise ValueError("complete five-arm M4 audit required")
    if any(r != indexed[r["run_id"]] for r in previous["corpora"]):
        raise ValueError("original corpus history differs from its earlier audit")
    if any(previous["arms"][a] != audit["arms"][a] for a in previous["arms"]):
        raise ValueError("original arm summaries changed")
    if any(
        not r["same_candidate_order"] or r["teacher_outcome_disagreements"]
        for r in audit["paired_contrast_replay"]
    ):
        raise ValueError("paired contrast/replay queues or physical teacher outcomes differ")
    captures = sum(r["physical_cost"]["assigned_episodes"] for r in audit["corpora"])
    steps = sum(r["physical_cost"]["total_recorded_steps"] for r in audit["corpora"])
    if captures != boundaries["total_captures"] or steps != boundaries["total_recorded_steps"]:
        raise ValueError("raw-record costs differ from completion receipts")

    rows, corpus_lines = [], []
    arm_lines = [
        "| Arm | Physics steps | Bank-solvable | Adaptation required | Best fixed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in LABELS.items():
        local = [r for r in audit["corpora"] if r["arm"] == arm]
        if len(local) != 3 or {r["seed"] for r in local} != {93201, 93202, 93203}:
            raise ValueError("all three assigned corpora required")
        pooled = audit["arms"][arm]["pooled_responses"]
        if not pooled["complete"] or pooled["assigned_tasks"] != 12:
            raise ValueError("all assigned pass/failure/unknown outcomes required")
        values = [
            pooled[k]
            for k in (
                "bank_solvable_lower",
                "adaptation_required_lower",
                "best_fixed_passages_lower",
            )
        ]
        arm_lines.append(
            f"| {label} | {audit['arms'][arm]['actual_recorded_steps']:,} | "
            + " | ".join(f"{v}/12" for v in values)
            + " |"
        )
        for corpus in sorted(local, key=lambda r: r["seed"]):
            r = corpus["responses"]
            row = dict(
                arm=arm,
                seed=corpus["seed"],
                assigned=r["assigned_tasks"],
                bank_solvable=r["bank_solvable_lower"],
                adaptation_required=r["adaptation_required_lower"],
                best_fixed=r["best_fixed_passages_lower"],
                physics_steps=corpus["physical_cost"]["total_recorded_steps"],
            )
            rows.append(row)
            corpus_lines.append(
                f"| {label} | {row['seed']} | {row['bank_solvable']}/4 | "
                f"{row['adaptation_required']}/4 | {row['best_fixed']}/4 | "
                f"{row['physics_steps']:,} |"
            )

    original_geometry = previous["geometric_cost_sources"]["proposal_accounting"]
    pool_refs = dict(
        original=previous["geometric_cost_sources"]["candidate_pool_result"],
        expanded=audit["geometric_cost_sources"]["candidate_pools"],
        reference_retained=artifact(DATA / "m2s-reference-candidate-pools-20260909-v1/result.json"),
        reference_active=audit["geometric_cost_sources"]["reference_pools"],
    )
    pools = {k: read_checked(v) for k, v in pool_refs.items()}
    search = [
        dict(
            component="Original shared pool",
            candidates=original_geometry["actual_shared_candidates"],
            queries=original_geometry["actual_shared_clearance_queries"],
            seconds=original_geometry["actual_shared_clearance_search_seconds"],
        ),
        dict(
            component="Expanded shared pool",
            candidates=pools["expanded"]["total_proposals"],
            queries=pools["expanded"]["actual_shared_clearance_queries"],
            seconds=sum(r["shared_clearance_search_seconds"] for r in pools["expanded"]["pools"]),
        ),
    ]
    for key, label in (
        ("reference_retained", "Reference v1, no acquisition queue"),
        ("reference_active", "Reference v2, active queue"),
    ):
        local = pools[key]["pools"]
        if [r["candidates"] for r in local] != [
            r["candidates"] for r in pools["expanded"]["pools"]
        ]:
            raise ValueError("reference screening must use the same expanded candidate identities")
        if key == "reference_retained" and any(r["queue"] is not None for r in local):
            raise ValueError("unexpected retained reference queue")
        search.append(
            dict(
                component=label,
                candidates=pools["expanded"]["total_proposals"],
                queries=sum(r["queries"] for r in local),
                seconds=sum(r["seconds"] for r in local),
            )
        )
    geometry_lines = [
        "| Recorded search component | Candidates evaluated | Clearance queries | Component seconds |",
        "| --- | ---: | ---: | ---: |",
    ] + [
        f"| {r['component']} | {r['candidates']:,} | {r['queries']:,} | {r['seconds']:.3f} |"
        for r in search
    ]

    figure_ref = artifact(DATA / "m2s-five-arm-M1-M4-evidence-20260910-v1/result.json")
    figure = read_checked(figure_ref)
    if completion["result"] not in figure["acquisition"]:
        raise ValueError("figure must use this completed M4 audit")
    outputs = []
    for ref in figure["outputs"]:
        path = checked(Path(ref["path"]), ref["sha256"])
        if path.name.startswith("acquisition_"):
            outputs.append(save(EVIDENCE / path.name, path.read_bytes()))
    save(EVIDENCE / "result.json", Path(completion["result"]["path"]).read_bytes())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    outputs.append(save(EVIDENCE / "M4_by_corpus.csv", stream.getvalue()))

    prefix = "evidence/research-progress-20260909/five_arm_M4/"
    report = "\n".join(
        [
            "# Five-arm M4 acquisition evidence",
            "",
            f"The completed raw-record audit covers {captures} acquisition captures "
            f"and {steps:,} measured physics steps. "
            "All twelve original corpus reports and four original arm summaries "
            "equal the earlier M4 audit exactly.",
            "",
            "There are three corpora per arm and four post-bootstrap encounter assignments per corpus. "
            "Costs include each corpus's bootstrap, pre-update students and all teacher branches. "
            "Adaptation required means neutral fails and at least one complete schedule passes.",
            "",
            *arm_lines,
            "",
            "Reference construction retains six bank-unsolvable tasks in its twelve-task denominator. "
            "No task outcome is unknown; four partial captures establish failure through observed contact, "
            "while their unobserved later fall/recovery status remains unknown.",
            "",
            "Execution-conditioned construction yields more physically solvable adaptation tasks here. "
            "The adaptation counts alone do not establish better teaching than target-only. "
            "Every corpus still admits one fixed schedule covering all its solvable tasks. "
            "These are construction-specific training scenes, not common-set or held-out policy results. "
            "Contrast/replay preserve candidate order and have identical teacher outcomes "
            "across all three corpora.",
            "",
            f"[Measured M1/M4 response figure]({prefix}acquisition_responses.pdf) · "
            f"[Per-corpus CSV]({prefix}M4_by_corpus.csv) · [Audit]({prefix}result.json)",
            "",
            "The figure shows individual corpora and corpus means against actual acquisition steps. "
            "Its endpoints contain different acquired tasks; it is not a policy learning curve. "
            "No confidence interval or significance claim is inferred from repeated episodes.",
            "",
            "| Arm | Corpus seed | Bank-solvable | Adaptation required | Best fixed | Physics steps |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *corpus_lines,
            "",
            "Geometric computation is reported separately below. "
            "Full pools include rejected and unselected candidates. "
            "Both reference screens reuse the expanded candidates; they are not new proposal draws. "
            "The retained v1 screen produced no acquisition queue and remains charged as actual search work.",
            "",
            *geometry_lines,
            "",
            "These are the recorded search-component timings, not end-to-end proposal wall time. "
            "Shared materialization, kinematics setup and other overhead are not inferred from these columns. "
            "The full pools support later checkpoints too; "
            "no unsupported per-arm or M4-only amortization is applied.",
            "",
            "Reproduce the independent audit with the original expanded plan and a fresh output directory:",
            "",
            "```bash",
            ".venv_isaaclab/bin/python scripts/research/motion2scene_response_diversity.py \\",
            f"  --plan {audit['plan']['path']} --budget 4 --out /tmp/m2s-five-arm-M4-audit",
            "```",
            "",
            "Regenerate this report and its evidence from the recorded source paths with "
            "` .venv_isaaclab/bin/python scripts/research/motion2scene_render_five_arm_m4.py`.",
            "The acquisition figure uses the unchanged `motion2scene_render_acquisition_evidence.py`; "
            "its source-bound output manifest also retains the previously measured WAIT control, "
            "with no new WAIT experiment.",
            "",
        ]
    )
    outputs.append(save(ROOT / "docs/motion2scene/FIVE_ARM_M4_ACQUISITION.md", report))
    return save_json(
        EVIDENCE / "verification_and_generation_v2.json",
        dict(
            implementation=artifact(Path(__file__)),
            completion=completion_ref,
            audit=completion["result"],
            previous=previous_ref,
            boundaries=boundaries_ref,
            original_corpus_reports_equal=12,
            original_arm_summaries_equal=4,
            captures=captures,
            actual_steps=steps,
            geometric_pool_sources=pool_refs,
            recorded_search_components=search,
            figure_generation=figure_ref,
            figure_visually_inspected=True,
            outputs=outputs,
            new_physics_steps=0,
            new_fits=0,
        ),
    )


if __name__ == "__main__":
    print(json.dumps(main()))
