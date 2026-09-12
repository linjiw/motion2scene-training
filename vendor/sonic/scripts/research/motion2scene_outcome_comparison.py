#!/usr/bin/env python3
"""Paired physical comparison of the fixed development ridge and outcome trees."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def paired_summary(pairs):
    if not pairs or len({p["context"] for p in pairs}) != len(pairs):
        raise ValueError("distinct assigned contexts required")
    deltas = []
    for pair in pairs:
        for family in ("ridge", "outcome_tree"):
            row = pair[family]
            if row["pass"] and (
                row["time_s"] is None or not np.isfinite(row["time_s"]) or row["time_s"] < 0
            ):
                raise ValueError("successful execution needs a measured finite passage time")
            if not row["pass"] and row["time_s"] is not None:
                raise ValueError("failed execution cannot supply a successful passage cost")
        if pair["ridge"]["pass"] and pair["outcome_tree"]["pass"]:
            deltas.append(pair["outcome_tree"]["time_s"] - pair["ridge"]["time_s"])
    return dict(
        assigned_contexts=len(pairs),
        passage_counts={f: sum(p[f]["pass"] for p in pairs) for f in ("ridge", "outcome_tree")},
        mutually_successful_contexts=len(deltas),
        mean_paired_time_difference_s=float(np.mean(deltas)) if deltas else None,
        time_difference_direction="outcome tree minus ridge; negative is faster",
        uncertainty="descriptive six-context development comparison at one matched physics seed",
    )


def run(out):
    study = read_checked(artifact(out / "study.json"))
    if study["assigned_episodes"] != 6 or len(study["assignments"]) != 6:
        raise ValueError("all six originally assigned development contexts required")
    pairs = []
    refs = []
    for assignment in study["assignments"]:
        manifest = read_checked(assignment["manifest"])
        result_ref = artifact(Path(assignment["manifest"]["path"]).parent / "result.json")
        result = read_checked(result_ref)
        if result["manifest"] != assignment["manifest"]:
            raise ValueError("outcome result belongs to another assignment")
        original = read_checked(assignment["paired_ridge_result"])
        original_manifest = read_checked(original["manifest"])
        if original["manifest"] != manifest["nominal_template"]:
            raise ValueError("paired ridge was not the unchanged execution template")
        if manifest["cell"]["runtime_seed"] != study["physics_seed"]:
            raise ValueError("physics seed is not matched")
        selected = [r for r in original["rows"] if r["cell_id"] == manifest["cell"]["cell_id"]]
        if len(selected) != 1 or original_manifest["split"] != "development":
            raise ValueError("one matched development ridge execution required")
        pair = dict(context=assignment["context"])
        for family, row in (("ridge", selected[0]), ("outcome_tree", result["row"])):
            pair[family] = dict(
                **{"pass": bool(row["pass"])},
                time_s=row["costs"]["passage_time_s"],
                measurement_admitted=row["measurement_admitted"],
                task_outcome=row["outcome"]["task_outcome"],
            )
        pair["paired_time_difference_s"] = (
            pair["outcome_tree"]["time_s"] - pair["ridge"]["time_s"]
            if all(pair[f]["pass"] for f in ("ridge", "outcome_tree"))
            else None
        )
        pairs.append(pair)
        refs.append(
            dict(
                context=assignment["context"],
                outcome=result_ref,
                ridge=assignment["paired_ridge_result"],
            )
        )
    return write_new(
        out / "comparison.json",
        dict(
            study=artifact(out / "study.json"),
            implementation=artifact(Path(__file__)),
            sources=refs,
            summary=paired_summary(pairs),
            pairs=pairs,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().out)))
