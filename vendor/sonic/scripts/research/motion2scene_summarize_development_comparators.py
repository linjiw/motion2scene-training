#!/usr/bin/env python3
"""Read the completed 48-comparator block using the original panel statistics."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
import motion2scene_development_panel_statistics as statistics  # noqa: E402
import motion2scene_development_ready as ready  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def physical_events(row):
    """Observed events survive partial capture; missing tails cannot prove absence."""
    complete = row["measurement_admitted"]
    kinds = {event["kind"] for event in row["outcome"]["physical_events"]}
    contact = row.get("contact_audit") or {}
    passage = row.get("passage") or {}
    # Single-beam fall_observed spans the first episode. The course scorer
    # names that horizon separately and narrows fall_observed to passage.
    whole_episode_fall = passage.get(
        "fall_anywhere_in_first_episode",
        passage.get("fall_observed") if "beam_count" not in passage else None,
    )
    if "verified_environment_contact" in kinds:
        contact_status = "observed"
    elif (
        contact.get("complete_synchronized_streams")
        and contact.get("no_undesired_measured_contact") is False
    ):
        contact_status = "observed"
    elif (
        complete
        and contact.get("complete_synchronized_streams")
        and contact.get("no_undesired_measured_contact") is True
    ):
        contact_status = "not_observed"
    else:
        contact_status = "unknown"
    if "recorded_fall_or_upright_threshold_failure" in kinds or whole_episode_fall is True:
        fall_status = "observed"
    elif complete and whole_episode_fall is False:
        fall_status = "not_observed"
    else:
        fall_status = "unknown"
    return dict(
        undesired_environment_contact=contact_status,
        fall_or_upright_threshold_failure=fall_status,
        maximum_recorded_undesired_environment_force_n=contact.get(
            "maximum_undesired_environment_force_n"
        ),
        recorded_episode_duration_s=row["costs"]["whole_episode_time_s"],
        whole_horizon_stable=row.get("whole_horizon_stable"),
        returned_to_neutral=row.get("returned_to_neutral"),
    )


def read_completed(panel):
    readiness_ref, study = ready.verify(panel)
    completion_ref = artifact(panel / "comparators_complete.json")
    complete = read_checked(completion_ref)
    expected = [r for r in study["assignments"] if r["mode"] != "learned"]
    if (
        len(expected) != 48
        or len(complete["assignments"]) != 48
        or complete["readiness"] != readiness_ref
        or complete["complete_panel"] is not False
        or complete["assigned_panel_episodes"] != 138
    ):
        raise ValueError("the complete original 48-comparator block is required")
    rows = []
    for assignment, source in zip(expected, complete["assignments"], strict=True):
        if source["assignment_id"] != assignment["assignment_id"]:
            raise ValueError("completed comparator identities or ordering changed")
        expected_path = panel / "episodes" / assignment["assignment_id"] / "result.json"
        if source["result"] != artifact(expected_path):
            raise ValueError("completed comparator result differs from its original capture")
        measured = read_checked(source["result"])
        if len(measured["rows"]) != 1:
            raise ValueError("one physical result per comparator assignment required")
        manifest = read_checked(measured["manifest"])
        ready.panel_api.check_assignment(assignment, manifest, study, {})
        native = measured["rows"][0]
        value = statistics.measure(native)
        if value["status"] not in ("pass", "failure"):
            raise ValueError("completed comparator block has an unknown physical outcome")
        if measured["physics_steps"] != value["recorded_physics_steps"]:
            raise ValueError("aggregate and recorded physical counter disagree")
        rows.append(
            dict(
                assignment_id=assignment["assignment_id"],
                policy_id=assignment["policy_id"],
                scene_id=assignment["scene_id"],
                physics_seed=statistics.PHYSICS_SEED,
                source_result=source["result"],
                chosen_option_id=(native.get("schedule_audit") or {}).get("chosen_option_id"),
                **value,
                **physical_events(native),
            )
        )
    fixed = sorted({r["policy_id"] for r in expected if r["mode"] == "forced"})
    summary = statistics.summarize(rows, fixed)
    if len(fixed) != 7 or summary["assigned_episodes"] != 48 or not summary["complete"]:
        raise ValueError("complete six-context script/seven-schedule matrix required")
    return dict(
        schema="motion2scene_development_comparator_summary_v1",
        study=artifact(panel / "study.json"),
        completed_comparators=completion_ref,
        implementation=[artifact(Path(__file__)), artifact(Path(statistics.__file__))],
        rows=rows,
        summary=summary,
        physical_event_counts={
            policy: {
                key: dict(Counter(r[key] for r in rows if r["policy_id"] == policy))
                for key in ("undesired_environment_contact", "fall_or_upright_threshold_failure")
            }
            for policy in sorted(summary["policies"])
        },
        measured_physics_steps=sum(r["recorded_physics_steps"] for r in rows),
        measured_physical_executions=len(rows),
        new_physics_steps_from_analysis=0,
        complete_M8_panel=False,
        learned_M8_policy_results=0,
        contact_definition=(
            "native per-body external counterpart normal force above 1 N; foot/floor support allowed; "
            "self contacts retained separately; partial observed contact establishes failure"
        ),
        fall_definition=(
            "recorded root height below 0.5 m or upright cosine below 0.5; "
            "absence requires complete admitted first-episode measurement"
        ),
        time_definition=(
            "passage_time_s is all-body-origin crossing plus 0.30 s stabilization; "
            "paired only on mutually successful identical conditions. Whole episode capture duration "
            "is separate and is not an adaptation cost or independently measured recovery latency."
        ),
        inference_scope=(
            "six reused development contexts, one execution seed, eight fixed comparators; "
            "descriptive matched results, no learned-policy or reserved generalization result"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = read_completed(args.panel)
    args.out.mkdir(parents=True, exist_ok=False)
    print(json.dumps(write_new(args.out / "result.json", result)))


if __name__ == "__main__":
    main()
