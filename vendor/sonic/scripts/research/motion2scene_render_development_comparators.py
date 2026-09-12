#!/usr/bin/env python3
"""Render the complete measured comparator table without fitting or simulation."""

import argparse
import hashlib
import json
from pathlib import Path


def render(path):
    source = json.loads(path.read_text())
    summary = source["summary"]
    if source["measured_physical_executions"] != 48 or not summary["complete"]:
        raise ValueError("completed original comparator block required")
    policies = summary["policies"]
    script_passes = policies["script"]["pass"]
    contexts = policies["script"]["assigned"]
    best_fixed = max(p["pass"] for name, p in policies.items() if name != "script")
    counts = summary["outcome_counts"]
    partial_steps = [
        r["recorded_physics_steps"] for r in source["rows"] if not r["measurement_admitted"]
    ]
    prior_pair = policies["fixed_prior_splice_e015_r265"]["paired_with_reference"]
    lines = [
        "# M8 development comparator block",
        "",
        f"The strong script passes {script_passes}/{contexts} assigned development contexts. "
        f"The seven-schedule bank covers {summary['capability_count_lower']}/{contexts}; "
        f"the best fixed schedules cover {best_fixed}/{contexts}. These are actual "
        "matched executions with seed 8732, not learned M8 policy results.",
        "",
        f"All 48 assignments completed: {counts['pass']} passes, {counts['failure']} failures and "
        f"**{source['measured_physics_steps']:,} measured physics steps**. "
        f"Partial captures have recorded step counts {partial_steps}; "
        "observed contact establishes task failure in this block, "
        "while its unobserved later fall outcome remains unknown. No task outcome is unknown.",
        "",
        "| Comparator | Passage | Contact observed | Fall threshold observed / unknown | "
        "Time minus script (s) | Mutually successful contexts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["script"] + sorted(p for p in summary["policies"] if p != "script"):
        policy = summary["policies"][name]
        events = source["physical_event_counts"][name]
        fall = events["fall_or_upright_threshold_failure"]
        pair = policy["paired_with_reference"]
        difference = pair["mean_time_difference_s"]
        time = "unavailable" if difference is None else f"{difference:+.3f}"
        lines.append(
            f"| {name.removeprefix('fixed_')} | {policy['pass']}/{policy['assigned']} | "
            f"{events['undesired_environment_contact'].get('observed', 0)} | "
            f"{fall.get('observed', 0)} / {fall.get('unknown', 0)} | {time} | "
            f"{pair['mutually_successful_contexts']} |"
        )
    lines += [
        "",
        f"The early prior-splice schedule's time minus script is "
        f"{prior_pair['mean_time_difference_s']:+.3f} s on "
        f"{prior_pair['mutually_successful_contexts']} mutually successful contexts. "
        "Its failures remain in the passage denominator. This is a passage/time comparison "
        "on development data. It does not establish which learner or constructor is better.",
        "",
        "| Development context | Measured passing bank schedules | Script choice | Script passage time (s) |",
        "|---|---|---|---:|",
    ]
    script = {r["scene_id"]: r for r in source["rows"] if r["policy_id"] == "script"}
    for condition in summary["capability"]:
        scene = condition["scene_id"]
        passing = ", ".join(
            p.removeprefix("fixed_") for p in condition["measured_passing_schedules"]
        )
        choice = script[scene]
        lines.append(
            f"| {scene} | {passing} | {choice['chosen_option_id']} | {choice['time_s']:.2f} |"
        )
    lines += [
        "",
        source["contact_definition"] + ". " + source["fall_definition"] + ".",
        "",
        source["time_definition"],
        "",
        "Complete admitted captures last 5.96 s. That recording duration is separate "
        "from crossing plus stabilization and is not a measured energy or recovery cost. "
        "The six layouts are reused across comparators; these 48 episodes are not 48 "
        "independent layouts. No population confidence interval or superiority claim is made.",
        "",
        "The first reader conservatively marked 39 complete single-beam fall outcomes "
        "unknown because it looked only for the course-specific whole-episode field. "
        "The corrected reader uses the existing single-beam field. Both outputs and the "
        "original reader bytes are preserved; passage, time, contact, capability and raw "
        "measurements are unchanged. Acquisition separately stopped at a preflight guard "
        "because an unused offline selector was added inside the frozen editable package. "
        "Moving that helper outside the runtime package restored the exact original identity; "
        "the next teacher had no attempt before this recovery. No physical outcome was replaced.",
        "",
        f"Source: `{path.resolve()}`; "
        f"SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`.",
        "",
        "Reproduce this table with `scripts/research/motion2scene_render_development_comparators.py "
        "--result <result.json> --out <new-report.md>`. The readout and renderer add no physics.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = render(args.result)
    with args.out.open("x") as handle:
        handle.write(text)
