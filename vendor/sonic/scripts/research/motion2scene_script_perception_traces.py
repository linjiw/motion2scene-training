#!/usr/bin/env python3
"""Extract delivered features and actual choices from the six completed script runs.

Recomputing the frozen script checks recorded readouts. It is not another
physical execution or a WAIT/one-time-choice performance ablation.
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination import (  # noqa: E402
    motion2scene_schedule_script as script,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)

EXAMPLES = ("early_prior_development_0284", "complementary_late_development_0100")


def trace(interface, bank, settings):
    """Only actual neutral decision phases can be represented as WAIT/COMMIT."""
    observations = interface["observations"]
    names = interface["feature_names"]
    phases = interface["phase_ticks"]
    if [o["tick"] for o in observations] != list(range(1, len(observations) + 1)):
        raise ValueError("consecutive recorded control ticks required")
    if interface["timed_schedule_mode"] != "scripted_multi":
        raise ValueError("these examples must come from actual frozen script executions")
    decisions = []
    for observation in observations:
        if observation["tick"] not in phases or observation["active_before"] != "neutral":
            continue
        legal = np.asarray(observation["legal_mask"], dtype=bool)
        if legal.sum() <= 1:
            continue
        capture = observation["capture_elapsed_s"]
        delivered = observation["delivered_capture_elapsed_s"]
        if (
            not all(math.isfinite(x) for x in (capture, delivered, observation["phase_s"]))
            or delivered > capture + 1e-8
            or capture > observation["phase_s"] + 1e-8
        ):
            raise ValueError("only packets captured by the preaction observation are causal")
        selected, evidence = script.choose_sensor_schedule(
            names, observation["features"], legal, observation["tick"], bank, settings
        )
        if selected != observation["selected_option_id"]:
            raise ValueError("recomputed frozen script differs from the recorded command")
        if not legal[bank.option_ids.index(selected)]:
            raise ValueError("the recorded selection must be legal before commitment")
        values = dict(zip(names, observation["features"], strict=True))
        decisions.append(
            dict(
                tick=observation["tick"],
                registered_phase_s=observation["phase_s"],
                preaction_capture_elapsed_s=capture,
                delivered_packet_capture_elapsed_s=delivered,
                selected_option_id=selected,
                decision=(
                    "COMMIT"
                    if selected != "neutral"
                    else "WAIT" if observation["tick"] < max(phases) else "CONTINUE_NEUTRAL"
                ),
                later_registered_entry_ticks=[p for p in phases if p > observation["tick"]],
                legal_option_ids=[
                    s for s, allowed in zip(bank.option_ids, legal, strict=True) if allowed
                ],
                script_readout=evidence,
                corridor_features={k: v for k, v in values.items() if k.startswith("corridor_")},
                root_pos_w=observation["root_pos_w"],
            )
        )
    committed = [d for d in decisions if d["decision"] == "COMMIT"]
    if len(committed) > 1:
        raise ValueError("the supported schedule has at most one adaptation commitment")
    first = decisions[0] if decisions else None
    commit = committed[0] if committed else None
    displacement = None
    if first is not None and commit is not None:
        displacement = math.dist(first["root_pos_w"][:2], commit["root_pos_w"][:2])
        switches = [s for s in interface["switches"] if s["from"] == "neutral"]
        if (
            len(switches) != 1
            or switches[0]["tick"] != commit["tick"]
            or switches[0]["to"] != commit["selected_option_id"]
        ):
            raise ValueError("recorded commitment and executed switch must agree")
    last_tick = commit["tick"] if commit else max(phases)
    series = []
    for o in observations:
        if o["tick"] > last_tick:
            break
        values = dict(zip(names, o["features"], strict=True))
        series.append(
            dict(
                tick=o["tick"],
                registered_phase_s=o["phase_s"],
                preaction_capture_elapsed_s=o["capture_elapsed_s"],
                upper_hit_by_band=[
                    values[f"corridor_{a:g}_{b:g}_upper_hit"] for a, b in script.BANDS
                ],
                unknown_fraction_by_band=[
                    values[f"corridor_{a:g}_{b:g}_unknown_fraction"] for a, b in script.BANDS
                ],
                root_xy_w=o["root_pos_w"][:2],
                active_before=o["active_before"],
            )
        )
    return dict(
        decisions=decisions,
        precommit_series=series,
        commitment=commit,
        net_planar_displacement_first_decision_to_commit_m=displacement,
        actual_switches=interface["switches"],
        recomputed_decisions_match_recording=True,
    )


def extract(summary_path, out):
    summary_ref = artifact(summary_path)
    summary = read_checked(summary_ref)
    if summary["measured_physical_executions"] != 48 or not summary["summary"]["complete"]:
        raise ValueError("complete original comparator block required")
    study = read_checked(summary["study"])
    bank = load_verified_registry(study["registry"]["path"], study["registry"]["sha256"])
    settings = read_checked(study["script"])
    selected = [r for r in summary["rows"] if r["policy_id"] == "script"]
    if len(selected) != 6 or len({r["scene_id"] for r in selected}) != 6:
        raise ValueError("all six assigned script executions must remain in the extraction")
    traces = []
    for assignment in selected:
        native_result = read_checked(assignment["source_result"])
        native = native_result["rows"][0]
        interface = read_checked(native["sensor"])
        if (
            interface["script_parameters_sha256"] != study["script"]["sha256"]
            or interface["timed_registry_sha256"] != study["registry"]["sha256"]
        ):
            raise ValueError("actual script or schedule registry differs from the frozen panel")
        checked(Path(native["trajectory"]["path"]), native["trajectory"]["sha256"])
        traces.append(
            dict(
                scene_id=assignment["scene_id"],
                source_result=assignment["source_result"],
                sensor=native["sensor"],
                trajectory=native["trajectory"],
                physical_outcome=native["outcome"]["task_outcome"],
                physical_passage_time_s=native["costs"]["passage_time_s"],
                observation_timing_audit=native["observation_timing"],
                **trace(interface, bank, settings),
            )
        )
    output = dict(
        schema="motion2scene_recorded_script_perception_trace_v1",
        source_comparator_summary=summary_ref,
        implementation=[artifact(Path(__file__)), artifact(Path(script.__file__))],
        registry=study["registry"],
        script_parameters=study["script"],
        traces=traces,
        displayed_examples=list(EXAMPLES),
        example_scope=(
            "the previously inspected prior-only/sustained-only development pair; all six traces retained"
        ),
        new_physical_executions=0,
        new_physics_steps=0,
        new_fits=0,
        limitations=(
            "recorded script examples, not learned-policy or reserved evidence. Upper-hit absence "
            "does not establish free space; unknown fractions are retained. Surface visibility alone "
            "does not prove alias resolution. These traces do not isolate additional information "
            "from availability of later schedules; the same-repertoire one-time control remains separate."
        ),
        clock_definition=(
            "registered phase labels commands; preaction capture and delivered packet capture "
            "timestamps are retained separately, not silently aligned as simultaneous measurements"
        ),
    )
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "result.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = extract(args.summary, args.out)
    print(json.dumps(dict(traces=len(output["traces"]), new_physical_executions=0)))
