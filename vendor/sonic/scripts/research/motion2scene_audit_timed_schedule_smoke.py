#!/usr/bin/env python3
"""Reconstruct every sensor feature and audit one registered late-entry smoke."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from motion2scene_collect_timed_schedules import analyze_cell, verify_manifest  # noqa: E402
from motion2scene_timing_diagnostic import artifact, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_integer_prefix import (  # noqa: E402
    paired_prefix_on_ticks,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (  # noqa: E402
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    checked_artifact,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    HISTORY_FRAMES,
    HISTORY_SECONDS,
    schedule_history_features,
)


def replay_sensor_features(bank, observations, names):
    """Reconstruct policy inputs from causally delivered rays and robot state only."""
    memory = FloorCeilingHistory(HistoryGrid(max_age_s=HISTORY_SECONDS, max_frames=HISTORY_FRAMES))
    errors = []
    for row in observations:
        captured = row["capture_elapsed_s"]
        delivered = row["delivered_capture_elapsed_s"]
        if delivered != captured or row["observation_age_s"] != 0:
            raise ValueError("this registered smoke requires zero sensor delivery delay")
        memory.push(
            [SensorRay(**ray) for ray in row["measurements"]], captured, delivered_time_s=captured
        )
        snapshot = memory.snapshot(row["root_pos_w"], row["root_quat_w"], captured)
        active = bank.option_ids.index(row["active_before"])
        actual_names, actual = schedule_history_features(
            snapshot,
            memory.grid,
            row["state"],
            row["tick"],
            active,
            np.asarray(row["legal_mask"], dtype=bool),
            row["observation_age_s"],
        )
        if actual_names != tuple(names):
            raise ValueError("raw sensor replay produced different named features")
        recorded = np.asarray(row["features"])
        errors.append(float(np.max(np.abs(actual - recorded))))
    return dict(
        rows=len(errors),
        exact_every_row=all(value == 0 for value in errors),
        maximum_absolute_error=max(errors),
        history_seconds=HISTORY_SECONDS,
        history_frames=HISTORY_FRAMES,
        scope="raw causal rays/robot state/legality only; no beam parameters",
    )


def analyze(out):
    manifest, bank, scene = verify_manifest(out, execution=False)
    if len(manifest["cells"]) != 1 or manifest["cells"][0]["timed_schedule_mode"] != "forced":
        raise ValueError("one registered forced branch required for this smoke audit")
    cell = manifest["cells"][0]
    option_id = cell["forced_option_id"]
    option = next(item for item in bank.request["options"] if item["option_id"] == option_id)
    row, payload, interface = analyze_cell(cell, manifest, bank, scene)
    observations = interface["observations"]
    replay = replay_sensor_features(bank, observations, interface["feature_names"])
    qualification = json.loads(checked_artifact(bank.definition["evidence"][option_id]).read_text())
    original = load_reset_capture(checked_artifact(qualification["artifacts"]["trajectory"]))
    prefix = paired_prefix_on_ticks(original, payload, option["entry_tick"] / 50)
    earlier = [item for item in observations if item["tick"] < option["entry_tick"]]
    all_neutral = all(
        item["active"] == "neutral" and item["selected_option_id"] == "neutral" for item in earlier
    )
    switches = [(item["tick"], item["from"], item["to"]) for item in interface["switches"]]
    expected = [
        (option["entry_tick"], "neutral", option_id),
        (option["return_tick"], option_id, "neutral"),
    ]
    valid = bool(
        row["measurement_admitted"]
        and row["schedule_audit"]["valid"]
        and replay["exact_every_row"]
        and prefix["exact_match"]
        and all_neutral
        and switches == expected
    )
    result = dict(
        schema="motion2scene_timed_schedule_runtime_smoke_audit_v1",
        manifest=artifact(out / "manifest.json"),
        implementation=artifact(Path(__file__)),
        chosen_option_id=option_id,
        runtime_smoke_admitted=valid,
        physical_pass=row["pass"],
        measured_passage_time_s=row["costs"]["passage_time_s"],
        full_sensor_feature_replay=replay,
        integer_tick_qualification_prefix=prefix,
        prefix_comparison_scope=(
            "qualified empty branch versus this development beam before entry; "
            "diagnostic exact state/action/token equality, not a teacher label"
        ),
        earlier_opportunities_held_neutral=all_neutral,
        actual_switches=switches,
        expected_switches=expected,
        sensor_alignment=row["sensor_alignment"],
        schedule_audit=row["schedule_audit"],
        contact_audit=row["contact_audit"],
        observation_timing=row["observation_timing"],
        phase_observation_receipt=row["phase_observation_receipt"],
        physics_steps=row["physics_steps"],
        sensor_queries=row["sensor_queries"],
        artifacts={
            key: row[key]
            for key in (
                "trajectory",
                "sensor",
                "features",
                "all_body_contacts",
                "environment_pairs",
                "environment_mapping",
                "physics_beam_contacts",
            )
        },
        scope="one actual late forced schedule; no learned-policy or course generalization claim",
    )
    write_new(out / "runtime_smoke_audit.json", result)
    print(
        json.dumps(
            dict(
                runtime_smoke_admitted=valid,
                physical_pass=row["pass"],
                feature_replay=replay,
                switches=switches,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args().out)
