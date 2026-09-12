#!/usr/bin/env python3
"""Replay the two declared development integration modes from actual measurements."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_audit_timed_schedule_smoke import replay_sensor_features  # noqa: E402
from motion2scene_collect_timed_schedules import analyze_cell, verify_manifest  # noqa: E402
from motion2scene_timing_diagnostic import artifact, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_integer_prefix import (  # noqa: E402
    paired_prefix_on_ticks,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    checked_artifact,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    choose_schedule_option,
    load_script_parameters,
)


def audit(out):
    manifest, bank, scene = verify_manifest(out, execution=False)
    if scene["split"] != "development" or len(manifest["cells"]) != 1:
        raise ValueError("one explicitly registered development smoke required")
    cell = manifest["cells"][0]
    if cell["timed_schedule_mode"] not in ("forced", "scripted_multi"):
        raise ValueError("this audit supports the two declared integration modes only")
    row, payload, interface = analyze_cell(cell, manifest, bank, scene)
    observations = interface["observations"]
    replay = replay_sensor_features(bank, observations, interface["feature_names"])
    chosen = row["schedule_audit"]["chosen_option_id"]
    option = next((item for item in bank.request["options"] if item["option_id"] == chosen), None)
    qualification = json.loads(checked_artifact(bank.definition["evidence"][chosen]).read_text())
    original = load_reset_capture(checked_artifact(qualification["artifacts"]["trajectory"]))
    prefix_tick = option["entry_tick"] if option else min(manifest["phase_ticks"])
    prefix = paired_prefix_on_ticks(original, payload, prefix_tick / 50)
    script_replay = None
    if cell["timed_schedule_mode"] == "scripted_multi":
        ref = manifest["script_parameters"]
        settings = load_script_parameters(ref["path"], ref["sha256"], bank)
        mismatches = []
        for index, packet in enumerate(observations):
            active = bank.option_ids.index(packet["active_before"])
            current = next(
                (o for o in bank.request["options"] if o["option_id"] == packet["active_before"]),
                None,
            )
            mandatory = (
                0 if current is not None and packet["tick"] == current["return_tick"] else None
            )
            selected, values = choose_schedule_option(
                "scripted_multi",
                interface["feature_names"],
                packet["features"],
                np.asarray(packet["legal_mask"], dtype=bool),
                active,
                packet["tick"],
                mandatory,
                bank,
                script_parameters=settings,
            )
            if selected != packet["selected_option_id"] or values != packet["policy_values"]:
                mismatches.append(index)
        script_replay = dict(
            rows=len(observations),
            mismatched_indices=mismatches,
            exact_every_row=not mismatches,
            parameters=ref,
        )
    valid = bool(
        row["measurement_admitted"]
        and row["schedule_audit"]["valid"]
        and replay["exact_every_row"]
        and prefix["exact_match"]
        and (script_replay is None or script_replay["exact_every_row"])
    )
    result = dict(
        schema="motion2scene_schedule_integration_audit_v1",
        manifest=artifact(out / "manifest.json"),
        actual_result=artifact(out / "result.json"),
        actual_beam_count=len(scene["beams"]),
        runtime_smoke_admitted=valid,
        physical_pass=row["pass"],
        chosen_option_id=chosen,
        costs=row["costs"],
        outcome=row["outcome"],
        feature_replay=replay,
        script_readout_replay=script_replay,
        qualification_prefix=prefix,
        prefix_tick=prefix_tick,
        phase_decisions=[
            {
                key: p[key]
                for key in (
                    "tick",
                    "active_before",
                    "active",
                    "selected_option_id",
                    "legal_mask",
                    "policy_values",
                )
            }
            for p in observations
            if p["tick"] in manifest["phase_ticks"]
        ],
        switches=interface["switches"],
        schedule_audit=row["schedule_audit"],
        passage=row["passage"],
        imported_geometry_audit=row["imported_geometry_audit"],
        contact_audit=row["contact_audit"],
        beam_contact_synchronization=row["beam_contact_synchronization"],
        observation_timing=row["observation_timing"],
        physics_steps=row["physics_steps"],
        sensor_queries=row["sensor_queries"],
        scope="Actual development integration only; no learned generalization or repeated adaptation claim",
    )
    sources = []
    for path in sorted(closure([Path(__file__)])):
        target = out / "integration_audit_source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError("immutable audit source differs")
        if not target.exists():
            shutil.copyfile(path, target)
        sources.append(dict(**artifact(path), snapshot=artifact(target)))
    result["analysis_implementation"] = sources
    write_new(out / "integration_audit.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "runtime_smoke_admitted",
                    "physical_pass",
                    "chosen_option_id",
                    "phase_decisions",
                    "actual_beam_count",
                    "costs",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args().out)
