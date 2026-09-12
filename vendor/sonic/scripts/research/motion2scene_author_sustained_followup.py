"""One preregistered .30-window followup after the .48-window return rejection."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_author_timed_profiles import author  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    HEIGHT_MEASUREMENT,
    REQUEST_SCHEMA,
    validate_request,
)


def register(out, previous):
    prior = json.loads((previous / "registration.json").read_text())
    previous_result = json.loads((previous / "result.json").read_text())
    rejected = next(row for row in previous_result["rows"] if row["label"] == "sustained")
    if rejected["later_return_guard_candidates"]:
        raise ValueError("followup premise requires the retained .48-window return rejection")
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    prior.update(
        schema="motion2scene_authored_sustained_followup_v1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        profiles=[{"label": "sustained", "window_half_width": 0.30}],
        implementation=sources,
        previous_proposals=artifact(previous / "result.json"),
        selection={
            "rule": "earliest native-guard-valid tick in the registered finite search interval",
            "minimum_return_tick": 250,
            "minimum_after_maintenance_ticks": 10,
            "minimum_recovery_ticks": 15,
            "last_recorded_physical_reference_tick": 297,
            "maximum_return_tick": 282,
            "entry_tick": 15,
            "maintenance_interval": "ceil(first core source time*50) through floor(last core source time*50)",
            "basis": "reference compatibility only; no physical outcomes or model training",
        },
        scope="one fresh authored .30-window proposal; preserve previous .48 candidate and every tick audit",
    )
    write_new(out / "registration.json", prior)
    print(json.dumps({"registration": artifact(out / "registration.json")}))


def run(out):
    author(out)
    reg = json.loads((out / "registration.json").read_text())
    previous = reg["previous_proposals"]
    previous_result = json.loads(checked(Path(previous["path"]), previous["sha256"]).read_text())
    short = next(row for row in previous_result["rows"] if row["label"] == "short")
    row = json.loads((out / "result.json").read_text())["rows"][0]
    guard = np.load(row["reference_guard"]["path"])
    maintenance_start = int(np.ceil(row["core_source_interval_s"][0] * 50))
    maintenance_end = int(np.floor(row["core_source_interval_s"][1] * 50))
    selection = reg["selection"]
    first = max(
        selection["minimum_return_tick"],
        maintenance_end + selection["minimum_after_maintenance_ticks"],
    )
    last = selection["maximum_return_tick"]
    candidates = [
        {
            "tick": tick,
            "time_s": tick / 50,
            "joint_jump_rad": float(guard["joint_jump_rad"][tick]),
            "root_jump_m": float(guard["root_jump_m"][tick]),
            "passes_reference_guard": bool(guard["eligible"][tick]),
            "physical_recovery_ticks_available": selection["last_recorded_physical_reference_tick"]
            - tick,
        }
        for tick in range(first, last + 1)
    ]
    eligible = [candidate for candidate in candidates if candidate["passes_reference_guard"]]
    chosen = eligible[0] if eligible else None
    receipt = {
        "registration": artifact(out / "registration.json"),
        "authored_result": artifact(out / "result.json"),
        "maintenance_ticks": [maintenance_start, maintenance_end],
        "audited_ticks": candidates,
        "selected_return": chosen,
        "selection_used_physical_outcomes": False,
        "physical_qualification": None,
        "physics_steps": 0,
    }
    write_new(out / "return_tick_audit.json", receipt)
    if chosen is None or not row["proposed_schedule_reference_checks"][0]["passes_reference_guard"]:
        print(
            json.dumps(
                {
                    "status": "no_eligible_complete_reference_schedule",
                    "receipt": artifact(out / "return_tick_audit.json"),
                }
            )
        )
        return
    controller = json.loads(Path(reg["neutral_qualification"]["path"]).read_text())["manifest"]
    manifest = json.loads(checked(Path(controller["path"]), controller["sha256"]).read_text())
    # The existing frozen controller is already hash-bound in the physical manifest.
    controller_refs = []

    def find_checkpoint(value):
        if isinstance(value, dict):
            if (
                "path" in value
                and "sha256" in value
                and str(value["path"]).endswith("/sonic_release/last.pt")
            ):
                controller_refs.append({"path": value["path"], "sha256": value["sha256"]})
            for item in value.values():
                find_checkpoint(item)
        elif isinstance(value, list):
            for item in value:
                find_checkpoint(item)

    find_checkpoint(manifest)
    if not controller_refs or any(ref != controller_refs[0] for ref in controller_refs):
        raise ValueError("could not bind one frozen physical controller")
    references = [
        {
            "reference_id": "neutral",
            "motion": reg["neutral"],
            "expected_loaded_frames": 299,
            "construction": "generated",
        }
    ]
    for reference_id, item in (("short", short), ("sustained", row)):
        references.append(
            {
                "reference_id": reference_id,
                "motion": item["motion"],
                "expected_loaded_frames": 299,
                "construction": "authored_local_crouch",
                "parent_motion": reg["neutral"],
            }
        )
    options = []
    for reference_id, item, return_tick in (
        ("short", short, 265),
        ("sustained", row, chosen["tick"]),
    ):
        options.append(
            {
                "option_id": f"{reference_id}_e015_r{return_tick:03d}",
                "reference_id": reference_id,
                "profile_label": reference_id,
                "entry_tick": 15,
                "return_tick": return_tick,
                "recovery_end_tick": 297,
                "maintenance": {
                    "start_tick": int(np.ceil(item["core_source_interval_s"][0] * 50)),
                    "end_tick": int(np.floor(item["core_source_interval_s"][1] * 50)),
                    "maximum_body_height_m": None,
                },
            }
        )
    request = {
        "schema": REQUEST_SCHEMA,
        "split": "development",
        "request_id": "six_second_authored_duration_v1",
        "controller": controller_refs[0],
        "reference_fps": 50,
        "expected_loaded_frames": 299,
        "max_entries_per_episode": 1,
        "height_measurement": HEIGHT_MEASUREMENT,
        "support_plane_z_m": 0.0,
        "references": references,
        "options": options,
        "authoring_registration": artifact(out / "registration.json"),
        "return_tick_audit": artifact(out / "return_tick_audit.json"),
        "minimum_recovery_ticks": 15,
        "height_scope": (
            "measure full executed outer height; no unmeasured clearance bound asserted before first qualification"
        ),
        "status": "experimental forced qualification request; not an online registry",
    }
    validate_request(request)
    write_new(out / "qualification_request.json", request)
    print(
        json.dumps(
            {"request": artifact(out / "qualification_request.json"), "selected_return": chosen}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "author"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.stage == "register":
        register(args.out, args.previous)
    else:
        run(args.out)
