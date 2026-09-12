#!/usr/bin/env python3
"""Export one recorded multi-beam development demonstration; no new simulation."""

import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    audit_imported_beams,
    beam_prim_name,
    score_course,
    synchronize_course_forces,
    validate_course_definition,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (  # noqa: E402
    audit_sensor_alignment,
)
from scripts.research.motion2scene_export_option_dataset import (  # noqa: E402
    FORBIDDEN_INPUT_KEYS,
    STUDENT_HISTORY_KEYS,
    checked,
    copy_npz,
    digest,
    executed_options,
    export_student_inputs,
    portable_path,
    save_pickle_free,
    sensor_packets,
    write_json,
)

SCHEMA = "motion2scene_one_adaptation_course_demonstration_v1"


def course_channels(contacts, definition, *, control_frames, physics_steps):
    """Require every named beam stream in its authored order; retain body channels."""
    count = len(definition["beams"])
    force = np.asarray(contacts["physics_force_w"])
    if (
        contacts["beam_names"].tolist() != [beam_prim_name(i) for i in range(count)]
        or force.ndim != 4
        or force.shape[1] != count
        or force.shape[2] != len(contacts["filter_paths"])
        or len(contacts["control_steps"]) != control_frames
        or len(contacts["physics_steps"]) != physics_steps
    ):
        raise ValueError("missing, reordered, or misaligned multi-beam contact channels")
    return synchronize_course_forces(contacts)


def check_sensor_privacy(value):
    if isinstance(value, dict):
        if set(value) & FORBIDDEN_INPUT_KEYS:
            raise ValueError("privileged identity, outcome or action in sensor measurements")
        for child in value.values():
            check_sensor_privacy(child)
    elif isinstance(value, list):
        for child in value:
            check_sensor_privacy(child)


def audit(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["episodes"] != 1:
        raise ValueError("requires one explicitly scoped course demonstration")
    for relative, expected in manifest["files"].items():
        path = portable_path(out, relative)
        if digest(path) != expected:
            raise ValueError(f"package hash mismatch: {relative}")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                if any(arrays[k].dtype.kind not in "biufcUS" for k in arrays.files):
                    raise ValueError("nonportable course array")
    row = json.loads((out / "episode.json").read_text())
    definition = json.loads((out / "course_definition.json").read_text())
    validate_course_definition(definition)
    capture = json.loads((out / "course_capture.json").read_text())
    if capture["definition"] != definition or capture["reference_exhaustion_events"]:
        raise ValueError("recorded course definition or reference-exhaustion disagreement")
    with np.load(out / "trajectory.npz", allow_pickle=False) as arrays:
        payload = {k: arrays[k] for k in arrays.files}
    if (
        audit_imported_beams(capture["imported_beams"], definition["beams"])
        != capture["import_geometry_audit"]
    ):
        raise ValueError("course imported beam geometry audit does not reproduce")
    payload["fps"] = json.loads((out / "trajectory_metadata.json").read_text())["fields"]["fps"]
    observations = json.loads((out / "sensor_history.json").read_text())
    check_sensor_privacy(observations)
    alignment = audit_sensor_alignment(
        payload, observations, reference_frames=definition["reference_budget"]["frames"]
    )
    if alignment != json.loads((out / "sensor_alignment.json").read_text()):
        raise ValueError("course sensor alignment does not reproduce")
    with np.load(out / "sensor_alignment.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["packet_eligible"], alignment["packet_eligible"]):
            raise ValueError("course eligibility mask disagrees")
    with np.load(out / "student_history.npz", allow_pickle=False) as arrays:
        if (
            set(arrays.files) - STUDENT_HISTORY_KEYS
            or arrays["features"].shape
            != (len(observations), row["student_input"]["feature_dimension"])
            or arrays["option_ids"].tolist() != row["registered_option_ids"]
            or not np.array_equal(arrays["phase_s"], [o["time_s"] for o in observations])
        ):
            raise ValueError("course student schema, identity or clock disagreement")
    commands = json.loads((out / "commands.json").read_text())
    actual = executed_options(
        row["registered_option_ids"],
        commands["records"],
        commands["switches"],
        alignment["physical_first_episode_frames"],
    )
    if actual != row["executed_option"]:
        raise ValueError("course executed option differs from actual switches")
    with np.load(out / "course_contacts.npz", allow_pickle=False) as arrays:
        forces, sync = course_channels(
            arrays,
            definition,
            control_frames=len(observations),
            physics_steps=manifest["physics_steps"],
        )
    scored = score_course(
        payload,
        forces,
        definition["beams"],
        commands_valid=row["recorded_result"]["commands_valid"],
        timeout_s=definition["timeout_s"],
        reference_frames=definition["reference_budget"]["frames"],
        final_reference_phase_s=observations[-1]["time_s"],
    )
    for key, value in scored.items():
        if row["recorded_result"][key] != value:
            raise ValueError(f"portable course scoring differs: {key}")
    if sync != row["recorded_result"]["all_beam_contact_synchronization_error"]:
        raise ValueError("portable course contact synchronization differs")
    return {
        "episodes": 1,
        "beams": len(definition["beams"]),
        "pass": scored["pass"],
        "physical_rows": len(observations),
        "eligible_sensor_packets": sum(alignment["packet_eligible"]),
        "physics_steps": manifest["physics_steps"],
        "files_checked": len(manifest["files"]),
    }


def export(result_path, out):
    result = json.loads(result_path.read_text())
    if result["schema"] != "motion2scene_development_course_result_v1" or len(result["rows"]) != 1:
        raise ValueError("requires exactly one recorded development course episode")
    row = result["rows"][0]
    manifest_path = checked(result["manifest"])
    manifest = json.loads(manifest_path.read_text())
    matches = [c for c in manifest["cells"] if c["cell_id"] == row["cell_id"]]
    if len(matches) != 1:
        raise ValueError("course row lacks unique registered cell")
    cell = matches[0]
    if json.loads((Path(cell["output"]) / "course_row.json").read_text()) != row:
        raise ValueError("course result differs from recorded branch row")
    registry = json.loads(checked(manifest["registry"]).read_text())
    definition = json.loads(checked(row["course_definition"]).read_text())
    validate_course_definition(definition)
    option_ids = manifest["option_ids"]
    sensor = json.loads(checked(row["sensor"]).read_text())
    if (
        definition["registry"] != manifest["registry"]
        or row["course_definition"] != manifest["course_definition"]
        or row["course_definition"] != cell["course_definition"]
        or definition["source"] != row["source"]
        or row["source"] != cell["generation_seed"]
        or registry["source"] != row["source"]
        or definition["neutral_motion"] != registry["references"][0]["motion"]
        or any(
            cell["motion"][key] != definition["neutral_motion"][key] for key in ("path", "sha256")
        )
        or definition["beams"] != row["beams"]
        or row["physics_seed"] != cell["runtime_seed"]
        or row["mode"] != cell["multi_option_mode"]
        or row["mode"] != sensor["mode"]
        or row["option_index"] != cell["option_index"]
        or row["decision_time_s"] != cell["decision_time_s"]
        or row["switches"] != sensor["switches"]
        or sensor["option_names"] != option_ids
        or sensor["option_registry_sha256"] != manifest["registry"]["sha256"]
        or [reference["name"] for reference in registry["references"]] != option_ids
        or manifest["implementation"]["checkpoint"] != registry["controller"]
    ):
        raise ValueError("course source, registry, mode or geometry provenance disagreement")
    paths = {
        key: checked(row[key])
        for key in ("trajectory", "sensor", "features", "course_contacts", "course_capture", "bank")
    }
    expected_folder = Path(cell["output"]) / "trajectories"
    if any(p.parent.resolve() != expected_folder.resolve() for p in paths.values()):
        raise ValueError("course artifacts do not belong to registered episode")
    with np.load(paths["bank"], allow_pickle=False) as bank:
        if bank["root_xyz"].shape[:2] != (
            len(option_ids),
            definition["reference_budget"]["frames"],
        ):
            raise ValueError("loaded bank differs from course reference budget")
    payload = load_reset_capture(paths["trajectory"])
    alignment = audit_sensor_alignment(
        payload, sensor["observations"], reference_frames=definition["reference_budget"]["frames"]
    )
    if alignment != row["sensor_alignment"]:
        raise ValueError("recorded course sensor chronology does not reproduce")
    out.mkdir(parents=True, exist_ok=False)
    meta, excluded = save_pickle_free(out / "trajectory.npz", payload)
    if excluded:
        raise ValueError(f"unexported physical fields: {excluded}")
    write_json(
        out / "trajectory_metadata.json", {"fields": meta, "excluded_nonportable_fields": excluded}
    )
    write_json(out / "sensor_history.json", sensor_packets(sensor))
    write_json(out / "sensor_alignment.json", alignment)
    np.savez_compressed(
        out / "sensor_alignment.npz",
        packet_eligible=np.asarray(alignment["packet_eligible"], dtype=bool),
        packet_confidence=np.asarray([p["confidence"] for p in alignment["packets"]]),
        sensor_phase_s=np.asarray([p["time_s"] for p in sensor["observations"]]),
        trajectory_phase_s=np.asarray(payload["motion_time_s"]),
    )
    provenance = {
        "result": {"path": str(result_path), "sha256": digest(result_path)},
        "manifest": result["manifest"],
        "registry": manifest["registry"],
        "recorded_artifacts": {k: row[k] for k in paths},
        "asset_scope": "hash provenance only; inherited weights and reference bank are not packaged",
    }
    student = export_student_inputs(out, row, sensor, provenance)
    records = [
        {
            k: o[k]
            for k in (
                "time_s",
                "capture_elapsed_s",
                "active_before",
                "active",
                "legal_mask",
                "transition",
                "policy_logits",
            )
            if k in o
        }
        for o in sensor["observations"]
    ]
    commands = {"records": records, "switches": sensor["switches"], "mode": sensor["mode"]}
    write_json(out / "commands.json", commands)
    actual = executed_options(
        option_ids, records, commands["switches"], row["first_episode_frames"]
    )
    if (
        actual["option_index"] != row["executed_option_index"]
        or actual["first_episode_option_sequence"] != row["executed_option_sequence"]
    ):
        raise ValueError("course actual switches differ from recorded executed action")
    copy_npz(paths["course_contacts"], out / "course_contacts.npz")
    for name, source in (
        ("course_definition.json", checked(row["course_definition"])),
        ("course_capture.json", paths["course_capture"]),
        ("original_result.json", result_path),
        ("original_manifest.json", manifest_path),
        ("option_registry.json", checked(manifest["registry"])),
    ):
        (out / name).write_bytes(source.read_bytes())
    prior_path = result_path.parent / "prior_attempts.json"
    prior = (
        json.loads(prior_path.read_text())
        if prior_path.exists()
        else {"attempts": [], "wall_seconds": 0}
    )
    for attempt in prior["attempts"]:
        for key in ("attempt", "manifest", "rollout_log"):
            checked(attempt[key])
    write_json(out / "prior_attempts.json", prior)
    if prior_path.exists():
        provenance["prior_attempts"] = {"path": str(prior_path), "sha256": digest(prior_path)}
    write_json(out / "provenance.json", provenance)
    write_json(
        out / "episode.json",
        {
            "episode_id": "course_episode_0000",
            "split": "development_only",
            "scope": "one authored adaptation across multiple constraints; development demonstration only",
            "registered_option_ids": option_ids,
            "configured_preference": {
                "option_index": row["option_index"],
                "entry_parameter_reference_phase_s": row["decision_time_s"],
            },
            "executed_option": actual,
            "measurement_admitted": row["measurement_admitted"],
            "student_input": student,
            "recorded_result": row,
        },
    )
    (out / "DATASET_CARD.md").write_text(
        "# One-adaptation course demonstration\n\n"
        f"One recorded development episode contains {len(definition['beams'])} overhead constraints "
        f"with recorded option {actual['option_id']}. "
        "This is a physical demonstration, not a navigation dataset, "
        "learned-course comparison or course benchmark.\n\n"
        "All beam geometries and ordered 200 Hz force channels are retained in `course_definition.json` and "
        "`course_contacts.npz`; physical trajectories are pickle-free. `episode.json` preserves every outcome and "
        "per-beam crossing/force label. `commands.json` preserves actual switches, "
        "separate from configured preference.\n\n"
        f"`student_history.npz` contains named {student['feature_dimension']}D causal inputs and legal masks; "
        "`sensor_history.json` "
        "contains only whitelisted measurements and poses. Labels, source identity and geometry are separate. "
        f"All {len(sensor['observations'])} physical and sensor rows are retained with `sensor_alignment.npz`; "
        "inspect its mask "
        "before pairing histories. Sensor/command time is reference phase, 0.02 s after same-index physical "
        "recording time. Traversal cost uses the physical clock. No learned weights "
        "or reference bank are bundled.\n\n"
        "The original result, registration and artifact hashes are retained for provenance. Prior infrastructure "
        "attempts carry wall cost but no invented course labels. Paths in provenance identify local originals; "
        "portable arrays and JSON need no original files to read. Use NumPy with `allow_pickle=False`.\n"
    )
    for source, name in (
        (Path(__file__), "exporter_source.py"),
        (Path(audit_sensor_alignment.__globals__["__file__"]), "sensor_alignment_source.py"),
        (Path(score_course.__globals__["__file__"]), "course_scoring_source.py"),
    ):
        (out / name).write_bytes(source.read_bytes())
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "episodes": 1,
            "beams": len(definition["beams"]),
            "role": "one-adaptation multi-constraint development demonstration",
            "physical_rows": len(payload["motion_time_s"]),
            "physics_steps": result["physics_steps"],
            "raw_sensor_packets": len(sensor["observations"]),
            "eligible_sensor_packets": sum(alignment["packet_eligible"]),
            "feature_dimension": student["feature_dimension"],
            "pass": row["pass"],
            "successful_acquisition_wall_seconds": result["wall_seconds"],
            "prior_infrastructure_wall_seconds": prior["wall_seconds"],
            "exporter_sha256": digest(Path(__file__)),
            "files": {
                str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()
            },
        },
    )
    result = audit(out)
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(out, arcname=out.name)
    return {
        **result,
        "manifest_sha256": digest(out / "manifest.json"),
        "archive": str(archive),
        "archive_sha256": digest(archive),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "audit"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "export" and args.result is None:
        parser.error("export requires --result")
    print(
        json.dumps(
            export(args.result, args.out) if args.mode == "export" else audit(args.out), indent=2
        )
    )


if __name__ == "__main__":
    main()
