#!/usr/bin/env python3
"""Fit the106D timed student from independently re-audited matched physical tables."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_collect_timed_history import COLLECTION_SCHEMA, analyze_cell  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import (  # noqa: E402
    paired_prefix,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_policy import (  # noqa: E402
    ENTRY_TICK,
    expected_feature_names,
    fit_timed_value_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    definition_digest,
    load_verified_registry,
)


def audit_collection(path, bank, registry_ref):
    result = json.loads(path.read_text())
    ref = result["manifest"]
    manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    if (
        result["schema"] != COLLECTION_SCHEMA
        or manifest["schema"] != COLLECTION_SCHEMA
        or manifest["split"] != "development"
        or manifest["registry"] != registry_ref
        or manifest["request_digest"] != definition_digest(bank.request)
    ):
        raise ValueError("matching development collection and verified registry required")
    scene_ref = manifest["scene_definition"]
    scene = json.loads(checked(Path(scene_ref["path"]), scene_ref["sha256"]).read_text())
    if scene["split"] != "development":
        raise ValueError("reserved evaluation is not permitted in training")
    teacher = result["teacher"]
    if teacher is None or not teacher["exact_shared_entry_observation"]:
        raise ValueError("complete exactly matched physical schedule teacher required")
    cells = {cell["forced_option_id"]: cell for cell in manifest["cells"]}
    if (
        len(manifest["cells"]) != len(bank.option_ids)
        or len(result["rows"]) != len(bank.option_ids)
        or len({row["forced_option_id"] for row in result["rows"]}) != len(bank.option_ids)
        or len({cell["runtime_seed"] for cell in manifest["cells"]}) != 1
        or any(
            type(cell["runtime_seed"]) is not int or cell["runtime_seed"] < 0
            for cell in manifest["cells"]
        )
    ):
        raise ValueError(
            "one unique branch per option and one common registered physics seed required"
        )
    if set(cells) != set(bank.option_ids) or any(
        c["timed_history_mode"] != "forced" for c in cells.values()
    ):
        raise ValueError("every named schedule must have its own forced physical branch")
    actual = {}
    for option in bank.option_ids:
        row = next(row for row in result["rows"] if row["forced_option_id"] == option)
        for name in (
            "trajectory",
            "sensor",
            "features",
            "all_body_contacts",
            "environment_pairs",
            "environment_mapping",
            "physics_beam_contacts",
            "attempt",
        ):
            checked(Path(row[name]["path"]), row[name]["sha256"])
        cell = cells[option]
        attempt_path = Path(cell["output"]) / "attempt.json"
        if row["attempt"] != artifact(attempt_path):
            raise ValueError("attempt identity differs from registered branch")
        attempt = json.loads(attempt_path.read_text())
        if attempt["exit_status"] != 0 or attempt["command"] != cell["command"]:
            raise ValueError("successful actual invocation must match registered command")
        command = cell["command"]
        if (
            command.count("--extra") != 1
            or command.index("--extra") + 1 >= len(command)
            or command[command.index("--extra") + 1] != " ".join(cell["hydra_overrides"])
        ):
            raise ValueError("actual command overrides differ from registered effective overrides")
        seed_overrides = [
            v for v in cell["hydra_overrides"] if v.lstrip("+").split("=", 1)[0] == "seed"
        ]
        if len(seed_overrides) != 1 or seed_overrides[0].split("=", 1)[1] != str(
            cell["runtime_seed"]
        ):
            raise ValueError("registered seed and executed override disagree")
        success = json.loads((Path(cell["output"]) / "success_manifest.json").read_text())
        captured_scene = success["capture_context"]["scene"]
        if (
            success["capture_context"]["scene_id"] != scene["scene_id"]
            or captured_scene["hash"] != scene["scene"]["sha256"]
            or Path(captured_scene["resolved"]).resolve() != Path(scene["scene"]["path"]).resolve()
        ):
            raise ValueError("actual runtime scene identity differs from registered scene")
        verified, payload, interface = analyze_cell(cell, manifest, bank, scene)
        for name in (
            "trajectory",
            "sensor",
            "features",
            "all_body_contacts",
            "environment_pairs",
            "environment_mapping",
            "physics_beam_contacts",
        ):
            if row[name] != verified[name]:
                raise ValueError("stored artifact identity differs from actual registered branch")
        for name in (
            "measurement_admitted",
            "pass",
            "costs",
            "schedule_audit",
            "observation_timing",
        ):
            if row[name] != verified[name]:
                raise ValueError(f"stored {name} differs from independent physical re-audit")
        actual[option] = (verified, payload, interface)
    neutral = actual["neutral"]
    features = []
    legality = []
    for option in bank.option_ids:
        row, payload, interface = actual[option]
        prefix = paired_prefix(neutral[1], payload, ENTRY_TICK / 50)
        if not prefix["exact_match"] or prefix["frames"] < ENTRY_TICK:
            raise ValueError("matched executed approach prefix required")
        decisions = [r for r in interface["observations"] if r["tick"] == ENTRY_TICK]
        if len(decisions) != 1:
            raise ValueError("one actual entry decision required")
        target = decisions[0]
        features.append(target["features"])
        legality.append(target["legal_mask"])
        if target["measurements"] != neutral[2]["observations"][ENTRY_TICK - 1]["measurements"]:
            raise ValueError("teacher branches differ in actual sensor measurements")
    if not all(np.array_equal(features[0], v) for v in features) or not all(
        np.array_equal(legality[0], v) for v in legality
    ):
        raise ValueError("teacher entry features and legality must agree exactly")
    recreated = dict(
        entry_tick=ENTRY_TICK,
        option_ids=list(bank.option_ids),
        feature_names=list(expected_feature_names()),
        features=features[0],
        legality=legality[0],
        passed=[actual[k][0]["pass"] for k in bank.option_ids],
        passage_time_s=[actual[k][0]["costs"]["passage_time_s"] for k in bank.option_ids],
        admitted=[actual[k][0]["measurement_admitted"] for k in bank.option_ids],
    )
    if any(teacher[k] != v for k, v in recreated.items()):
        raise ValueError("cached teacher table differs from re-audited recorded branches")
    return {
        **recreated,
        "scene_id": scene["scene_id"],
        "collection": artifact(path),
        "trajectory_identities": [actual[k][0]["trajectory"] for k in bank.option_ids],
        "physics_steps": sum(actual[k][0]["physics_steps"] for k in bank.option_ids),
    }


def run(registry, collections, out, l2):
    registry_ref = artifact(registry)
    bank = load_verified_registry(registry, registry_ref["sha256"])
    if len(set(p.resolve() for p in collections)) != len(collections) or not collections:
        raise ValueError("distinct nonempty collection paths required")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("finite nonnegative ridge penalty required")
    refs = [artifact(path) for path in collections]
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        out / "registration.json",
        dict(
            schema="motion2scene_timed_policy_training_v1",
            registry=registry_ref,
            collections=refs,
            l2=l2,
            implementation=sources,
            weighting="uniform development groups",
            objective=(
                "one fixed ridge head; complete legal physical regret targets; "
                "passage first then measured time"
            ),
            scope="offline training; actual selected schedule execution remains required",
        ),
    )
    rows = [audit_collection(path, bank, registry_ref) for path in collections]
    identities = [
        (str(Path(r["path"]).resolve()), r["sha256"])
        for row in rows
        for r in row["trajectory_identities"]
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("same captured episode supplied through multiple collections")
    model, report = fit_timed_value_policy(
        bank,
        np.asarray([r["features"] for r in rows]),
        np.asarray([r["passed"] for r in rows], dtype=bool),
        np.asarray([r["passage_time_s"] for r in rows], dtype=float),
        np.asarray([r["admitted"] for r in rows], dtype=bool),
        np.asarray([r["legality"] for r in rows], dtype=bool),
        l2=l2,
    )
    np.savez_compressed(out / "policy.npz", **model)
    write_new(out / "teachers.json", rows)
    write_new(
        out / "result.json",
        dict(
            registration=artifact(out / "registration.json"),
            policy=artifact(out / "policy.npz"),
            teachers=artifact(out / "teachers.json"),
            fit=report,
            source_physics_steps=sum(r["physics_steps"] for r in rows),
            new_physics_steps=0,
        ),
    )
    print(json.dumps({"policy": artifact(out / "policy.npz"), "fit": report}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1e-6)
    args = parser.parse_args()
    run(args.registry, args.collections, args.out, args.l2)
