#!/usr/bin/env python3
"""Reconstruct frozen acquisition fits using only portable records and NumPy.

The inspector rebuilds complete WAIT continuations from bundled histories.
Replay fits use the original audited historical weights, not recomputed physical
gaps or a new online replay trajectory. No original absolute source is opened.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_schedule_dataset_baseline import (  # noqa: E402
    arrays,
    inspect_package,
    read_json,
    relative_path,
    sha256,
)
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    fit_timed_schedule_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    expected_feature_names,
    validate_schedule_policy,
)


def bound(root, item):
    path = relative_path(root, item["path"])
    if "sha256:" + sha256(path) != item["sha256"]:
        raise ValueError("portable provenance hash mismatch")
    return path


def prefix_targets(targets, original_groups, collections):
    if [g["collection"] for g in original_groups] != collections:
        raise ValueError("the frozen fit must retain its exact historical teacher prefix")
    original = [r for g in original_groups for r in g["targets"]]
    subset = targets[: len(original)]
    if len(subset) != len(original):
        raise ValueError("the portable prefix omits original teacher rows")
    keys = (
        "features",
        "phase_tick",
        "pass_labels",
        "passage_time_s",
        "admitted",
        "legal_mask",
        "continuation_option_indices",
    )
    if any(
        not a.get("available", True) or any(a[k] != b[k] for k in keys)
        for a, b in zip(original, subset, strict=True)
    ):
        raise ValueError("recomputed portable targets differ from the frozen fit")
    return subset


def archived_weights(registration, result, slots):
    if registration["replay_weights"] is None:
        if result["replay_audit"] is not None:
            raise ValueError("uniform arm unexpectedly contains replay weights")
        return slots, None
    replay = result["replay_audit"]
    if replay is None or replay["weights_artifact"] != registration["replay_weights"]:
        raise ValueError("replay fit must bind its originally audited historical weights")
    weights = np.asarray(replay["weights_in_group_target_order"], float)
    if weights.shape != (len(slots),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(
            "this complete M2 release requires a positive weight for every available target"
        )
    return slots, weights


def compare_models(actual, expected, targets):
    if set(actual) != set(expected):
        raise ValueError("reconstructed model schema differs")
    differences = {}
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        if a.shape != b.shape:
            raise ValueError("reconstructed model shape differs")
        if a.dtype.kind in "fc":
            difference = float(np.max(np.abs(a - b)))
            if not np.isfinite(difference) or difference > 1e-10:
                raise ValueError("reconstructed coefficients exceed declared numerical tolerance")
            differences[key] = difference
        elif not np.array_equal(a, b):
            raise ValueError("reconstructed discrete schema differs")
    checked = 0
    maximum = 0.0
    for row in targets:
        phase = list(actual["phase_ticks"]).index(row["phase_tick"])
        values = [
            ((np.asarray(row["features"]) - m["mean"][phase]) / m["std"][phase])
            @ m["weights"][phase]
            + m["bias"][phase]
            for m in (actual, expected)
        ]
        legal = np.asarray(row["legal_mask"], bool)
        actions = [int(np.argmax(np.where(legal, v, -np.inf))) for v in values]
        if actions[0] != actions[1]:
            raise ValueError("reconstructed actual argmax differs on a recorded teacher packet")
        maximum = max(maximum, float(np.abs(values[0][legal] - values[1][legal]).max()))
        checked += 1
    return dict(
        maximum_array_differences=differences,
        recorded_teacher_packet_choices_checked=checked,
        maximum_legal_value_difference=maximum,
    )


def verify_student_models(package, original_results):
    """Retain each actual pre-update visit's generating historical policy."""
    manifest = read_json(package / "manifest.json")
    if len(manifest["source_results"]) != 5 or len(original_results) != 3:
        raise ValueError("three teacher collections and two student collections required")
    bindings = []
    for round_index in (1, 2):
        study = package / "provenance" / f"study_{round_index + 2:03d}"
        source = read_json(study / "manifest.json")
        expected = original_results[round_index - 1]["policy"]
        if source["policy"] != expected or any(
            c["timed_schedule_mode"] != "learned" for c in source["cells"]
        ):
            raise ValueError("a pre-update student was rebound to a different generating policy")
        bindings.append(
            dict(round=round_index, generating_checkpoint=round_index - 1, policy=expected)
        )
    return bindings


def measured_task_yield(outcomes):
    if (
        not outcomes
        or "neutral" not in outcomes
        or any(value not in ("pass", "failure", "unknown") for value in outcomes.values())
    ):
        raise ValueError(
            "explicit known or unknown outcomes for the complete schedule bank required"
        )
    alternatives = [v for k, v in outcomes.items() if k != "neutral"]
    return dict(
        bank_solvable_lower=int("pass" in outcomes.values()),
        bank_solvable_upper=int(any(v != "failure" for v in outcomes.values())),
        adaptation_required_lower=int(outcomes["neutral"] == "failure" and "pass" in alternatives),
        adaptation_required_upper=int(
            outcomes["neutral"] != "pass" and any(v != "failure" for v in alternatives)
        ),
        unknown_branch_outcomes=sum(v == "unknown" for v in outcomes.values()),
    )


def acquisition_curve(package):
    """Recompute M1/M2 physical yield and prefix costs from inspected portable records."""
    manifest = read_json(package / "manifest.json")
    teachers = read_json(package / "teachers.json")
    if len(teachers) != 3 or len(manifest["source_results"]) != 5:
        raise ValueError("exact bootstrap and two-encounter portable prefix required")
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    rows = []
    for checkpoint in (1, 2):
        sources = {
            manifest["source_results"][i]["sha256"]
            for i in [*range(checkpoint + 1), *range(3, 3 + checkpoint)]
        }
        tasks = []
        for teacher in teachers[1 : checkpoint + 1]:
            outcomes = {
                episodes[identifier]["configured_forced_option_id"]: episodes[identifier][
                    "assessment"
                ]["outcome"]["task_outcome"]
                for identifier in teacher["episode_ids"]
            }
            if len(outcomes) != 7:
                raise ValueError("all seven complete schedules must remain in task yield")
            tasks.append(measured_task_yield(outcomes))
        selected = [e for e in episodes.values() if e["collection_sha256"] in sources]
        if len(selected) != 7 + checkpoint * 8:
            raise ValueError(
                "teacher and actual pre-update student prefix costs must both be retained"
            )
        rows.append(
            dict(
                checkpoint=checkpoint,
                assigned_tasks=checkpoint,
                acquisition_episodes=len(selected),
                recorded_physics_steps=sum(e["physics_steps"] for e in selected),
                **{key: sum(t[key] for t in tasks) for key in tasks[0]},
            )
        )
    return rows


def run(release, out):
    index = read_json(release / "release.json")
    if (
        index["schema"] != "motion2scene_acquisition_portable_release_v1"
        or index["budget"] != 2
        or len(index["corpora"]) != 12
        or {c["run_id"] for c in index["corpora"]}
        != {
            f"seed{seed}_{arm}"
            for seed in (93201, 93202, 93203)
            for arm in ("uniform", "target_only", "analytic_contrast", "observation_curriculum")
        }
        or any([m["checkpoint"] for m in c["models"]] != [0, 1, 2] for c in index["corpora"])
        or index["total_episodes"] != 276
        or index["total_physics_steps"] != 328992
    ):
        raise ValueError("complete twelve-corpus M2 release required")
    if out.resolve().is_relative_to(release.resolve()):
        raise ValueError("reconstruction outputs must not modify the release")
    for item in index["toolkit_files"]:
        bound(release, item)
    out.mkdir(parents=True, exist_ok=False)
    records, curves = [], []
    for corpus in index["corpora"]:
        print(json.dumps(dict(status="inspecting_corpus", run_id=corpus["run_id"])), flush=True)
        started = time.monotonic()
        bank, targets, inspection = inspect_package(
            relative_path(release, corpus["dataset"]), corpus["dataset_manifest_sha256"]
        )
        inspection["student_policy_bindings"] = verify_student_models(
            relative_path(release, corpus["dataset"]),
            [read_json(bound(release, m["files"]["result.json"])) for m in corpus["models"]],
        )
        curves.append(
            dict(
                run_id=corpus["run_id"],
                rows=acquisition_curve(relative_path(release, corpus["dataset"])),
            )
        )
        for model in corpus["models"]:
            files = {key: bound(release, ref) for key, ref in model["files"].items()}
            result, registration = read_json(files["result.json"]), read_json(
                files["registration.json"]
            )
            if (
                result["status"] != "complete"
                or registration["l2"] != 10.0
                or registration["allow_measured_tie_initialization"] is not True
            ):
                raise ValueError(
                    "retain the complete frozen ridge fit and measured-tie initializer"
                )
            subset = prefix_targets(
                targets, read_json(files["teachers.json"]), registration["collections"]
            )
            subset, weights = archived_weights(registration, result, subset)
            expected = arrays(files["policy.npz"])
            fitted, fit = fit_timed_schedule_policy(
                bank,
                np.asarray([r["features"] for r in subset]),
                expected_feature_names(7),
                np.asarray([r["phase_tick"] for r in subset]),
                np.asarray([r["pass_labels"] for r in subset], bool),
                np.asarray([r["passage_time_s"] for r in subset], float),
                np.asarray([r["admitted"] for r in subset], bool),
                np.asarray([r["legal_mask"] for r in subset], bool),
                sample_weights=weights,
                l2=registration["l2"],
                allow_measured_tie_initialization=True,
            )
            validate_schedule_policy(fitted, bank)
            comparison = compare_models(fitted, expected, targets)
            folder = out / corpus["run_id"] / f"model_{model['checkpoint']:03d}"
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "policy.npz", **fitted)
            record = dict(
                run_id=corpus["run_id"],
                checkpoint=model["checkpoint"],
                teacher_rows=len(subset),
                sample_weights=None if weights is None else weights.tolist(),
                comparison=comparison,
                fit=fit,
                policy_sha256="sha256:" + sha256(folder / "policy.npz"),
            )
            (folder / "result.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            records.append(record)
        (out / corpus["run_id"] / "inspection.json").write_text(
            json.dumps(inspection, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                dict(
                    status="corpus_reconstructed",
                    run_id=corpus["run_id"],
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
    report = dict(
        schema="motion2scene_acquisition_portable_reconstruction_v1",
        release_sha256="sha256:" + sha256(release / "release.json"),
        models=records,
        acquisition_curves=curves,
        fits=len(records),
        new_physics_steps=0,
        scope=__doc__,
    )
    (out / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.release.resolve(), args.out.resolve())
    print(json.dumps(dict(fits=result["fits"], new_physics_steps=0)))
