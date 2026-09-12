#!/usr/bin/env python3
"""Inspect or fit an offline value baseline from portable traversal teacher datasets.

No simulator, original absolute artifact paths, teacher search or held-out layout
access is needed. Fitting produces a development model and training diagnostics,
not a traversal evaluation. Use the recorded WAIT continuation targets.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_value_imitation import (  # noqa: E402
    fit_value_policy,
)
from scripts.research.motion2scene_export_option_dataset import (  # noqa: E402
    MULTI_SCHEMA,
    audit,
    digest,
    portable_path,
    write_json,
)


def verify_teacher_row(package, target, episodes):
    """Resolve a sensor-only input and physical continuations inside one package."""
    ref = target["student_input"]
    episode = episodes[ref["episode_id"]]
    if episode["split"] not in ("development_only", "train"):
        raise ValueError("teacher input is not training/development data")
    folder = portable_path(package, episode["directory"])
    expected_file = folder / episode["student_input"]["file"]
    if portable_path(package, ref["file"]) != expected_file:
        raise ValueError("teacher input is not its declared episode's exact student file")
    alignment = json.loads((folder / "sensor_alignment.json").read_text())
    frame = ref["row_index"]
    if type(frame) is not int or not 0 <= frame < len(alignment["packet_eligible"]):
        raise ValueError("teacher row is outside the recorded episode")
    if not alignment["packet_eligible"][frame]:
        raise ValueError("teacher input uses an ineligible sensor packet")
    with np.load(expected_file, allow_pickle=False) as arrays:
        if str(arrays["schema_version"]) != MULTI_SCHEMA:
            raise ValueError("baseline requires the exact multi-option feature schema")
        features = arrays["features"][frame].copy()
        names, option_ids = arrays["feature_names"].tolist(), arrays["option_ids"].tolist()
        if (
            arrays["phase_s"][frame] != target["phase_s"]
            or arrays["phase_s"][frame] != ref["phase_s"]
            or arrays["active_before"][frame] != 0
            or not np.array_equal(arrays["legal_mask"][frame], target["legal_mask"])
        ):
            raise ValueError("teacher target and causal phase/legality do not match")
    count = len(option_ids)
    for key in (
        "pass_labels",
        "passage_time_s",
        "admitted",
        "legal_mask",
        "continuation_episode_ids",
    ):
        if len(target[key]) != count:
            raise ValueError("teacher action dimension differs from recorded options")
    for index, continuation_id in enumerate(target["continuation_episode_ids"]):
        if not target["admitted"][index]:
            continue
        if not target["legal_mask"][index] or continuation_id is None:
            raise ValueError("admitted teacher action needs a legal physical continuation")
        if (
            target["admitted_continuation_counts"][index]
            != target["expected_continuation_counts"][index]
        ):
            raise ValueError("admitted target omits an expected finite continuation")
        continuation = episodes[continuation_id]
        if any(
            continuation[key] != episode[key]
            for key in ("source", "physics_seed", "condition", "beam")
        ):
            raise ValueError("teacher continuation is from a different physical encounter")
        matches = [r for r in target["matching_audit"] if r["episode_id"] == continuation_id]
        if (
            len(matches) != 1
            or not matches[0]["matched"]
            or not matches[0]["prefix"]["exact_match"]
        ):
            raise ValueError("teacher continuation lacks a matched recorded prefix")
        if continuation["split"] not in ("development_only", "train"):
            raise ValueError("teacher continuation crosses into evaluation data")
        if continuation["registered_option_ids"] != option_ids:
            raise ValueError("teacher continuation uses a different option registry")
        if not continuation["measurement_admitted"]:
            raise ValueError("teacher target uses an unadmitted physical continuation")
        if target["pass_labels"][index]:
            actual = continuation["executed_option"]
            if (
                not continuation["pass"]
                or continuation["reset_count"]
                or continuation["failure_flags"].get("fall_observed", False)
                or actual["first_episode_final_option_index"] != 0
                or (not actual["stayed_neutral"] and actual["return_time_s"] is None)
            ):
                raise ValueError("passing teacher target lacks successful physical recovery")
            if target["passage_time_s"][index] != continuation["costs"]["passage_time_s"]:
                raise ValueError("teacher passage cost differs from its physical continuation")
        elif target["passage_time_s"][index] is not None:
            raise ValueError("failed teacher continuation has an invented passage time")
    return features, names, option_ids


def load_training_datasets(packages):
    features, targets, inputs, source_manifests = [], [], [], []
    names, options, physical_sources = None, None, set()
    for package in packages:
        report = audit(package)
        manifest_hash = digest(package / "manifest.json")
        if manifest_hash in {x["sha256"] for x in source_manifests}:
            raise ValueError("the same dataset was supplied more than once")
        table = json.loads((package / "schedule_teachers.json").read_text())
        episodes = {
            row["episode_id"]: row for row in json.loads((package / "episodes.json").read_text())
        }
        provenance = {
            row["episode_id"]: row for row in json.loads((package / "provenance.json").read_text())
        }
        source_manifests.append(
            {"path": str(package / "manifest.json"), "sha256": manifest_hash, "audit": report}
        )
        for target in table:
            x, row_names, row_options = verify_teacher_row(package, target, episodes)
            if names is None:
                names, options = row_names, row_options
            elif names != row_names or options != row_options:
                raise ValueError("cannot concatenate different feature names or option identities")
            ref = target["student_input"]
            trajectory = provenance[ref["episode_id"]]["artifacts"]["trajectory"]
            # Distinct physical runs can produce byte-identical trajectories.
            # Only the same recorded artifact repeated across releases is a duplicate.
            identity = (trajectory["path"], trajectory["sha256"], ref["row_index"])
            if identity in physical_sources:
                raise ValueError("duplicate physical decision across dataset releases")
            physical_sources.add(identity)
            features.append(x)
            targets.append(target)
            inputs.append(
                {
                    "dataset_manifest_sha256": manifest_hash,
                    "student_input": ref,
                    "schedule_teacher_id": target["schedule_teacher_id"],
                }
            )
    if not targets:
        raise ValueError(
            "no portable finite-schedule targets; policy-only datasets cannot train this baseline"
        )
    return {
        "features": np.asarray(features),
        "feature_names": names,
        "option_ids": options,
        "passed": np.array([r["pass_labels"] for r in targets], dtype=bool),
        "times": [[np.nan if t is None else t for t in r["passage_time_s"]] for r in targets],
        "admitted": np.array([r["admitted"] for r in targets], dtype=bool),
        "legal": np.array([r["legal_mask"] for r in targets], dtype=bool),
        "inputs": inputs,
        "datasets": source_manifests,
        "semantics": "action0 means WAIT now with recorded future continuation; walking commitment is separate",
    }


def fit(packages, out, l2):
    data = load_training_datasets(packages)
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("ridge regularization must be finite and nonnegative")
    out.mkdir(parents=True, exist_ok=False)
    source_files = {
        "cli": Path(__file__),
        "dataset_audit": Path(audit.__globals__["__file__"]),
        "fitter": Path(fit_value_policy.__globals__["__file__"]),
        "physical_regret": Path(
            fit_value_policy.__globals__["physical_regret"].__globals__["__file__"]
        ),
    }
    registration = {
        "schema": "motion2scene_portable_dataset_baseline_v1",
        "datasets": data["datasets"],
        "inputs": data["inputs"],
        "option_ids": data["option_ids"],
        "feature_names": data["feature_names"],
        "method": "per-option ridge regression of measured finite-schedule regret",
        "l2": l2,
        "decision_weights": "uniform over recorded decision targets",
        "normalization": "complete consequential training rows only; std floor0.05",
        "implementation": {name: digest(path) for name, path in source_files.items()},
        "action_semantics": data["semantics"],
        "scope": "offline development baseline; no simulator, new labels, policy rollout or held-out result",
    }
    write_json(out / "registration.json", registration)
    (out / "source_snapshot").mkdir()
    for name, path in source_files.items():
        (out / "source_snapshot" / (name + ".py")).write_bytes(path.read_bytes())
    model, report = fit_value_policy(
        data["features"],
        data["feature_names"],
        data["option_ids"],
        data["passed"],
        data["times"],
        data["admitted"],
        data["legal"],
        l2=l2,
    )
    np.savez_compressed(out / "policy.npz", **model)
    write_json(
        out / "training_report.json",
        {
            **report,
            "model_sha256": digest(out / "policy.npz"),
            "new_physics_steps": 0,
            "evaluation_episodes": 0,
        },
    )
    return {
        "model": str(out / "policy.npz"),
        "decisions": len(data["features"]),
        "new_physics_steps": 0,
        "scope": "training diagnostics only; no traversal evaluation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "fit"))
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--l2", type=float, default=1e-6)
    args = parser.parse_args()
    if args.mode == "fit":
        if args.out is None:
            parser.error("fit requires a new --out directory")
        report = fit(args.dataset, args.out, args.l2)
    else:
        data = load_training_datasets(args.dataset)
        report = {
            "datasets": len(data["datasets"]),
            "teacher_decisions": len(data["features"]),
            "feature_dimension": len(data["feature_names"]),
            "option_ids": data["option_ids"],
            "action_semantics": data["semantics"],
            "new_physics_steps": 0,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
