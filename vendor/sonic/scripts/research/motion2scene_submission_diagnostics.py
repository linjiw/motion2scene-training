#!/usr/bin/env python3
"""Registered-prediction reconciliation and explicitly post-hoc selector refits."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from audit_motion2scene_submission import ARMS, DATA, DOC, Audit, condition, digest, dump
from motion2scene_selector_diagnosis_v2 import low_capacity
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import select_action

OUT = DOC / "submission/evidence"


def predict(saved, x):
    # Deployment-shaped single-row calls avoid changing float32 reduction order.
    return np.asarray(
        [
            torch.sigmoid(
                (torch.tensor(row) / torch.tensor(saved["scale"])) @ torch.tensor(saved["weights"])
                + torch.tensor(saved["bias"])
            ).numpy()
            for row in x
        ]
    )


def registered_predictions(studies, seed_rows):
    result = {}
    for name, study in studies.items():
        props = study["proposals"]["rows"]
        generated = [r for r in props if r["assigned"] and r["eligible"]]
        pairs = {p["group_id"]: p for p in study["pairs"]}
        useful = [r for r in generated if pairs[r["group_id"]]["outcomes"] == [False, True]]
        source_useful = {
            a: {
                str(s): sum(r["arm"] == a and r["source"] == s for r in useful)
                for s in (41001, 41002, 41003)
            }
            for a in ARMS
        }
        target = [r for r in generated if r["arm"] == "no_contrast"]
        target_both_pass = sum(pairs[r["group_id"]]["outcomes"] == [True, True] for r in target)
        if name == "robust":
            clear = [r for r in generated if r["target_clear_geometry"]]
            clear_ids = {r["group_id"] for r in clear}
            target_rows = [r for r in seed_rows if r["group_id"] in clear_ids and r["action"] == 1]
            assert len(target_rows) == len(clear)
            result[name] = [
                {
                    "id": 1,
                    "prediction": (
                        "Both contrast arms acquire at least one useful generated group on "
                        "every carrier"
                    ),
                    "met": all(
                        v >= 1
                        for a in ("analytic", "motion2scene")
                        for v in source_useful[a].values()
                    ),
                    "counts": source_useful,
                },
                {
                    "id": 2,
                    "prediction": "Uniform useful yield at most 1/18 requested generated groups",
                    "met": sum(source_useful["uniform"].values()) <= 1,
                    "numerator": sum(source_useful["uniform"].values()),
                    "denominator": 18,
                },
                {
                    "id": 3,
                    "prediction": "Target-only both-pass at least 12/18 requested generated groups",
                    "met": target_both_pass >= 12,
                    "numerator": target_both_pass,
                    "denominator": 18,
                },
                {
                    "id": 4,
                    "prediction": (
                        "Zero target-screen false-clears among executed target-clear generated "
                        "predictions"
                    ),
                    "met": all(r["pass"] for r in target_rows),
                    "numerator": sum(not r["pass"] for r in target_rows),
                    "denominator": len(target_rows),
                    "tracker_rejections": sum(
                        r["tracker_outcome"] == "rejected" for r in target_rows
                    ),
                    "definition": "Accepted assigned generated scenes with target_clear_geometry, d040 command",
                },
                {
                    "id": 5,
                    "prediction": "Each contrast arm improves over uniform by at least three net discordances",
                    "met": all(
                        study["primary"]["per_arm"][a]["pass"]
                        - study["primary"]["per_arm"]["uniform"]["pass"]
                        >= 3
                        for a in ("analytic", "motion2scene")
                    ),
                    "net_passages": {
                        a: study["primary"]["per_arm"][a]["pass"]
                        - study["primary"]["per_arm"]["uniform"]["pass"]
                        for a in ("analytic", "motion2scene")
                    },
                },
            ]
        else:
            eligible = [r for r in props if r["eligible"]]
            selected_contrasts = [r for r in generated if r["arm"] in ("analytic", "motion2scene")]
            accepted_by_source = {
                a: {
                    str(s): sum(r["arm"] == a and r["source"] == s for r in generated)
                    for s in (41001, 41002, 41003)
                }
                for a in ARMS
            }
            result[name] = [
                {
                    "id": 1,
                    "prediction": "Both contrast arms yield at least one eligible group per carrier",
                    "met": all(
                        v >= 1
                        for a in ("analytic", "motion2scene")
                        for v in accepted_by_source[a].values()
                    ),
                    "counts": accepted_by_source,
                },
                {
                    "id": 2,
                    "prediction": "An arm other than analytic acquires a useful contrast",
                    "met": any(r["arm"] != "analytic" for r in useful),
                    "counts": source_useful,
                },
                {
                    "id": 3,
                    "prediction": "Uniform useful yield at most one across assigned groups",
                    "met": sum(source_useful["uniform"].values()) <= 1,
                    "numerator": sum(source_useful["uniform"].values()),
                    "denominator": 9,
                },
                {
                    "id": 4,
                    "prediction": "Target-only predominantly both-pass",
                    "met": target_both_pass > len(target) / 2,
                    "numerator": target_both_pass,
                    "denominator": len(target),
                    "including_backgrounds": study["composition"]["no_contrast"]["outcomes"],
                },
                {
                    "id": 5,
                    "prediction": "At least one useful training contrast iff at least one evaluation d040 request",
                    "met": all(
                        (study["composition"][a]["outcomes"]["01"] > 0)
                        == (study["primary"]["per_arm"][a]["d040_requests"] > 0)
                        for a in ARMS
                    ),
                },
                {
                    "id": 6,
                    "prediction": "Majority of nominally accepted proposals fail 113-offset diagnostic",
                    "met": sum(not r["robust_113_offset"] for r in eligible) > len(eligible) / 2,
                    "all_eligible": [
                        sum(not r["robust_113_offset"] for r in eligible),
                        len(eligible),
                    ],
                    "all_assigned": [
                        sum(not r["robust_113_offset"] for r in generated),
                        len(generated),
                    ],
                    "assigned_contrast_arms": [
                        sum(not r["robust_113_offset"] for r in selected_contrasts),
                        len(selected_contrasts),
                    ],
                },
            ]
    return result


def sensitivities(audit, study, commands, seed_rows):
    pairs = {p["group_id"]: p for p in study["pairs"]}
    observations = [
        next(
            r
            for r in seed_rows
            if r["source"] == c["source"]
            and r["layout"] == c["layout"]
            and r["physics_seed"] == 8511
            and r["cell_id"].startswith("nom_eval_")
            and r["arm"] == "uniform"
        )
        for c in commands
    ]
    eval_x = np.asarray(
        [
            audit.read(r["decision"]["path"], r["decision"]["sha256"])["features"]
            for r in observations
        ],
        np.float32,
    )
    labels = np.asarray([r["outcomes"] for r in commands], bool)
    assert labels.shape == (36, 2)
    output = []
    (OUT / "sensitivity_models").mkdir(exist_ok=True)
    started = time.monotonic()
    for arm in ARMS:
        ids = study["composition"][arm]["training_ids"]
        folds = [
            ("leave_carrier", str(source), [k for k in ids if pairs[k]["source"] == source])
            for source in (41001, 41002, 41003)
        ]
        if arm in ("analytic", "motion2scene"):
            folds += [
                ("leave_contrast", k, [k]) for k in ids if pairs[k]["outcomes"] == [False, True]
            ]
        for kind, key, withheld in folds:
            retained = [k for k in ids if k not in withheld]
            x = np.asarray([pairs[k]["features"] for k in retained], np.float32)
            y = np.asarray([pairs[k]["outcomes"] for k in retained], np.float32)
            saved, _, loss = low_capacity(x, y, eval_x)
            probabilities = predict(saved, eval_x)
            selections = select_action(probabilities)
            actions = np.maximum(selections, 0)
            passed = labels[np.arange(36), actions]
            model = OUT / "sensitivity_models" / f"{arm}_{kind}_{key}.npz"
            np.savez_compressed(model, **saved)
            per_carrier = {
                str(s): {"pass": int(passed[[r["source"] == s for r in commands]].sum()), "n": 12}
                for s in (41001, 41002, 41003)
            }
            output.append(
                {
                    "arm": arm,
                    "kind": kind,
                    "withheld_key": key,
                    "withheld_ids": withheld,
                    "training_ids": retained,
                    "training_bce": loss,
                    "model": audit.pin(model),
                    "full_bank_pass": int(passed.sum()),
                    "n": 36,
                    "requests": int((actions == 1).sum()),
                    "refusals": int((selections == -1).sum()),
                    "per_carrier": per_carrier,
                    "actions": actions.tolist(),
                    "probabilities": probabilities.tolist(),
                    "post_hoc": True,
                    "evaluation": "Matched command lookup, not additional policy execution",
                }
            )
    return {
        "refits": output,
        "cpu_wall_seconds": time.monotonic() - started,
        "scope": (
            "Selector-only removal diagnostics. All labels/backgrounds of a removed carrier "
            "stay together. Generator pretraining still includes development carriers; no "
            "generator-level source-held-out transfer."
        ),
    }


def original_refits(audit, robust):
    fitted = audit.read(DATA / "m2s-icra-learning-v1/fit.json")
    pairs = {p["group_id"]: p for p in robust["pairs"]}
    result = []
    for fold in fitted["folds"]:
        audit.ref(fold["model"])
        with np.load(fold["model"]["path"]) as f:
            saved = {k: f[k] for k in f.files}
        withheld = fold["withheld_labelled"]
        if withheld:
            x = np.asarray([pairs[k]["features"] for k in withheld], np.float32)
            probabilities = predict(saved, x)
            assert np.max(np.abs(probabilities - fold["probabilities"])) <= 1e-6
            assert select_action(probabilities).tolist() == fold["selected_actions"]
        result.append(fold)
    original = audit.read(DATA / "m2s-icra-learning-v1/comparison_540.json")["rows"]
    unique = {r["group_id"]: r for r in original if r["arm"] == "analytic"}
    x = np.asarray(
        [
            audit.read(r["decision"]["path"], r["decision"]["sha256"])["features"]
            for r in unique.values()
        ],
        np.float32,
    )
    request_counts = {}
    for f in result:
        if f["arm"] == "analytic":
            with np.load(f["model"]["path"]) as s:
                request_counts[str(f["fold"])] = int((select_action(predict(s, x)) == 1).sum())
    with np.load(DATA / "m2s-icra-learning-v1/models/motion2scene.npz") as base:
        with np.load(DATA / "m2s-icra-learning-v1/models/analytic_leave4_0.npz") as removed:
            equals = all(np.array_equal(base[k], removed[k]) for k in base.files)
    return {
        "folds": result,
        "analytic_requests_on_90_recorded_conditions": request_counts,
        "analytic_removal_equals_background_model": equals,
        "registered_evaluation": "Only withheld physically labeled groups are the registered refit endpoint",
        "post_hoc_evaluation": (
            "The 90-recorded-input request comparison is a post-hoc sensitivity, not executed "
            "refit policies"
        ),
    }


def provenance(audit, analysis):
    earlier = audit.read(DATA / "m2s-icra-learning-v1/comparison_366.json")
    prior = {condition(r) for r in earlier["rows"]}
    nominal = {condition(r) for r in analysis["command_lookup"]["conditions"]}
    names = (
        "M2S_ICRA_V1.md",
        "M2S_ICRA_EXECUTION_V1.md",
        "M2S_ICRA_NOMINAL_V1.md",
        "M2S_ICRA_366_RESULT.md",
        "M2S_ICRA_540_RESULT.md",
        "ENVELOPE_TRADEOFF_V1.md",
        "LEARNING_UTILITY_PLAN_V1.md",
        "NOMINAL_TRANSITION_NEXT_STAGE.md",
    )
    history = {}
    for name in names:
        path = DOC / name
        audit.pin(path)
        revisions = subprocess.check_output(
            ["git", "log", "--format=%H|%aI|%cI|%s", "--", str(path)], text=True
        ).splitlines()
        versions = []
        for line in revisions:
            commit, authored, committed, subject = line.split("|", 3)
            content = subprocess.check_output(["git", "show", f"{commit}:docs/motion2scene/{name}"])
            versions.append(
                {
                    "commit": commit,
                    "author_time": authored,
                    "commit_time": committed,
                    "subject": subject,
                    "document_sha256": "sha256:"
                    + __import__("hashlib").sha256(content).hexdigest(),
                }
            )
        history[name] = versions
    overlap = {}

    def beamkey(r):
        return r["source"], r["beam"]["route_progress"], r["beam"]["underside_m"]

    for arm in ARMS:
        chosen = [
            {
                beamkey(r)
                for r in analysis["studies"][name]["proposals"]["rows"]
                if r["arm"] == arm and r["assigned"] and r["eligible"]
            }
            for name in ("robust", "nominal")
        ]
        overlap[arm] = len(chosen[0] & chosen[1])
    return {
        "protocol_git_versions": history,
        "nominal_conditions_observed_in_366_panel": sorted(prior & nominal),
        "count_observed_before_nominal_registration": len(prior & nominal),
        "partial_panel_conditions": len(prior),
        "same_seed_generated_group_overlap": overlap,
        "same_proposal_seed": 8841,
        "same_label_seed": 8722,
        "nominal_metadata_discrepancies": {
            "generated_slots_per_source_arm": (
                "JSON retains 6, but operative assigned_per_source_arm=3 and source code "
                "first-three-eligible rule agree with protocol and records"
            ),
            "requested_generated": (
                "proposals.json retains 72 slot metadata; actual selected group count is 36 and "
                "command count is 72"
            ),
            "changed_and_only_this": (
                "Prose overstates isolation: acceptance, selection order, quota and evaluation "
                "seed count differ, with six shared backgrounds reused"
            ),
        },
        "envelope_protocol_discrepancies": {
            "grid_size": (
                "Protocol's 7181 total is arithmetic error; specified 81 stations x 141 heights "
                "and code/records yield 11421"
            ),
            "monotonicity": (
                "Zero offset only justifies retaining nominal witnesses. Scaled finite sets "
                "need not be nested; observed count decrease is not a general monotonicity "
                "proof."
            ),
        },
        "timestamp_limit": (
            "Git and run timestamps plus local file mtimes establish documented availability; "
            "they are not an external timestamping service or a log of every human inspection"
        ),
        "interpretation": (
            "Nominal study is an outcome-informed design revision on an already partly observed "
            "bank, with proposals and labels frozen prospectively for that revision."
        ),
    }


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    audit = Audit()
    analysis = audit.read(OUT / "analysis.json")
    assert analysis["command_lookup"]["missing_actions"] == 0
    assert not analysis["command_lookup"]["reuse_mismatches"]
    predictions = registered_predictions(analysis["studies"], analysis["seed_level_rows"])
    original = original_refits(audit, analysis["studies"]["robust"])
    sensitivity = sensitivities(
        audit,
        analysis["studies"]["nominal"],
        analysis["command_lookup"]["conditions"],
        analysis["seed_level_rows"],
    )
    history = provenance(audit, analysis)
    inputs = list(audit.files.values())
    result = {
        "post_hoc": True,
        "registered_predictions": predictions,
        "registered_refits": original,
        "nominal_sensitivity": sensitivity,
        "provenance": history,
        "analysis_script": audit.pin(Path(__file__)),
        "inputs": inputs,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_wall_seconds": time.monotonic() - started,
    }
    dump(OUT / "diagnostics.json", result)
    print(
        json.dumps(
            {
                "predictions": {s: [p["met"] for p in ps] for s, ps in predictions.items()},
                "sensitivity_fits": len(sensitivity["refits"]),
                "observed_nominal_conditions": history[
                    "count_observed_before_nominal_registration"
                ],
                "output_sha256": digest(OUT / "diagnostics.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
