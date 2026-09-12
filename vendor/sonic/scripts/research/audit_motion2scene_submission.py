#!/usr/bin/env python3
"""Reproduce frozen Motion2Scene records and emit a post-hoc submission audit.

No original file is written, no primary model is replaced, and no physics is run.
Run from the repository with PYTHONPATH=.:scripts/research .venv_research/bin/python.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from motion2scene_icra_compare import PAIRS, compare
from motion2scene_reactive_interface import physics_windows
from motion2scene_selector_diagnosis_v2 import low_capacity
import numpy as np
from scipy.stats import binomtest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_icra_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import (
    decision_features,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "research-data/groot-wbc"
DOC = ROOT / "docs/motion2scene"
ARMS = ("uniform", "analytic", "no_contrast", "motion2scene")
STUDIES = {
    "robust": ("m2s-icra-v1", "m2s-icra-learning-v1", "comparison_540.json"),
    "nominal": ("m2s-icra-nominal-v1", "m2s-icra-nominal-learning-v1", "comparison_144.json"),
}
DECISION_FIELDS = (
    "packet",
    "features",
    "state",
    "root_pos_w",
    "root_quat_w",
    "active_before",
    "phase_s",
    "observation_age_s",
    "joint_names",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def condition(row):
    return row["source"], row["layout"], row["physics_seed"]


def counts(rows):
    """Parallel outcome counts, without turning a refusal into a successful stop."""
    return {
        "n": len(rows),
        "pass": sum(bool(r["pass"]) for r in rows),
        "d040_requests": sum(r["action"] == 1 for r in rows),
        "successful_d040_passages": sum(r["action"] == 1 and r["pass"] for r in rows),
        "refusals": sum(r.get("readout", {}).get("refusal", False) for r in rows),
        "successful_walk_refusals": sum(
            r.get("readout", {}).get("refusal", False) and r["pass"] for r in rows
        ),
        "falls": sum(r["fall_observed"] for r in rows),
        "resets": sum(r["reset_count"] > 0 for r in rows),
        "denied_commands": sum(not r["command_executed"] for r in rows),
        "entries": sum(r["entry_logged"] for r in rows),
        "returns": sum(r["return_logged"] for r in rows),
        "tracker_rejections": sum(r["tracker_outcome"] == "rejected" for r in rows),
        "pass_with_tracker_rejection": sum(
            r["pass"] and r["tracker_outcome"] == "rejected" for r in rows
        ),
        "max_force_n": max(
            (r["maximum_beam_normal_force_n_through_passage"] for r in rows), default=None
        ),
    }


def grouped(rows, field):
    return {
        str(k): counts([r for r in rows if r[field] == k]) for k in sorted({r[field] for r in rows})
    }


class Audit:
    def __init__(self):
        self.files = {}
        self.pin_checks = []
        self.raw_checks = []
        self.contexts = {}
        self.decisions = {}
        self.prefixes = {}
        self.histories = {}
        self.post_decision_fingerprints = {}
        self.batch_costs = []
        self.policies = {}

    def pin(self, path, expected=None, strict=True):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        key = str(path)
        if key not in self.files:
            if not path.is_file():
                self.pin_checks.append(
                    {"path": key, "expected": expected, "matches": False, "missing": True}
                )
                if strict:
                    raise FileNotFoundError(path)
                return {"path": key, "missing": True}
            self.files[key] = {"path": key, "sha256": digest(path), "bytes": path.stat().st_size}
        ref = self.files[key]
        if expected:
            ok = ref["sha256"] == expected
            self.pin_checks.append({"path": key, "expected": expected, "matches": ok})
            if strict and not ok:
                raise ValueError(f"Frozen artifact hash mismatch: {path}")
        return ref

    def read(self, path, expected=None):
        ref = self.pin(path, expected)
        return json.loads(Path(ref["path"]).read_text())

    def ref(self, obj):
        return self.pin(obj["path"], obj["sha256"])

    def batches(self, index_path, study, stage):
        index = self.read(index_path)
        rows, pairs = [], []
        for b in index["batches"]:
            folder = Path(b["directory"])
            admission = self.read(folder / "admission.json")
            assert admission["admitted"], folder
            result = self.read(admission["result"]["path"], admission["result"]["sha256"])
            manifest = self.read(folder / "manifest.json")
            run = self.read(folder / "run_record.json")
            for ref in manifest.get("new_dependencies", []):
                self.ref(ref)
            for field in ("checkpoint", "rollout_driver", "manifest_driver"):
                self.ref(manifest["implementation"][field])
            self.pin(folder / "manifest.json", run["manifest_sha256"])
            assert run["status"] == "completed"
            self.batch_costs.append(
                {
                    "study": study,
                    "stage": stage,
                    "record": self.pin(folder / "run_record.json"),
                    "hours": run["budget"]["actual_contended_gpu_hours"],
                    "started_at": run.get("started_at"),
                    "ended_at": run.get("ended_at"),
                    "cells": len(result["rows"]),
                    "cell_elapsed_seconds": {
                        k: v["elapsed_seconds"] for k, v in run["cells"].items()
                    },
                }
            )
            cells = {c["cell_id"]: c for c in manifest["cells"]}
            assert set(cells) == {r["cell_id"] for r in result["rows"]}
            for row in result["rows"]:
                self.raw_row(row, cells[row["cell_id"]])
                self.contexts[row["cell_id"]] = {
                    "manifest": self.pin(folder / "manifest.json"),
                    "result": self.pin(folder / "result.json"),
                    "run_record": self.pin(folder / "run_record.json"),
                    "cell": cells[row["cell_id"]],
                    "implementation": manifest["implementation"],
                    "execution_policy": manifest["execution_policy"],
                    "runtime_dependencies": manifest.get("new_dependencies", []),
                }
                rows.append(row)
            for pair in result.get("pairs", []):
                assert pair["valid"] and all(pair["commands_executed"])
                members = sorted(
                    [r for r in result["rows"] if r["group_id"] == pair["group_id"]],
                    key=lambda r: r["action"],
                )
                assert [r["action"] for r in members] == [0, 1]
                assert pair["outcomes"] == [r["pass"] for r in members]
                assert all(
                    self.decisions[r["cell_id"]]["features"] == pair["features"] for r in members
                )
                pairs.append(
                    {
                        **pair,
                        "arm": members[0]["arm"],
                        "suite": cells[members[0]["cell_id"]]["suite"],
                        "row_ids": [r["cell_id"] for r in members],
                    }
                )
            for rr in self.by_group(result["rows"]).values():
                for other in rr[1:]:
                    assert self.prefix_equal(rr[0], other), (rr[0]["cell_id"], other["cell_id"])
        return rows, pairs

    @staticmethod
    def by_group(rows):
        result = defaultdict(list)
        for r in rows:
            result[r["group_id"]].append(r)
        return result

    def raw_row(self, row, cell):
        for key in ("trajectory", "contacts", "physics", "decision", "sensor", "bank", "inventory"):
            self.ref(row[key])
        payload = load_reset_capture(row["trajectory"]["path"])
        with np.load(row["contacts"]["path"]) as sampled:
            with np.load(row["physics"]["path"]) as physics:
                _, forces, sync = physics_windows(physics, sampled["force_w"])
        computed = score_passage(payload, forces, row["beam"])
        assert all(row[k] == v for k, v in computed.items()), row["cell_id"]
        assert all(row["predicates"].values())
        decision = self.read(row["decision"]["path"])
        x = decision_features(
            decision["packet"],
            decision["state"],
            decision["phase_s"],
            decision["active_before"],
            decision["observation_age_s"],
        )
        assert np.array_equal(x, np.asarray(decision["features"], np.float32))
        if cell.get("policy"):
            policy = cell["policy"]
            self.ref(policy)
            assert decision["policy_sha256"] == policy["sha256"]
            if policy["sha256"] not in self.policies:
                self.policies[policy["sha256"]] = load_readout(policy["path"], policy["sha256"])
            prediction = readout(self.policies[policy["sha256"]], x, packet=decision["packet"])
            assert all(
                prediction[k] == row["readout"][k]
                for k in ("requested_action", "selected_action", "refusal")
            )
            if prediction["probabilities"] is not None:
                assert (
                    np.max(
                        np.abs(
                            np.asarray(prediction["probabilities"])
                            - row["readout"]["probabilities"]
                        )
                    )
                    <= 1e-6
                )
        self.decisions[row["cell_id"]] = decision
        fields = (
            "dof_pos",
            "dof_vel",
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "applied_joint_action",
            "action_motion_token",
            "reference_g1_qpos",
            "motion_time_s",
        )
        self.prefixes[row["cell_id"]] = {k: np.asarray(payload[k])[:15].copy() for k in fields}
        self.histories[row["cell_id"]] = {
            k: "sha256:" + hashlib.sha256(np.ascontiguousarray(payload[k]).tobytes()).hexdigest()
            for k in fields
        }
        self.raw_checks.append(
            {
                "cell_id": row["cell_id"],
                "pass": row["pass"],
                "scorer_equal": True,
                "contact_sync_error": sync,
                "frames": len(payload["motion_time_s"]),
                "last_recorded_phase_s": float(payload["motion_time_s"][-1]),
                "first_episode_frames": computed["first_episode_frames"],
            }
        )

    def prefix_equal(self, first, second):
        a, b = [self.decisions[r["cell_id"]] for r in (first, second)]
        return (
            all(a[k] == b[k] for k in DECISION_FIELDS)
            and paired_prefix(
                self.prefixes[first["cell_id"]], self.prefixes[second["cell_id"]], 0.3
            )["exact_match"]
        )

    def reusable(self, first, second):
        """Compare physical configuration and recorded history; omit policy metadata only."""
        a, b = [self.contexts[r["cell_id"]] for r in (first, second)]
        fields = (
            "beam",
            "runtime_seed",
            "decision_time_s",
            "delay_s",
            "motion",
            "alternate_motion",
            "reference",
            "body_mode",
            "scene_start_xyz_expected",
        )
        differences = [k for k in fields if a["cell"].get(k) != b["cell"].get(k)]
        if a["cell"]["scene"]["sha256"] != b["cell"]["scene"]["sha256"]:
            differences.append("scene_hash")
        for k in ("checkpoint", "rollout_driver", "manifest_driver", "python"):
            if a["implementation"][k] != b["implementation"][k]:
                differences.append("implementation." + k)

        def clean(cell):
            return [v for v in cell["hydra_overrides"] if "learned_policy_" not in v]

        if clean(a["cell"]) != clean(b["cell"]):
            differences.append("non_policy_hydra_overrides")
        if a["execution_policy"]["runtime"] != b["execution_policy"]["runtime"]:
            differences.append("runtime")
        if not self.prefix_equal(first, second):
            differences.append("recorded_prefix_or_decision")
        if first["bank"]["sha256"] != second["bank"]["sha256"]:
            differences.append("bank")
        return differences


def fitted_composition(fits, pairs):
    by_id = {p["group_id"]: p for p in pairs}
    result = {}
    for f in fits["fits"]:
        if f.get("fold") is not None:
            continue
        ids = f["training_ids"]
        if f["arm"] in result:
            continue
        chosen = [by_id[k] for k in ids]
        c = Counter("".join(str(int(v)) for v in p["outcomes"]) for p in chosen)
        result[f["arm"]] = {
            "groups": len(ids),
            "outcomes": {k: c[k] for k in ("00", "01", "10", "11")},
            "training_ids": ids,
            "reported_bce": f["training_bce"],
            "fit": f,
        }
    return result


def acquisition(proposals, pairs):
    by_id = {p["group_id"]: p for p in pairs}
    result = []
    for arm in ARMS:
        for source in (41001, 41002, 41003):
            rows = [r for r in proposals["rows"] if r["arm"] == arm and r["source"] == source]
            if not rows:
                continue
            assigned = [r for r in rows if r["assigned"]]
            accepted = [r for r in assigned if r["eligible"]]
            useful = [r for r in accepted if by_id[r["group_id"]]["outcomes"] == [False, True]]
            cost = next(c for c in proposals["costs"] if c["arm"] == arm and c["source"] == source)
            result.append(
                {
                    "arm": arm,
                    "source": source,
                    "proposed_outputs": len(rows),
                    "eligible_outputs": sum(r["eligible"] for r in rows),
                    "assigned_slots": len(assigned),
                    "accepted_selected_groups": len(accepted),
                    "physically_labeled_groups": sum(r["group_id"] in by_id for r in rows),
                    "useful_contrasts": len(useful),
                    "useful_with_proposal_ray_hits": sum(
                        bool(r["visibility_proposal_hits"]) for r in useful
                    ),
                    "selected_with_proposal_ray_hits": sum(
                        bool(r["visibility_proposal_hits"]) for r in accepted
                    ),
                    "nominal_critical": sum(r.get("nominal_critical", False) for r in rows),
                    "nominal_critical_with_proposal_ray_hits": sum(
                        r.get("nominal_critical", False) and bool(r["visibility_proposal_hits"])
                        for r in rows
                    ),
                    "refusal_reasons_all_outputs": dict(
                        Counter(x for r in rows for x in r["reasons"])
                    ),
                    "cost": cost,
                }
            )
    return result


def primary(rows):
    traversal = [r for r in rows if r["suite"] == "traversal"]
    result = {"per_arm": {}, "registered_pairs": [compare(traversal, a, b) for a, b in PAIRS]}
    for arm in sorted({r["arm"] for r in traversal}):
        rr = [r for r in traversal if r["arm"] == arm]
        result["per_arm"][arm] = {
            **counts(rr),
            "per_carrier": grouped(rr, "source"),
            "per_layout": grouped(rr, "layout"),
            "per_seed": grouped(rr, "physics_seed"),
        }
    result["backgrounds"] = {
        suite: {
            arm: counts([r for r in rows if r["suite"] == suite and r["arm"] == arm])
            for arm in sorted({r["arm"] for r in rows})
        }
        for suite in sorted({r["suite"] for r in rows} - {"traversal"})
    }
    result["clustered_descriptions"] = []
    for a, b in PAIRS:
        for cluster in ("source", "layout"):
            diffs = []
            for k in sorted({r[cluster] for r in traversal}):
                rr = [r for r in traversal if r[cluster] == k]
                comparison = compare(rr, a, b)
                c = comparison["counts"]
                diffs.append(
                    {
                        "cluster": k,
                        "net_passages": c["only_first"] - c["only_second"],
                        "paired_conditions": comparison["paired_conditions"],
                    }
                )
            positive = sum(d["net_passages"] > 0 for d in diffs)
            negative = sum(d["net_passages"] < 0 for d in diffs)
            result["clustered_descriptions"].append(
                {
                    "first": a,
                    "second": b,
                    "cluster_axis": cluster,
                    "post_hoc": True,
                    "differences": diffs,
                    "positive": positive,
                    "negative": negative,
                    "ties": len(diffs) - positive - negative,
                    "sign_test_p_if_clusters_independent": (
                        float(binomtest(positive, positive + negative).pvalue)
                        if positive + negative
                        else None
                    ),
                    "limitation": (
                        "Descriptive sensitivity only; reused layouts and three development carriers "
                        "are not independent population draws"
                    ),
                }
            )
    return result


def empirical_aliases(x, y, ids, precision=None):
    """Only exact complete-feature collisions justify the empirical entropy bound."""
    keys = defaultdict(list)
    xx = np.round(x, precision) if precision is not None else x
    for i, row in enumerate(xx):
        keys[tuple(float(v) for v in row)].append(i)
    collisions, entropy, errors = [], 0.0, 0
    for indices in keys.values():
        yy = y[indices]
        if len(np.unique(yy, axis=0)) > 1:
            collisions.append(
                {
                    "ids": [ids[i] for i in indices],
                    "labels": yy.astype(int).tolist(),
                    "max_feature_difference": float(np.ptp(x[indices], axis=0).max()),
                }
            )
        mean = yy.mean(0)
        for p in mean:
            if 0 < p < 1:
                entropy += len(indices) * float(-p * np.log(p) - (1 - p) * np.log(1 - p))
        errors += int(np.minimum(yy.sum(0), len(yy) - yy.sum(0)).sum())
    return {
        "unique_observations": len(keys),
        "conflicting_classes": collisions,
        "bce_infimum_for_identical_features": entropy / y.size if precision is None else None,
        "head_error_minimum": errors if precision is None else None,
        "rounded_decimal_places": precision,
    }


def fit_diagnostics(audit, fits, pairs, proposals, replay=True):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    by_id = {p["group_id"]: p for p in pairs}
    props = {p["group_id"]: p for p in proposals["rows"]}
    diagnostics = {}
    for arm, composition in fits.items():
        f, ids = composition["fit"], composition["training_ids"]
        p = [by_id[k] for k in ids]
        x = np.asarray([r["features"] for r in p], np.float32)
        y = np.asarray([r["outcomes"] for r in p], np.float32)
        model = Path(f["model"]["path"]).with_suffix(".npz")
        audit.pin(model)
        audit.ref(f["model"])
        with np.load(model) as data:
            saved = {k: data[k] for k in data.files}
        w = torch.tensor(saved["weights"], requires_grad=True)
        bias = torch.tensor(saved["bias"], requires_grad=True)
        tx = torch.tensor(x) / torch.tensor(saved["scale"])
        logits = tx @ w + bias
        per_head = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor(y), reduction="none"
        )
        bce = per_head.mean()
        penalty = 0.01 * w.square().mean()
        objective = bce + penalty
        objective.backward()
        assert abs(bce.item() - f["training_bce"]) < 1e-7
        reconstructed = None
        if replay:
            check, _, loss = low_capacity(x, y, x)
            reconstructed = {k: float(np.abs(check[k] - saved[k]).max()) for k in saved}
            assert all(v == 0 for v in reconstructed.values()), (arm, reconstructed)
            assert loss == f["training_bce"]
        probabilities = logits.detach().sigmoid().numpy()
        errors = (probabilities >= 0.5) != y
        visibility = []
        for r in p:
            d = audit.decisions[r["row_ids"][0]]
            visibility.append(any(ray["hit"] for ray in d["packet"]["rays"]))
        details = []
        for i, ident in enumerate(ids):
            details.append(
                {
                    "group_id": ident,
                    "source": p[i]["source"],
                    "outcomes": y[i].astype(int).tolist(),
                    "probabilities": probabilities[i].tolist(),
                    "bce": float(per_head[i].detach().mean()),
                    "head_errors": int(errors[i].sum()),
                    "actual_upper_ray_visible": visibility[i],
                    "proposal_visible": (
                        bool(props[ident]["visibility_proposal_hits"]) if ident in props else None
                    ),
                }
            )
        distances = []
        for i, j in itertools.combinations(range(len(x)), 2):
            if np.array_equal(y[i], y[j]):
                continue
            delta = np.abs((x[i] - x[j]) / saved["scale"])
            distances.append(
                {
                    "first": ids[i],
                    "second": ids[j],
                    "scaled_linf": float(delta.max()),
                    "scaled_l2": float(np.linalg.norm(delta)),
                }
            )
        diagnostics[arm] = {
            "groups": len(x),
            "labels": int(y.size),
            "positives_per_head": y.sum(0).astype(int).tolist(),
            "bce": bce.item(),
            "weight_penalty": penalty.item(),
            "objective": objective.item(),
            "gradient_linf": max(float(w.grad.abs().max()), float(bias.grad.abs().max())),
            "loss_reduction": "unweighted mean over groups and both heads; no masked or duplicate ID rows",
            "scale": saved["scale"].tolist(),
            "replay_max_errors": reconstructed,
            "head_errors": int(errors.sum()),
            "groups_with_any_head_error": int(errors.any(1).sum()),
            "exact_aliasing": empirical_aliases(x, y, ids),
            "near_aliasing": [empirical_aliases(x, y, ids, precision) for precision in (6, 4, 2)],
            "nearest_conflicting_pairs": sorted(distances, key=lambda r: r["scaled_linf"])[:10],
            "by_actual_visibility": {
                str(visible): {
                    "groups": int(sum(v == visible for v in visibility)),
                    "bce": (
                        float(per_head.detach().numpy()[np.asarray(visibility) == visible].mean())
                        if visible in visibility
                        else None
                    ),
                    "head_errors": int(errors[np.asarray(visibility) == visible].sum()),
                }
                for visible in (False, True)
            },
            "groups_detail": details,
        }
    return diagnostics


def command_table(audit, rows):
    by_condition = defaultdict(list)
    for r in rows:
        if r["suite"] == "traversal" and r["physics_seed"] == 8511:
            by_condition[condition(r)].append(r)
    records, mismatches, repeated = [], [], []
    for key, rr in sorted(by_condition.items()):
        baseline = next(r for r in rr if r["arm"] == "uniform" and r["cell_id"].startswith("nom_"))
        for r in rr:
            differences = audit.reusable(baseline, r)
            if differences:
                mismatches.append(
                    {"first": baseline["cell_id"], "second": r["cell_id"], "fields": differences}
                )
        outcomes, selected = [], []
        for action in (0, 1):
            subset = [r for r in rr if r["action"] == action]
            if not subset:
                outcomes.append(None)
                selected.append([])
                continue
            assert len({r["pass"] for r in subset}) == 1, (key, action)
            outcomes.append(subset[0]["pass"])
            selected.append([r["cell_id"] for r in subset])
            reference = audit.histories[subset[0]["cell_id"]]
            for other in subset[1:]:
                repeated.append(
                    {
                        "first": subset[0]["cell_id"],
                        "second": other["cell_id"],
                        "same_action": action,
                        "passage_agrees": True,
                        "trajectory_fields_identical": reference
                        == audit.histories[other["cell_id"]],
                    }
                )
        records.append(
            {
                "source": key[0],
                "layout": key[1],
                "physics_seed": key[2],
                "outcomes": outcomes,
                "row_ids_by_action": selected,
                "beam": baseline["beam"],
            }
        )
    return {
        "conditions": records,
        "missing_actions": sum(v is None for r in records for v in r["outcomes"]),
        "reuse_mismatches": mismatches,
        "same_command_repeats": repeated,
        "post_hoc": True,
        "hidden_state_limit": (
            "Exact recorded histories and pinned deterministic runtime; "
            "hidden state was not independently snapshotted"
        ),
    }


def simple_policies(rows, command_lookup):
    commands = {condition(r): r for r in command_lookup["conditions"]}
    policies = {}
    for arm in (*ARMS, "scripted_rays", "privileged_geometry"):
        rr = [
            r
            for r in rows
            if r["arm"] == arm
            and r["suite"] == "traversal"
            and r["physics_seed"] == 8511
            and (r["cell_id"].startswith("nom_") if arm in ARMS else True)
        ]
        policies[arm] = {
            **counts(rr),
            "per_carrier": grouped(rr, "source"),
            "per_layout": grouped(rr, "layout"),
        }
        for outcome, label in (
            ([True, True], "both_pass"),
            ([False, False], "both_fail"),
            ([False, True], "d040_only"),
            ([True, False], "walk_only"),
        ):
            policies[arm][label] = counts(
                [r for r in rr if commands[condition(r)]["outcomes"] == outcome]
            )
    if not command_lookup["missing_actions"] and not command_lookup["reuse_mismatches"]:
        for name, action in (("always_walk", 0), ("always_d040", 1)):
            policies[name] = {
                "n": len(commands),
                "pass": sum(r["outcomes"][action] for r in commands.values()),
                "d040_requests": len(commands) * action,
                "refusals": 0,
                "post_hoc_offline_lookup": True,
                "both_pass": {
                    "n": sum(r["outcomes"] == [True, True] for r in commands.values()),
                    "d040_requests": action
                    * sum(r["outcomes"] == [True, True] for r in commands.values()),
                },
                "both_fail": {
                    "n": sum(r["outcomes"] == [False, False] for r in commands.values()),
                    "d040_requests": action
                    * sum(r["outcomes"] == [False, False] for r in commands.values()),
                    "refusals": 0,
                },
            }
        policies["hindsight_command_oracle"] = {
            "n": len(commands),
            "pass": sum(any(r["outcomes"]) for r in commands.values()),
            "d040_requests": sum(r["outcomes"] == [False, True] for r in commands.values()),
            "refusals": sum(r["outcomes"] == [False, False] for r in commands.values()),
            "post_hoc_offline_lookup": True,
        }
    return policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DOC / "submission/evidence")
    parser.add_argument("--skip-fit-replay", action="store_true")
    args = parser.parse_args()
    audit = Audit()
    all_rows, studies = [], {}
    for name, (construction, learning, comparison) in STUDIES.items():
        c, learned = DATA / construction, DATA / learning
        regs = [audit.read(p / "registration.json") for p in (c, learned)]
        for reg in regs:
            for ref in reg["references"]:
                audit.pin(ref["path"], ref["sha256"], strict=False)
        label_rows, pairs = audit.batches(c / "prepared.json", name, "labels")
        eval_rows, _ = audit.batches(learned / "evaluation_master.json", name, "evaluation")
        original = audit.read(learned / comparison)
        assert original["rows"] == eval_rows
        fitted = audit.read(learned / "fit.json")
        if name == "nominal":
            pairs += [p for p in studies["robust"]["pairs"] if p["arm"] == "shared"]
        proposals = audit.read(c / "proposals.json")
        composition = fitted_composition(fitted, pairs)
        result = primary(eval_rows)
        assert result["registered_pairs"] == original["comparisons"]
        studies[name] = {
            "construction": construction,
            "learning": learning,
            "registrations": [audit.pin(p / "registration.json") for p in (c, learned)],
            "proposals": proposals,
            "pairs": pairs,
            "composition": composition,
            "primary": result,
            "acquisition": acquisition(proposals, pairs),
            "raw_labels": len(label_rows),
            "raw_evaluations": len(eval_rows),
        }
        if name == "nominal":
            studies[name]["fit_diagnostics"] = fit_diagnostics(
                audit, composition, pairs, proposals, not args.skip_fit_replay
            )
        all_rows.extend(label_rows + eval_rows)
        print(
            f"Reproduced {name}: {len(label_rows)} label commands, {len(eval_rows)} evaluations",
            flush=True,
        )
    audit_path = DATA / "m2s-submission-command-audit-v1/registration.json"
    if audit_path.exists():
        reg = audit.read(audit_path)
        if all((Path(b["directory"]) / "admission.json").exists() for b in reg["batches"]):
            rows, _ = audit.batches(audit_path, "post_hoc_command_audit", "evaluation")
            all_rows.extend(rows)
    evaluation_rows = [r for r in all_rows if "layout" in r]
    lookup = command_table(audit, evaluation_rows)
    comparators = simple_policies(evaluation_rows, lookup)
    envelope = audit.read(DATA / "m2s-envelope-tradeoff-v1/envelope-tradeoff.json")
    records = []
    for r in all_rows:
        context = audit.contexts[r["cell_id"]]
        records.append(
            {
                k: r[k]
                for k in (
                    "cell_id",
                    "group_id",
                    "source",
                    "physics_seed",
                    "arm",
                    "pass",
                    "action",
                    "reset_count",
                    "fall_observed",
                    "entry_logged",
                    "return_logged",
                    "command_executed",
                    "tracker_outcome",
                    "maximum_beam_normal_force_n_through_passage",
                )
            }
            | {
                "suite": context["cell"]["suite"],
                "layout": r.get("layout"),
                "decision": r["decision"],
                "trajectory": r["trajectory"],
                "physics": r["physics"],
                "manifest": context["manifest"],
                "result": context["result"],
                "configuration_sha256": "sha256:"
                + hashlib.sha256(json.dumps(context["cell"], sort_keys=True).encode()).hexdigest(),
                "controller": context["implementation"]["checkpoint"],
                "inherited_source_commit": context["implementation"].get("source_commit"),
                "policy": context["cell"].get("policy"),
            }
        )
    history = []
    for p in sorted(DOC.glob("*.md")):
        ref = audit.pin(p)
        history.append(
            {
                **ref,
                "title": p.read_text().splitlines()[0],
                "role": (
                    "historical_result" if "RESULT" in p.name else "protocol_or_project_document"
                ),
            }
        )
    outputs = {
        "studies": studies,
        "command_lookup": lookup,
        "comparators": comparators,
        "envelope": envelope,
        "costs": audit.batch_costs,
        "raw_checks": audit.raw_checks,
        "seed_level_rows": records,
    }
    dump(args.out / "analysis.json", outputs)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_hoc_audit": True,
        "original_experiments_unchanged": True,
        "analysis_script": audit.pin(Path(__file__)),
        "audit_checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "inputs": list(audit.files.values()),
        "expected_pin_checks": audit.pin_checks,
        "historical_documents": history,
        "rows": records,
        "unresolved_pin_checks": [r for r in audit.pin_checks if not r["matches"]],
        "analysis_output": {
            "path": str(args.out / "analysis.json"),
            "sha256": digest(args.out / "analysis.json"),
        },
        "scope": (
            "Finite evaluation bank on three inspected development carriers. "
            "Post-hoc diagnostics do not convert redesign into an untouched confirmatory study."
        ),
    }
    dump(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "rows_rescored": len(records),
                "pin_mismatches": len(manifest["unresolved_pin_checks"]),
                "missing_commands": lookup["missing_actions"],
                "reuse_mismatches": len(lookup["reuse_mismatches"]),
                "comparators": {k: v["pass"] for k, v in comparators.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
