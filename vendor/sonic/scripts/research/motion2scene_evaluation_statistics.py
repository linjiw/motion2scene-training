#!/usr/bin/env python3
"""NumPy-only paired statistics for the complete assigned V4 evaluation matrix.

This reader checks report identities and denominators, not simulator evidence or
model weights. Missing outcomes remain assigned and bound all completion claims.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = "motion2scene_reserved_execution_v4"
OUTPUT_SCHEMA = "motion2scene_reserved_statistics_v1"
ARMS = ("uniform", "target_only", "analytic_contrast", "analytic_observation_curriculum")
CORPORA = (93201, 93202, 93203)
EVALUATION_SEEDS = (94301, 94302)
CHECKPOINTS = {"M2": 27416, "M4": 46488}
BASELINES = ("always_walk", "constant_development_selected", "scripted_multi_option")
FAMILY_COUNTS = {
    "short": 4,
    "sustained": 4,
    "early_constraint": 4,
    "short_then_short": 3,
    "short_then_sustained": 3,
}
STATUSES = ("pass", "failure", "technical_missing", "not_run")
DRAWS = 20000
RANDOM_SEED = 202609081822


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def model_identity(value):
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not value["path"]:
        raise ValueError("each learned policy requires a model artifact identity")
    digest = value.get("sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", digest):
        raise ValueError("model artifact requires a complete SHA256 digest")
    return value["path"], digest.removeprefix("sha256:")


def validate_report(value):
    """Accept final receipts or paused event wrappers; require all972 assigned rows."""
    if isinstance(value, dict) and "receipt" in value:
        if value.get("kind") != "evaluation_status":
            raise ValueError("unsupported event wrapper")
        value = value["receipt"]
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("a V4 evaluation receipt is required")
    if value.get("status") not in ("complete", "paused"):
        raise ValueError("receipt must explicitly be complete or paused")
    report = value.get("report", {})
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 972 or report.get("assigned_episodes") != 972:
        raise ValueError("all972 assigned outcomes, including not_run rows, are required")
    layouts, policies, keys, model_paths, model_owners = {}, {}, set(), {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each assignment must be a row object")
        policy, layout, seed = (row.get(k) for k in ("policy_id", "layout_id", "physics_seed"))
        if not all(isinstance(s, str) and s for s in (policy, layout)):
            raise ValueError("nonempty policy and layout identities required")
        if type(seed) is not int or seed not in EVALUATION_SEEDS:
            raise ValueError("evaluation seeds must remain the two fixed repeated measurements")
        key = policy, layout, seed
        if key in keys:
            raise ValueError("duplicate policy/layout/evaluation-seed assignment")
        keys.add(key)
        family = row.get("layout_family")
        if family not in FAMILY_COUNTS or layouts.setdefault(layout, family) != family:
            raise ValueError("inconsistent or unsupported layout family")
        if row.get("variant_id", "nominal") != "nominal":
            raise ValueError("nominal evaluation cannot pool stress variants")
        status, cost = row.get("status"), row.get("successful_passage_time_s")
        if status not in STATUSES:
            raise ValueError("unknown status encoding")
        if status == "pass":
            if type(cost) not in (int, float) or not np.isfinite(cost) or cost <= 0:
                raise ValueError("a passed episode requires finite positive measured passage time")
        elif cost is not None:
            raise ValueError("failed or unknown episodes cannot carry successful passage time")
        if "recorded_physics_steps" in row:
            steps = row["recorded_physics_steps"]
            if steps is not None and (type(steps) is not int or not 0 <= steps <= 1192):
                raise ValueError("invalid measured evaluation step count")
            if status == "not_run" and steps is not None:
                raise ValueError("unlaunched assignments cannot have recorded physics steps")
        fields = (
            "training_arm",
            "acquisition_seed",
            "checkpoint",
            "acquisition_actual_physics_steps",
            "acquisition_assigned_maximum_steps",
        )
        if policy in BASELINES:
            if any(row.get(k) is not None for k in fields) or row.get("model") is not None:
                raise ValueError(
                    "shared baselines cannot be copied into acquired-corpus/model slots"
                )
            metadata = (None,) * 6
        else:
            arm, corpus, checkpoint, actual, maximum = (row.get(k) for k in fields)
            if (
                arm not in ARMS
                or type(corpus) is not int
                or corpus not in CORPORA
                or checkpoint not in CHECKPOINTS
            ):
                raise ValueError("unexpected training arm, corpus or checkpoint")
            if policy != f"{arm}__seed{corpus}__{checkpoint}":
                raise ValueError("policy identifier differs from its training metadata")
            if (
                type(actual) is not int
                or type(maximum) is not int
                or maximum != CHECKPOINTS[checkpoint]
                or not 0 <= actual <= maximum
            ):
                raise ValueError("invalid measured/assigned acquisition budget")
            identity = model_identity(row.get("model"))
            if model_paths.setdefault(identity[0], identity[1]) != identity[1]:
                raise ValueError("one model path has inconsistent digests")
            if model_owners.setdefault(identity[0], policy) != policy:
                raise ValueError("distinct checkpoint/corpus slots require distinct model paths")
            metadata = arm, corpus, checkpoint, actual, maximum, identity
        if policies.setdefault(policy, metadata) != metadata:
            raise ValueError("model or acquisition metadata changes within one policy")
    expected_policies = set(BASELINES) | {
        f"{arm}__seed{corpus}__{checkpoint}"
        for arm in ARMS
        for corpus in CORPORA
        for checkpoint in CHECKPOINTS
    }
    if (
        set(policies) != expected_policies
        or len(layouts) != 18
        or Counter(layouts.values()) != FAMILY_COUNTS
    ):
        raise ValueError(
            "exactly24 learned checkpoints, three shared baselines and18 specified-family layouts required"
        )
    expected = {
        (p, layout, seed)
        for p in expected_policies
        for layout in layouts
        for seed in EVALUATION_SEEDS
    }
    if keys != expected:
        raise ValueError("evaluation matrix has missing or extra pairings")
    for arm in ARMS:
        for corpus in CORPORA:
            before = policies[f"{arm}__seed{corpus}__M2"][3]
            after = policies[f"{arm}__seed{corpus}__M4"][3]
            if before > after or after - before > CHECKPOINTS["M4"] - CHECKPOINTS["M2"]:
                raise ValueError("M2/M4 must be cumulative checkpoints of the same acquisition")
    complete = all(row["status"] in ("pass", "failure") for row in rows)
    if type(report.get("complete")) is not bool or report["complete"] != complete:
        raise ValueError("report completion flag disagrees with its assigned outcomes")
    if value["status"] == "complete" and not complete:
        raise ValueError("unknown outcomes cannot produce a complete receipt")
    return value, sorted(rows, key=lambda r: (r["policy_id"], r["layout_id"], r["physics_seed"]))


def completion_summary(rows):
    counts = Counter(row["status"] for row in rows)
    n = len(rows)
    unknown = counts["technical_missing"] + counts["not_run"]
    return {
        "assigned": n,
        **{status: counts[status] for status in STATUSES},
        "completion_lower": counts["pass"] / n,
        "completion_upper": (counts["pass"] + unknown) / n,
        "complete_outcome_mean": None if unknown else counts["pass"] / n,
        "measured_outcome_denominator": n - unknown,
    }


def variation(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def bootstrap_weights(corpora, layouts, *, draws=DRAWS, seed=RANDOM_SEED):
    """Shared corpus/layout draws; the two evaluation seeds are never resampled."""
    if type(draws) is not int or draws <= 0 or corpora < 1 or layouts < 1:
        raise ValueError("positive bootstrap dimensions required")
    rng = np.random.default_rng(seed)
    c = rng.integers(corpora, size=(draws, corpora))
    layout_indices = rng.integers(layouts, size=(draws, layouts))
    cw = np.eye(corpora)[c].mean(axis=1)
    lw = np.eye(layouts)[layout_indices].mean(axis=1)
    return cw, lw


def weighted_draws(values, corpus_weights, layout_weights):
    """Crossed product weights preserve paired policy and shared-baseline identity."""
    return np.sum((corpus_weights @ values) * layout_weights, axis=1)


def percentile(values):
    return np.quantile(values, [0.025, 0.975], method="linear").tolist()


def paired_comparison(left, right, left_times, right_times, weights, *, shared_right=False):
    """Arrays are corpus/layout/evaluation-seed; NaN is an assigned unknown outcome.

    A shared baseline has only layout/evaluation-seed rows. Broadcasting lets the
    SAME draw contribute once to the corpus-average contrast; it does not give
    the baseline independently resampled copies or increase its unique count.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    lt, rt = np.asarray(left_times, float), np.asarray(right_times, float)
    if left.ndim != 3 or lt.shape != left.shape or left.shape[-1] != 2:
        raise ValueError("left outcomes require corpus/layout/two-seed arrays")
    expected = left.shape[1:] if shared_right else left.shape
    if right.shape != expected or rt.shape != expected:
        raise ValueError("right outcomes do not match the declared shared/paired identities")
    for outcome, time in ((left, lt), (right, rt)):
        if np.any(~(np.isnan(outcome) | (outcome == 0) | (outcome == 1))):
            raise ValueError("outcomes are binary or explicitly unknown")
        if np.any((outcome == 1) & (~np.isfinite(time) | (time <= 0))) or np.any(
            (outcome != 1) & ~np.isnan(time)
        ):
            raise ValueError("costs must exist exactly for successful outcomes")
    known = np.isfinite(left) & np.isfinite(right)
    lower = np.nan_to_num(left, nan=0) - np.nan_to_num(right, nan=1)
    upper = np.nan_to_num(left, nan=1) - np.nan_to_num(right, nan=0)
    complete = bool(known.all())
    cw, lw = weights
    if (
        cw.ndim != 2
        or lw.ndim != 2
        or cw.shape[0] != lw.shape[0]
        or cw.shape[1] != left.shape[0]
        or lw.shape[1] != left.shape[1]
    ):
        raise ValueError("bootstrap weights differ from pairing dimensions")
    if any(
        not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(axis=1), 1)
        for w in (cw, lw)
    ):
        raise ValueError("bootstrap weights must be finite normalized frequencies")
    fixed_cw = np.full_like(cw, 1 / left.shape[0])
    joint = (left == 1) & (right == 1)
    n_joint = int(joint.sum())
    sums = np.where(joint, lt - rt, 0).sum(axis=-1)
    support = joint.sum(axis=-1)
    cost = {
        "mutually_successful_pairs": n_joint,
        "unique_left_successful_episodes_used": n_joint,
        "unique_right_successful_episodes_used": (
            int(joint.any(axis=0).sum()) if shared_right else n_joint
        ),
        "mean_difference_s": None if not n_joint else float(sums.sum() / n_joint),
        "crossed_95_percentile_s": None,
        "layout_only_95_percentile_s": None,
        "scope": "left minus right on matched episodes where BOTH succeed; no unpaired-success comparison",
    }
    completion = {
        "difference_lower": float(lower.mean()),
        "difference_upper": float(upper.mean()),
        "mean_difference": float(lower.mean()) if complete else None,
        "crossed_95_percentile": None,
        "layout_only_95_percentile": None,
    }
    if complete:
        difference = (left - right).mean(axis=-1)
        completion["crossed_95_percentile"] = percentile(weighted_draws(difference, cw, lw))
        completion["layout_only_95_percentile"] = percentile(
            weighted_draws(difference, fixed_cw, lw)
        )
    else:
        completion["interval_withheld_reason"] = (
            "assigned unknown outcomes; retain identification bounds"
        )
    if not complete:
        cost["interval_withheld_reason"] = "assigned unknown outcomes in this comparison"
    elif not n_joint:
        cost["interval_withheld_reason"] = "no mutually successful matched episodes"
    else:
        for name, weights_c in (("crossed", cw), ("layout_only", fixed_cw)):
            denominator = weighted_draws(support, weights_c, lw)
            empty = int((denominator == 0).sum())
            cost[name + "_zero_support_draws"] = empty
            # Do not silently condition a cost interval on nonempty resamples.
            if empty:
                cost[name + "_interval_withheld_reason"] = (
                    "some bootstrap draws have no mutual-success support"
                )
            else:
                cost[name + "_95_percentile_s"] = percentile(
                    weighted_draws(sums, weights_c, lw) / denominator
                )
    per_corpus = []
    for i in range(left.shape[0]):
        mask = joint[i]
        differences = (lt - rt)[i] if not shared_right else lt[i] - rt
        per_corpus.append(
            {
                "corpus_index": i,
                "difference_lower": float(lower[i].mean()),
                "difference_upper": float(upper[i].mean()),
                "mutually_successful_pairs": int(mask.sum()),
                "mean_time_difference_s": float(differences[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "assigned_matched_pairs": int(left.size),
        "complete_matched_pairs": int(known.sum()),
        "unknown_matched_pairs": int((~known).sum()),
        "unique_left_episodes": int(left.size),
        "unique_right_episodes": int(right.size),
        "shared_right_baseline": shared_right,
        "completion": completion,
        "paired_success_time": cost,
        "per_corpus": per_corpus,
        "per_corpus_difference_lower_variation": variation(
            [r["difference_lower"] for r in per_corpus]
        ),
        "per_corpus_difference_upper_variation": variation(
            [r["difference_upper"] for r in per_corpus]
        ),
        "per_corpus_time_difference_variation": {
            "defined_corpora": sum(r["mean_time_difference_s"] is not None for r in per_corpus),
            "statistics": (
                variation(
                    [
                        r["mean_time_difference_s"]
                        for r in per_corpus
                        if r["mean_time_difference_s"] is not None
                    ]
                )
                if any(r["mean_time_difference_s"] is not None for r in per_corpus)
                else None
            ),
            "scope": "descriptive variation of defined per-corpus paired-success means",
        },
    }


def summarize(value, *, draws=DRAWS, seed=RANDOM_SEED):
    receipt, rows = validate_report(value)
    layouts = sorted({row["layout_id"] for row in rows})
    by_policy = {
        p: [r for r in rows if r["policy_id"] == p] for p in sorted({r["policy_id"] for r in rows})
    }
    by_key = {(r["policy_id"], r["layout_id"], r["physics_seed"]): r for r in rows}
    weights = bootstrap_weights(3, 18, draws=draws, seed=seed)
    curves, aggregates, comparisons = [], [], []
    for arm in ARMS:
        for checkpoint in CHECKPOINTS:
            policies = [f"{arm}__seed{c}__{checkpoint}" for c in CORPORA]
            corpus_rows = []
            for corpus, p in zip(CORPORA, policies, strict=True):
                model = by_policy[p][0]
                point = {
                    "training_arm": arm,
                    "acquisition_seed": corpus,
                    "checkpoint": checkpoint,
                    "policy_id": p,
                    "model": model["model"],
                    "acquisition_actual_physics_steps": model["acquisition_actual_physics_steps"],
                    "acquisition_assigned_maximum_steps": model[
                        "acquisition_assigned_maximum_steps"
                    ],
                    **completion_summary(by_policy[p]),
                }
                curves.append(point)
                corpus_rows.append(point)
            aggregates.append(
                {
                    "training_arm": arm,
                    "checkpoint": checkpoint,
                    "corpus_count": 3,
                    "per_corpus": corpus_rows,
                    "completion_lower_across_corpora": variation(
                        [r["completion_lower"] for r in corpus_rows]
                    ),
                    "completion_upper_across_corpora": variation(
                        [r["completion_upper"] for r in corpus_rows]
                    ),
                    "acquisition_actual_physics_steps": variation(
                        [r["acquisition_actual_physics_steps"] for r in corpus_rows]
                    ),
                    "acquisition_assigned_maximum_steps": CHECKPOINTS[checkpoint],
                }
            )

    def tensors(arm, checkpoint=None):
        shared = arm in BASELINES
        ids = [arm] if shared else [f"{arm}__seed{c}__{checkpoint}" for c in CORPORA]
        y = np.full((len(ids), 18, 2), np.nan)
        t = np.full_like(y, np.nan)
        for c, p in enumerate(ids):
            for layout_index, layout in enumerate(layouts):
                for e, eval_seed in enumerate(EVALUATION_SEEDS):
                    row = by_key[p, layout, eval_seed]
                    if row["status"] in ("pass", "failure"):
                        y[c, layout_index, e] = int(row["status"] == "pass")
                    if row["status"] == "pass":
                        t[c, layout_index, e] = row["successful_passage_time_s"]
        return (y[0], t[0]) if shared else (y, t)

    def compare(left, right, left_checkpoint, right_checkpoint):
        ly, lt = tensors(left, left_checkpoint)
        ry, rt = tensors(right, right_checkpoint)
        result = paired_comparison(ly, ry, lt, rt, weights, shared_right=right in BASELINES)
        for item, corpus in zip(result["per_corpus"], CORPORA, strict=True):
            item["acquisition_seed"] = corpus
        comparisons.append(
            {
                "comparison_id": (
                    f"{left}:{left_checkpoint}__minus__{right}:{right_checkpoint or 'fixed'}"
                ),
                "contrast_direction": "left_minus_right",
                "kind": (
                    "checkpoint_vs_shared_baseline"
                    if right in BASELINES
                    else (
                        "within_corpus_checkpoint_update"
                        if left_checkpoint != right_checkpoint
                        else "constructor_at_matched_checkpoint"
                    )
                ),
                "left": left,
                "right": right,
                "left_checkpoint": left_checkpoint,
                "right_checkpoint": right_checkpoint,
                **result,
            }
        )

    for checkpoint in CHECKPOINTS:
        for left, right in ((ARMS[3], ARMS[2]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
            compare(left, right, checkpoint, checkpoint)
        for arm in ARMS:
            for baseline in BASELINES:
                compare(arm, baseline, checkpoint, None)
    for arm in ARMS:
        compare(arm, arm, "M4", "M2")
    strata = {}
    family_by_layout = {r["layout_id"]: r["layout_family"] for r in rows}
    stratum_layouts = {
        "all18_nominal": set(layouts),
        "12_single": {
            layout for layout, f in family_by_layout.items() if not f.startswith("short_then_")
        },
        "6_two_beam": {
            layout for layout, f in family_by_layout.items() if f.startswith("short_then_")
        },
    }
    stratum_layouts.update(
        {
            family: {layout for layout, f in family_by_layout.items() if f == family}
            for family in FAMILY_COUNTS
        }
    )
    for name, subset in stratum_layouts.items():
        strata[name] = {
            p: completion_summary([r for r in selected if r["layout_id"] in subset])
            for p, selected in by_policy.items()
        }
    return {
        "schema": OUTPUT_SCHEMA,
        "input_status": receipt["status"],
        "complete": receipt["report"]["complete"],
        "assigned_outcomes": completion_summary(rows),
        "assigned_rows": rows,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "confidence_level": 0.95,
            "rng": "numpy.random.default_rng / PCG64; corpus draws precede layout draws",
            "corpus_order": list(CORPORA),
            "layout_order": layouts,
            "evaluation_seed_order": list(EVALUATION_SEEDS),
            "corpus_weight_matrix_sha256": hashlib.sha256(weights[0].tobytes()).hexdigest(),
            "layout_weight_matrix_sha256": hashlib.sha256(weights[1].tobytes()).hexdigest(),
            "weight_matrix_dtype": str(weights[0].dtype),
            "crossed_units": (
                "resample 3 acquired-corpus blocks and 18 layout blocks with replacement; "
                "use the same draws across policies and keep both evaluation seeds together"
            ),
            "layout_only_units": (
                "resample 18 layout blocks, averaging the three fixed corpora; "
                "keep both evaluation seeds together"
            ),
            "shared_baselines": (
                "one 36-episode baseline; same layout draw is shared across corpora, "
                "with no independent copies"
            ),
            "cost_zero_support_rule": "withhold interval if any bootstrap draw has zero mutually successful pairs",
        },
        "protocol_alignment": {
            "draw_count_and_seed_match_v4_proposal": draws == DRAWS and seed == RANDOM_SEED,
            "clarified_primary_interval": "crossed_95_percentile; paired corpus/layout resampling",
            "original_proposal_sensitivity": (
                "layout_only_95_percentile holds corpus weights fixed; "
                "both intervals remain descriptive with separate per-corpus variation"
            ),
            "adoption_scope": "Reporting clarification does not itself adopt the evaluation protocol",
        },
        "policy_summaries": {p: completion_summary(selected) for p, selected in by_policy.items()},
        "stratum_summaries": strata,
        "per_corpus_learning_curves": curves,
        "mean_learning_curves": aggregates,
        "paired_comparisons": comparisons,
        "scope": (
            "Statistics of supplied assigned report rows only; model/physical evidence "
            "is not requalified here. No new physics or hidden outcome imputation."
        ),
        "limitations": [
            (
                "Only three independently acquired corpora; descriptive intervals can be unstable "
                "and are not a large-sample guarantee."
            ),
            (
                "M2 and M4 share each acquisition trajectory and are paired checkpoints, "
                "not independent training replicates."
            ),
            "The two evaluation seeds are repeated measurements within layouts, not independent scene samples.",
            (
                "Completion intervals are withheld for comparisons containing assigned unknown "
                "outcomes; lower/upper bounds retain every assignment."
            ),
            (
                "Time differences condition on mutually successful matched episodes; "
                "neither failure times nor mechanical energy are imputed."
            ),
            (
                "No multiplicity correction; these are descriptive paired intervals, "
                "not simultaneous significance claims."
            ),
        ],
    }


def write_tables(out, result):
    curves = result["per_corpus_learning_curves"]
    keys = [
        "training_arm",
        "acquisition_seed",
        "checkpoint",
        "policy_id",
        "acquisition_actual_physics_steps",
        "acquisition_assigned_maximum_steps",
        "assigned",
        "pass",
        "failure",
        "technical_missing",
        "not_run",
        "completion_lower",
        "completion_upper",
        "complete_outcome_mean",
    ]
    with (out / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(curves)
    with (out / "paired_comparisons.csv").open("w", newline="") as handle:
        keys = [
            "left",
            "right",
            "left_checkpoint",
            "right_checkpoint",
            "assigned_matched_pairs",
            "unknown_matched_pairs",
            "difference_lower",
            "difference_upper",
            "mean_difference",
            "crossed_95_percentile",
            "layout_only_95_percentile",
            "mutually_successful_pairs",
            "mean_difference_s",
            "crossed_95_percentile_s",
            "layout_only_95_percentile_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in result["paired_comparisons"]:
            writer.writerow({**row, **row["completion"], **row["paired_success_time"]})


def run(source, out, *, expected_sha256=None, reporting_clarification=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    bound = file_ref(source)
    if expected_sha256 and bound["sha256"].removeprefix("sha256:") != expected_sha256.removeprefix(
        "sha256:"
    ):
        raise ValueError("input receipt digest differs")
    value = json.loads(source.read_text())
    validate_report(value)
    if out == source.parent or source.parent in out.parents or out in source.parents:
        raise ValueError("write statistics outside the immutable input directory")
    out.mkdir(parents=True, exist_ok=False)
    registration = {
        "schema": "motion2scene_evaluation_statistics_registration_v1",
        "input": bound,
        "implementation": file_ref(__file__),
        "numpy_version": np.__version__,
        "draws": DRAWS,
        "seed": RANDOM_SEED,
        "new_physics_steps": 0,
        "reporting_clarification": (
            None if reporting_clarification is None else file_ref(reporting_clarification)
        ),
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=2))
    result = summarize(value)
    result["registration"] = file_ref(out / "registration.json")
    result["protocol_alignment"]["reporting_clarification"] = registration[
        "reporting_clarification"
    ]
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_tables(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--reporting-clarification", type=Path)
    args = parser.parse_args()
    report = run(
        args.input,
        args.out,
        expected_sha256=args.expected_sha256,
        reporting_clarification=args.reporting_clarification,
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "complete": report["complete"],
                "assigned": report["assigned_outcomes"],
                "comparisons": len(report["paired_comparisons"]),
            }
        )
    )
