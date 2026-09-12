"""Fixed-phase value heads on the existing sensor/state and physical-regret table.

Only the three qualified entry ticks have learned decisions. The runtime owns
holding and mandatory guarded recovery; this model does not learn termination.
"""

import hashlib
from pathlib import Path

import numpy as np

from .motion2scene_multi_option_imitation import physical_regret
from .motion2scene_value_imitation import fit_value_policy

SCHEMA = "motion2scene_history_phase_value_v1"
PHASES = np.array([0.2, 0.3, 0.4])
OPTION_IDS = ("neutral", "d040", "d055", "d070", "d085")
QUALIFIED = np.array([[1, 1, 0, 0, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 1]], dtype=bool)
RIDGE_L2 = 1e-6


def phase_index(phase_s):
    if not np.isfinite(phase_s):
        raise ValueError("finite qualified decision phase required")
    matches = np.flatnonzero(abs(PHASES - phase_s) < 1e-8)
    if len(matches) != 1:
        raise ValueError("unsupported decision phase; only .2/.3/.4 are qualified")
    return int(matches[0])


def _check_features(x, names, phases, legality):
    if (
        x.ndim != 2
        or x.shape[1] != len(names)
        or len(set(names)) != len(names)
        or not np.isfinite(x).all()
        or not len(x)
        or len(phases) != len(x)
        or legality.dtype != bool
        or legality.shape != (len(x), 5)
    ):
        raise ValueError("finite named features and aligned boolean legality required")
    indices = np.array([phase_index(t) for t in phases])
    if np.any(legality & ~QUALIFIED[indices]):
        raise ValueError("legal option lacks qualification at this phase")
    if not legality[:, 0].all():
        raise ValueError("phase heads require neutral-state entry decisions")
    # These are existing sensor/controller features, never additional scene truth.
    checks = {"phase_s": phases, "active_skill": np.zeros(len(x))}
    checks.update({f"active_option_{k}": np.full(len(x), float(k == 0)) for k in range(5)})
    checks.update({f"option_{k}_legal": legality[:, k] for k in range(5)})
    for name, expected in checks.items():
        if name in names and not np.allclose(x[:, names.index(name)], expected, rtol=0, atol=2e-8):
            raise ValueError(f"feature {name} disagrees with the decision interface")
    return indices


def fit_phase_value_policy(
    features,
    feature_names,
    option_ids,
    phases_s,
    passed,
    passage_time_s,
    admitted,
    legality,
    *,
    sample_weights=None,
):
    """One fixed ridge fit per phase and observed legal option, no hyperparameter search."""
    if tuple(option_ids) != OPTION_IDS:
        raise ValueError("phase learner requires the exact qualified five-option order")
    x, legal = np.asarray(features, dtype=float), np.asarray(legality)
    names = tuple(feature_names)
    phases = np.asarray(phases_s, dtype=float)
    phase_ids = _check_features(x, names, phases, legal)
    regret, supervised = physical_regret(passed, passage_time_s, admitted, legal)
    replay = np.ones(len(x)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if replay.shape != (len(x),) or not np.isfinite(replay).all() or (replay <= 0).any():
        raise ValueError("finite positive original replay weights required")
    passed, admitted = np.asarray(passed), np.asarray(admitted)
    times = np.asarray(passage_time_s, dtype=float)
    model = {
        "schema_version": np.array(SCHEMA),
        "feature_names": np.array(names),
        "option_ids": np.array(option_ids),
        "classes": np.arange(5),
        "phases_s": PHASES.copy(),
        "qualified_mask": QUALIFIED.copy(),
        "trained_mask": np.zeros((3, 5), dtype=bool),
        "mean": np.zeros((3, x.shape[1])),
        "std": np.ones((3, x.shape[1])),
        "weights": np.zeros((3, x.shape[1], 5)),
        "bias": np.zeros((3, 5)),
        "l2": np.array(RIDGE_L2),
    }
    reports = []
    for phase in range(3):
        rows = np.flatnonzero(phase_ids == phase)
        complete = rows[supervised[rows]]
        if not len(complete):
            raise ValueError(f"phase {PHASES[phase]} has no complete consequential targets")
        columns = np.flatnonzero(legal[complete].any(axis=0))
        if len(columns) < 2:
            raise ValueError("phase requires at least two measured legal alternatives")
        head, report = fit_value_policy(
            x[complete],
            names,
            [option_ids[c] for c in columns],
            passed[np.ix_(complete, columns)],
            times[np.ix_(complete, columns)],
            admitted[np.ix_(complete, columns)],
            legal[np.ix_(complete, columns)],
            l2=RIDGE_L2,
            sample_weights=replay[complete],
        )
        # Completeness is decided on the full physical table before projecting
        # to the phase's observed options. Missing branches never become failures.
        if report["excluded_decision_indices"]:
            raise ValueError("phase projection altered physical target completeness")
        for key in ("mean", "std"):
            model[key][phase] = head[key]
        model["weights"][phase][:, columns] = head["weights"]
        model["bias"][phase, columns] = head["bias"]
        model["trained_mask"][phase, columns] = True
        scaled = (x[complete] - head["mean"]) / head["std"]
        report.pop("fitted_decisions")  # The top-level readout uses global row/option indices.
        report.update(
            phase_s=float(PHASES[phase]),
            recorded_decisions=len(rows),
            recorded_decision_indices=rows.tolist(),
            complete_decision_indices=complete.tolist(),
            excluded_decision_indices=rows[~supervised[rows]].tolist(),
            global_option_indices=columns.tolist(),
            augmented_design_rank=int(np.linalg.matrix_rank(np.c_[scaled, np.ones(len(scaled))])),
        )
        for option_report in report["option_fits"]:
            option_report["option_index"] = int(columns[option_report["option_index"]])
            option_report["recorded_decision_indices"] = complete[
                option_report["recorded_decision_indices"]
            ].tolist()
        reports.append(report)
    validate_phase_model(model, option_ids)
    rows = []
    for index in np.flatnonzero(supervised):
        action, logits = choose_phase_option(model, names, x[index], legal[index], phases[index])
        rows.append(
            {
                "recorded_decision_index": int(index),
                "phase_s": float(phases[index]),
                "action": action,
                "measured_regret": float(regret[index, action]),
                "predicted_regret": [None if value is None else -value for value in logits],
            }
        )
    return model, {
        "method": "three phase-specific per-option ridge fits of measured physical regret",
        "l2": RIDGE_L2,
        "recorded_decisions": len(x),
        "complete_consequential_decisions": int(supervised.sum()),
        "excluded_decision_indices": np.flatnonzero(~supervised).tolist(),
        "phase_fits": reports,
        "fitted_decisions": rows,
        "scope": "finite teacher-table learner development; no physical execution or generalization evidence",
    }


def validate_phase_model(model, option_ids):
    names = tuple(model["feature_names"])
    dimension = len(names)
    shapes = {
        "mean": (3, dimension),
        "std": (3, dimension),
        "weights": (3, dimension, 5),
        "bias": (3, 5),
    }
    if (
        model["schema_version"].item() != SCHEMA
        or tuple(option_ids) != OPTION_IDS
        or tuple(model["option_ids"]) != OPTION_IDS
        or not dimension
        or len(set(names)) != dimension
        or not np.array_equal(model["classes"], np.arange(5))
        or not np.array_equal(model["phases_s"], PHASES)
        or not np.array_equal(model["qualified_mask"], QUALIFIED)
        or model["qualified_mask"].dtype != bool
        or model["trained_mask"].shape != (3, 5)
        or model["trained_mask"].dtype != bool
        or np.any(model["trained_mask"] & ~QUALIFIED)
        or not model["trained_mask"][:, 0].all()
        or (model["trained_mask"].sum(axis=1) < 2).any()
        or model["l2"].shape != ()
        or model["l2"].item() != RIDGE_L2
        or any(
            model[k].shape != shape or not np.isfinite(model[k]).all()
            for k, shape in shapes.items()
        )
        or (model["std"] < 0.05).any()
    ):
        raise ValueError("malformed phase-value policy")
    return model


def load_phase_policy(path, expected_sha256, option_ids):
    raw = Path(path).read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("phase-value policy hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        model = {key: data[key].copy() for key in data.files}
    return validate_phase_model(model, option_ids)


def choose_phase_option(model, names, features, legality, phase_s):
    """Choose at a consequential neutral entry tick; unsupported actions error."""
    if tuple(model["feature_names"]) != tuple(names):
        raise ValueError("phase-value feature schema mismatch")
    features = np.asarray(features, dtype=float)
    legal = np.asarray(legality)
    phase = _check_features(features[None], tuple(names), np.array([phase_s]), legal[None])[0]
    if np.any(legal & ~model["trained_mask"][phase]):
        raise ValueError("legal action has no trained physical value at this phase")
    logits = ((features - model["mean"][phase]) / model["std"][phase]) @ model["weights"][
        phase
    ] + model["bias"][phase]
    action = int(np.argmax(np.where(legal, logits, -np.inf)))
    return action, [
        float(value) if allowed else None for value, allowed in zip(logits, legal, strict=True)
    ]
