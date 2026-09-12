#!/usr/bin/env python3
"""Choose a common ridge penalty by registered development-layout cross-validation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (  # noqa: E402
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    fit_timed_schedule_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    expected_feature_names,
    validate_schedule_policy,
)

PENALTIES = [1e-6, 0.01, 0.1, 1.0, 10.0]


def arrays(rows):
    return dict(
        features=np.asarray([r["features"] for r in rows], dtype=float),
        phase_ticks=np.asarray([r["phase_tick"] for r in rows], dtype=int),
        passed=np.asarray([r["pass_labels"] for r in rows], dtype=bool),
        passage_time_s=np.asarray([r["passage_time_s"] for r in rows], dtype=float),
        admitted=np.asarray([r["admitted"] for r in rows], dtype=bool),
        legality=np.asarray([r["legal_mask"] for r in rows], dtype=bool),
    )


def score(model, rows):
    data = arrays(rows)
    if not np.isfinite(data["features"]).all():
        raise ValueError("finite held-out sensor features required")
    regret, eligible = physical_regret(
        data["passed"], data["passage_time_s"], data["admitted"], data["legality"]
    )
    measured = []
    for index, row in enumerate(rows):
        if not eligible[index]:
            continue
        phase = np.flatnonzero(model["phase_ticks"] == row["phase_tick"])
        if len(phase) != 1:
            raise ValueError("every held-out decision phase needs a trained head")
        phase = int(phase[0])
        legal = data["legality"][index]
        if np.any(legal & ~model["trained_mask"][phase]):
            raise ValueError("held-out legal option has no training evidence")
        values = ((data["features"][index] - model["mean"][phase]) / model["std"][phase]) @ model[
            "weights"
        ][phase] + model["bias"][phase]
        if not np.isfinite(values[legal]).all():
            raise ValueError("finite legal held-out predictions required")
        choice = int(np.where(legal, values, -np.inf).argmax())
        measured.append(
            dict(
                phase_tick=row["phase_tick"],
                selected_immediate_action=choice,
                measured_teacher_regret=float(regret[index, choice]),
                selected_continuation_passed=bool(data["passed"][index, choice]),
            )
        )
    if not measured:
        raise ValueError("held-out layout has no consequential complete teacher decisions")
    return dict(
        rows=measured,
        excluded_phase_indices=np.flatnonzero(~eligible).tolist(),
        mean_teacher_regret=float(np.mean([r["measured_teacher_regret"] for r in measured])),
        failure_choice_fraction=float(
            np.mean([not r["selected_continuation_passed"] for r in measured])
        ),
    )


def prepare(batch, out):
    registration = json.loads((batch / "registration.json").read_text())
    children = registration["children"]
    if len(children) < 3:
        raise ValueError("at least three registered development layouts required")
    manifests = [
        json.loads(checked(Path(c["manifest"]["path"]), c["manifest"]["sha256"]).read_text())
        for c in children
    ]
    if (
        any(m["split"] != "development" for m in manifests)
        or len({m["scene_definition"]["sha256"] for m in manifests}) != len(manifests)
        or len({m["registry"]["sha256"] for m in manifests}) != 1
    ):
        raise ValueError("distinct development layouts with one common registry required")
    scenes = [
        json.loads(
            checked(
                Path(m["scene_definition"]["path"]), m["scene_definition"]["sha256"]
            ).read_text()
        )
        for m in manifests
    ]
    geometry_keys = [
        json.dumps({k: s[k] for k in ("beams", "beam_collision_enabled")}, sort_keys=True)
        for s in scenes
    ]
    if len(set(geometry_keys)) != len(scenes) or len({s["scene"]["sha256"] for s in scenes}) != len(
        scenes
    ):
        raise ValueError("aliased native geometry may not cross development folds")
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
            schema="motion2scene_schedule_development_ridge_selection_v1",
            batch=artifact(batch / "registration.json"),
            manifests=[c["manifest"] for c in children],
            registry=manifests[0]["registry"],
            penalties=PENALTIES,
            implementation=artifact(Path(__file__)),
            source_closure=sources,
            fold_rule="leave one complete development layout out; all its phases stay together",
            selection_rule=(
                "minimum mean layout failure-choice fraction, then mean layout teacher regret, "
                "then larger penalty"
            ),
            purpose="common learner selection before primary acquisition; not constructor-specific tuning",
            scope="offline finite-teacher continuation proxy; actual student traversal remains unmeasured",
        ),
    )


def run(out):
    registration = json.loads((out / "registration.json").read_text())
    checked(Path(__file__), registration["implementation"]["sha256"])
    for ref in registration["source_closure"]:
        checked(Path(ref["path"]), ref["sha256"])
        checked(Path(ref["snapshot"]["path"]), ref["sha256"])
    ref = registration["registry"]
    bank = load_verified_registry(Path(ref["path"]), ref["sha256"])
    groups = []
    for item in registration["manifests"]:
        path = checked(Path(item["path"]), item["sha256"])
        groups.append(audit_collection(path.parent / "result.json", bank, ref))
    candidates = []
    for penalty in registration["penalties"]:
        folds = []
        for held, group in enumerate(groups):
            training = [
                r
                for i, g in enumerate(groups)
                if i != held
                for r in g["targets"]
                if r.get("features") is not None
            ]
            try:
                model, _ = fit_timed_schedule_policy(
                    bank,
                    feature_names=expected_feature_names(len(bank.option_ids)),
                    l2=penalty,
                    **arrays(training),
                )
                validate_schedule_policy(model, bank)
                assessment = score(
                    model, [r for r in group["targets"] if r.get("features") is not None]
                )
                folds.append(dict(collection=group["collection"], **assessment, status="scored"))
            except ValueError as error:
                folds.append(
                    dict(
                        collection=group["collection"],
                        status="insufficient_evidence",
                        error=str(error),
                    )
                )
        complete = all(f["status"] == "scored" for f in folds)
        candidates.append(
            dict(
                l2=penalty,
                folds=folds,
                complete=complete,
                mean_layout_teacher_regret=(
                    float(np.mean([f["mean_teacher_regret"] for f in folds])) if complete else None
                ),
                mean_layout_failure_choice_fraction=(
                    float(np.mean([f["failure_choice_fraction"] for f in folds]))
                    if complete
                    else None
                ),
            )
        )
    eligible = [c for c in candidates if c["complete"]]
    best = (
        min(
            eligible,
            key=lambda c: (
                c["mean_layout_failure_choice_fraction"],
                c["mean_layout_teacher_regret"],
                -c["l2"],
            ),
        )
        if eligible
        else None
    )
    write_new(
        out / "result.json",
        dict(
            registration=artifact(out / "registration.json"),
            candidates=candidates,
            selected_l2=None if best is None else best["l2"],
            source_physics_steps=sum(g["physics_steps"] for g in groups),
            new_physics_steps=0,
            source_collections=[g["collection"] for g in groups],
            scope=registration["scope"],
        ),
    )
    print(json.dumps(dict(selected_l2=None if best is None else best["l2"], folds=len(groups))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.batch, args.out)
    else:
        run(args.out)
