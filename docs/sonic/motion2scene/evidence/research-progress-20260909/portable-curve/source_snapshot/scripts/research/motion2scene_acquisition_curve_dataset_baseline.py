#!/usr/bin/env python3
"""Reconstruct declared acquisition curves from portable records and NumPy.

Supports the immutable M2 release and the later five-arm release format. Historical
student outcomes stay with their generating policies. This is offline validation,
not a new physical policy evaluation or a refreshed physical-gap measurement.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_acquisition_dataset_baseline import (  # noqa: E402
    arrays,
    bound,
    compare_models,
    expected_feature_names,
    fit_timed_schedule_policy,
    inspect_package,
    measured_task_yield,
    prefix_targets,
    read_json,
    relative_path,
    sha256,
    validate_schedule_policy,
)
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    encounter_replay_weights,
)

CURVE_SCHEMA = "motion2scene_acquisition_curve_portable_release_v1"
ORIGINAL_SCHEMA = "motion2scene_acquisition_portable_release_v1"
ARMS = ("uniform", "target_only", "analytic_contrast", "observation_curriculum")
SEEDS = (93201, 93202, 93203)


def artifact_identity(ref):
    """Export manifests additionally record size; source identities bind path/hash."""
    return {key: ref[key] for key in ("path", "sha256")}


def hashable_coverage(value):
    """Restore tuple coverage identities after the sidecar's JSON round trip."""
    return tuple(hashable_coverage(v) for v in value) if isinstance(value, list) else value


def verify_history_snapshot(release, registration):
    folder = release / "historical_startup"
    path = folder / "history.json"
    if "sha256:" + sha256(path) != registration["history"]["sha256"]:
        raise ValueError("retained startup history differs from its registered snapshot")
    history = read_json(path)
    if history["schema"] != "motion2scene_portable_historical_startup_v1":
        raise ValueError("the original startup assessment and bootstrap provenance are required")
    for item in history["sources"].values():
        bound(folder, dict(path=item["file"], sha256=item["source"]["sha256"]))
    return history


def validate_index(index, plan):
    legacy = index["schema"] == ORIGINAL_SCHEMA
    budget = index["budget"]
    arms = ARMS if legacy else (*ARMS, "reference_contrast")
    if legacy:
        if budget != 2:
            raise ValueError("the original immutable release is the M2 comparison")
        checkpoints = [1, 2]
    else:
        if (
            index["schema"] != CURVE_SCHEMA
            or type(budget) is not int
            or budget not in (8, 16, 32)
            or plan["schema"] != "motion2scene_expanded_acquisition_v1"
            or budget not in plan["checkpoints"]
        ):
            raise ValueError("a declared five-arm M8/M16/M32 acquisition release is required")
        checkpoints = [n for n in (8, 16, 32) if n <= budget]
        if index["reported_checkpoints"] != checkpoints:
            raise ValueError("reported checkpoints differ from the fixed learning curve")
    expected = {f"seed{seed}_{arm}" for seed in SEEDS for arm in arms}
    corpora = index["corpora"]
    if (
        len(corpora) != len(expected)
        or {c["run_id"] for c in corpora} != expected
        or len(plan["runs"]) != len(expected)
        or {r["run_id"] for r in plan["runs"]} != expected
        or any([m["checkpoint"] for m in c["models"]] != list(range(budget + 1)) for c in corpora)
    ):
        raise ValueError("every assigned corpus and historical model slot is required")
    for run in plan["runs"]:
        seed = run.get("seed", run.get("physics_seed"))
        if (
            run["arm"] not in arms
            or seed not in SEEDS
            or run["run_id"] != f"seed{seed}_{run['arm']}"
        ):
            raise ValueError("registered corpus seed and construction arm differ from its identity")
    episodes = 7 + 8 * budget
    if (
        index["total_episodes"] != episodes * len(expected)
        or any(c["counts"]["episodes"] != episodes for c in corpora)
        or index["total_physics_steps"] != sum(c["counts"]["physics_steps"] for c in corpora)
        or any(
            c["counts"]["physics_steps"] != c["physical_cost"]["total_recorded_steps"]
            for c in corpora
        )
    ):
        raise ValueError("all actual acquisition episodes and measured costs must be retained")
    return budget, checkpoints


def student_bindings(package, originals):
    budget = len(originals) - 1
    manifest = read_json(package / "manifest.json")
    if len(manifest["source_results"]) != 2 * budget + 1:
        raise ValueError("complete teacher and actual student prefix required")
    result = []
    for encounter in range(1, budget + 1):
        source = read_json(
            package / "provenance" / f"study_{budget + encounter:03d}" / "manifest.json"
        )
        expected = originals[encounter - 1]["policy"]
        if (
            source["policy"] != expected
            or len(source["cells"]) != 1
            or source["cells"][0]["timed_schedule_mode"] != "learned"
        ):
            raise ValueError("actual student visit has a different generating historical policy")
        result.append(dict(round=encounter, generating_checkpoint=encounter - 1, policy=expected))
    return result


def measured_curve(package, budget, checkpoints):
    manifest = read_json(package / "manifest.json")
    teachers = read_json(package / "teachers.json")
    if len(teachers) != budget + 1 or len(manifest["source_results"]) != 2 * budget + 1:
        raise ValueError("complete ordered bootstrap/teacher/student prefix required")
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    rows = []
    for checkpoint in checkpoints:
        if not 1 <= checkpoint <= budget:
            raise ValueError("curve point lies outside the acquired prefix")
        sources = {
            manifest["source_results"][i]["sha256"]
            for i in [*range(checkpoint + 1), *range(budget + 1, budget + checkpoint + 1)]
        }
        tasks = []
        for teacher in teachers[1 : checkpoint + 1]:
            outcomes = {
                episodes[e]["configured_forced_option_id"]: episodes[e]["assessment"]["outcome"][
                    "task_outcome"
                ]
                for e in teacher["episode_ids"]
            }
            if len(outcomes) != 7:
                raise ValueError("every assigned complete schedule must remain in task yield")
            tasks.append(measured_task_yield(outcomes))
        selected = [e for e in episodes.values() if e["collection_sha256"] in sources]
        if len(selected) != 7 + 8 * checkpoint:
            raise ValueError("prefix cost omitted or duplicated a bootstrap, teacher or student")
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


def select_weighted_rows(rows, weights):
    """Mirror the native fitter's filtering, preserving all original slot indices."""
    if weights is None:
        return [r for r in rows if r.get("available", True)], None
    weights = np.asarray(weights, float)
    if weights.shape != (len(rows),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("finite nonnegative weights for every historical target slot required")
    selected, values = [], []
    for row, weight in zip(rows, weights, strict=True):
        if not row.get("available", True):
            if weight != 0:
                raise ValueError("unavailable phase cannot receive replay supervision")
            continue
        if weight == 0:
            if (
                row["complete_legal_action_table"]
                and row["teacher_action"] is not None
                and sum(row["legal_mask"]) >= 2
            ):
                raise ValueError("zero weighting removed a complete consequential target")
            continue
        selected.append(row)
        values.append(weight)
    return selected, np.asarray(values, float)


def historical_weights(registration, result, files, groups, originals, student_sources):
    schema = registration["schema"]
    if schema == "motion2scene_timed_schedule_training_v1":
        ref = registration["replay_weights"]
        audit = result["replay_audit"]
        if ref is None:
            if audit is not None:
                raise ValueError("uniform historical fit unexpectedly contains replay weighting")
            return None
        if (
            audit is None
            or audit["weights_artifact"] != ref
            or "sha256:" + sha256(files["replay_weights.json"]) != ref["sha256"]
        ):
            raise ValueError("historical replay weights lost their original artifact binding")
        return audit["weights_in_group_target_order"]
    if schema != "motion2scene_expanded_fit_v1":
        raise ValueError("unsupported historical fitting schema")
    if registration["weighting"] == "uniform_per_phase":
        if "replay.json" in files:
            raise ValueError("uniform expanded fit unexpectedly contains replay weighting")
        return None
    if registration["weighting"] != "historical_observation_gap":
        raise ValueError("the registered historical physical-gap weighting must be retained")
    replay = read_json(files["replay.json"])
    proof = read_json(files["replay_verification.json"])
    if (
        proof["original_sidecar"]["sha256"] != "sha256:" + sha256(files["replay.json"])
        or proof["teacher_collections"] != [g["collection"] for g in groups]
        or proof["historical_outcomes_independently_reaudited"] is not True
        or proof["new_physics_steps"] != 0
        or replay["signal"] != "historical measured generating-policy outcomes"
    ):
        raise ValueError(
            "expanded sidecar requires its exact export-time physical re-audit binding"
        )
    records = replay["records"]
    expected = [(i, target) for i, group in enumerate(groups) for target in group["targets"]]
    if len(records) != len(expected):
        raise ValueError("expanded replay lost historical phase slots")
    for row, (index, target) in zip(records, expected, strict=True):
        if (
            row["teacher_collection"] != groups[index]["collection"]
            or row["phase_tick"] != target["phase_tick"]
            or row["generating_model"] != (None if index == 0 else originals[index - 1]["policy"])
            or row["student_collection"] != (None if index == 0 else student_sources[index - 1])
        ):
            raise ValueError(
                "expanded physical gap belongs to a different historical encounter or policy"
            )
    calculated = encounter_replay_weights(
        [dict(row, coverage_key=hashable_coverage(row["coverage_key"])) for row in records]
    )
    if calculated != replay["weights"]:
        raise ValueError("expanded replay weights differ from their recorded historical signals")
    return calculated["weights"]


def verify_model_provenance(files, original, collections, checkpoint, arm, plan, replay_rule):
    result = read_json(files["result.json"])
    if "sha256:" + sha256(files["result.json"]) != original["sha256"]:
        raise ValueError("historical result snapshot differs from its original identity")
    for name in ("registration", "teachers", "policy"):
        extension = ".npz" if name == "policy" else ".json"
        if "sha256:" + sha256(files[name + extension]) != result[name]["sha256"]:
            raise ValueError("historical model dependency differs from the original result")
    registration = read_json(files["registration.json"])
    if registration["collections"] != collections[: checkpoint + 1]:
        raise ValueError("historical model must retain its complete chronological teacher prefix")
    if registration["schema"] == "motion2scene_expanded_fit_v1":
        if registration["plan"] != plan:
            raise ValueError("expanded model belongs to a different acquisition plan")
        replay = registration["weighting"] == "historical_observation_gap"
        if replay:
            proof = read_json(files["replay_verification.json"])
            if proof["replay_rule"] != replay_rule:
                raise ValueError("expanded replay must preserve the adopted historical rule")
    else:
        replay = registration["replay_weights"] is not None
    # Inherited bootstrap fits are common uniform initializers; later replay
    # fits must preserve their assigned arm, including their physical history.
    expected = arm == "observation_curriculum" and (
        checkpoint > 0 or registration["schema"] == "motion2scene_expanded_fit_v1"
    )
    if replay != expected:
        raise ValueError("historical fitting weights differ from the assigned construction arm")
    return result


def reconstruct_model(bank, targets, files, originals, student_sources):
    result, registration = read_json(files["result.json"]), read_json(files["registration.json"])
    if (
        result["status"] != "complete"
        or result["audit_errors"]
        or registration["l2"] != 10.0
        or registration["allow_measured_tie_initialization"] is not True
    ):
        raise ValueError("the complete common ridge fit and adopted initializer are required")
    groups = read_json(files["teachers.json"])
    rows = prefix_targets(targets, groups, registration["collections"])
    weights = historical_weights(registration, result, files, groups, originals, student_sources)
    rows, weights = select_weighted_rows(rows, weights)
    fitted, fit = fit_timed_schedule_policy(
        bank,
        np.asarray([r["features"] for r in rows]),
        expected_feature_names(7),
        np.asarray([r["phase_tick"] for r in rows]),
        np.asarray([r["pass_labels"] for r in rows], bool),
        np.asarray([r["passage_time_s"] for r in rows], float),
        np.asarray([r["admitted"] for r in rows], bool),
        np.asarray([r["legal_mask"] for r in rows], bool),
        sample_weights=weights,
        l2=10.0,
        allow_measured_tie_initialization=True,
    )
    validate_schedule_policy(fitted, bank)
    return fitted, dict(
        teacher_rows=len(rows),
        sample_weights=None if weights is None else weights.tolist(),
        fit=fit,
        comparison=compare_models(fitted, arrays(files["policy.npz"]), targets),
    )


def run(release, out):
    index = read_json(release / "release.json")
    audit = read_json(release / "source_audit.json")
    plan = read_json(release / "source_plan.json")
    registration = read_json(release / "registration.json")
    if (
        "sha256:" + sha256(release / "source_plan.json") != audit["plan"]["sha256"]
        or "sha256:" + sha256(release / "source_audit.json") != index["source_audit"]["sha256"]
        or "sha256:" + sha256(release / "registration.json") != index["registration"]["sha256"]
        or registration["audit"] != index["source_audit"]
    ):
        raise ValueError("source plan or physical audit snapshot changed")
    budget, checkpoints = validate_index(index, plan)
    snapshots = {c["run_id"]: c for c in registration["corpora"]}
    if len(snapshots) != len(index["corpora"]) or len(registration["corpora"]) != len(snapshots):
        raise ValueError("registration must preserve each separate assigned corpus")
    replay_rule = registration.get("replay_rule")
    if index["schema"] == CURVE_SCHEMA:
        if (
            registration["plan"] != audit["plan"]
            or "sha256:" + sha256(release / "replay_rule.json") != replay_rule["sha256"]
            or "sha256:" + sha256(release / "source_adoption.json")
            != registration["adoption"]["sha256"]
        ):
            raise ValueError("original adoption, expanded plan or replay rule snapshot changed")
        verify_history_snapshot(release, registration)
    if out.resolve().is_relative_to(release.resolve()):
        raise ValueError("reconstruction must not write into the immutable release")
    for item in index["toolkit_files"]:
        bound(release, item)
    out.mkdir(parents=True, exist_ok=False)
    models, curves = [], []
    for corpus in index["corpora"]:
        started = time.monotonic()
        print(json.dumps(dict(status="inspecting_corpus", run_id=corpus["run_id"])), flush=True)
        package = relative_path(release, corpus["dataset"])
        bank, targets, inspection = inspect_package(package, corpus["dataset_manifest_sha256"])
        originals = [read_json(bound(release, m["files"]["result.json"])) for m in corpus["models"]]
        inspection["student_policy_bindings"] = student_bindings(package, originals)
        manifest = read_json(package / "manifest.json")
        snapshot = snapshots[corpus["run_id"]]
        if (
            [m["original"] for m in corpus["models"]] != snapshot["models"]
            or [artifact_identity(r) for r in manifest["source_results"]] != snapshot["collections"]
            or corpus["physical_cost"] != snapshot["physical_cost"]
            or corpus["counts"] != manifest["counts"]
        ):
            raise ValueError(
                "portable corpus differs from its registered chronological acquisition"
            )
        students = [artifact_identity(r) for r in manifest["source_results"][budget + 1 :]]
        curves.append(
            dict(run_id=corpus["run_id"], rows=measured_curve(package, budget, checkpoints))
        )
        for model in corpus["models"]:
            files = {key: bound(release, item) for key, item in model["files"].items()}
            arm = corpus["run_id"].split("_", 1)[1]
            verify_model_provenance(
                files,
                model["original"],
                snapshot["collections"],
                model["checkpoint"],
                arm,
                audit["plan"],
                replay_rule,
            )
            fitted, record = reconstruct_model(bank, targets, files, originals, students)
            folder = out / corpus["run_id"] / f"model_{model['checkpoint']:03d}"
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "policy.npz", **fitted)
            record.update(
                run_id=corpus["run_id"],
                checkpoint=model["checkpoint"],
                policy_sha256="sha256:" + sha256(folder / "policy.npz"),
            )
            (folder / "result.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            models.append(record)
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
        schema="motion2scene_acquisition_curve_portable_reconstruction_v1",
        release_sha256="sha256:" + sha256(release / "release.json"),
        budget=budget,
        models=models,
        fits=len(models),
        acquisition_curves=curves,
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
