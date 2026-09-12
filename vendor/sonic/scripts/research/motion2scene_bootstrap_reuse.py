"""Validate explicit reuse of one corpus's completed, unchanged bootstrap.

This supplies no new measurements and never imports a policy or a failed fit.
The acquisition controller separately recomputes the physical outcome rows.
"""

import hashlib
import json
from pathlib import Path


def checked(ref):
    path = Path(ref["path"]).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"].removeprefix("sha256:"):
        raise ValueError("bootstrap reuse artifact differs from its frozen identity")
    return path


def read(ref):
    return json.loads(checked(ref).read_text())


def same(left, right):
    if checked(left) != checked(right):
        raise ValueError("bootstrap reuse artifact does not occupy its original slot")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_bootstrap_reuse(plan, adoption, run):
    """Permit only an adopted exact predecessor bootstrap for the same run.

    Other corpora, current/future encounter teachers, partial measurements and
    successful predecessor models cannot be substituted through this interface.
    """
    declarations = plan.get("bootstrap_reuse", {})
    require(isinstance(declarations, dict), "explicit bootstrap reuse mapping required")
    require(
        declarations == adoption.get("bootstrap_reuse", {}),
        "plan and adoption must bind exactly the same bootstrap reuse declarations",
    )
    require(
        set(declarations).issubset({item["run_id"] for item in plan["runs"]}),
        "bootstrap reuse names an unregistered corpus",
    )
    if run["run_id"] not in declarations:
        return None
    declaration_ref = declarations[run["run_id"]]
    declaration = read(declaration_ref)
    require(
        declaration.get("schema") == "motion2scene_same_corpus_bootstrap_reuse_v1"
        and declaration.get("run_id") == run["run_id"]
        and declaration.get("recorded_physics_steps") == 8344,
        "only a complete same-corpus seven-schedule bootstrap can be reused",
    )
    source_plan, source_adoption = read(declaration["source_plan"]), read(
        declaration["source_adoption"]
    )
    require(source_adoption.get("status") == "adopted", "original adoption required")
    same(source_adoption["plan"], declaration["source_plan"])
    source_runs = [item for item in source_plan["runs"] if item["run_id"] == run["run_id"]]
    require(len(source_runs) == 1, "source corpus is absent or duplicated")
    source_run = source_runs[0]
    require(
        source_run["physics_seed"] == run["physics_seed"]
        and source_run["arm"] == run["arm"]
        and source_plan["option_ids"] == plan["option_ids"],
        "reused bootstrap cannot change corpus seed, arm or options",
    )
    old, current = source_run["rounds"][0], run["rounds"][0]
    for key in (
        "round_index",
        "physics_seed",
        "scene_definition",
        "teacher_branch_order",
    ):
        require(old[key] == current[key], "reused bootstrap scientific assignment changed")
    for key in ("teacher_directory", "expected_teacher_result_path"):
        require(
            old[key] == current[key],
            "bootstrap must retain its original physical paths",
        )
    require(
        checked(declaration["collection"]) == Path(old["expected_teacher_result_path"]).resolve(),
        "bootstrap collection is outside its originally assigned slot",
    )
    registration = read(declaration["source_controller_registration"])
    same(registration["plan"], declaration["source_plan"])
    same(registration["adoption"], declaration["source_adoption"])
    require(registration["run_id"] == run["run_id"], "source controller corpus differs")
    source_controller = checked(declaration["source_controller_registration"]).parent
    source_root = Path(source_plan["intended_execution_root"]) / run["run_id"]
    require(
        source_controller == (source_root / "controller").resolve(),
        "source controller slot differs",
    )
    for row in source_run["rounds"][1:]:
        require(
            not Path(row["teacher_directory"]).exists()
            and not Path(row["student_directory"]).exists(),
            "bootstrap continuation is only valid before predecessor encounter acquisition",
        )
    fit = read(declaration["failed_fit"])
    require(
        checked(declaration["failed_fit"])
        == Path(old["expected_postupdate_training_result_path"]).resolve()
        and fit.get("status") == "fit_validation_failed"
        and fit.get("policy") is None
        and fit.get("audit_errors") == [],
        "retain an audited failed bootstrap fit; never inherit a predecessor policy",
    )
    require(
        not any((source_root / "models").glob("model_*/policy.npz")),
        "a bootstrap-only continuation cannot inherit a completed predecessor model",
    )
    collection = read(declaration["collection"])
    same(collection["manifest"], declaration["manifest"])
    manifest = read(declaration["manifest"])
    require(
        checked(declaration["manifest"])
        == (Path(old["teacher_directory"]) / "manifest.json").resolve()
        and manifest.get("policy") is None
        and manifest.get("script_parameters") is None,
        "only the original forced-teacher bootstrap manifest is reusable",
    )
    same(manifest["registry"], plan["registry"])
    same(manifest["scene_definition"], current["scene_definition"])
    rows, cells = collection["rows"], manifest["cells"]
    require(
        len(rows) == len(cells) == len(declaration["attempts"]) == 7
        and [cell["forced_option_id"] for cell in cells] == current["teacher_branch_order"]
        and [row["cell_id"] for row in rows] == [cell["cell_id"] for cell in cells],
        "complete original seven-branch order required",
    )
    intents, raw_paths = [], set()
    for item, row, cell in zip(declaration["attempts"], rows, cells, strict=True):
        intent_path, assessment_path = checked(item["intent"]), checked(item["assessment"])
        expected = source_controller / "attempts" / ("round000_teacher_" + cell["cell_id"])
        require(
            intent_path == expected / "intent.json"
            and assessment_path == expected / "assessment.json",
            "reuse must name the original intent and assessment",
        )
        intent, assessment = read(item["intent"]), read(item["assessment"])
        same(intent["manifest"], declaration["manifest"])
        require(intent["command"] == cell["command"], "original command differs")
        require(
            cell["timed_schedule_mode"] == "forced"
            and cell["runtime_seed"] == run["physics_seed"]
            and Path(cell["output"]).resolve()
            == (Path(old["teacher_directory"]) / "rollouts" / cell["cell_id"]).resolve(),
            "original forced schedule seed or output differs",
        )
        attempt_path = checked(assessment["attempt"])
        require(
            attempt_path == Path(intent["attempt_path"]).resolve()
            and attempt_path == (Path(cell["output"]) / "attempt.json").resolve()
            and read(assessment["attempt"])["exit_status"] == 0
            and assessment["row"] == row
            and row.get("measurement_admitted") is True
            and row.get("task_outcome_admitted") is True
            and row.get("outcome", {}).get("task_outcome") in ("pass", "failure")
            and row.get("physics_steps") == 1192,
            "every reused branch must retain a complete verified physical outcome",
        )
        raw = row.get("trajectory") or row.get("raw_artifacts", {}).get("trajectory")
        raw_path = checked(raw)
        require(raw_path not in raw_paths, "duplicate physical capture in reused bootstrap")
        raw_paths.add(raw_path)
        intents.append(intent_path)
    require(collection["physics_steps"] == 8344, "bootstrap physics accounting differs")
    require(
        set(intents) == set((source_controller / "attempts").glob("*/intent.json")),
        "every original launch reservation must remain accounted for",
    )
    return dict(
        declaration=declaration_ref,
        collection=declaration["collection"],
        manifest=declaration["manifest"],
        intents=intents,
        attempt_artifacts=declaration["attempts"],
        failed_fit=declaration["failed_fit"],
        recorded_physics_steps=8344,
    )


def revalidate_reused_attempts(reused):
    """Keep the original artifact identities binding on every ledger read."""
    if reused is None:
        return []
    declaration = read(reused["declaration"])
    require(
        declaration["attempts"] == reused["attempt_artifacts"]
        and declaration["collection"] == reused["collection"]
        and declaration["manifest"] == reused["manifest"],
        "context reuse identities differ from the declared predecessor",
    )
    for ref in (reused["collection"], reused["manifest"], reused["failed_fit"]):
        checked(ref)
    paths = []
    for item in reused["attempt_artifacts"]:
        paths.append(checked(item["intent"]))
        assessment = read(item["assessment"])
        checked(assessment["attempt"])
    require(paths == reused["intents"], "context original intent paths changed")
    return paths
