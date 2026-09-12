"""Immutable acquisition ordering and budget contracts, without simulation.

An unsigned proposed plan is never execution authority or a completed corpus.
Hash-bound receipts establish recorded provenance, not independent attestation
that unregistered experiments do not exist. Physical admission is audited by
the separate replay/collection readers, not inferred from this chronology.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ARMS = ("uniform", "target_only", "analytic_contrast", "observation_curriculum")
SEEDS = (93201, 93202, 93203)
PLAN_SCHEMA = "motion2scene_primary_acquisition_plan_v1"
RECEIPT_SCHEMA = "motion2scene_primary_acquisition_completion_v1"
ADOPTION_SCHEMA = "motion2scene_primary_acquisition_adoption_v1"


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())


def checked_path(ref):
    if not isinstance(ref, dict) or not {"path", "sha256"} <= ref.keys():
        raise ValueError("hash-bound artifact required")
    path = Path(ref["path"])
    if not path.is_absolute() or artifact(path)["sha256"].removeprefix("sha256:") != ref[
        "sha256"
    ].removeprefix("sha256:"):
        raise ValueError("artifact path/hash mismatch")
    return path


def read_bound(ref):
    return json.loads(checked_path(ref).read_text())


def same_artifact(left, right):
    return Path(left["path"]).resolve() == Path(right["path"]).resolve() and left[
        "sha256"
    ].removeprefix("sha256:") == right["sha256"].removeprefix("sha256:")


def source_identities(refs):
    identities = []
    for ref in refs:
        # Original paths may later change; immutable copies preserve evidence.
        if "snapshot" in ref:
            checked_path(ref["snapshot"])
            if ref["snapshot"]["sha256"].removeprefix("sha256:") != ref["sha256"].removeprefix(
                "sha256:"
            ):
                raise ValueError("source snapshot differs from the declared implementation")
        else:
            checked_path(ref)
        identities.append((str(Path(ref["path"]).resolve()), ref["sha256"].removeprefix("sha256:")))
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("distinct nonempty common training source closure required")
    return sorted(identities)


def write_new(path, value):
    """Refuse replacing any prior registration or completion marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return artifact(path)


def budget_contract():
    episode = 298 * 4
    bootstrap = 7 * episode
    round_steps = 8 * episode
    total = bootstrap + 4 * round_steps
    return dict(
        budget_physics_steps=50000,
        loaded_reference_frames=299,
        recorded_control_frames_per_complete_episode=298,
        physics_substeps_per_record=4,
        maximum_physics_steps_per_episode=episode,
        bootstrap_teacher_episodes=7,
        bootstrap_physics_steps=bootstrap,
        acquisition_rounds=4,
        teacher_episodes_per_round=7,
        student_episodes_per_round=1,
        maximum_physics_steps_per_round=round_steps,
        total_planned_episodes=39,
        maximum_planned_physics_steps=total,
        unallocated_physics_steps=50000 - total,
    )


def attempt_accounting(attempts, limit=50000):
    """Separate actual recorded steps from reservations for unknown attempts.

    Resource-only prelaunch pauses have zero physical steps. Each launched
    attempt without a persisted count reserves its full declared upper bound;
    the reservation is never substituted for an actual measurement.
    """
    known = unknown = pauses = 0
    ids = []
    for attempt in attempts:
        ids.append(attempt["attempt_id"])
        maximum = attempt["maximum_physics_steps"]
        steps = attempt.get("actual_physics_steps")
        if type(maximum) is not int or not 0 < maximum <= 1192:
            raise ValueError("a bounded native attempt is required")
        if attempt.get("launched") is False:
            if steps != 0:
                raise ValueError("unlaunched resource pause must explicitly record zero steps")
            pauses += 1
        elif attempt.get("launched") is True:
            if steps is None:
                unknown += maximum
            elif type(steps) is not int or not 0 <= steps <= maximum:
                raise ValueError("invalid actual physical step count")
            else:
                known += steps
        else:
            raise ValueError("explicit launch state required")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate attempt identity; count each actual attempt once")
    return dict(
        actual_recorded_physics_steps=known,
        unknown_attempt_reserved_steps=unknown,
        resource_only_pauses=pauses,
        conservative_charged_steps=known + unknown,
        budget_remaining_after_reservations=limit - known - unknown,
        within_budget=known + unknown <= limit,
    )


def validate_plan(plan):
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("adoption_status") != "proposed_not_adopted"
        or plan.get("execution_authorized") is not False
        or plan.get("budget") != budget_contract()
        or len(plan.get("option_ids", [])) != 7
        or len(set(plan["option_ids"])) != 7
        or plan["option_ids"][0] != "neutral"
    ):
        raise ValueError("explicit proposed seven-option 50k plan required")
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    actual = {(r["physics_seed"], r["arm"]) for r in plan["runs"]}
    if actual != expected or len(plan["runs"]) != 12:
        raise ValueError("exactly three independent seeds and four arms required")
    outputs = []
    by_pair = {}
    for run in plan["runs"]:
        seed, arm = run["physics_seed"], run["arm"]
        if run["run_id"] != f"seed{seed}_{arm}":
            raise ValueError("stable corpus/arm run identity required")
        rounds = run["rounds"]
        if [r["round_index"] for r in rounds] != list(range(5)):
            raise ValueError("bootstrap followed by four acquisition rounds required")
        for row in rounds:
            index = row["round_index"]
            if (
                row["physics_seed"] != seed
                or sorted(row["teacher_branch_order"]) != sorted(plan["option_ids"])
                or row["allowed_teacher_rounds"] != list(range(index))
                or row["student_before_current_teachers"] != bool(index)
            ):
                raise ValueError(
                    "matched seed, complete branch order and causal label set required"
                )
            outputs.append(row["expected_teacher_result_path"])
            if index:
                if row["candidate_id"] is None or row["observation_availability"] is not None:
                    raise ValueError(
                        "candidate ID with unmeasured observation availability required"
                    )
                outputs.append(row["expected_student_result_path"])
            elif any(
                row[key] is not None
                for key in (
                    "expected_student_result_path",
                    "expected_preupdate_policy_path",
                    "expected_preupdate_training_result_path",
                )
            ):
                raise ValueError("bootstrap has teachers only")
        by_pair[(seed, arm)] = [
            (r["candidate_id"], r["scene_definition"], r["teacher_branch_order"])
            for r in rounds[1:]
        ]
    if len(outputs) != len(set(outputs)):
        raise ValueError("each arm executes fresh physical collections, including bootstrap")
    for seed in SEEDS:
        if by_pair[(seed, "analytic_contrast")] != by_pair[(seed, "observation_curriculum")]:
            raise ValueError("replay comparison must use exactly the same candidate order")
    return plan


def _adopted(plan, plan_ref, adoption_ref):
    validate_plan(plan)
    if read_bound(plan_ref) != plan:
        raise ValueError("receipt must name this exact immutable plan")
    adoption = read_bound(adoption_ref)
    if (
        adoption.get("schema") != ADOPTION_SCHEMA
        or adoption.get("status") != "adopted"
        or not same_artifact(adoption.get("plan", {}), plan_ref)
    ):
        raise ValueError("separate adopted protocol required; proposed plan cannot authorize a run")
    learner = read_bound(adoption["common_learner"])
    if (
        learner.get("status") != "frozen"
        or learner.get("feature_dimension") != 114
        or learner.get("option_ids") != plan["option_ids"]
        or learner.get("phase_ticks") != [15, 50, 70]
        or type(learner.get("allow_measured_tie_initialization", False)) is not bool
        or not isinstance(learner.get("l2"), (float, int))
        or not 0 <= learner["l2"] < float("inf")
    ):
        raise ValueError("one frozen common 114D seven-option learner and finite penalty required")
    checked_path(adoption["runtime_freeze"])
    source_identities(learner["training_implementation"])
    return learner


def _expected(ref, path):
    if str(checked_path(ref).resolve()) != str(Path(path).resolve()):
        raise ValueError("artifact does not occupy its registered round slot")


def _training_provenance(plan, row, entry, earlier, learner):
    _expected(entry["student_model"], row["expected_preupdate_policy_path"])
    _expected(entry["model_training_result"], row["expected_preupdate_training_result_path"])
    training = read_bound(entry["model_training_result"])
    if training.get("status") != "complete" or not same_artifact(
        training["policy"], entry["student_model"]
    ):
        raise ValueError("a completed pre-update model fit is required")
    registration = read_bound(training["registration"])
    registered_ties = registration.get("allow_measured_tie_initialization", False)
    adopted_ties = learner.get("allow_measured_tie_initialization", False)
    if (
        type(registered_ties) is not bool
        or type(adopted_ties) is not bool
        or registered_ties != adopted_ties
    ):
        raise ValueError("measured-tie initialization must match the adopted boolean setting")
    actual = registration["collections"]
    if len(actual) != len(earlier) or any(
        not same_artifact(a, b) for a, b in zip(actual, earlier, strict=True)
    ):
        raise ValueError("model labels must equal bootstrap plus strictly earlier teacher rounds")
    if (
        not same_artifact(registration["registry"], plan["registry"])
        or registration.get("l2") != learner["l2"]
    ):
        raise ValueError("all arms require the adopted common registry and penalty")
    if source_identities(registration["implementation"]) != source_identities(
        learner["training_implementation"]
    ):
        raise ValueError("all arms require the adopted common training implementation")
    for ref in actual:
        checked_path(ref)


def validate_completed_replay_order(plan, receipt):
    """Validate an adopted completed prefix; separately audit physical contents.

    The returned rows identify each gap's generating model and age. It never
    relabels a historical pre-update error as an error of the final learner.
    """
    learner = _adopted(plan, receipt["plan"], receipt["adoption"])
    end = receipt.get("completed_through_round")
    status = receipt.get("status")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or type(end) is not int
        or not 1 <= end <= 4
        or status not in ("complete_prefix", "complete")
        or (status == "complete" and end != 4)
        or [e["round_index"] for e in receipt["entries"]] != list(range(end + 1))
    ):
        raise ValueError("explicit completed bootstrap/round prefix required")
    runs = [r for r in plan["runs"] if r["run_id"] == receipt["run_id"]]
    if len(runs) != 1:
        raise ValueError("registered run ID required")
    run = runs[0]
    earlier, gaps = [], []
    for entry, row in zip(receipt["entries"], run["rounds"], strict=False):
        index = row["round_index"]
        _expected(entry["teacher_collection"], row["expected_teacher_result_path"])
        if index:
            _expected(entry["student_collection"], row["expected_student_result_path"])
            _training_provenance(plan, row, entry, earlier, learner)
            for key in ("preupdate_binding", "teacher_release"):
                _expected(entry[key], row[f"expected_{key}_path"])
            binding = read_bound(entry["preupdate_binding"])
            release = read_bound(entry["teacher_release"])
            if (
                binding.get("schema") != "motion2scene_preupdate_binding_v1"
                or binding.get("run_id") != run["run_id"]
                or binding.get("round_index") != index
                or not same_artifact(binding["plan"], receipt["plan"])
                or not same_artifact(binding["adoption"], receipt["adoption"])
                or not same_artifact(binding["student_model"], entry["student_model"])
                or not same_artifact(
                    binding["model_training_result"], entry["model_training_result"]
                )
                or binding.get("current_teacher_directory_absent") is not True
                or release.get("schema") != "motion2scene_current_teachers_release_v1"
                or not same_artifact(release["preupdate_binding"], entry["preupdate_binding"])
                or not same_artifact(release["student_collection"], entry["student_collection"])
                or release.get("current_teacher_directory_absent") is not True
                or binding["created_utc"] > release["created_utc"]
            ):
                raise ValueError(
                    "pre-update commitment and student-before-teacher release required"
                )
            validate_student_release(read_bound(entry["student_collection"]), binding, plan, row)
            gaps.append(
                dict(
                    round_index=index,
                    generating_student_model=entry["student_model"],
                    gap_age_rounds=end - index,
                    student_collection=entry["student_collection"],
                    teacher_collection=entry["teacher_collection"],
                )
            )
        elif any(
            entry.get(key) is not None
            for key in ("student_collection", "student_model", "model_training_result")
        ):
            raise ValueError("bootstrap has no pre-update student")
        earlier.append(entry["teacher_collection"])
    for row in run["rounds"][end + 1 :]:
        if Path(row["teacher_directory"]).exists() or Path(row["student_directory"]).exists():
            raise ValueError("completed prefix requires later rounds to remain unlaunched")
    return dict(
        run_id=run["run_id"],
        completed_through_round=end,
        teacher_collections=earlier,
        historical_gaps=gaps,
        physical_admission_audited=False,
        scope="Recorded ordering/model provenance; replay reader must audit raw physical outcomes",
    )


def bind_preupdate_round(plan_ref, adoption_ref, run_id, round_index, earlier_teachers):
    """Commit an existing causal model before current student/teacher dispatch."""
    plan = read_bound(plan_ref)
    learner = _adopted(plan, plan_ref, adoption_ref)
    run = next(r for r in plan["runs"] if r["run_id"] == run_id)
    if type(round_index) is not int or not 1 <= round_index <= 4:
        raise ValueError("an acquisition round from one to four is required")
    row = run["rounds"][round_index]
    if len(earlier_teachers) != round_index:
        raise ValueError("every earlier teacher round is required")
    for ref, prior in zip(earlier_teachers, run["rounds"][:round_index], strict=True):
        _expected(ref, prior["expected_teacher_result_path"])
    if Path(row["teacher_directory"]).exists() or Path(row["student_directory"]).exists():
        raise ValueError("current student/teacher collection directories must not exist yet")
    entry = dict(
        student_model=artifact(row["expected_preupdate_policy_path"]),
        model_training_result=artifact(row["expected_preupdate_training_result_path"]),
    )
    _training_provenance(plan, row, entry, earlier_teachers, learner)
    return write_new(
        row["expected_preupdate_binding_path"],
        dict(
            schema="motion2scene_preupdate_binding_v1",
            plan=plan_ref,
            adoption=adoption_ref,
            run_id=run_id,
            round_index=round_index,
            earlier_teachers=earlier_teachers,
            **entry,
            current_teacher_directory_absent=True,
            current_student_directory_absent=True,
            created_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


def release_current_teachers(binding_ref):
    """Record a completed student receipt before preparing its current teachers."""
    binding = read_bound(binding_ref)
    plan = read_bound(binding["plan"])
    _adopted(plan, binding["plan"], binding["adoption"])
    run = next(r for r in plan["runs"] if r["run_id"] == binding["run_id"])
    row = run["rounds"][binding["round_index"]]
    _expected(binding_ref, row["expected_preupdate_binding_path"])
    if Path(row["teacher_directory"]).exists():
        raise ValueError("teacher collection cannot precede its student completion release")
    student_ref = artifact(row["expected_student_result_path"])
    student = read_bound(student_ref)
    validate_student_release(student, binding, plan, row)
    return write_new(
        row["expected_teacher_release_path"],
        dict(
            schema="motion2scene_current_teachers_release_v1",
            preupdate_binding=binding_ref,
            student_collection=student_ref,
            current_teacher_directory_absent=True,
            created_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


def validate_student_release(student, binding, plan, row):
    """Accept a measured pass or fail under the collector's actual schema.

    A physically verified early failure can have a nonzero process exit. It is
    retained if the independent outcome classification is known and its steps
    are measured. Unknown infrastructure failures cannot release new teachers.
    This checks receipt structure; the replay reader re-audits native arrays.
    """
    rows = student.get("rows", [])
    if student.get("schema") != "motion2scene_timed_schedule_collection_v1" or len(rows) != 1:
        raise ValueError("one native student collection row is required")
    result = rows[0]
    steps = result.get("physics_steps")
    outcome = result.get("outcome", {})
    attempt = read_bound(result["attempt"])
    manifest = read_bound(student["manifest"])
    if (
        result.get("mode") != "learned"
        or result.get("task_outcome_admitted") is not True
        or outcome.get("task_outcome") not in ("pass", "failure")
        or outcome.get("exit_status") != attempt.get("exit_status")
        or type(steps) is not int
        or not 0 < steps <= 1192
        or student.get("physics_steps") != steps
        or student.get("unmeasured_failed_attempts") != 0
        or not same_artifact(manifest["policy"], binding["student_model"])
        or not same_artifact(manifest["registry"], plan["registry"])
        or manifest["scene_definition"] != row["scene_definition"]
    ):
        raise ValueError(
            "measured pre-update student outcome and exact registered context required"
        )
