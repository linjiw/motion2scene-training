#!/usr/bin/env python3
"""Serial, resumable execution of a separately adopted primary acquisition plan.

``check`` is read-only. ``run`` can launch physics and requires an adopted plan,
an exact frozen runtime, and the nonblocking common acquisition lock. An intent
without a completed attempt is never automatically retried.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import motion2scene_audit_reused_bootstrap as bootstrap_audit  # noqa: E402
from motion2scene_bootstrap_reuse import (  # noqa: E402
    revalidate_reused_attempts,
    validate_bootstrap_reuse,
)
import motion2scene_build_timed_replay as replay  # noqa: E402
import motion2scene_collect_timed_schedules as collector  # noqa: E402
from motion2scene_preflight_runtime import validate_environment  # noqa: E402
import motion2scene_train_timed_schedules as trainer  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    RECEIPT_SCHEMA,
    _adopted,
    artifact,
    attempt_accounting,
    bind_preupdate_round,
    checked_path,
    read_bound,
    release_current_teachers,
    same_artifact,
    source_identities,
    validate_completed_replay_order,
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    validate_rule,
)

SCHEMA = "motion2scene_primary_acquisition_controller_v1"
RUNTIME_SCHEMA = "motion2scene_primary_acquisition_runtime_v1"
EXECUTION_CONTRACT = {
    "serial": True,
    "automatic_physical_retries": False,
    "unknown_attempt_policy": "pause_and_reserve_full_episode",
    "known_physical_failure_policy": "retain_and_continue",
    "bootstrap_fit": "uniform",
    "baseline_round_fit": "uniform_all_available_teacher_groups",
    "observation_round_fit": "cpu_uniform_audit_then_verified_historical_replay_fit",
    "unweighted_audit_subdirectory": "unweighted_audit",
    "replay_subdirectory": "replay",
    "minimum_free_gpu_mib": 7500,
    "timeout_s": 375,
    "maximum_physics_steps_per_episode": 1192,
    "lock_relative_path": ".primary-acquisition.lock",
}


class Paused(RuntimeError):
    """A retained condition requiring an external change or explicit repair."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def commit(path, value):
    """Idempotent exclusive receipt; a changed existing receipt is an error."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"immutable controller receipt differs: {path}")
        return artifact(path)
    return write_new(path, value)


def current_sources(refs, expected):
    actual = [artifact(path) for path in sorted(expected)]
    if source_identities(refs) != source_identities(actual):
        raise ValueError("executing source closure differs from the adopted implementation")
    # Source snapshots alone cannot authorize running changed original files.
    for ref in refs:
        checked_path({key: ref[key] for key in ("path", "sha256")})


def runtime_description(plan_ref, request, template, assets, environment, rule, cell="neutral"):
    """Return a reviewable runtime declaration; this does not adopt or execute it."""
    runtime_sources = closure([ROOT / (collector.RUNTIME.replace(".", "/") + ".py")]) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    scoring_sources = closure([Path(collector.__file__)])
    return dict(
        schema=RUNTIME_SCHEMA,
        status="frozen",
        plan=plan_ref,
        registry=read_bound(plan_ref)["registry"],
        request=artifact(request),
        template=artifact(template),
        template_cell=cell,
        runtime_assets_declaration=artifact(assets),
        environment=artifact(environment),
        replay_rule=artifact(rule),
        collection_implementation=[artifact(p) for p in sorted(runtime_sources | scoring_sources)],
        runtime_artifacts=[artifact(p) for p in sorted(runtime_sources)] + read(assets),
        scoring_artifacts=[artifact(p) for p in sorted(scoring_sources)],
        controller_implementation=[artifact(p) for p in sorted(closure([Path(__file__)]))],
        execution_contract=EXECUTION_CONTRACT,
    )


def load_context(plan_path, adoption_path, run_id):
    """Validate adoption and current code without creating execution directories."""
    plan_ref, adoption_ref = artifact(plan_path), artifact(adoption_path)
    plan, adoption = read_bound(plan_ref), read_bound(adoption_ref)
    learner = _adopted(plan, plan_ref, adoption_ref)
    runtime = read_bound(adoption["runtime_freeze"])
    if (
        runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("status") != "frozen"
        or not same_artifact(runtime["plan"], plan_ref)
        or not same_artifact(runtime["registry"], plan["registry"])
        or runtime.get("execution_contract") != EXECUTION_CONTRACT
    ):
        raise ValueError("exact adopted acquisition runtime and execution contract required")
    expected = runtime_description(
        plan_ref,
        checked_path(runtime["request"]),
        checked_path(runtime["template"]),
        checked_path(runtime["runtime_assets_declaration"]),
        checked_path(runtime["environment"]),
        checked_path(runtime["replay_rule"]),
        runtime["template_cell"],
    )
    for key in (
        "collection_implementation",
        "runtime_artifacts",
        "scoring_artifacts",
        "controller_implementation",
    ):
        current_sources(runtime[key], [Path(ref["path"]) for ref in expected[key]])
    current_sources(learner["training_implementation"], closure([Path(trainer.__file__)]))
    assets = read_bound(runtime["runtime_assets_declaration"])
    if not isinstance(assets, list) or runtime["environment"] not in assets:
        raise ValueError("the frozen environment must be explicitly declared among runtime assets")
    for ref in assets:
        checked_path(ref)
    validate_rule(read_bound(runtime["replay_rule"]))
    bank = collector.load_verified_registry(plan["registry"]["path"], plan["registry"]["sha256"])
    if read_bound(runtime["request"]) != bank.request:
        raise ValueError("adopted request differs from the qualified registry")
    template = read_bound(runtime["template"])
    bases = [row for row in template["cells"] if row["cell_id"] == runtime["template_cell"]]
    if (
        len(bases) != 1
        or template["implementation"]["checkpoint"] != bank.request["controller"]
        or bases[0]["motion"] != bank.request["references"][0]["motion"]
    ):
        raise ValueError("common template source/controller/cell is not the qualified bank")
    run = next((row for row in plan["runs"] if row["run_id"] == run_id), None)
    if run is None:
        raise ValueError("unknown registered run ID")
    reused_bootstrap = validate_bootstrap_reuse(plan, adoption, run)
    execution_root = Path(plan["intended_execution_root"]).resolve()
    run_root = execution_root / run_id
    for row in run["rounds"]:
        collector.validate_scene(read_bound(row["scene_definition"]))
        for key, value in row.items():
            if value is not None and (key.endswith("_directory") or key.endswith("_path")):
                reused_path = (
                    reused_bootstrap is not None
                    and row["round_index"] == 0
                    and key in ("teacher_directory", "expected_teacher_result_path")
                )
                if not reused_path and not Path(value).resolve().is_relative_to(run_root):
                    raise ValueError("registered execution slot escapes its run directory")
    return dict(
        plan=plan,
        plan_ref=plan_ref,
        adoption_ref=adoption_ref,
        learner=learner,
        runtime=runtime,
        runtime_ref=adoption["runtime_freeze"],
        run=run,
        execution_root=execution_root,
        run_root=run_root,
        reused_bootstrap=reused_bootstrap,
    )


@contextmanager
def acquisition_lock(path):
    """One lock for all arms/corpora; never wait behind another controller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Paused("another acquisition controller holds the common lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LocalBackend:
    """Local APIs only; tests substitute a file-backed, nonphysical backend."""

    def prepare(self, context, row, kind, model):
        runtime = context["runtime"]
        collector.prepare(
            argparse.Namespace(
                out=Path(row[f"{kind}_directory"]),
                registry=checked_path(context["plan"]["registry"]),
                request=checked_path(runtime["request"]),
                scene_definition=checked_path(row["scene_definition"]),
                template=checked_path(runtime["template"]),
                cell=runtime["template_cell"],
                policy_mode="forced" if kind == "teacher" else "learned",
                policy=None if model is None else checked_path(model),
                policy_id=f"{context['run']['run_id']}_round{row['round_index']:03d}_{kind}",
                forced_option_ids=(row["teacher_branch_order"] if kind == "teacher" else None),
                preferred_option_id="neutral",
                preferred_reference_id="sustained",
                script_parameters=None,
                runtime_assets=checked_path(runtime["runtime_assets_declaration"]),
                seed=row["physics_seed"],
            )
        )

    def collection(self, context, row, kind, model):
        out = Path(row[f"{kind}_directory"])
        manifest, bank, scene = collector.verify_manifest(out)
        runtime = context["runtime"]
        for key, expected in (
            ("registry", context["plan"]["registry"]),
            ("request", runtime["request"]),
            ("template", runtime["template"]),
            ("scene_definition", row["scene_definition"]),
            ("runtime_assets_declaration", runtime["runtime_assets_declaration"]),
        ):
            if not same_artifact(manifest[key], expected):
                raise ValueError(f"prepared collection differs from adopted {key}")
        for key, frozen in (
            ("dependencies", "collection_implementation"),
            ("runtime_artifacts", "runtime_artifacts"),
            ("scoring_artifacts", "scoring_artifacts"),
        ):
            if source_identities(manifest[key]) != source_identities(runtime[frozen]):
                raise ValueError(f"prepared {key} differs from the adopted runtime")
        ids = row["teacher_branch_order"] if kind == "teacher" else ["neutral"]
        if (
            manifest["policy"] != model
            or manifest.get("script_parameters") is not None
            or manifest["limits"]
            != {"minimum_free_gpu_mib": 7500, "timeout_s": 375, "serial": True}
            or [cell["forced_option_id"] for cell in manifest["cells"]] != ids
        ):
            raise ValueError("collection policy, complete branch order, or resource limits differ")
        for cell in manifest["cells"]:
            if (
                cell["timed_schedule_mode"] != ("forced" if kind == "teacher" else "learned")
                or cell["runtime_seed"] != row["physics_seed"]
                or Path(cell["output"]).resolve() != (out / "rollouts" / cell["cell_id"]).resolve()
            ):
                raise ValueError("collection mode, seed or actual output slot differs")
        return manifest, bank, scene

    def reuse_bootstrap(self, context):
        reused = context["reused_bootstrap"]
        revalidate_reused_attempts(reused)
        manifest = read_bound(reused["manifest"])
        report = bootstrap_audit.audit(reused["declaration"], manifest["implementation"]["python"])
        if not same_artifact(report["collection"], reused["collection"]):
            raise ValueError("archived bootstrap audit returned a different collection")
        commit(context["run_root"] / "controller" / "bootstrap_physical_reaudit.json", report)
        return reused["collection"]

    def preflight(self, context, manifest):
        runtime = context["runtime"]
        for key in ("controller_implementation", "collection_implementation"):
            for ref in runtime[key]:
                checked_path({name: ref[name] for name in ("path", "sha256")})
        for ref in context["learner"]["training_implementation"]:
            checked_path({name: ref[name] for name in ("path", "sha256")})
        return validate_environment(
            runtime["environment"],
            runtime["runtime_assets_declaration"],
            manifest["implementation"]["python"],
        )

    def free_gpu_mib(self):
        return collector.free_gpu_mib()

    def launch(self, command, timeout):
        return collector.run_with_process_group(command, timeout=timeout)

    def audit(self, cell, manifest, bank, scene):
        attempt = read(Path(cell["output"]) / "attempt.json")
        collector.validate_collection_context(cell, manifest, bank, scene, actual=True)
        if attempt["exit_status"] != 0:
            result = collector.analyze_incomplete(
                cell, manifest, bank, scene, attempt, "nonzero_process_exit"
            )
        else:
            try:
                result, _, _ = collector.analyze_cell(cell, manifest, bank, scene)
            except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
                result = collector.analyze_incomplete(
                    cell, manifest, bank, scene, attempt, str(error)
                )
        result["attempt"] = artifact(Path(cell["output"]) / "attempt.json")
        return result

    def analyze(self, out):
        collector.analyze(out)

    def fit(self, context, collections, out, weights):
        trainer.run(
            checked_path(context["plan"]["registry"]),
            [checked_path(ref) for ref in collections],
            out,
            context["learner"]["l2"],
            None if weights is None else checked_path(weights),
            allow_measured_tie_initialization=context["learner"].get(
                "allow_measured_tie_initialization", False
            ),
        )

    def build_replay(self, context, teachers, students, out, receipt):
        result = replay.build(
            checked_path(teachers),
            [checked_path(ref) for ref in students],
            read_bound(context["runtime"]["replay_rule"]),
            out,
            checked_path(receipt),
        )
        verified = replay.verify_weights(
            checked_path(result["weights"]),
            read_bound(teachers),
            context["plan"]["registry"],
        )
        return dict(**result, verification=verified, new_physics_steps=0, new_model_fits=0)


class Controller:
    def __init__(self, context, backend=None):
        self.context = context
        self.backend = LocalBackend() if backend is None else backend
        self.folder = context["run_root"] / "controller"
        self.entries = []

    def event(self, kind, **fields):
        folder = self.folder / "events"
        folder.mkdir(parents=True, exist_ok=True)
        index = len(list(folder.glob("*.json")))
        return write_new(folder / f"{index:06d}.json", dict(kind=kind, created_utc=utc(), **fields))

    def accounting(self):
        attempts, original_captures = [], set()
        unknown_tails = 0
        reused = self.context.get("reused_bootstrap")
        inherited = revalidate_reused_attempts(reused)
        current = list((self.folder / "attempts").glob("*/intent.json"))
        intent_paths = sorted(inherited + current)
        identifiers = [read(path)["attempt_id"] for path in intent_paths]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("reused and new attempts cannot duplicate an assigned slot")
        for path in intent_paths:
            intent = read(path)
            assessment_path = path.parent / "assessment.json"
            assessment = read(assessment_path) if assessment_path.exists() else None
            steps = None if assessment is None else assessment["row"].get("physics_steps")
            if assessment is not None:
                checked_path(assessment["attempt"])
                row = assessment["row"]
                ref = row.get("trajectory") or row.get("raw_artifacts", {}).get("trajectory")
                if ref is None:
                    ref = row.get("raw_artifacts", {}).get("aborted_raw_recording")
                if ref is not None:
                    checked_path(ref)
                    identity = str(Path(ref["path"]).resolve())
                    if identity in original_captures:
                        raise ValueError(
                            "same original physical capture occupies two attempt slots"
                        )
                    original_captures.add(identity)
                if steps is not None and (
                    row.get("task_outcome_admitted") is not True
                    or row.get("outcome", {}).get("task_outcome") not in ("pass", "failure")
                ):
                    unknown_tails += 1192 - steps
            attempts.append(
                dict(
                    attempt_id=intent["attempt_id"],
                    origin=(
                        "reused_predecessor_bootstrap" if path in inherited else "current_execution"
                    ),
                    maximum_physics_steps=1192,
                    launched=True,
                    actual_process_receipt=(
                        artifact(intent["attempt_path"])
                        if Path(intent["attempt_path"]).exists()
                        else None
                    ),
                    actual_physics_steps=steps,
                    intent=artifact(path),
                    assessment=(artifact(assessment_path) if assessment is not None else None),
                    launch_state=(
                        "completed_process_receipt"
                        if assessment is not None
                        else "dispatch_committed_unknown"
                    ),
                )
            )
        # An intent is a conservative dispatch reservation, not evidence that
        # physics actually began. Unknown startup steps are never called measured.
        pauses, cpu_attempts = [], []
        for path in sorted((self.folder / "events").glob("*.json")):
            event = read(path)
            if event["kind"] == "cpu_fit_started":
                cpu_attempts.append(event)
            if event["kind"] == "resource_pause":
                pauses.append(
                    dict(
                        attempt_id=str(path),
                        maximum_physics_steps=1192,
                        launched=False,
                        actual_physics_steps=0,
                    )
                )
        ledger = attempt_accounting(
            attempts + pauses, self.context["plan"]["budget"]["budget_physics_steps"]
        )
        ledger["unknown_outcome_tail_reserved_steps"] = unknown_tails
        ledger["conservative_charged_steps"] += unknown_tails
        ledger["budget_remaining_after_reservations"] -= unknown_tails
        ledger["within_budget"] = ledger["budget_remaining_after_reservations"] >= 0
        # The helper's `launched=True` also accounts conservatively for an
        # intent interrupted before dispatch. Public receipts distinguish that
        # unknown state from an observed completed process.
        public_attempts = []
        for attempt in attempts:
            public = {key: value for key, value in attempt.items() if key != "launched"}
            public["actual_launch_confirmed"] = (
                True if attempt["actual_process_receipt"] is not None else None
            )
            public_attempts.append(public)
        return dict(
            schema="motion2scene_primary_acquisition_budget_v1",
            run_id=self.context["run"]["run_id"],
            attempts=public_attempts,
            **ledger,
            unlaunched_assigned_episode_slots=39 - len(attempts),
            reserved_future_assigned_steps=(39 - len(attempts)) * 1192,
            distinct_original_recorded_captures=len(original_captures),
            cpu_fit_attempts_started=len(cpu_attempts),
            auxiliary_cpu_fit_attempts=sum(event["auxiliary"] for event in cpu_attempts),
            completed_cpu_fit_receipts=len(list((self.folder / "cpu_fits").glob("*.json"))),
            completed_replay_receipts=len(list((self.folder / "replay").glob("*.json"))),
            new_training_physics_steps=0,
            reused_bootstrap_recorded_steps=0 if reused is None else 8344,
            retained_predecessor_failed_fit=(None if reused is None else reused["failed_fit"]),
            interpretation="intent-only reservations are not measured launch/physics counts",
        )

    def ledger(self):
        value = self.accounting()
        self.event("budget", ledger=value)
        return value

    def step(self, row, kind, cell, manifest, bank, scene):
        identifier = f"round{row['round_index']:03d}_{kind}_{cell['cell_id']}"
        folder = self.folder / "attempts" / identifier
        intent_path, assessment_path = (
            folder / "intent.json",
            folder / "assessment.json",
        )
        output = Path(cell["output"])
        attempt_path = output / "attempt.json"
        manifest_ref = artifact(Path(row[f"{kind}_directory"]) / "manifest.json")
        if intent_path.exists():
            intent = read(intent_path)
            if (
                intent["manifest"] != manifest_ref
                or intent["command"] != cell["command"]
                or intent["attempt_id"] != identifier
            ):
                raise ValueError("existing dispatch intent differs from registered cell")
            checked_path(intent["runtime_preflight"])
            if not attempt_path.exists():
                raise Paused(f"unresolved launch intent retained; no automatic retry: {identifier}")
        else:
            if output.exists():
                raise Paused(f"unowned existing rollout directory retained: {output}")
            ledger = self.accounting()
            if ledger["budget_remaining_after_reservations"] < 1192:
                raise Paused("physical budget cannot reserve another complete assigned episode")
            preflight = self.backend.preflight(self.context, manifest)
            if self.backend.free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
                self.event(
                    "resource_pause",
                    attempt_id=identifier,
                    launched=False,
                    physics_steps=0,
                )
                raise Paused(
                    "resource preflight paused before intent; assigned cell remains unlaunched"
                )
            preflight_ref = commit(folder / "runtime_preflight.json", preflight)
            write_new(
                intent_path,
                dict(
                    schema="motion2scene_primary_launch_intent_v1",
                    attempt_id=identifier,
                    attempt_path=str(attempt_path.resolve()),
                    manifest=manifest_ref,
                    command=cell["command"],
                    runtime_preflight=preflight_ref,
                    maximum_physics_steps=1192,
                    created_utc=utc(),
                    scope="exclusive dispatch commitment; not proof that process/physics started",
                ),
            )
            self.ledger()
            started = time.monotonic()
            try:
                status = self.backend.launch(cell["command"], manifest["limits"]["timeout_s"])
            except BaseException as error:
                self.event("launch_unresolved", attempt_id=identifier, error=repr(error))
                # The process-group helper terminates children on timeout or
                # interruption. A missing real exit receipt remains unknown.
                raise Paused(
                    f"launch interrupted; retained full reservation: {identifier}"
                ) from error
            if type(status) is not int:
                raise Paused("process runner returned no actual integer exit status")
            write_new(
                attempt_path,
                dict(
                    exit_status=status,
                    wall_seconds=time.monotonic() - started,
                    command=cell["command"],
                ),
            )
        attempt = read(attempt_path)
        if type(attempt.get("exit_status")) is not int or attempt.get("command") != cell["command"]:
            raise ValueError("actual process receipt differs from the registered invocation")
        result = self.backend.audit(cell, manifest, bank, scene)
        assessment = dict(
            manifest=manifest_ref,
            attempt=artifact(attempt_path),
            row=result,
            scope="independently re-audited actual attempted capture",
        )
        commit(assessment_path, assessment)
        self.ledger()
        if (
            result.get("task_outcome_admitted") is not True
            or result.get("outcome", {}).get("task_outcome") not in ("pass", "failure")
            or type(result.get("physics_steps")) is not int
            or not 0 < result["physics_steps"] <= 1192
        ):
            raise Paused(f"unknown physical/infrastructure outcome retained: {identifier}")
        return result

    def collect(self, row, kind, model=None):
        reused = self.context.get("reused_bootstrap")
        if reused is not None and row["round_index"] == 0:
            if kind != "teacher" or model is not None:
                raise ValueError("a reused bootstrap cannot supply student data or a policy")
            ref = self.backend.reuse_bootstrap(self.context)
            if ref != reused["collection"]:
                raise ValueError("bootstrap audit returned a different collection")
            commit(
                self.folder / "collections" / "round000_teacher.json",
                dict(collection=ref),
            )
            commit(
                self.folder / "bootstrap_reuse.json",
                dict(
                    declaration=reused["declaration"],
                    collection=ref,
                    recorded_physics_steps=8344,
                    new_physics_steps=0,
                ),
            )
            return ref
        out = Path(row[f"{kind}_directory"])
        if not (out / "manifest.json").exists():
            if out.exists():
                raise Paused(f"partial collection preparation retained: {out}")
            self.backend.prepare(self.context, row, kind, model)
        manifest, bank, scene = self.backend.collection(self.context, row, kind, model)
        rows = [self.step(row, kind, cell, manifest, bank, scene) for cell in manifest["cells"]]
        if not (out / "result.json").exists():
            self.backend.analyze(out)
        result = read(out / "result.json")
        if result["manifest"] != artifact(out / "manifest.json") or result["rows"] != rows:
            raise ValueError("published collection differs from re-audited assigned attempts")
        ref = artifact(out / "result.json")
        commit(
            self.folder / "collections" / f"round{row['round_index']:03d}_{kind}.json",
            dict(collection=ref),
        )
        return ref

    def fit(self, collections, out, weights=None, *, auxiliary=False):
        out = Path(out)
        if not (out / "result.json").exists():
            if out.exists():
                raise Paused(f"interrupted CPU fit retained without overwriting: {out}")
            self.event(
                "cpu_fit_started",
                output=str(out),
                auxiliary=auxiliary,
                new_physics_steps=0,
            )
            self.backend.fit(self.context, collections, out, weights)
        result_ref = artifact(out / "result.json")
        result = read_bound(result_ref)
        registration = read_bound(result["registration"])
        if (
            registration["collections"] != collections
            or not same_artifact(registration["registry"], self.context["plan"]["registry"])
            or registration["l2"] != self.context["learner"]["l2"]
            or registration.get("replay_weights") != weights
            or type(registration.get("allow_measured_tie_initialization", False)) is not bool
            or type(self.context["learner"].get("allow_measured_tie_initialization", False))
            is not bool
            or registration.get("allow_measured_tie_initialization", False)
            != self.context["learner"].get("allow_measured_tie_initialization", False)
            or source_identities(registration["implementation"])
            != source_identities(self.context["learner"]["training_implementation"])
        ):
            raise ValueError(
                "completed fit differs from the frozen common learner or assigned labels"
            )
        commit(
            self.folder / "cpu_fits" / (out.parent.name + "_" + out.name + ".json"),
            dict(result=result_ref, auxiliary=auxiliary, new_physics_steps=0),
        )
        if result["status"] != "complete" or result.get("policy") is None:
            raise Paused(
                f"CPU teacher/fit evidence insufficient; retained {result['status']}: {out}"
            )
        if checked_path(result["policy"]).resolve() != (out / "policy.npz").resolve():
            raise ValueError("fitted policy does not occupy the registered output slot")
        checked_path(result["teachers"])
        return result

    def receipt(self, index, *, final=False, model=None):
        value = dict(
            schema=RECEIPT_SCHEMA,
            plan=self.context["plan_ref"],
            adoption=self.context["adoption_ref"],
            run_id=self.context["run"]["run_id"],
            status="complete" if final else "complete_prefix",
            completed_through_round=index,
            entries=self.entries[: index + 1],
        )
        if final:
            value.update(
                final_model=model["policy"],
                final_model_training_result=artifact(
                    self.context["run"]["rounds"][index]["expected_postupdate_training_result_path"]
                ),
            )
        path = self.folder / ("completion.json" if final else f"prefix_{index:03d}.json")
        if not path.exists():
            validate_completed_replay_order(self.context["plan"], value)
        # Historical prefix checks deliberately reject directories from future
        # rounds. Preserve their original exclusive receipt when resuming later.
        return commit(path, value)

    def replay_weights(self, row, teachers, receipt):
        out = Path(row["expected_replay_result_path"]).parent
        students = [entry["student_collection"] for entry in self.entries[1:]]
        if not (out / "result.json").exists():
            if out.exists():
                raise Paused(f"interrupted CPU replay audit retained: {out}")
            result = self.backend.build_replay(self.context, teachers, students, out, receipt)
            write_new(out / "result.json", result)
        result_ref = artifact(out / "result.json")
        result = read_bound(result_ref)
        registration = read_bound(read_bound(result["evidence"])["registration"])
        if (
            registration["teachers"] != teachers
            or registration["student_results"] != students
            or registration["acquisition_receipt"] != receipt
            or registration["rule"] != read_bound(self.context["runtime"]["replay_rule"])
        ):
            raise ValueError("replay output differs from actual causal teacher/student prefix")
        commit(
            self.folder / "replay" / f"round{row['round_index']:03d}.json",
            dict(result=result_ref),
        )
        checked_path(result["weights"])
        return result["weights"]

    def acquire(self, *, stop_after_round=4):
        if type(stop_after_round) is not int or not 0 <= stop_after_round <= 4:
            raise ValueError("stop-after-round must be an assigned boundary from zero through four")
        context = self.context
        self.entries = []
        registration = dict(
            schema=SCHEMA,
            plan=context["plan_ref"],
            adoption=context["adoption_ref"],
            runtime=context["runtime_ref"],
            run_id=context["run"]["run_id"],
            execution_contract=EXECUTION_CONTRACT,
        )
        commit(self.folder / "registration.json", registration)
        teachers = []
        final_model = None
        for row in context["run"]["rounds"][: stop_after_round + 1]:
            index = row["round_index"]
            binding_ref = student_ref = release_ref = None
            if index:
                binding_path = Path(row["expected_preupdate_binding_path"])
                if not binding_path.exists():
                    bind_preupdate_round(
                        context["plan_ref"],
                        context["adoption_ref"],
                        context["run"]["run_id"],
                        index,
                        teachers,
                    )
                binding_ref = artifact(binding_path)
                binding = read_bound(binding_ref)
                if (
                    binding["plan"] != context["plan_ref"]
                    or binding["adoption"] != context["adoption_ref"]
                    or binding["run_id"] != context["run"]["run_id"]
                    or binding["round_index"] != index
                    or binding["earlier_teachers"] != teachers
                    or binding["student_model"] != artifact(row["expected_preupdate_policy_path"])
                    or binding["model_training_result"]
                    != artifact(row["expected_preupdate_training_result_path"])
                    or binding["current_student_directory_absent"] is not True
                    or binding["current_teacher_directory_absent"] is not True
                ):
                    raise ValueError(
                        "retained pre-update binding differs from causal model and history"
                    )
                student_ref = self.collect(row, "student", binding["student_model"])
                release_path = Path(row["expected_teacher_release_path"])
                if not release_path.exists():
                    release_current_teachers(binding_ref)
                release_ref = artifact(release_path)
                release = read_bound(release_ref)
                if (
                    release["preupdate_binding"] != binding_ref
                    or release["student_collection"] != student_ref
                    or release["current_teacher_directory_absent"] is not True
                ):
                    raise ValueError(
                        "retained teacher release differs from the actual pre-update student"
                    )
            teacher_ref = self.collect(row, "teacher")
            teachers.append(teacher_ref)
            self.entries.append(
                dict(
                    round_index=index,
                    teacher_collection=teacher_ref,
                    student_collection=student_ref,
                    student_model=None if not index else binding["student_model"],
                    model_training_result=(None if not index else binding["model_training_result"]),
                    preupdate_binding=binding_ref,
                    teacher_release=release_ref,
                )
            )
            prefix = None if not index else self.receipt(index)
            weights = None
            if index and context["run"]["arm"] == "observation_curriculum":
                audit = self.fit(
                    teachers,
                    Path(row["teacher_directory"]).parent / "unweighted_audit",
                    auxiliary=True,
                )
                weights = self.replay_weights(row, audit["teachers"], prefix)
            final_model = self.fit(
                teachers,
                Path(row["expected_postupdate_training_result_path"]).parent,
                weights,
            )
        ledger = self.ledger()
        if stop_after_round == 4:
            completion = self.receipt(4, final=True, model=final_model)
            return dict(status="complete", completion=completion, ledger=ledger)
        return dict(
            status="paused_at_round_boundary",
            completed_through_round=stop_after_round,
            final_model=final_model["policy"],
            ledger=ledger,
        )

    def run(self, *, stop_after_round=4):
        with acquisition_lock(
            self.context["execution_root"] / EXECUTION_CONTRACT["lock_relative_path"]
        ):
            try:
                return self.acquire(stop_after_round=stop_after_round)
            except Paused as error:
                self.event("paused", reason=str(error))
                return dict(status="paused", reason=str(error), ledger=self.ledger())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "describe-runtime"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--adoption", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-round", type=int, default=4)
    for name in (
        "request",
        "template",
        "runtime-assets",
        "environment",
        "replay-rule",
        "out",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--cell", default="neutral")
    args = parser.parse_args()
    if args.action == "describe-runtime":
        if any(
            getattr(args, key) is None
            for key in (
                "request",
                "template",
                "runtime_assets",
                "environment",
                "replay_rule",
                "out",
            )
        ):
            parser.error(
                "describe-runtime requires request, template, runtime-assets, environment, replay-rule and out"
            )
        value = runtime_description(
            artifact(args.plan),
            args.request,
            args.template,
            args.runtime_assets,
            args.environment,
            args.replay_rule,
            args.cell,
        )
        print(
            json.dumps(
                {
                    "runtime_freeze": write_new(args.out, value),
                    "adopted": False,
                    "new_physics_steps": 0,
                }
            )
        )
        return
    if args.adoption is None or args.run_id is None:
        parser.error("check/run require --adoption and --run-id")
    context = load_context(args.plan, args.adoption, args.run_id)
    try:
        result = (
            dict(
                status="adopted_configuration_verified",
                run_id=args.run_id,
                new_physics_steps=0,
            )
            if args.action == "check"
            else Controller(context).run(stop_after_round=args.stop_after_round)
        )
    except Paused as error:
        # A busy common flock is handled before this controller owns any file.
        result = dict(status="paused", reason=str(error), new_physics_steps=0)
    print(json.dumps(result, indent=2))
    if result["status"] == "paused":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
