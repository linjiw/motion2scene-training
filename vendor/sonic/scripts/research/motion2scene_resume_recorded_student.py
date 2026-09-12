#!/usr/bin/env python3
"""Resume one declared, fully accounted reference-arm student with unknown exit.

Retain the frozen scorer's UNKNOWN outcome. The reference arm fits uniform
teacher targets only, so the missing student outcome cannot change its fit.
This adapter never retries the episode or changes a native source or queue.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_collect_timed_schedules as collector  # noqa: E402
import motion2scene_resume_expanded_acquisition as resume_driver  # noqa: E402
import motion2scene_run_primary_acquisition as primary  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

REASON = "native_recording_saved_original_process_exit_unknown"


def read_bound(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def audit(declaration_path):
    declaration = json.loads(declaration_path.read_text())
    if declaration["adapter"] != artifact(Path(__file__)):
        raise ValueError("recorded-student adapter differs from declaration")
    for source in declaration["sources"]:
        checked(Path(source["path"]), source["sha256"])
    plan = read_bound(declaration["plan"])
    run = next(r for r in plan["runs"] if r["run_id"] == declaration["run_id"])
    if run["arm"] != "reference_contrast":
        raise ValueError("unknown-student continuation is limited to uniform reference teaching")
    index = declaration["round_index"]
    slot = run["rounds"][index]
    out = Path(slot["student_directory"])
    if artifact(out / "manifest.json") != declaration["manifest"]:
        raise ValueError("assigned student manifest changed")
    manifest, bank, scene = collector.verify_manifest(out, execution=False)
    if len(manifest["cells"]) != 1 or manifest["cells"][0]["timed_schedule_mode"] != "learned":
        raise ValueError("one actual learned student required")
    cell = manifest["cells"][0]
    attempt_ref = artifact(Path(cell["output"]) / "attempt.json")
    if attempt_ref != declaration["attempt"]:
        raise ValueError("reconciled attempt changed")
    attempt = read_bound(attempt_ref)
    intent = read_bound(attempt["intent"])
    preupdate = read_bound(declaration["preupdate"])
    if (
        attempt.get("schema") != "motion2scene_reconciled_launch_receipt_v1"
        or attempt["exit_status"] is not None
        or attempt["command"] != cell["command"]
        or intent["command"] != cell["command"]
        or intent["manifest"] != declaration["manifest"]
        or preupdate["model"] != manifest["policy"]
        or preupdate["round_index"] != index
        or preupdate["expanded_plan"] != declaration["plan"]
        or preupdate["scene"] != slot["scene"]
    ):
        raise ValueError("recovery must retain the original invocation and pre-update policy")
    read_bound(preupdate["training_result"])
    checked(Path(manifest["policy"]["path"]), manifest["policy"]["sha256"])
    read_bound(intent["runtime_preflight"])
    success = read_bound(attempt["success_manifest"])
    log = checked(Path(attempt["rollout_log"]["path"]), attempt["rollout_log"]["sha256"])
    if (
        success.get("status") != "success"
        or success.get("exit_reason") != "max_render_steps"
        or "SONIC_EVAL_SUCCESS" not in log.read_text(errors="replace")
    ):
        raise ValueError("saved native completion evidence required")
    collector.validate_collection_context(cell, manifest, bank, scene, actual=True)
    row = collector.analyze_incomplete(cell, manifest, bank, scene, attempt, REASON)
    if (
        row["physics_steps"] != 1192
        or row["outcome"]["physical_rows_recorded"] != 298
        or row["outcome"]["sensor_packets_recorded"] != 298
        or row["outcome"]["task_outcome"] != "unknown"
        or not row["contact_audit"]["complete_synchronized_streams"]
    ):
        raise ValueError("this declaration requires full measured cost and an unknown outcome")
    if row["raw_artifacts"] != declaration["raw_artifacts"]:
        raise ValueError("saved physical recording changed")
    row["attempt"] = attempt_ref
    return declaration, out, row


def publish(declaration_path):
    declaration, out, row = audit(declaration_path)
    sources = []
    for source in sorted(collector.closure([Path(collector.__file__)])) + [Path(__file__)]:
        destination = out / "analysis_source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("existing analysis snapshot differs")
        if not destination.exists():
            shutil.copyfile(source, destination)
        sources.append({**artifact(source), "snapshot": artifact(destination)})
    result = dict(
        schema=collector.COLLECTION_SCHEMA,
        manifest=declaration["manifest"],
        rows=[row],
        teacher=None,
        teacher_scope="teacher branches remain separately required",
        analysis_implementation=sources,
        physics_steps=row["physics_steps"],
        unmeasured_failed_attempts=0,
        reconciliation=artifact(declaration_path),
        scope=(
            "One native student capture, fully measured cost; task outcome and OS exit "
            "remain unknown. Legacy failed_attempt status denotes infrastructure "
            "admission, not a measured task failure. No retry or outcome imputation."
        ),
    )
    ref = primary.commit(out / "result.json", result)
    folder = out.parent.parent / "controller"
    attempt_id = read_bound(read_bound(declaration["attempt"])["intent"])["attempt_id"]
    primary.commit(
        folder / "attempts" / attempt_id / "assessment.json",
        dict(
            manifest=declaration["manifest"],
            attempt=declaration["attempt"],
            row=row,
            reconciliation=artifact(declaration_path),
            scope="Independent recorded-cost audit; original process exit and task outcome unknown",
        ),
    )
    primary.commit(
        folder / "collections" / f"round{declaration['round_index']:03d}_student.json",
        dict(collection=ref),
    )
    return ref


@contextmanager
def recorded_student(declaration_path):
    declaration, out, expected_row = audit(declaration_path)
    result_ref = artifact(out / "result.json")
    if read_bound(result_ref)["rows"] != [expected_row]:
        raise ValueError("published student differs from re-audited recording")
    original_collect = primary.Controller.collect
    original_accounting = primary.Controller.accounting
    attempt_id = read_bound(read_bound(declaration["attempt"])["intent"])["attempt_id"]

    def collect(self, row, kind, model=None):
        if Path(row[f"{kind}_directory"]) != out:
            return original_collect(self, row, kind, model)
        if (
            kind != "student"
            or self.context["run"]["run_id"] != declaration["run_id"]
            or model != read_bound(declaration["manifest"])["policy"]
            or row["round_index"] != declaration["round_index"]
        ):
            raise ValueError("reconciled collection requested under another policy or slot")
        _, _, verified = audit(declaration_path)
        if read_bound(result_ref)["rows"] != [verified]:
            raise ValueError("reconciled collection changed")
        self.ledger()
        return result_ref

    def accounting(self):
        report = original_accounting(self)
        for attempt in report["attempts"]:
            if attempt.get("actual_process_receipt") == declaration["attempt"]:
                if attempt["attempt_id"] != attempt_id:
                    raise ValueError("recovered recording duplicated into another slot")
                attempt.update(
                    actual_process_receipt=None,
                    recording_reconciliation=declaration["attempt"],
                    actual_launch_confirmed=True,
                    launch_state="native_recording_verified_process_exit_unknown",
                )
        return report

    primary.Controller.collect = collect
    primary.Controller.accounting = accounting
    try:
        yield
    finally:
        primary.Controller.collect = original_collect
        primary.Controller.accounting = original_accounting


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--until-budget", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    resume_driver.validate_repair(args.repair, args.plan, args.adoption)
    declaration = json.loads(args.reconciliation.read_text())
    if declaration["plan"] != artifact(args.plan) or declaration["path_repair"] != artifact(
        args.repair
    ):
        raise ValueError("continuation declaration differs from the supplied plan or repair")
    if args.publish_only:
        print(json.dumps(publish(args.reconciliation)), flush=True)
    else:
        with recorded_student(args.reconciliation):
            result = resume_driver.resume(
                args.plan, args.adoption, args.until_budget, args.repair, watch=True
            )
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result["status"] == "complete" else 75)
