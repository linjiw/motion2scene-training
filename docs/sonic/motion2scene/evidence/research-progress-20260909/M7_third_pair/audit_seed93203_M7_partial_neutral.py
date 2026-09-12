"""Recheck the measured neutral failure in either matched seed-93203 M7 corpus."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/home/linjiw/groot-wbc-sonic-sim-trackb")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_collect_timed_schedules as collector


def ref(path):
    return dict(
        path=str(path.absolute()),
        sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def bound(reference):
    path = Path(reference["path"])
    assert ref(path) == reference
    return json.loads(path.read_text())


def run(arm, out):
    assert not out.exists()
    root = (
        Path(
            "/home/linjiw/research-data/groot-wbc/m2s-expanded-acquisition-20260909-v1"
        )
        / f"seed93203_{arm}"
    )
    assessment = ref(
        root / "controller/attempts/round007_teacher_forced_neutral/assessment.json"
    )
    saved = bound(assessment)
    row = saved["row"]
    manifest = bound(saved["manifest"])
    scene = bound(manifest["scene_definition"])
    attempt = bound(row["attempt"])
    assert saved["attempt"] == row["attempt"]
    assert manifest["split"] == "development"
    cell = next(c for c in manifest["cells"] if c["forced_option_id"] == "neutral")
    assert cell["runtime_seed"] == 93203
    assert row["mode"] == "forced" and row["forced_option_id"] == "neutral"
    assert row["outcome"]["measurement_status"] == "partial"
    assert attempt["exit_status"] != 0
    for reference in row["raw_artifacts"].values():
        assert ref(Path(reference["path"])) == reference
    bank = collector.load_verified_registry(
        manifest["registry"]["path"], manifest["registry"]["sha256"]
    )
    collector.validate_collection_context(cell, manifest, bank, scene, actual=True)
    fresh = collector.analyze_incomplete(
        cell, manifest, bank, scene, attempt, "nonzero_process_exit"
    )
    assert fresh == {k: v for k, v in row.items() if k != "attempt"}
    assert fresh["outcome"]["task_outcome"] == "failure"
    assert fresh["contact_audit"]["complete_synchronized_streams"]
    assert fresh["physics_steps"] == 756
    report = dict(
        schema="motion2scene_bounded_partial_failure_reaudit_v1",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_id=f"seed93203_{arm}",
        budget=7,
        assessment=assessment,
        manifest=saved["manifest"],
        attempt=row["attempt"],
        raw_artifacts=row["raw_artifacts"],
        script=ref(Path(__file__)),
        collector_source=ref(Path(collector.__file__)),
        recomputed_row_equals_original=True,
        measured_physics_steps=fresh["physics_steps"],
        outcome=fresh["outcome"],
        contact_audit=fresh["contact_audit"],
        new_physical_executions=0,
        new_fits=0,
        interpretation=(
            "Independent recomputation from hash-checked raw partial capture, sensor and synchronized "
            "contact streams reproduces the original known task failure and actual measured cost. "
            "Later uncaptured fall, passage and recovery outcomes remain unknown. This does not "
            "retry the episode, re-audit unrelated captures or validate the complete native inventory."
        ),
    )
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            dict(
                output=ref(out),
                measured_steps=fresh["physics_steps"],
                outcome=fresh["outcome"],
                contact_audit=fresh["contact_audit"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=["analytic_contrast", "observation_curriculum"], required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.arm, args.out)
