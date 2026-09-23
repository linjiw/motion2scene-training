import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "navigation_distill" / "export_results.py"
)
spec = importlib.util.spec_from_file_location("export_results", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

HEADER = [
    "stage",
    "run",
    "success",
    "reached",
    "hold",
    "final_distance_m",
    "max_force_n",
    "fell",
    "steps",
    "stop",
    "actor_profile",
    "student",
]


def task_result(success=True):
    return {
        "navigation_success": success,
        "goal_ever_reached": True,
        "max_hold_ticks": 50 if success else 12,
        "final_goal_distance_m": 0.08123,
        "max_undesired_force_n": 0.0,
        "fell": False,
        "control_steps": 238,
        "stop_reason": "goal_hold" if success else "deadline",
        "actor_profile": "nav_goal_map_localization_v2",
        "student_sha256": "ab" * 32,
    }


def recovery(tick):
    return {
        "takeover_tick": tick,
        "supported": True,
        "rows": 200,
        "supported_rows": 180,
        "suffix_score": {"max_hold_ticks": 50, "final_goal_distance_m": 0.07},
        "behavior_checkpoint": {"sha256": "cd" * 32},
    }


def run_dir(packet, rel, result, rec=None, extras=True):
    d = packet / rel
    (d / "task").mkdir(parents=True)
    (d / "task/task-result.json").write_text(json.dumps(result))
    if rec:
        (d / "task/recovery.json").write_text(json.dumps(rec))
    if extras:
        (d / "task/trace.npz").write_bytes(b"\0" * 128)
        (d / "process-result.json").write_text('{"exit_code": 0, "wall_seconds": 38}')
        (d / "config.json").write_text('{"student_sha256": "ab"}')
        (d / "command.json").write_text('["python", "++seed=91260"]')
        (d / "native.log").write_text("log")
        (d / "hydra").mkdir()
        (d / "hydra/eval.log").write_text("log")
    return d


@pytest.fixture
def packet(tmp_path):
    p = tmp_path / "nav"
    run_dir(p, "eval/dag-approach-c2-91260/00908-stop-clear", task_result())
    run_dir(p, "eval/dag-approach-c2-91260/00120-stop-clear", task_result(False))
    run_dir(p, "collect/motor-demo-91260/00908-stop-clear", task_result(), recovery(0))
    run_dir(p, "dagger/dag-approach/c2-recovery/00908-stop-clear-t26", task_result(), recovery(26))
    return p


def read(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_default_export_keeps_format_and_skips_dagger(packet, tmp_path):
    out = tmp_path / "out"
    export.main(["--packet", str(packet), "--out", str(out)])
    rows = read(out / "task-results.csv")
    assert list(rows[0]) == HEADER
    assert [(r["stage"], r["run"], r["success"]) for r in rows] == [
        ("motor-demo-91260", "00908-stop-clear", "True"),
        ("dag-approach-c2-91260", "00120-stop-clear", "False"),
        ("dag-approach-c2-91260", "00908-stop-clear", "True"),
    ]
    assert rows[0]["final_distance_m"] == "0.081"
    rec = read(out / "recovery-attempts.csv")
    assert [(r["stage"], r["takeover_tick"], r["behavior"]) for r in rec] == [
        ("motor-demo-91260", "0", "cd" * 32)
    ]
    assert sorted(p.name for p in out.iterdir()) == ["recovery-attempts.csv", "task-results.csv"]


def test_copy_receipts_copies_only_small_json(packet, tmp_path):
    out = tmp_path / "out"
    export.main(
        ["--packet", str(packet), "--out", str(out), "--copy-receipts", "dag-approach-c2-91260"]
    )
    copied = sorted(
        str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.suffix != ".csv"
    )
    assert copied == [
        f"dag-approach-c2-91260/{run}/{name}"
        for run in ("00120-stop-clear", "00908-stop-clear")
        for name in ("command.json", "config.json", "process-result.json", "task-result.json")
    ]
    assert not list(out.rglob("*.npz")) and not list(out.rglob("*.log"))
    assert not (out / "motor-demo-91260").exists()


def test_dagger_glob_prefix_and_nested_stage(packet, tmp_path):
    out = tmp_path / "out"
    export.main(
        [
            "--packet",
            str(packet),
            "--out",
            str(out),
            "--prefix",
            "dagger-c2-",
            "--glob",
            "dagger/*/c2-recovery/*/task/task-result.json",
        ]
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "dagger-c2-recovery-attempts.csv",
        "dagger-c2-task-results.csv",
    ]
    rec = read(out / "dagger-c2-recovery-attempts.csv")
    assert [(r["stage"], r["run"], r["takeover_tick"]) for r in rec] == [
        ("dag-approach/c2-recovery", "00908-stop-clear-t26", "26")
    ]


def test_copy_receipts_rejects_unknown_stage_and_large_files(packet, tmp_path):
    with pytest.raises(SystemExit, match="no runs for stage"):
        export.main(
            ["--packet", str(packet), "--out", str(tmp_path / "a"), "--copy-receipts", "nope"]
        )
    big = packet / "eval/dag-approach-c2-91260/00908-stop-clear/config.json"
    big.write_text("x" * (export.RECEIPT_MAX_BYTES + 1))
    with pytest.raises(SystemExit, match="expected to be small"):
        export.main(
            [
                "--packet",
                str(packet),
                "--out",
                str(tmp_path / "b"),
                "--copy-receipts",
                "dag-approach-c2-91260",
            ]
        )
