"""Run the existing CPU M4 audit after every original dispatcher boundary closes."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/home/linjiw/groot-wbc-sonic-sim-trackb")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent


def ref(path):
    return dict(
        path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def checked(item):
    path = Path(item["path"])
    if ref(path)["sha256"] != item["sha256"]:
        raise ValueError("pinned source or receipt changed: " + str(path))
    return path


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def readiness(registration):
    for source in registration["sources"]:
        checked(source)
    dispatcher = Path(registration["dispatcher_directory"])
    missing = []
    for row in registration["boundaries"]:
        path = dispatcher / f"complete_{row['index']:03d}.json"
        if not path.is_file():
            missing.append(row["run_id"])
            continue
        result = json.loads(path.read_text())
        if result["assignment"] != row:
            raise ValueError("dispatcher boundary differs from original assignment")
        checked(result["training_result"])
        checked(result["policy"])
    return missing


def main(check_only):
    registration = json.loads((HERE / "registration.json").read_text())
    missing = readiness(registration)
    if check_only:
        print(
            json.dumps(
                dict(
                    status="waiting_for_complete_original_M4" if missing else "ready",
                    missing=missing,
                    checked_references=len(registration["sources"]),
                    new_physics_steps=0,
                )
            )
        )
        return
    write_new(
        HERE / "started.json",
        dict(
            pid=os.getpid(),
            registration=ref(HERE / "registration.json"),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    previous = None
    while missing:
        proc = Path("/proc") / str(registration["dispatcher_pid"]) / "cmdline"
        command = proc.read_bytes().split(b"\0") if proc.exists() else []
        if (
            str(Path(registration["dispatcher_directory"]) / "dispatch.py").encode()
            not in command
        ):
            raise RuntimeError(
                "original dispatcher ended with incomplete M4; inspect retained attempts"
            )
        if missing != previous:
            print(
                json.dumps(
                    dict(status="waiting_for_complete_original_M4", missing=missing)
                ),
                flush=True,
            )
            previous = missing
        time.sleep(45)
        missing = readiness(registration)
    target = Path(registration["analysis_output"])
    if target.exists():
        raise FileExistsError(
            "retain existing analysis output; do not replace or repeat it"
        )
    write_new(
        HERE / "ready.json",
        dict(
            registration=ref(HERE / "registration.json"),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    command = registration["command"]
    write_new(HERE / "launch.json", dict(command=command, new_physics_steps=0))
    with (HERE / "analysis.log").open("x") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        write_new(
            HERE / "failed.json",
            dict(exit_code=process.returncode, log=ref(HERE / "analysis.log")),
        )
        raise RuntimeError(
            "M4 audit failed; retain output and inspect before any retry"
        )
    result = target / "result.json"
    audit = json.loads(result.read_text())
    if (
        audit["budget"] != 4
        or audit["plan"] != registration["plan"]
        or audit["new_physics_steps"] != 0
    ):
        raise ValueError("unexpected completed audit identity or scope")
    write_new(
        HERE / "complete.json",
        dict(result=ref(result), log=ref(HERE / "analysis.log"), new_physics_steps=0),
    )
    print(
        json.dumps(
            dict(status="M4_recorded_prefix_audit_complete", result=ref(result))
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
