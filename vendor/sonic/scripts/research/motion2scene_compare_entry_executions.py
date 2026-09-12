#!/usr/bin/env python3
"""Report observed whole-execution differences for predeclared entry-time pairs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)


def difference(first, second):
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape:
        return {"identical": False, "first_shape": list(a.shape), "second_shape": list(b.shape)}
    differing = np.any((a != b).reshape(len(a), -1), axis=1)
    return {
        "identical": bool(np.array_equal(a, b)),
        "maximum_absolute_difference": float(np.max(np.abs(a - b), initial=0)),
        "first_differing_recorded_index": (
            int(np.flatnonzero(differing)[0]) if differing.any() else None
        ),
    }


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    ref = manifest["whole_execution_comparison_contract"]
    contract = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    result = json.loads((out / "environment_result.json").read_text())
    captures, rows = {}, []
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        if attempt["exit_status"] != 0:
            continue
        path = folder / "trajectories"
        payload = load_reset_capture(path / "000000.trajectory.pkl")
        capture = json.loads((path / "decision_capture.json").read_text())
        interface = json.loads((path / "reactive_interface.json").read_text())
        tick = cell["required_exact_prefix_frames"]
        frame = interface["observations"][tick - 1]
        entry_matches = all(
            capture[k] == frame[k]
            for k in ("tick", "physics_step", "root_pos_w", "root_quat_w", "state")
        )
        prefix = json.loads((path / "prefix_audit.json").read_text())
        roots = np.asarray(payload["root_pos_w"])
        rows.append(
            {
                "option_id": cell["cell_id"],
                "qualified": next(
                    r["qualified"] for r in result["rows"] if r["cell_id"] == cell["cell_id"]
                ),
                "entry_capture_tick": capture["tick"],
                "own_entry_capture_synchronized": entry_matches and capture["tick"] == tick,
                "prefix_exact": prefix["exact_match"],
                "prefix_frames": prefix["frames"],
                "expected_prefix_frames": tick,
                "actual_loaded_bank_shape": interface["actual_bank_shape"],
                "root_displacement_xyz_m": (roots[-1] - roots[0]).tolist(),
                "minimum_root_height_m": float(roots[:, 2].min()),
                "minimum_upright_gravity": float(
                    (-np.asarray(payload["projected_gravity_b"])[:, 2]).min()
                ),
                "actual_simulation_elapsed_s": len(payload["motion_time_s"]) * 0.02,
                "mechanical_work_j": None,
                "trajectory": artifact(path / "000000.trajectory.pkl"),
                "entry_capture": artifact(path / "decision_capture.json"),
            }
        )
        captures[cell["cell_id"]] = (path, payload)
    comparisons = []
    for first, second in contract["pairs"]:
        if first not in captures or second not in captures:
            comparisons.append({"pair": [first, second], "status": "incomplete execution retained"})
            continue
        path_a, a = captures[first]
        path_b, b = captures[second]
        compared = {key: difference(a[key], b[key]) for key in contract["arrays"]}
        for filename, key in (
            ("all_body_contacts.npz", "net_force_w"),
            ("environment_pair_contacts.npz", "force_w"),
        ):
            with np.load(path_a / filename) as x, np.load(path_b / filename) as y:
                compared[filename + ":" + key] = difference(x[key], y[key])
        comparisons.append(
            {
                "pair": [first, second],
                "status": "complete",
                "all_compared_execution_arrays_identical": all(
                    x["identical"] for x in compared.values()
                ),
                "arrays": compared,
            }
        )
    write_new(
        out / "entry_execution_comparison.json",
        {
            "schema": "motion2scene_entry_execution_comparison_v1",
            "contract": ref,
            "source": artifact(Path(__file__)),
            "qualification_results": artifact(out / "environment_result.json"),
            "rows": rows,
            "comparisons": comparisons,
            "scope": (
                "Later commitment does not imply changed physical adaptation or cost; "
                "reference clocks remain unchanged."
            ),
        },
    )
    print(
        json.dumps(
            {"comparisons": [{k: v for k, v in x.items() if k != "arrays"} for x in comparisons]}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args().out)
