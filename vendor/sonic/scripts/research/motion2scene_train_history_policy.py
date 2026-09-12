#!/usr/bin/env python3
"""Fit the repeated-decision integration policy from exact physical teacher pairs."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_timing_diagnostic import artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_history_imitation import (
    fit_history_policy,
)


def train(results, out, steps=1000):
    feature_names = None
    features, labels, times, admission, provenance = [], [], [], [], []
    seen = set()
    for path in results:
        result = json.loads(path.read_text())
        if result["schema"] != "motion2scene_sensor_history_acquisition_v1":
            raise ValueError("unsupported physical teacher acquisition")
        manifest = json.loads(
            checked(Path(result["manifest"]["path"]), result["manifest"]["sha256"]).read_text()
        )
        rows = {row["cell_id"]: row for row in result["rows"]}
        cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
        for target in result["teachers"]:
            pair = [rows[key] for key in target["branch_cell_ids"]]
            identity = tuple(row["trajectory"]["sha256"] for row in pair)
            if identity in seen:
                raise ValueError("duplicate physical teacher pair across input results")
            seen.add(identity)
            if any(row["mode"] != mode for row, mode in zip(pair, ("forced_walk", "forced_adapt"))):
                raise ValueError("teacher target must bind the executed walk/adapt branches")
            if len({cells[row["cell_id"]]["runtime_seed"] for row in pair}) != 1:
                raise ValueError("teacher physics seeds differ")
            admitted = all(
                (
                    target["paired_prefix"]["exact_match"],
                    target["decision_state_exact_match"],
                    target["decision_features_exact_match"],
                )
            )
            for row in pair:
                checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
                with np.load(
                    checked(Path(row["features"]["path"]), row["features"]["sha256"]),
                    allow_pickle=False,
                ) as arrays:
                    names = tuple(arrays["feature_names"].tolist())
                    if feature_names is None:
                        feature_names = names
                    elif feature_names != names:
                        raise ValueError("inconsistent student sensor/state feature schema")
                    ids = np.flatnonzero(
                        np.isclose(
                            arrays["phase_s"][: row["first_episode_frames"]],
                            target["phase_s"],
                            atol=1e-8,
                        )
                    )
                    if len(ids) != 1 or not np.array_equal(
                        arrays["features"][ids[0]], np.array(target["features"], np.float32)
                    ):
                        raise ValueError("teacher features do not match recorded pre-action inputs")
            features.append(target["features"])
            labels.append([bool(row["pass"]) for row in pair])
            times.append([row["costs"]["passage_time_s"] for row in pair])
            admission.append([bool(v and admitted) for v in target.get("admitted", [False, False])])
            provenance.append(
                {
                    "result": artifact(path),
                    "base_cell_id": target["base_cell_id"],
                    "phase_s": target["phase_s"],
                    "branches": [row["trajectory"] for row in pair],
                    "source": pair[0]["source"],
                    "admitted": admitted,
                }
            )
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "role": "development imitation integration fit; no held-out performance claim",
            "inputs": [artifact(path) for path in results],
            "implementation": [
                artifact(Path(__file__)),
                artifact(
                    Path(__file__).resolve().parents[2]
                    / "gear_sonic/dataset_generation/hallucination/motion2scene_history_imitation.py"
                ),
            ],
            "steps": steps,
            "learning_rate": 0.03,
            "l2": 0.001,
            "target": "passage first, then recorded passage-time regret",
            "provenance": provenance,
        },
    )
    started = time.monotonic()
    model, report = fit_history_policy(
        features, feature_names, labels, times, admission, steps=steps
    )
    np.savez_compressed(out / "policy.npz", **model)
    np.savez_compressed(
        out / "training.npz",
        features=features,
        pass_labels=labels,
        passage_time_s=np.asarray(times, float),
        admitted=admission,
    )
    write_new(
        out / "result.json",
        {
            **report,
            "fit_seconds": time.monotonic() - started,
            "policy": artifact(out / "policy.npz"),
            "registration": artifact(out / "registration.json"),
        },
    )
    print(json.dumps({"policy": artifact(out / "policy.npz"), **report}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    train(args.results, args.out, args.steps)
