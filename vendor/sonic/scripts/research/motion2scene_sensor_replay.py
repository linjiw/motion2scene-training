#!/usr/bin/env python3
"""Replay recorded rays through prospective sensor sensitivity interventions."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import evaluate, read_checked, ridge_choice  # noqa: E402
from motion2scene_sensor_perturbations import (  # noqa: E402
    PerturbedObservationStream,
    SensorPerturbation,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (  # noqa: E402
    SensorRay,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    load_schedule_policy,
    schedule_history_features,
)

CONDITIONS = dict(
    nominal={},
    dropout_10=dict(dropout_probability=0.1),
    dropout_30=dict(dropout_probability=0.3),
    range_1cm=dict(range_noise_std_m=0.01),
    range_3cm=dict(range_noise_std_m=0.03),
    latency_40ms=dict(latency_s=0.04),
    latency_100ms=dict(latency_s=0.1),
)


def replay_features(interface, bank, settings):
    stream = PerturbedObservationStream(settings)
    phases = interface["phase_ticks"]
    decisions = []
    for packet in interface["observations"]:
        tick = packet["tick"]
        if tick > max(phases):
            break
        cache, _ = stream.push([SensorRay(**r) for r in packet["measurements"]], tick / 50)
        if tick not in phases:
            continue
        observation = stream.history.snapshot(
            packet["root_pos_w"], packet["root_quat_w"], cache["capture_elapsed_s"]
        )
        names, features = schedule_history_features(
            observation,
            stream.history.grid,
            packet["state"],
            tick,
            bank.option_ids.index(packet["active_before"]),
            np.asarray(packet["legal_mask"], dtype=bool),
            cache["observation_age_s"],
        )
        if tuple(names) != tuple(interface["feature_names"]):
            raise ValueError("sensor reconstruction changed the student interface")
        original = np.asarray(packet["features"])
        if settings == SensorPerturbation(seed=settings.seed) and not np.array_equal(
            features, original
        ):
            raise ValueError("zero perturbation does not reproduce nominal recorded features")
        decisions.append(
            dict(
                phase_tick=tick,
                features=features.tolist(),
                changed_feature_coordinates=int(np.sum(features != original)),
                observation_age_s=cache["observation_age_s"],
            )
        )
    if [r["phase_tick"] for r in decisions] != phases:
        raise ValueError("recorded neutral prefix must contain every decision")
    return decisions


def run(source, out, seed):
    original = read_checked(artifact(source / "result.json"))
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    ref = registration["registry"]
    bank = load_verified_registry(ref["path"], ref["sha256"])
    policy = load_schedule_policy(original["policy"]["path"], original["policy"]["sha256"], bank)
    _, _, entries = schedule_layout(bank)
    settings = {name: SensorPerturbation(seed=seed, **args) for name, args in CONDITIONS.items()}
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            source=artifact(source / "result.json"),
            teachers=original["teachers"],
            policy=original["policy"],
            conditions={name: asdict(value) for name, value in settings.items()},
            implementation=[
                artifact(Path(__file__)),
                artifact(Path(__file__).with_name("motion2scene_sensor_perturbations.py")),
            ],
            scope=(
                "recorded neutral-prefix ray reconstruction and branch proxies; "
                "new physical policy executions=0"
            ),
        ),
    )
    altered = {name: [] for name in settings}
    sources = []
    for group in groups:
        collection = read_checked(group["collection"])
        if read_checked(collection["manifest"])["split"] != "development":
            raise ValueError("sensor development cannot inspect reserved outcomes")
        neutral = next(r for r in collection["rows"] if r["forced_option_id"] == "neutral")
        interface = read_checked(neutral["sensor"])
        sources.append(neutral["sensor"])
        for name, config in settings.items():
            decisions = replay_features(interface, bank, config)
            updated = copy.deepcopy(group)
            indexed = {r["phase_tick"]: r for r in decisions}
            for target in updated["targets"]:
                target.update(indexed[target["phase_tick"]])
            altered[name].append(updated)
            print(
                json.dumps(
                    dict(
                        scene=group["scene_id"], condition=name, reconstructed_phases=len(decisions)
                    )
                ),
                flush=True,
            )
    results = {}
    for name, changed in altered.items():
        assessment = evaluate(lambda row: ridge_choice(policy, row), changed, entries)
        results[name] = dict(
            passing_branch_proxies=sum(r["branch_proxy_passed"] for r in assessment),
            contexts=len(changed),
            assessments=assessment,
            changed_decision_vectors=sum(
                t["changed_feature_coordinates"] > 0 for g in changed for t in g["targets"]
            ),
            reconstructed_decisions=[
                dict(scene_id=g["scene_id"], targets=g["targets"]) for g in changed
            ],
        )
    return write_new(
        out / "result.json",
        dict(experiment=experiment, sensors=sources, conditions=results, new_physics_steps=0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=95001)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out, args.seed)))
