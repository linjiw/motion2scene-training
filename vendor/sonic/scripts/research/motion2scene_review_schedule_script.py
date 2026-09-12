#!/usr/bin/env python3
"""Registered offline history reconstruction and provisional script sensitivity.

New schedule outcomes are unavailable here. Existing three-option outcomes are
reported only as a reference-family surrogate; no late-entry label is imputed.
"""

import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_closed_loop_policy import (  # noqa: E402
    history_policy_features,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (  # noqa: E402
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_script import (  # noqa: E402
    DEFAULT_CONFIG,
    choose_sensor_schedule,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    TimedOptionState,
    legal_timed_actions,
    load_verified_registry,
)

PHASES = (15, 50, 70)


def read_bound(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def prepare(args):
    parents = [artifact(args.teachers), artifact(args.comparison)]
    children = []
    for parent in parents:
        for row in read_bound(parent)["children"]:
            read_bound(row["manifest"])
            children.append(dict(parent=parent, manifest=row["manifest"]))
    configurations = [
        dict(
            minimum_observed_free_height_m=height,
            immediate_prior_band_count=immediate,
            later_prior_band_count=later,
            sustained_minimum_hazard_bands=sustained,
        )
        for height, immediate, later, sustained in itertools.product(
            (1.28, 1.32, 1.36, 1.40), (1, 2), (1, 2, 3), (1, 2, 3)
        )
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for path in sorted(closure([Path(__file__)])):
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        snapshots.append({**artifact(path), "snapshot": artifact(destination)})
    write_new(
        args.out / "registration.json",
        dict(
            schema="motion2scene_schedule_script_offline_review_v1",
            split="development",
            parents=parents,
            children=children,
            registry=artifact(args.registry),
            implementation=snapshots,
            configurations=configurations,
            provisional_default=DEFAULT_CONFIG,
            command_ticks=PHASES,
            old_history=dict(max_age_s=0.5, max_frames=26),
            reconstructed_history=dict(max_age_s=2.0, max_frames=101),
            reconstruction_admission=(
                "All archived100 base features must match the old history reconstruction "
                "exactly at all three phases, on all24 actual episodes, before2s readout."
            ),
            sequential_contexts=(
                "Only the six actual never-adapted neutral trajectories: forced_neutral "
                "on three8731 scenes and always_walk on same three8732 scenes. "
                "Read neutral history until first selected adaptation, then stop readout."
            ),
            outcome_scope=(
                "No physically validated new7-option actions are labeled here. Reference-family "
                "agreement with cheapest successful old3-option branch is a surrogate only; "
                "prior family is unmeasured and late timing is not granted an old early label."
            ),
            selection="Report all72 settings; no automatic adoption or physical performance ranking.",
            expected_episodes=24,
            expected_contexts=6,
            physics_steps=0,
            heldout_geometry_or_outcomes_used=False,
        ),
    )
    print(artifact(args.out / "registration.json"), flush=True)


def reconstruct(interface):
    names = tuple(interface["feature_names"][:100])
    if len(interface["observations"]) != 298:
        raise ValueError("complete298 sensor packets required")
    histories = [
        FloorCeilingHistory(HistoryGrid(max_age_s=0.5, max_frames=26)),
        FloorCeilingHistory(HistoryGrid(max_age_s=2.0, max_frames=101)),
    ]
    result = []
    for packet in interface["observations"][: max(PHASES)]:
        elapsed = packet["capture_elapsed_s"]
        delivered = packet["delivered_capture_elapsed_s"]
        if elapsed != delivered or len(packet["measurements"]) != 65:
            raise ValueError("this replay requires the recorded zero-delay65-ray sensor")
        rays = [SensorRay(**ray) for ray in packet["measurements"]]
        for memory in histories:
            memory.push(rays, delivered, delivered_time_s=elapsed)
        if packet["tick"] not in PHASES:
            continue
        values = []
        for memory in histories:
            snapshot = memory.snapshot(packet["root_pos_w"], packet["root_quat_w"], elapsed)
            current_names, x = history_policy_features(
                snapshot,
                memory.grid,
                packet["state"],
                packet["tick"] / 50,
                int(packet["active_before"] != "neutral"),
                [packet["legal_mask"][0], any(packet["legal_mask"][1:])],
                packet["observation_age_s"],
            )
            if current_names != names:
                raise ValueError("base feature schemas differ")
            values.append(x)
        original = np.asarray(packet["features"][:100], dtype=np.float32)
        if not np.array_equal(values[0], original):
            bad = np.flatnonzero(values[0] != original)
            raise ValueError(f"old reconstruction mismatch at tick{packet['tick']}: {bad.tolist()}")
        result.append(
            dict(
                tick=packet["tick"],
                active_before=packet["active_before"],
                original100=original.tolist(),
                reconstructed100=values[1].tolist(),
                changed_corridor_fields=[
                    names[i] for i in np.flatnonzero(values[0][:28] != values[1][:28])
                ],
                exact_old_reconstruction=True,
            )
        )
    return names, result


def family(option):
    if option == "neutral":
        return "neutral"
    return option.split("_e")[0]


def run(out):
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    for ref in registration["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    bank = load_verified_registry(
        registration["registry"]["path"], registration["registry"]["sha256"]
    )
    masks = {
        tick: legal_timed_actions(bank, TimedOptionState(), tick, np.zeros(7), np.zeros(7))[0]
        for tick in PHASES
    }
    # Verify every episode is complete before processing any predictions.
    sources = []
    for child in registration["children"]:
        manifest = read_bound(child["manifest"])
        path = Path(child["manifest"]["path"]).parent / "result.json"
        if not path.exists():
            raise RuntimeError(f"remaining physical result pending; no prediction made: {path}")
        result = json.loads(path.read_text())
        if result["manifest"] != child["manifest"]:
            raise ValueError("result manifest binding mismatch")
        sources.append((manifest, result, artifact(path)))
    audits, contexts, receipts, teachers = [], [], [], {}
    for manifest, result, result_ref in sources:
        scene = read_bound(manifest["scene_definition"])["scene_id"]
        cells = {row["cell_id"]: row for row in manifest["cells"]}
        for row in result["rows"]:
            if not row["measurement_admitted"]:
                raise ValueError("this planned replay requires all24 admitted measurements")
            cell = cells[row["cell_id"]]
            seed = cell["runtime_seed"]
            key = f"{scene}/seed{seed}"
            interface = read_bound(row["sensor"])
            feature_path = checked(Path(row["features"]["path"]), row["features"]["sha256"])
            with np.load(feature_path, allow_pickle=False) as archive:
                if not np.array_equal(
                    archive["features"], [r["features"] for r in interface["observations"]]
                ):
                    raise ValueError("raw sensor and archived features disagree")
            names, reconstructed = reconstruct(interface)
            audits.append(
                dict(scene=scene, seed=seed, cell_id=row["cell_id"], phases=reconstructed)
            )
            receipts.append(dict(result=result_ref, sensor=row["sensor"], features=row["features"]))
            if row["mode"] in ("forced", "constant_short", "constant_sustained", "always_walk"):
                chosen = row["schedule_audit"]["chosen_option_id"]
                teachers.setdefault(key, []).append(
                    dict(
                        option=chosen,
                        family=family(chosen),
                        passed=row["pass"],
                        time_s=row["costs"]["passage_time_s"],
                    )
                )
            if row["cell_id"] == "forced_neutral" or row["mode"] == "always_walk":
                if any(p["active_before"] != "neutral" for p in reconstructed):
                    raise ValueError("sequential replay must retain the actual neutral prefix")
                contexts.append(dict(context_id=key, feature_names=names, phases=reconstructed))
            print(json.dumps({"reconstructed_episodes": len(audits), "expected": 24}), flush=True)
    if len(audits) != 24 or len(contexts) != 6 or len({c["context_id"] for c in contexts}) != 6:
        raise ValueError("requires exactly24 episodes and six distinct neutral contexts")
    configurations = registration["configurations"]
    outputs = []
    for index, settings in enumerate(configurations):
        decisions = []
        for context in contexts:
            chosen, trace = "neutral", []
            for phase in context["phases"]:
                tick = phase["tick"]
                selected, detail = choose_sensor_schedule(
                    context["feature_names"],
                    phase["reconstructed100"],
                    masks[tick],
                    tick,
                    bank,
                    settings,
                )
                trace.append(dict(tick=tick, selected=selected, diagnostic=detail))
                if selected != "neutral":
                    chosen = selected
                    break
            successful = [row for row in teachers[context["context_id"]] if row["passed"]]
            minimum = min(row["time_s"] for row in successful)
            preferred = sorted(
                {row["family"] for row in successful if abs(row["time_s"] - minimum) <= 1e-9}
            )
            selected_family = family(chosen)
            decisions.append(
                dict(
                    context_id=context["context_id"],
                    selected=chosen,
                    selected_family=selected_family,
                    trace=trace,
                    old_three_option_cheapest_successful_families=preferred,
                    family_surrogate_agreement=(
                        (selected_family in preferred)
                        if selected_family != "prior_splice"
                        else None
                    ),
                    new_schedule_physical_outcome=None,
                )
            )
        outputs.append(dict(index=index, settings=settings, decisions=decisions))
    default = next(row for row in outputs if row["settings"] == registration["provisional_default"])
    write_new(
        out / "result.json",
        dict(
            schema="motion2scene_schedule_script_offline_review_result_v1",
            registration=artifact(registration_path),
            actual_inputs=receipts,
            reconstruction_audits=audits,
            neutral_contexts=contexts,
            configurations=outputs,
            provisional_default=default,
            old_three_option_teacher_family_table=teachers,
            selected_settings=None,
            physical_episode_predictions_evaluated=0,
            scope="Offline development readout only. Prior and later schedules receive no copied physical labels.",
        ),
    )
    print(json.dumps({"default": default, "configurations": len(outputs)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teachers", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else run(args.out)
