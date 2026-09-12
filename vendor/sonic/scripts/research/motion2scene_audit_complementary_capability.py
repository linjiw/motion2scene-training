#!/usr/bin/env python3
"""Post-acquisition CPU audit of a complementary capability/observation example."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance  # noqa: E402
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_timing import (  # noqa: E402
    audit_decision_visibility,
    observed_beam_faces,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_baselines import (  # noqa: E402
    context_oracle,
    script_context,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)


def read_ref(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def all_capsule_clearance(capsules, beam, offset):
    """Recompute every capsule without the proposal's AABB prune or clearance cap."""
    starts, ends, radii, _ = capsules
    angle = beam["yaw_rad"] + offset[3]
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    center = np.r_[beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
    center += offset[:3]
    half = np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]]) / 2
    values = capsule_box_clearance(
        (starts - center) @ rotation, (ends - center) @ rotation, radii, -half, half
    )
    return float(values.min())


def forecast_audit(screen_path, scene):
    screen = json.loads(screen_path.read_text())
    registration = read_ref(screen["registration"])
    selected = screen["selected"]
    index = selected["grid_index"]
    if (
        scene != read_ref(screen["scene"])
        or scene["beams"] != [screen["selected_beam"]]
        or registration["grid_world_beams"][index] != scene["beams"][0]
        or scene["provenance"]["registration"] != screen["registration"]
        or registration["margin_m"] != 0.01
    ):
        raise ValueError("physical definition must exactly bind the registered selected proposal")
    with np.load(checked(Path(screen["nominal"]["path"]), screen["nominal"]["sha256"])) as archive:
        nominal = archive["clearance_m"]
        if archive["option_ids"].tolist() != registration["option_ids"]:
            raise ValueError("proposal option order differs")
    margin, positive, negative = (
        registration["margin_m"],
        selected["positive"],
        selected["negatives"],
    )
    # Verify deterministic shortlist and winner from the retained complete grid.
    expected_candidates = []
    for option in registration["positive_indices"]:
        eligible = [
            i
            for i in range(len(nominal))
            if nominal[i, option, 0] >= margin
            and np.all(nominal[i, registration["negative_indices"], 1] <= -margin)
        ]

        def slack(i):
            return min(
                nominal[i, option, 0] - margin,
                float((-nominal[i, registration["negative_indices"], 1] - margin).min()),
            )

        expected_candidates.extend(
            (option, i) for i in sorted(eligible, key=lambda i: (-slack(i), i))[:16]
        )
    if expected_candidates != [
        (row["positive_index"], row["grid_index"]) for row in screen["candidates"]
    ]:
        raise ValueError("retained shortlist differs from the registered deterministic rule")
    for row in screen["candidates"]:
        values = np.asarray(row["positive_offset_clearance_m"])
        if values.shape != (81,) or not np.isfinite(values).all():
            raise ValueError("all registered positive offset values required")
        negatives = nominal[row["grid_index"], registration["negative_indices"], 1]
        expected_slack = min(float(values.min()) - margin, float((-negatives - margin).min()))
        if (
            row["eligible"] != bool(values.min() >= margin)
            or row["robust_slack_m"] != expected_slack
        ):
            raise ValueError("candidate eligibility/slack differs from recorded values")
    winner = max(
        (row for row in screen["candidates"] if row["eligible"]),
        key=lambda row: (row["robust_slack_m"], -row["grid_index"], -row["positive_index"]),
    )
    if winner != selected:
        raise ValueError("selected proposal differs from the registered ranking")
    source = read_ref(registration["references"][0])
    manifest = read_ref(source["manifest"])
    geometry = read_ref(manifest["geometry"])
    for layer in geometry["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
    shapes = {"outer": {}, "inner": {}}
    for shape in geometry["shapes"]:
        capsule = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        shapes["outer"].setdefault(shape["owner"], []).append(capsule)
        if shape["role"] == "native_primitive_subset":
            shapes["inner"].setdefault(shape["owner"], []).append(capsule)
    measured, positive_offsets = {}, None
    for row in source["rows"]:
        evidence = read_ref(row["evidence"])
        if not row["qualified"] or not all(evidence["checks"].values()):
            raise ValueError("forecast requires qualified source execution")
        ref = evidence["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
        option = row["cell_id"]
        measured[option] = {}
        for kind, members in shapes.items():
            capsules = body_capsules_world(
                payload["body_pos_w"],
                payload["body_quat_w"],
                payload["body_names"],
                capsules=members,
            )
            measured[option][kind] = all_capsule_clearance(capsules, scene["beams"][0], np.zeros(4))
            if option == positive and kind == "outer":
                positive_offsets = [
                    all_capsule_clearance(capsules, scene["beams"][0], np.asarray(offset))
                    for offset in registration["offsets_world_xyz_yaw"]
                ]
    maximum_error = max(
        abs(measured[option][kind] - screen["all_selected_nominal_clearance_m"][option][kind])
        for option in measured
        for kind in shapes
    )
    offset_error = float(
        np.max(np.abs(np.asarray(positive_offsets) - selected["positive_offset_clearance_m"]))
    )
    if (
        maximum_error > 1e-12
        or offset_error > 1e-12
        or min(positive_offsets) < margin
        or any(measured[option]["inner"] > -margin for option in negative)
    ):
        raise ValueError("full-capsule recomputation does not confirm the proposed finite contrast")
    return dict(
        screen=artifact(screen_path),
        registration=screen["registration"],
        selected_grid_index=index,
        selected_positive=positive,
        negative_options=negative,
        beam=scene["beams"][0],
        nominal_clearance_m=measured,
        minimum_positive_81_offset_clearance_m=min(positive_offsets),
        maximum_recomputed_nominal_error_m=maximum_error,
        maximum_recomputed_offset_error_m=offset_error,
        selection_rule_verified=True,
        recorded_grid_size=len(nominal),
        recorded_proposal_queries=screen["clearance_queries"],
        recorded_proposal_seconds=screen["seconds"],
        audit_full_capsule_queries=14 + 81,
        physical_offsets_evaluated=1,
        scope=(
            "Full native-primitive replay of the chosen finite geometric forecast; "
            "only nominal scene received new physical executions."
        ),
    )


def beam_force_trace(row, beam_path):
    mapping = read_ref(row["environment_mapping"])
    with np.load(
        checked(Path(row["environment_pairs"]["path"]), row["environment_pairs"]["sha256"])
    ) as archive:
        force, steps, dt = (
            archive["force_w"],
            archive["physics_steps"],
            float(archive["physics_dt_s"]),
        )
        bodies = archive["body_names"].tolist()
    if bodies != mapping["pair_subject_body_names"] or not np.array_equal(
        steps, np.arange(1, 1193)
    ):
        raise ValueError("exact articulation body ordering and complete physics clock required")
    vectors = []
    for index, body in enumerate(bodies):
        paths = mapping["sensors"][body]["filter_paths"]
        if paths.count(beam_path) != 1:
            raise ValueError("every subject needs exactly one measured beam counterpart")
        vectors.append(force[:, index, paths.index(beam_path), :])
    norms = np.linalg.norm(np.stack(vectors, axis=1), axis=-1)
    trace = norms.max(axis=1)
    contacts = np.flatnonzero(trace > 1.0)
    return (
        steps * dt,
        trace,
        dict(
            maximum_beam_normal_force_n=float(trace.max()),
            first_beam_force_above_1n_s=(
                None if not len(contacts) else float(steps[contacts[0]] * dt)
            ),
            physics_steps_above_1n=int(len(contacts)),
            contacting_bodies=[bodies[i] for i in np.flatnonzero(norms.max(axis=0) > 1.0)],
        ),
    )


def sensor_comparisons(group, original_groups):
    rows = []
    for target in group["targets"]:
        names, values = target["feature_names"], np.asarray(target["features"])
        selected = np.array([name.startswith("corridor_") for name in names])
        comparisons = []
        for other in original_groups:
            original = next(t for t in other["targets"] if t["phase_tick"] == target["phase_tick"])
            if original["feature_names"] != names:
                raise ValueError("cross-context comparison needs exact common sensor/state schema")
            delta = np.abs(values - np.asarray(original["features"]))
            comparisons.append(
                dict(
                    scene_id=other["scene_id"],
                    exact_full114_alias=bool(np.all(delta == 0)),
                    exact_sensor28_alias=bool(np.all(delta[selected] == 0)),
                    changed_sensor_feature_count=int(np.sum(delta[selected] != 0)),
                    maximum_sensor_feature_delta=float(delta[selected].max()),
                    differing_sensor_features=[
                        dict(
                            name=names[i],
                            complementary=float(values[i]),
                            original=float(original["features"][i]),
                        )
                        for i in np.flatnonzero(selected & (delta != 0))
                    ],
                )
            )
        rows.append(
            dict(
                phase_tick=target["phase_tick"],
                corridor_features={
                    name: float(value)
                    for name, value in zip(names, values, strict=True)
                    if name.startswith("corridor_")
                },
                comparisons=comparisons,
            )
        )
    return rows


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source_files = []
    for path in sorted(closure([Path(__file__)])):
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        source_files.append({**artifact(path), "snapshot": artifact(destination)})
    write_new(
        args.out / "registration.json",
        dict(
            status="post_acquisition_independent_audit",
            collection=artifact(args.collection),
            screen=artifact(args.screen),
            original_baseline_selection=artifact(args.original_baseline),
            implementation=source_files,
            new_physics_steps=0,
            scope=(
                "Nominal complementary development capability, forecast and observation audit; "
                "no change to original tuning or model."
            ),
        ),
    )
    result = json.loads(args.collection.read_text())
    manifest = read_ref(result["manifest"])
    bank = load_verified_registry(manifest["registry"]["path"], manifest["registry"]["sha256"])
    scene = read_ref(manifest["scene_definition"])
    group = audit_collection(args.collection, bank, manifest["registry"])
    if (
        group["assigned_branches"] != 7
        or group["physics_steps"] != 8344
        or any(not b["task_outcome_admitted"] for b in group["branch_assessments"])
    ):
        raise ValueError("complete independently known seven-branch physical panel required")
    write_new(args.out / "teacher.json", group)
    forecast = forecast_audit(args.screen, scene)
    write_new(args.out / "forecast.json", forecast)
    traces, physical = [], []
    for row in result["rows"]:
        times, trace, contact = beam_force_trace(row, manifest["environment_beam_paths"][0])
        traces.append(trace)
        physical.append(
            dict(
                option_id=row["forced_option_id"],
                passed=row["pass"],
                passage_time_s=row["costs"]["passage_time_s"],
                measurement_admitted=row["measurement_admitted"],
                physics_steps=row["physics_steps"],
                return_verified=row["returned_to_neutral"],
                whole_horizon_stable=row["whole_horizon_stable"],
                reset_count=row["passage"]["reset_count"],
                fall_observed=row["passage"]["fall_observed"],
                maximum_environment_force_n=row["contact_audit"][
                    "maximum_undesired_environment_force_n"
                ],
                external_contacts=row["contact_audit"]["external_contacts"],
                **contact
            )
        )
    np.savez_compressed(
        args.out / "beam_force_traces.npz",
        elapsed_s=times,
        maximum_beam_normal_force_n=np.asarray(traces),
        option_ids=np.asarray([r["option_id"] for r in physical]),
    )
    neutral = next(r for r in result["rows"] if r["forced_option_id"] == "neutral")
    observations = read_ref(neutral["sensor"])["observations"]
    phases = []
    for target in group["targets"]:
        tick = target["phase_tick"]
        visibility = audit_decision_visibility(observations, scene["beams"][0], tick)
        phase = dict(
            phase_tick=tick,
            command_time_s=tick / 50,
            capture_elapsed_s=observations[tick - 1]["capture_elapsed_s"],
            visibility=visibility,
            entry_packet_beam_ray_hits=observed_beam_faces(
                observations[tick - 1]["measurements"], scene["beams"][0]
            ),
            target={
                key: value
                for key, value in target.items()
                if key not in ("features", "feature_names")
            },
            prefix_comparisons=[r for r in group["prefix_comparisons"] if r["phase_tick"] == tick],
        )
        phases.append(phase)
    original = json.loads(args.original_baseline.read_text())
    reaudits = read_ref(original["actual_reaudits"])
    # Bind and independently reconstruct the original five histories, avoiding
    # comparisons based only on a copied summary or privileged geometry.
    original_groups = []
    for archived in reaudits["groups"]:
        ref = archived["collection"]
        checked(Path(ref["path"]), ref["sha256"])
        actual = audit_collection(Path(ref["path"]), bank, manifest["registry"])
        if actual != archived:
            raise ValueError("original comparison group differs from independent re-audit")
        original_groups.append(actual)
    comparisons = sensor_comparisons(group, original_groups)
    oracle = context_oracle(bank, group, result["rows"])
    parameters = read_ref(original["script_parameters"])
    scripted = script_context(bank, group, oracle, parameters)
    report = dict(
        registration=artifact(args.out / "registration.json"),
        teacher=artifact(args.out / "teacher.json"),
        forecast=artifact(args.out / "forecast.json"),
        physical_branches=physical,
        phases=phases,
        sensor_comparisons=comparisons,
        frozen_selected_script_proxy=scripted,
        beam_force_traces=artifact(args.out / "beam_force_traces.npz"),
        new_physics_steps=0,
        source_physics_steps=8344,
        original_comparison_physics_steps_reaudited=41720,
        scope=(
            "Post-acquisition nominal development example. No trained policy is modified or executed. "
            "Exact finite-corpus sensor distinctions and successful continuations do not prove "
            "general perceptual sufficiency or noise robustness."
        ),
    )
    write_new(args.out / "result.json", report)
    print(
        json.dumps(
            {
                "result": artifact(args.out / "result.json"),
                "pass_count": sum(r["passed"] for r in physical),
                "branches": physical,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--original-baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
