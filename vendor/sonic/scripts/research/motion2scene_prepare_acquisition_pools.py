#!/usr/bin/env python3
"""Prepare and search four common-domain training candidate queues; CPU only.

The complete draws, target assignments, RNG streams and fixed perturbations are
frozen before any clearance query. Observation curriculum remains pending native
sensor/physics evidence. Shared computation is counted once; independent-arm
equivalent query costs are separately reported without claiming timing speedups.
"""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import joblib  # noqa: E402
from motion2scene_materialize_evaluation import place_beams  # noqa: E402
from motion2scene_timed_scene_screen import beam_clearance  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (  # noqa: E402
    ARMS,
    SEEDS,
    acquisition_queues,
    budget_prefixes,
    draw_pool,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    author_course,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)


def read_bound(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def beam_key(beam):
    return tuple(
        np.round(
            [
                *beam["center_xy_m"],
                beam["yaw_rad"],
                beam["length_m"],
                beam["width_m"],
                beam["thickness_m"],
                beam["underside_m"],
            ],
            9,
        )
    )


def prepare(args):
    started = time.monotonic()
    result_ref = artifact(args.executions)
    result = read_bound(result_ref)
    manifest = read_bound(result["manifest"])
    registry_ref = result["registry"]
    bank = load_verified_registry(registry_ref["path"], registry_ref["sha256"])
    if (
        len(bank.option_ids) != 7
        or bank.frame_count != 299
        or any(not r["qualified"] for r in result["rows"])
    ):
        raise ValueError("the physically verified complete seven-schedule bank is required")
    geometry = read_bound(manifest["geometry"])
    references = [result_ref, result["manifest"], registry_ref, manifest["geometry"]]
    for layer in geometry["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
        references.append(layer)
    for row in result["rows"]:
        evidence = read_bound(row["evidence"])
        ref = evidence["artifacts"]["trajectory"]
        checked(Path(ref["path"]), ref["sha256"])
        references.extend([row["evidence"], ref])
    neutral_ref = manifest["cells"][0]["motion"]
    neutral = next(
        iter(joblib.load(checked(Path(neutral_ref["path"]), neutral_ref["sha256"])).values())
    )
    route = np.asarray(neutral["root_trans_offset"])[:, :2]
    scene_ref = manifest["cells"][0]["scene"]
    checked(Path(scene_ref["path"]), scene_ref["sha256"])
    lock_ref = artifact(args.lock)
    lock = read_bound(lock_ref)
    reserved = {
        beam_key(beam)
        for layout in lock["evaluation"]["layouts"]
        for variant in layout["fixed_world_variants"]
        for beam in variant["beams"]
    }
    old_contexts, integration_beams = [], set()
    for path in args.integration_definitions:
        ref = artifact(path)
        definition = read_bound(ref)
        old_contexts.append(ref)
        integration_beams.update(
            beam_key(b) for b in definition.get("beams", [definition.get("beam")]) if b is not None
        )
    pools = []
    for seed in SEEDS:
        candidates = draw_pool(lock["generation"], seed, args.draws_per_stratum, bank.option_ids)
        for candidate in candidates:
            candidate["beams"] = place_beams(route, candidate["beam_specifications"])
            keys = {beam_key(beam) for beam in candidate["beams"]}
            candidate["excluded_reason"] = (
                "exact_reserved_beam"
                if keys & reserved
                else "exact_integration_context_beam" if keys & integration_beams else None
            )
        pools.append(dict(seed=seed, candidates=candidates))
    offsets = [[0.0, 0.0, 0.0, 0.0]] + [
        list(v)
        for v in itertools.product(
            (-0.02, 0, 0.02), (-0.02, 0, 0.02), (-0.005, 0, 0.005), (-0.02, 0, 0.02)
        )
        if any(v)
    ]
    assert len(offsets) == 81
    args.out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(destination)})
    write_new(args.out / "draws.json", dict(pools=pools))
    write_new(
        args.out / "registration.json",
        dict(
            schema="motion2scene_four_arm_acquisition_pool_v1",
            registered_before_clearance=True,
            status="candidate_geometry_only",
            domain_lock=lock_ref,
            generation_domain=lock["generation"],
            excluded_integration_definitions=old_contexts,
            new_scene_or_heldout_outcome_files_loaded=False,
            common_option_qualification_outcomes_used=True,
            option_qualification_inputs=references,
            neutral_route=neutral_ref,
            room_template=scene_ref,
            registry=registry_ref,
            implementation=sources,
            draws=artifact(args.out / "draws.json"),
            acquisition_seeds=list(SEEDS),
            proposal_rng="Python random.Random MT19937; independent seed streams described in each draw",
            geometry_draw_order="round, five strata, beam, seven keys; round9 decimals; no rejection resampling",
            assigned_positive=(
                "Uniform option ID from separate RNG(seed+1000000), recorded for every "
                "candidate before queries; independent of negative outcomes"
            ),
            future_physics=(
                "One physics seed equal to acquisition seed for every matched group and every"
                " arm in that pool; branch orders frozen per candidate"
            ),
            draws_per_stratum=args.draws_per_stratum,
            arms=list(ARMS),
            margin_m=0.01,
            capped_clearance_m=0.20,
            offsets_world_xyz_yaw=offsets,
            positive_screen=(
                "All81 offsets, every beam, whole recorded50Hz trajectory outer envelope "
                ">=.01m. Nominal failure rejects before remaining80; every nominal-positive "
                "option receives all80 even after a later failure."
            ),
            negative_screen=(
                "Contrast arms only: at least one distinct option has native primitive-subset"
                " nominal interference <=-.01m at some beam."
            ),
            arm_selection={
                "uniform": "Any robust positive; original random draw order; no negative outcome is read",
                "target_only": (
                    "Assigned positive must clear all81; rank tightest positive fit above "
                    "common margin; no negative outcome is read"
                ),
                "analytic_contrast": (
                    "All executable pairs; maximum min(positive slack, negative nominal "
                    "slack); tie positive index then negative index then draw index"
                ),
                "observation_curriculum": (
                    "Identical analytic candidate queue initially; sensor deadline and "
                    "physical teacher/student-gap eligibility remain pending actual65-ray "
                    "full executions; future uniform.2/coverage.2/verified-gap.6 mixture"
                ),
            },
            candidate_identity=(
                "Three independently drawn candidate pools; paired common pools across arms. "
                "No physical corpora acquired and no reused five-context corpus independence "
                "claim."
            ),
            coverage=(
                "Report all five scene strata plus selected option/entry identity; "
                "observation timing class remains unknown until physical sensor audit"
            ),
            checkpoints=lock["acquisition"]["budget_checkpoints"],
            shared_option_qualification_cost=(
                "Inherited source qualification counted once in its original receipt, to be "
                "allocated equally in later arm reporting"
            ),
            geometry_accounting=(
                "Actual shared queries counted once; all logically consumed proposal rows and"
                " standalone arm-equivalent outer/inner query counts reported separately. "
                "Shared CPU search time is not an independent arm runtime benchmark."
            ),
            failures=(
                "Keep every draw, exclusion and geometric rejection. No silent resampling or "
                "selection from physical outcomes. Shortfall is geometric candidate "
                "availability, not unobservability."
            ),
            learned_proposal=(
                "Optional future acceleration only; existing negative learned-proposal "
                "development result retained; no fit or inference in this pipeline"
            ),
            physics_steps=0,
            sensor_queries=0,
            independent_physical_corpora=0,
        ),
    )
    write_new(
        args.out / "preparation_execution.json",
        dict(
            registration=artifact(args.out / "registration.json"),
            preparation_wall_seconds=time.monotonic() - started,
            clearance_queries=0,
            physics_steps=0,
        ),
    )
    print(
        json.dumps(
            {
                "registration": artifact(args.out / "registration.json"),
                "proposals": sum(len(p["candidates"]) for p in pools),
            }
        ),
        flush=True,
    )


def run(out):
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    for ref in registration["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    for ref in registration["option_qualification_inputs"]:
        checked(Path(ref["path"]), ref["sha256"])
    checked(Path(registration["domain_lock"]["path"]), registration["domain_lock"]["sha256"])
    source = read_bound(registration["option_qualification_inputs"][0])
    manifest = read_bound(source["manifest"])
    geometry = read_bound(manifest["geometry"])
    bank = load_verified_registry(
        registration["registry"]["path"], registration["registry"]["sha256"]
    )
    shapes = {"outer": {}, "inner": {}}
    for shape in geometry["shapes"]:
        capsule = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        shapes["outer"].setdefault(shape["owner"], []).append(capsule)
        if shape["role"] == "native_primitive_subset":
            shapes["inner"].setdefault(shape["owner"], []).append(capsule)
    executions = {}
    for row in source["rows"]:
        ref = read_bound(row["evidence"])["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
        if len(payload["body_pos_w"]) != bank.frame_count - 1:
            raise ValueError("complete299-frame-bank execution required")
        executions[row["cell_id"]] = {
            kind: body_capsules_world(
                payload["body_pos_w"], payload["body_quat_w"], payload["body_names"], capsules=value
            )
            for kind, value in shapes.items()
        }
    template = checked(
        Path(registration["room_template"]["path"]), registration["room_template"]["sha256"]
    ).read_text()
    pools = read_bound(registration["draws"])["pools"]
    offsets = registration["offsets_world_xyz_yaw"]
    summaries = []
    for pool in pools:
        folder = out / f"seed_{pool['seed']}"
        folder.mkdir(exist_ok=False)
        start = time.monotonic()
        candidates = [c for c in pool["candidates"] if c["excluded_reason"] is None]
        count, actions = len(candidates), len(bank.option_ids)
        outer = np.full((count, 81, actions), np.nan)
        inner = np.zeros((count, actions))
        evaluated = np.ones((count, actions), dtype=np.int64)
        query_outer = np.zeros((count, actions), dtype=np.int64)
        query_inner = np.zeros((count, actions), dtype=np.int64)
        for i, candidate in enumerate(candidates):
            for a, option_id in enumerate(bank.option_ids):
                for kind in ("outer", "inner"):
                    value = min(
                        beam_clearance(executions[option_id][kind], beam, offsets[0])
                        for beam in candidate["beams"]
                    )
                    if kind == "outer":
                        outer[i, 0, a] = value
                        query_outer[i, a] += len(candidate["beams"])
                    else:
                        inner[i, a] = value
                        query_inner[i, a] += len(candidate["beams"])
                if outer[i, 0, a] >= registration["margin_m"]:
                    for j, offset in enumerate(offsets[1:], 1):
                        outer[i, j, a] = min(
                            beam_clearance(executions[option_id]["outer"], beam, offset)
                            for beam in candidate["beams"]
                        )
                        query_outer[i, a] += len(candidate["beams"])
                    evaluated[i, a] = 81
            if (i + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "seed": pool["seed"],
                            "screened": i + 1,
                            "total": count,
                            "seconds": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
        clearance_seconds = time.monotonic() - start
        selection_started = time.monotonic()
        minima = np.nanmin(outer, axis=1)
        queues = acquisition_queues(candidates, bank.option_ids, minima, evaluated, inner)
        selection_seconds = time.monotonic() - selection_started
        raw_path = folder / "geometry_queries.npz"
        np.savez_compressed(
            raw_path,
            candidate_ids=np.asarray([c["candidate_id"] for c in candidates]),
            option_ids=np.asarray(bank.option_ids),
            outer_clearance_by_offset_m=outer,
            nominal_inner_clearance_m=inner,
            evaluated_offset_counts=evaluated,
            outer_beam_queries=query_outer,
            inner_beam_queries=query_inner,
        )
        candidate_rows = []
        for i, candidate in enumerate(candidates):
            scene_path = folder / "scenes" / f"{candidate['candidate_id']}.usda"
            scene_path.parent.mkdir(exist_ok=True)
            scene_path.write_text(
                author_course(template, candidate["beams"], course_id=candidate["candidate_id"])
            )
            definition_path = folder / "definitions" / f"{candidate['candidate_id']}.json"
            write_new(
                definition_path,
                dict(
                    schema="motion2scene_timed_schedule_scene_v1",
                    split="development",
                    usage="new random primary-training acquisition candidate; no outcomes yet",
                    scene_id=candidate["candidate_id"],
                    scene=artifact(scene_path),
                    beams=candidate["beams"],
                    beam_collision_enabled=[True] * len(candidate["beams"]),
                    provenance=dict(
                        pool_registration=artifact(registration_path),
                        candidate_id=candidate["candidate_id"],
                        source_ancestry="source41002 development motion ancestry",
                    ),
                    physical_labels=None,
                    observation_timing_status="pending native65-ray execution",
                ),
            )
            candidate_rows.append(
                dict(
                    **candidate,
                    definition=artifact(definition_path),
                    positive_offset_counts=evaluated[i].tolist(),
                    minimum_evaluated_positive_clearance_m=minima[i].tolist(),
                    robust_positive_eligible=((evaluated[i] == 81) & (minima[i] >= 0.01)).tolist(),
                    nominal_negative_clearance_m=inner[i].tolist(),
                )
            )
        definitions = {r["candidate_id"]: r["definition"] for r in candidate_rows}
        for queue in queues.values():
            for row in queue:
                row["definition"] = definitions[row["candidate_id"]]
                option = next(
                    (
                        o
                        for o in bank.request["options"]
                        if o["option_id"] == row["positive_option_id"]
                    ),
                    None,
                )
                row["coverage_key"] = dict(
                    scene_stratum=row["stratum"],
                    positive_option=row["positive_option_id"],
                    positive_entry_tick=None if option is None else option["entry_tick"],
                    observation_timing_class="pending_actual_sensor",
                )
        target_queries = sum(
            int(query_outer[i, bank.option_ids.index(c["assigned_positive_option_id"])])
            for i, c in enumerate(candidates)
        )
        actual_outer, actual_inner = int(query_outer.sum()), int(query_inner.sum())
        costs = {
            "uniform": dict(outer_clearance_queries=actual_outer, inner_clearance_queries=0),
            "target_only": dict(outer_clearance_queries=target_queries, inner_clearance_queries=0),
            "analytic_contrast": dict(
                outer_clearance_queries=actual_outer, inner_clearance_queries=actual_inner
            ),
            "observation_curriculum": dict(
                outer_clearance_queries=actual_outer, inner_clearance_queries=actual_inner
            ),
        }
        arm_refs = {}
        for arm in ARMS:
            path = folder / f"{arm}.json"
            write_new(
                path,
                dict(
                    schema="motion2scene_acquisition_candidate_queue_v1",
                    arm=arm,
                    pool_registration=artifact(registration_path),
                    geometry=artifact(raw_path),
                    acquisition_seed=pool["seed"],
                    future_physics_seed=pool["seed"],
                    proposed_candidates=len(pool["candidates"]),
                    screened_candidates=count,
                    eligible_candidates=len(queues[arm]),
                    rows=queues[arm],
                    coverage_counts=dict(Counter(r["stratum"] for r in queues[arm])),
                    standalone_equivalent_query_costs=costs[arm],
                    budget_prefixes=budget_prefixes(
                        len(queues[arm]),
                        actions,
                        bank.frame_count,
                        tuple(registration["checkpoints"]),
                    ),
                    future_group_maximum_physics_steps=actions * 4 * (bank.frame_count - 1),
                    physical_rollout_steps=0,
                    actual_sensor_queries=0,
                    acquired_corpora=0,
                    observation_gate_pending=True,
                ),
            )
            arm_refs[arm] = artifact(path)
        write_new(
            folder / "candidates.json",
            dict(
                rows=candidate_rows,
                excluded=[c for c in pool["candidates"] if c["excluded_reason"] is not None],
            ),
        )
        summary = dict(
            seed=pool["seed"],
            proposed_candidates=len(pool["candidates"]),
            screened_candidates=count,
            excluded_before_geometry=len(pool["candidates"]) - count,
            actual_shared_clearance_queries=actual_outer + actual_inner,
            actual_shared_outer_queries=actual_outer,
            actual_shared_inner_queries=actual_inner,
            shared_clearance_search_seconds=clearance_seconds,
            all_arm_queue_selection_seconds=selection_seconds,
            shared_search_and_materialization_seconds=time.monotonic() - start,
            candidates=artifact(folder / "candidates.json"),
            geometry=artifact(raw_path),
            arms=arm_refs,
            eligible_counts={arm: len(queues[arm]) for arm in ARMS},
            physical_steps=0,
            observation_curriculum_ready=False,
        )
        write_new(folder / "result.json", summary)
        summaries.append({**summary, "result": artifact(folder / "result.json")})
        print(
            json.dumps(
                {
                    "seed": pool["seed"],
                    "eligible": summary["eligible_counts"],
                    "queries": summary["actual_shared_clearance_queries"],
                }
            ),
            flush=True,
        )
    write_new(
        out / "result.json",
        dict(
            schema="motion2scene_four_arm_acquisition_pool_result_v1",
            registration=artifact(registration_path),
            pools=summaries,
            total_proposals=sum(r["proposed_candidates"] for r in summaries),
            actual_shared_clearance_queries=sum(
                r["actual_shared_clearance_queries"] for r in summaries
            ),
            physical_steps=0,
            actual_sensor_queries=0,
            independent_physical_corpora=0,
            pending=(
                "Actual matched full-course7-option branches, causal sensor timing, student "
                "decisions and verified gaps; all candidate shortages retained"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executions", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--integration-definitions", type=Path, nargs="*", default=[])
    parser.add_argument("--draws-per-stratum", type=int, default=32)
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else run(args.out)
