"""Deterministic candidate pools and four geometric acquisition queues.

Candidate metadata never becomes a policy input. No function here creates
physical outcomes, sensor visibility, independent corpora, or learner regret.
"""

import random

import numpy as np

ARMS = ("uniform", "target_only", "analytic_contrast", "observation_curriculum")
SEEDS = (93201, 93202, 93203)
DRAW_KEYS = (
    "route_progress_fraction",
    "underside_m",
    "length_m",
    "width_m",
    "thickness_m",
    "lateral_offset_m",
    "yaw_offset_rad",
)


def proposal_strata(generation):
    """Use the locked generation ranges, without reading held-out outcomes."""
    single = generation["single_families_in_order"]
    if [row["name"] for row in single] != ["short", "sustained", "early_constraint"]:
        raise ValueError("requires the declared three single-beam generation families")
    strata = [
        dict(name=row["name"], beam_ranges=[(row["station_range"], row["length_range_m"])])
        for row in single
    ]
    stations = generation["course_station_ranges"]
    for name, last_length in (
        ("short_then_short", [0.1, 0.25]),
        ("short_then_sustained", [0.5, 0.9]),
    ):
        strata.append(
            dict(name=name, beam_ranges=[(stations[0], [0.1, 0.25]), (stations[1], last_length)])
        )
    return strata


def draw_pool(generation, seed, draws_per_stratum, option_ids):
    """Interleave strata; preassign positive references and future branch orders.

    Separate RNG streams prevent target assignment or branch shuffling from
    changing the geometry draws. Every draw is rounded to nine decimal places.
    """
    if (
        seed not in SEEDS
        or type(draws_per_stratum) is not int
        or draws_per_stratum < 1
        or len(option_ids) < 2
        or len(set(option_ids)) != len(option_ids)
        or option_ids[0] != "neutral"
    ):
        raise ValueError(
            "explicit acquisition seed, positive count and unique option bank required"
        )
    geometry_rng = random.Random(seed)
    target_rng = random.Random(seed + 1000000)
    branch_rng = random.Random(seed + 2000000)
    common = generation["common_ranges"]
    result = []
    for round_index in range(draws_per_stratum):
        for stratum in proposal_strata(generation):
            beams = []
            for station_range, length_range in stratum["beam_ranges"]:
                ranges = [
                    station_range,
                    common["underside_m"],
                    length_range,
                    common["width_m"],
                    common["thickness_m"],
                    common["lateral_offset_m"],
                    common["yaw_offset_rad"],
                ]
                if any(
                    len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]
                    for bounds in ranges
                ):
                    raise ValueError("finite ordered proposal bounds required")
                beams.append(
                    dict(
                        zip(
                            DRAW_KEYS,
                            [round(geometry_rng.uniform(*bounds), 9) for bounds in ranges],
                            strict=True,
                        )
                    )
                )
            order = list(option_ids)
            branch_rng.shuffle(order)
            result.append(
                dict(
                    candidate_id=f"primary_candidate_{seed}_{len(result):04d}",
                    candidate_index=len(result),
                    stratum=stratum["name"],
                    stratum_round=round_index,
                    proposal_rng_seed=seed,
                    target_rng_seed=seed + 1000000,
                    future_physics_seed=seed,
                    future_branch_order=order,
                    assigned_positive_option_id=option_ids[target_rng.randrange(len(option_ids))],
                    beam_specifications=beams,
                )
            )
    return result


def acquisition_queues(
    candidates,
    option_ids,
    evaluated_positive_min,
    evaluated_offsets,
    nominal_inner,
    *,
    offset_count=81,
    margin_m=0.01,
):
    """Rank shared candidates without using physical labels or sensor proxies.

    Uniform uses any robust positive in original draw order. Target-only uses
    its preassigned positive and prioritizes a close fit above the same margin.
    Contrast searches all distinct pairs and maximizes the smaller positive or
    negative slack. Observation curriculum shares this queue until actual
    physical and sensor evidence is available; it is never marked ready here.
    """
    minimum = np.asarray(evaluated_positive_min, dtype=float)
    offsets = np.asarray(evaluated_offsets)
    negative = np.asarray(nominal_inner, dtype=float)
    shape = (len(candidates), len(option_ids))
    if (
        minimum.shape != shape
        or negative.shape != shape
        or offsets.shape != shape
        or offsets.dtype.kind not in "iu"
        or not np.isfinite(minimum).all()
        or not np.isfinite(negative).all()
        or not np.isfinite(margin_m)
        or margin_m < 0
        or not np.isin(offsets, (1, offset_count)).all()
        or np.any((offsets == 1) & (minimum >= margin_m))
    ):
        raise ValueError(
            "complete positives or explicit nominal rejections required for every option"
        )
    eligible = (offsets == offset_count) & (minimum >= margin_m)
    queues = {arm: [] for arm in ARMS}

    def entry(index, positive, negative_index=None, slack=None):
        candidate = candidates[index]
        return dict(
            candidate_id=candidate["candidate_id"],
            candidate_index=candidate["candidate_index"],
            stratum=candidate["stratum"],
            positive_option_id=option_ids[positive],
            negative_option_id=None if negative_index is None else option_ids[negative_index],
            positive_minimum_clearance_m=float(minimum[index, positive]),
            negative_nominal_clearance_m=(
                None if negative_index is None else float(negative[index, negative_index])
            ),
            geometric_slack_m=float(
                minimum[index, positive] - margin_m if slack is None else slack
            ),
            geometric_screen_passed=True,
            physical_measurement_admitted=None,
            physical_passage=None,
            observation_availability=None,
            verified_learner_gap=None,
            curriculum_ready=False,
            future_physics_seed=candidate["future_physics_seed"],
        )

    for i, candidate in enumerate(candidates):
        positives = np.flatnonzero(eligible[i])
        if len(positives):
            queues["uniform"].append(entry(i, int(positives[0])))
        positive = option_ids.index(candidate["assigned_positive_option_id"])
        if eligible[i, positive]:
            queues["target_only"].append(entry(i, positive))
        pairs = []
        for positive in positives:
            for negative_index in np.flatnonzero(negative[i] <= -margin_m):
                if positive != negative_index:
                    slack = min(
                        minimum[i, positive] - margin_m, -negative[i, negative_index] - margin_m
                    )
                    pairs.append((float(slack), int(positive), int(negative_index)))
        if pairs:
            slack, positive, negative_index = min(pairs, key=lambda row: (-row[0], row[1], row[2]))
            queues["analytic_contrast"].append(entry(i, positive, negative_index, slack))
    queues["target_only"].sort(key=lambda row: (row["geometric_slack_m"], row["candidate_index"]))
    queues["analytic_contrast"].sort(
        key=lambda row: (-row["geometric_slack_m"], row["candidate_index"])
    )
    queues["observation_curriculum"] = [dict(row) for row in queues["analytic_contrast"]]
    return queues


def budget_prefixes(available, option_count, frame_count, checkpoints=(50000, 150000, 450000)):
    """Predeclare complete-group capacity; no group is truncated or overspent."""
    if (
        any(type(value) is not int or value < 1 for value in (option_count, frame_count))
        or type(available) is not int
        or available < 0
        or any(type(value) is not int or value < 1 for value in checkpoints)
        or list(checkpoints) != sorted(set(checkpoints))
    ):
        raise ValueError("nonnegative availability and positive integer bank dimensions required")
    group_steps = option_count * 4 * (frame_count - 1)
    if group_steps <= 0:
        raise ValueError("physical groups require at least two loaded frames")
    return [
        dict(
            budget_physics_steps=budget,
            maximum_complete_groups=budget // group_steps,
            available_candidate_groups=available,
            available_prefix_groups=min(available, budget // group_steps),
            planned_prefix_maximum_steps=min(available, budget // group_steps) * group_steps,
            candidate_shortfall=max(0, budget // group_steps - available),
            acquired_groups=0,
        )
        for budget in checkpoints
    ]
