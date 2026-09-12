"""Bounded acquisition-only intervention against an earlier common response.

Geometric predictions select proposals; they never stand in for physical
outcomes. The original acquisition constructor and its queues remain unchanged.
"""

import math

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (
    acquisition_queues,
)


def common_response(tasks, option_ids):
    """Choose a common passer by measured time, then the fixed registry order.

    All-failing and incomplete tasks remain assigned but cannot manufacture an
    empty passing-set intersection. Only complete, bank-solvable tasks define it.
    """
    if not option_ids or len(option_ids) != len(set(option_ids)):
        raise ValueError("distinct ordered complete schedules required")
    common = set(option_ids)
    eligible, unsolvable, incomplete = [], [], []
    for index, task in enumerate(tasks):
        outcomes = task["outcomes"]
        times = task["passage_time_s"]
        if set(outcomes) != set(option_ids) or set(times) != set(option_ids):
            raise ValueError("every assigned branch outcome and passage time required")
        if not set(outcomes.values()) <= {"pass", "failure", "unknown"}:
            raise ValueError("explicit pass/failure/unknown labels required")
        for schedule in option_ids:
            value = times[schedule]
            if outcomes[schedule] == "pass":
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise ValueError("passing branches require measured passage times")
                if not math.isfinite(value) or value < 0:
                    raise ValueError("finite nonnegative measured passage times required")
            elif value is not None:
                raise ValueError("failed/unknown passage times must remain unavailable")
        passing = {s for s in option_ids if outcomes[s] == "pass"}
        if "unknown" in outcomes.values():
            incomplete.append(index)
        elif not passing:
            unsolvable.append(index)
        else:
            eligible.append(index)
            common &= passing
    ordered = [s for s in option_ids if s in common] if eligible else []
    means = {
        s: math.fsum(tasks[i]["passage_time_s"][s] for i in eligible) / len(eligible)
        for s in ordered
    }
    selected = min(ordered, key=lambda s: (means[s], option_ids.index(s))) if ordered else None
    return dict(
        assigned_tasks=len(tasks),
        complete_bank_solvable_task_indices=eligible,
        all_failing_task_indices=unsolvable,
        incomplete_task_indices=incomplete,
        common_passing_schedules=ordered,
        mean_measured_passage_time_s=means,
        selected_fixed_schedule=selected,
        selection_rule="common passage, then lowest mean measured passage time, then registry order",
        new_physical_executions=0,
    )


def complementary_queue(
    candidates,
    option_ids,
    minimum,
    offsets,
    negative,
    fixed_schedule,
    *,
    margin_m=0.01,
    offset_count=81,
):
    """Enumerate all robust positives against the chosen negative schedule.

    Filtering the original best-pair queue by negative identity is incorrect:
    a candidate's second-best pair may be the valid targeted intervention.
    """
    if fixed_schedule not in option_ids or len(set(option_ids)) != len(option_ids):
        raise ValueError("the targeted fixed schedule must belong to the fixed bank")
    ordinary = acquisition_queues(
        candidates,
        option_ids,
        minimum,
        offsets,
        negative,
        margin_m=margin_m,
        offset_count=offset_count,
    )["analytic_contrast"]
    original = {row["candidate_index"]: row for row in ordinary}
    minimum, offsets, negative = map(np.asarray, (minimum, offsets, negative))
    fixed = option_ids.index(fixed_schedule)
    queue = []
    for i, candidate in enumerate(candidates):
        if negative[i, fixed] > -margin_m:
            continue
        positives = [
            p
            for p in range(len(option_ids))
            if p != fixed and offsets[i, p] == offset_count and minimum[i, p] >= margin_m
        ]
        if not positives:
            continue

        def slack(p):
            return min(minimum[i, p] - margin_m, -negative[i, fixed] - margin_m)

        positive = min(positives, key=lambda p: (-slack(p), p))
        row = dict(original[candidate["candidate_index"]])
        row.update(
            positive_option_id=option_ids[positive],
            negative_option_id=fixed_schedule,
            positive_minimum_clearance_m=float(minimum[i, positive]),
            negative_nominal_clearance_m=float(negative[i, fixed]),
            geometric_slack_m=float(slack(positive)),
        )
        queue.append(row)
    return sorted(queue, key=lambda row: (-row["geometric_slack_m"], row["candidate_index"]))


def paired_selection(
    candidates,
    option_ids,
    minimum,
    offsets,
    negative,
    fixed_schedule,
    excluded_geometries,
    *,
    strata=("early_constraint", "short_then_sustained"),
    proposals_per_stratum=256,
):
    """One ordinary encounter, then ordinary versus targeted contrast.

    Exclusions consume the declared draw budget. A shortage is retained; there
    is no outcome-conditioned refill. A targeted shortage uses the declared
    ordinary fallback and is explicitly marked as an inactive intervention.
    """
    if type(proposals_per_stratum) is not int or proposals_per_stratum < 1:
        raise ValueError("a positive bounded proposal count is required")
    if len(strata) != 2 or len(set(strata)) != 2:
        raise ValueError("two distinct matched task strata required")
    identities = [c["candidate_id"] for c in candidates]
    indices = [c["candidate_index"] for c in candidates]
    if len(identities) != len(set(identities)) or len(indices) != len(set(indices)):
        raise ValueError("unique candidate identities and draw indices required")
    ordinary = acquisition_queues(candidates, option_ids, minimum, offsets, negative)[
        "analytic_contrast"
    ]
    targeted = complementary_queue(
        candidates, option_ids, minimum, offsets, negative, fixed_schedule
    )
    by_id = {c["candidate_id"]: c for c in candidates}
    excluded = set(excluded_geometries)
    rounds = []
    for position, stratum in enumerate(strata):
        draws = sorted(
            (c for c in candidates if c["stratum"] == stratum),
            key=lambda c: c["candidate_index"],
        )[:proposals_per_stratum]
        allowed = {
            c["candidate_id"]
            for c in draws
            if not c.get("excluded_reason") and c["geometry_key"] not in excluded
        }
        controls = [r for r in ordinary if r["candidate_id"] in allowed]
        interventions = [r for r in targeted if r["candidate_id"] in allowed]
        control = controls[0] if controls else None
        selected = interventions[0] if position == 1 and interventions else control
        rounds.append(
            dict(
                encounter=position + 3,
                stratum=stratum,
                proposed_candidate_ids=[c["candidate_id"] for c in draws],
                proposal_shortfall=proposals_per_stratum - len(draws),
                excluded_candidate_count=len(draws) - len(allowed),
                ordinary_eligible_count=len(controls),
                targeted_eligible_count=len(interventions),
                ordinary=control,
                complementary=selected,
                targeted_selection=position == 1 and bool(interventions),
                ordinary_fallback=position == 1 and not interventions,
                distinct_intervention=control is not None
                and selected is not None
                and by_id[control["candidate_id"]]["geometry_key"]
                != by_id[selected["candidate_id"]]["geometry_key"],
            )
        )
        # Both continuations exclude their shared first encounter thereafter.
        if position == 0 and control is not None:
            excluded.add(by_id[control["candidate_id"]]["geometry_key"])
    return dict(
        rounds=rounds,
        assigned_proposal_budget=2 * proposals_per_stratum,
        proposals_considered=sum(len(r["proposed_candidate_ids"]) for r in rounds),
        matched_physical_pair_ready=all(
            r["ordinary"] is not None and r["complementary"] is not None for r in rounds
        ),
        new_physical_executions=0,
        interpretation="geometric selection only; complete physical branches remain required",
    )
