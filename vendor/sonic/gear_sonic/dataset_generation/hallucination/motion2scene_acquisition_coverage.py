"""Common-support scene stratification without physical or sensor outcomes."""

ARMS = ("uniform", "target_only", "analytic_contrast", "observation_curriculum")
SEEDS = (93201, 93202, 93203)
SINGLES = ("short", "sustained", "early_constraint")
COURSES = ("short_then_short", "short_then_sustained")


def coverage_selection(queues, seed):
    """Choose matched strata, preserving each constructor's ranking within them.

    With four slots and five common strata, cover all three single types and
    one course subtype, rotated by acquisition-seed index. With fewer common
    single types, take every available course subtype before repeating a type.
    This version never repeats a stratum or silently fills a missing fourth.
    Positive witness identities are reported, never used to rerank: uniform's
    arbitrary first eligible witness differs from target-only's fixed target.
    """
    if set(queues) != set(ARMS) or seed not in SEEDS:
        raise ValueError("four declared constructor queues and acquisition seed required")
    universe = set(SINGLES + COURSES)
    support = {}
    for arm, rows in queues.items():
        if len({r["candidate_id"] for r in rows}) != len(rows):
            raise ValueError("candidate identities must be unique within each queue")
        support[arm] = {r["stratum"] for r in rows}
        if not support[arm] <= universe:
            raise ValueError("unknown geometric stratum")
    common = set.intersection(*(support[arm] for arm in ARMS))
    singles = [s for s in SINGLES if s in common]
    shift = SEEDS.index(seed) % len(COURSES)
    course_priority = COURSES[shift:] + COURSES[:shift]
    courses = [s for s in course_priority if s in common]
    strata = (singles + courses)[:4]
    if common and courses and not set(strata) & set(COURSES):
        raise ValueError("a common eligible course must precede repeated single strata")
    selected = {}
    for arm, rows in queues.items():
        selected[arm] = []
        for stratum in strata:
            rank, row = next((i, r) for i, r in enumerate(rows) if r["stratum"] == stratum)
            selected[arm].append(dict(**row, original_queue_rank=rank))
    if selected["analytic_contrast"] != selected["observation_curriculum"]:
        raise ValueError("analytic and replay curriculum must retain identical geometric rows")
    return dict(
        schema="motion2scene_common_support_coverage_v1",
        acquisition_seed=seed,
        course_subtype_priority=list(course_priority),
        selected_strata_in_order=strata,
        common_support=sorted(common),
        omitted_common_strata=sorted(common - set(strata)),
        unsupported_common_strata=sorted(universe - common),
        support_by_arm={arm: sorted(values) for arm, values in support.items()},
        excluded_from_common_support_by_arm={
            arm: sorted(values - common) for arm, values in support.items()
        },
        planned_group_shortfall=4 - len(strata),
        selected=selected,
        outcomes_consulted=False,
        new_clearance_queries=0,
        physical_steps=0,
    )
