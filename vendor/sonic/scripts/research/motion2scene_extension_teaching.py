"""Matched arm-specific teaching while retaining the complete physical motion bank.

The logical view is a subset of an already qualified bank, not a new physical
qualification. WAIT is recomputed using only the selected arm's continuations.
Both arms retain the same sensor/state channels and the shared neutral motion.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    TimedScheduledOutcome,
    schedule_layout,
    timed_schedule_teacher,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    choose_schedule_option,
    expected_feature_names,
)


def longitudinal_folds(tasks):
    """Hold out each entire longitudinal center, including all heights/lengths."""
    if len(tasks) != 12 or len({t["task_id"] for t in tasks}) != 12:
        raise ValueError("all twelve distinct pre-generation capability tasks required")
    centers = (1.7, 2.25, 2.8)
    for center in centers:
        selected = [t for t in tasks if t["center_xy_m"][0] == center]
        if len(selected) != 4 or {(t["length_m"], t["underside_m"]) for t in selected} != {
            (length, height) for length in (0.1, 0.75) for height in (1.24, 1.30)
        }:
            raise ValueError("fixed center-by-length-by-height panel required")
    return [
        dict(
            held_center_m=center,
            training_task_ids=[t["task_id"] for t in tasks if t["center_xy_m"][0] != center],
            evaluation_task_ids=[t["task_id"] for t in tasks if t["center_xy_m"][0] == center],
        )
        for center in centers
    ]


def arm_view(bank, arm):
    """Derive the exact neutral-plus-arm menu from a verified full registry."""
    phases, _, _ = schedule_layout(bank)
    if arm not in ("generated", "authored"):
        raise ValueError("declared generated or authored arm required")
    request = deepcopy(bank.request)
    options = [o for o in request["options"] if o["reference_id"].startswith(arm + "_")]
    if not options:
        raise ValueError("arm has no qualified schedules")
    ids = ("neutral", *[o["option_id"] for o in options])
    if len(set(ids)) != len(ids):
        raise ValueError("unique arm schedule identities required")
    indices = np.asarray([bank.option_ids.index(name) for name in ids], dtype=int)
    request["options"] = options
    references = {"neutral", *[o["reference_id"] for o in options]}
    request["references"] = [r for r in request["references"] if r["reference_id"] in references]
    if {r["reference_id"] for r in request["references"]} != references:
        raise ValueError("every selected motion reference must exist")
    view = SimpleNamespace(online_verified=bank.online_verified, option_ids=ids, request=request)
    if not np.array_equal(schedule_layout(view)[0], phases):
        raise ValueError("equivalent teaching requires both arms at every original phase")
    return view, indices


def project_observation(bank, indices, names, features, legality, active):
    """Keep sensing and state; project only action identity and legality channels."""
    count = len(bank.option_ids)
    x, legal, indices = np.asarray(features), np.asarray(legality), np.asarray(indices)
    if (
        tuple(names) != expected_feature_names(count)
        or x.shape != (100 + 2 * count,)
        or not np.isfinite(x).all()
        or legal.shape != (count,)
        or legal.dtype.kind != "b"
        or indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or indices[0] != 0
        or len(set(indices.tolist())) != len(indices)
        or np.any((indices < 0) | (indices >= count))
        or active not in indices
        or not legal[active]
    ):
        raise ValueError("finite full-bank input and admissible arm action indices required")
    if (
        not np.array_equal(x[100 : 100 + count], np.eye(count)[active])
        or not np.array_equal(x[100 + count :], legal)
        or x[96] != int(active != 0)
        or x[98] != legal[0]
        or x[99] != bool(legal[1:].any())
    ):
        raise ValueError("recorded feature channels disagree with the actual full-bank interface")
    projected = np.concatenate((x[:100], x[100 + indices], legal[indices])).copy()
    projected[98] = legal[0]
    projected[99] = bool(legal[indices[1:]].any())
    return (
        expected_feature_names(len(indices)),
        projected,
        legal[indices].copy(),
        int(np.flatnonzero(indices == active)[0]),
    )


def restricted_teacher(bank, arm, branches, tick, prefix_hash, seed, names, features, legality):
    """Recompute WAIT from original branch outcomes; never slice full-bank WAIT."""
    view, indices = arm_view(bank, arm)
    projected_names, x, legal, _ = project_observation(
        bank, indices, names, features, legality, active=0
    )
    mapping = {int(full): local for local, full in enumerate(indices)}
    branches = tuple(branches)
    if len({b.option_index for b in branches}) != len(branches):
        raise ValueError("unique original physical branches required")
    selected = [
        replace(branch, option_index=mapping[branch.option_index])
        for branch in branches
        if branch.option_index in mapping
    ]
    target = timed_schedule_teacher(view, selected, tick, prefix_hash, seed, legal)
    return dict(
        **target,
        features=x.tolist(),
        feature_names=list(projected_names),
        original_option_indices=indices.tolist(),
        arm=arm,
        assigned_arm_branch_ids=[b.branch_id for b in selected],
        restriction="arm-specific continuations from shared full-bank physical executions",
    )


def outcomes_from_audited_group(bank, group, result):
    """Recover branches after the caller independently rechecks the collection.

    The upstream collection audit establishes matching recorded histories and
    complete physical outcomes. Only those exact matched histories are admitted
    here. This function does not replace that raw-recording verification.
    """
    ids = bank.option_ids
    rows = {r["forced_option_id"]: r for r in result["rows"]}
    assessments = {r["option_id"]: r for r in group["branch_assessments"]}
    if (
        tuple(group["option_ids"]) != ids
        or set(rows) != set(ids)
        or len(result["rows"]) != len(ids)
        or set(assessments) != set(ids)
        or len(group["branch_assessments"]) != len(ids)
    ):
        raise ValueError("complete original bank assignment and audited branch identities required")
    targets = {t["phase_tick"]: t for t in group["targets"]}
    hashes = {name: {} for name in ids}
    seen = set()
    for comparison in group["prefix_comparisons"]:
        tick, name = comparison["phase_tick"], comparison["option_id"]
        if (tick, name) in seen or name not in ids or tick not in targets:
            raise ValueError("unique registered prefix comparisons required")
        seen.add((tick, name))
        if comparison["matched"]:
            target = targets[tick]
            if not target["available"] or not target["recorded_history_sha256"]:
                raise ValueError("matched branch requires an available actual neutral history")
            hashes[name][tick] = target["recorded_history_sha256"]
    branches = []
    for index, name in enumerate(ids):
        row, assessment = rows[name], assessments[name]
        outcome = row["outcome"]["task_outcome"]
        if (
            outcome not in ("pass", "failure", "unknown")
            or outcome != assessment["task_outcome"]
            or bool(row["pass"]) != (outcome == "pass")
            or row["cell_id"] != assessment["cell_id"]
            or row["physics_steps"] != assessment["physics_steps"]
        ):
            raise ValueError("source physical labels differ from the independently audited group")
        steps = assessment["physics_steps"]
        if hashes[name] and type(steps) is int and steps > 0:
            branches.append(
                TimedScheduledOutcome(
                    branch_id=row["cell_id"],
                    option_index=index,
                    physics_seed=group["physics_seed"],
                    passed=outcome == "pass",
                    passage_time_s=row["costs"]["passage_time_s"],
                    admitted=outcome != "unknown",
                    prefix_hash_by_tick=hashes[name],
                    physics_steps=steps,
                )
            )
    return branches


def choose_arm_schedule(bank, arm, model, names, features, legality, active, tick, mandatory):
    """Use the unchanged ridge readout on the projected legal action interface."""
    view, indices = arm_view(bank, arm)
    projected_names, x, legal, local_active = project_observation(
        bank, indices, names, features, legality, active
    )
    selected, values = choose_schedule_option(
        "learned",
        projected_names,
        x,
        legal,
        local_active,
        tick,
        mandatory,
        view,
        policy=model,
    )
    # Return the original identity, not a local index into a different bank.
    return selected, dict(
        original_option_indices=indices.tolist(),
        features=x.tolist(),
        legal_mask=legal.tolist(),
        policy_values=values,
    )
