"""Short-horizon teacher targets from full physically executed entry schedules.

Waiting and committing to walk have different continuations. This module groups
matched full schedules by their immediate action so a later successful adaptation
can supply a verified value for waiting now. It never creates a stopped branch.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScheduledOutcome:
    branch_id: str
    option_index: int
    entry_time_s: float | None
    physics_seed: int
    passed: bool
    passage_time_s: float | None
    admitted: bool
    prefix_hash_by_tick: dict[int, str]
    physics_steps: int

    def __post_init__(self):
        if (
            not self.branch_id
            or not isinstance(self.option_index, int)
            or self.option_index < 0
            or not isinstance(self.passed, bool)
            or not isinstance(self.admitted, bool)
            or not isinstance(self.physics_steps, int)
            or self.physics_steps < 1
            or (self.option_index == 0) != (self.entry_time_s is None)
        ):
            raise ValueError("invalid executed schedule identity or action")
        if self.entry_time_s is not None and (
            not np.isfinite(self.entry_time_s)
            or not 0.2 <= self.entry_time_s <= 0.4
            or abs(self.entry_time_s * 50 - round(self.entry_time_s * 50)) > 1e-8
        ):
            raise ValueError("entry time must be a supported 50 Hz decision tick")
        if self.passed and (
            self.passage_time_s is None
            or not np.isfinite(self.passage_time_s)
            or self.passage_time_s < 0
        ):
            raise ValueError("passing schedule requires measured passage time")


def schedule_teacher(
    branches, decision_time_s, prefix_hash, physics_seed, legality, *, expected_schedules
):
    """Group verified future schedules into immediate wait/adapt actions.

    ``prefix_hash`` must bind pre-action state and complete controller history,
    including active command. The caller audits hashes from actual recordings.
    Schedules already adapted before this neutral-state decision are excluded.
    This teacher is currently restricted to neutral approach states.
    """
    branches = tuple(branches)
    expected_schedules = tuple(tuple(schedule) for schedule in expected_schedules)
    legal = np.asarray(legality)
    tick = round(decision_time_s * 50)
    if (
        not np.isfinite(decision_time_s)
        or abs(decision_time_s * 50 - tick) > 1e-8
        or not 0.2 <= decision_time_s <= 0.4
        or not prefix_hash
        or legal.ndim != 1
        or legal.dtype != bool
        or not len(legal)
        or not legal[0]
        or len({b.branch_id for b in branches}) != len(branches)
        or not expected_schedules
        or len(set(expected_schedules)) != len(expected_schedules)
    ):
        raise ValueError("valid neutral-state decision, legal mask, and unique branches required")
    by_action = [[] for _ in legal]
    expected_by_action = [set() for _ in legal]
    for index, entry in expected_schedules:
        if (
            type(index) is not int
            or not 0 <= index < len(legal)
            or (index == 0) != (entry is None)
            or (
                entry is not None
                and (
                    not np.isfinite(entry)
                    or not 0.2 <= entry <= 0.4
                    or abs(entry * 50 - round(entry * 50)) > 1e-8
                )
            )
        ):
            raise ValueError("invalid expected finite schedule set")
        if entry is not None and entry < decision_time_s - 1e-8:
            continue
        action = index if entry is not None and abs(entry - decision_time_s) < 1e-8 else 0
        if legal[action]:
            expected_by_action[action].add((index, entry))
    actual_schedules = set()
    for branch in branches:
        identity = (branch.option_index, branch.entry_time_s)
        if not 0 <= branch.option_index < len(legal):
            raise ValueError("executed option outside current registry")
        if identity not in expected_schedules or identity in actual_schedules:
            raise ValueError("duplicate or unregistered physical schedule")
        actual_schedules.add(identity)
        if (
            not branch.admitted
            or branch.physics_seed != physics_seed
            or branch.prefix_hash_by_tick.get(tick) != prefix_hash
            or (branch.entry_time_s is not None and branch.entry_time_s < decision_time_s - 1e-8)
        ):
            continue
        action = (
            branch.option_index
            if branch.entry_time_s is not None and abs(branch.entry_time_s - decision_time_s) < 1e-8
            else 0
        )
        if legal[action]:
            by_action[action].append(branch)
    selected = []
    for candidates in by_action:
        selected.append(
            min(
                candidates,
                key=lambda b: (not b.passed, b.passage_time_s if b.passed else 0, b.branch_id),
            )
            if candidates
            else None
        )
    pass_labels = [bool(branch is not None and branch.passed) for branch in selected]
    admitted = [
        bool(expected)
        and expected == {(branch.option_index, branch.entry_time_s) for branch in candidates}
        for expected, candidates in zip(expected_by_action, by_action, strict=True)
    ]
    times = [
        branch.passage_time_s if branch is not None and branch.passed else None
        for branch in selected
    ]
    feasible = [i for i in range(len(legal)) if legal[i] and admitted[i] and pass_labels[i]]
    complete = all(admitted[i] for i in np.flatnonzero(legal))
    teacher_action = min(feasible, key=lambda i: (times[i], i)) if complete and feasible else None
    return {
        "phase_s": decision_time_s,
        "teacher_action": teacher_action,
        "pass_labels": pass_labels,
        "passage_time_s": times,
        "admitted": admitted,
        "legal_mask": legal.tolist(),
        "complete_legal_action_table": complete,
        "expected_continuation_counts": [len(expected) for expected in expected_by_action],
        "admitted_continuation_counts": [len(candidates) for candidates in by_action],
        "continuation_branch_ids": [
            branch.branch_id if branch is not None else None for branch in selected
        ],
        "waiting_has_future_adaptation": bool(
            selected[0] is not None and selected[0].entry_time_s is not None
        ),
        "scope": (
            "verified finite future schedules; replan/student continuation "
            "still requires closed-loop execution"
        ),
    }
