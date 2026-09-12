"""Finite, observation-consistent supervision for a single-entry schedule tree.

Equivalence keys describe *student information*, never physical state hashes or
scene IDs. Partitions must refine over time (perfect recall). A successful WAIT
requires one compatible continuation policy in every later information group.
This is a finite recorded-data calculation, not a partial-observability guarantee.
"""

from collections import defaultdict

import numpy as np


def observation_consistent_teacher(keys, phase_ticks, entry_ticks, passed, times, admitted):
    """Return all successful actions and a fastest common continuation per group.

    Passage is required for every encounter in a group. Among such policies,
    minimize the unweighted mean of their measured passage times. Unknown
    branches withhold the group's table; they are never converted into failures.
    All groups at all neutral phases are reported, including unreachable ones.
    """
    keys = np.asarray(keys)
    phases = np.asarray(phase_ticks)
    passed, admitted = np.asarray(passed), np.asarray(admitted)
    times = np.asarray(times, dtype=float)
    entries = tuple(entry_ticks)
    if (
        keys.ndim != 2
        or keys.dtype.kind not in "US"
        or not keys.size
        or np.any(keys == "")
        or phases.shape != (keys.shape[1],)
        or phases.dtype.kind not in "iu"
        or phases[0] <= 0
        or np.any(np.diff(phases) <= 0)
        or not entries
        or entries[0] is not None
        or any(type(t) is not int or t not in phases for t in entries[1:])
        or passed.shape != (len(keys), len(entries))
        or passed.dtype.kind != "b"
        or admitted.shape != passed.shape
        or admitted.dtype.kind != "b"
        or times.shape != passed.shape
        or np.any(~np.isfinite(times[passed & admitted]))
        or np.any(times[passed & admitted] < 0)
    ):
        raise ValueError("finite aligned outcomes, legal phases and information keys required")
    partitions = []
    for k in range(len(phases)):
        groups = defaultdict(list)
        for i, key in enumerate(keys[:, k]):
            groups[str(key)].append(i)
        if k and any(len({keys[i, k - 1] for i in ids}) != 1 for ids in groups.values()):
            raise ValueError(
                "information groups must retain past information (refining partitions)"
            )
        partitions.append(dict(groups))
    solved, rows = {}, []
    for k in reversed(range(len(phases))):
        tick = int(phases[k])
        remaining = [s for s, t in enumerate(entries) if t is None or t >= tick]
        legal = [0] + [s for s, t in enumerate(entries) if t == tick]
        for key, ids in partitions[k].items():
            complete = bool(admitted[np.ix_(ids, remaining)].all())
            candidates = {}
            for action in legal:
                if action or k == len(phases) - 1:
                    if complete and passed[ids, action].all():
                        candidates[action] = {i: (action, float(times[i, action])) for i in ids}
                elif complete:
                    continuations = [solved[k + 1, str(keys[i, k + 1])] for i in ids]
                    if all(c["selected"] is not None for c in continuations):
                        candidates[0] = {
                            i: c["selected"][i] for i, c in zip(ids, continuations, strict=True)
                        }
            means = {a: float(np.mean([v[1] for v in c.values()])) for a, c in candidates.items()}
            best = min(means, key=lambda a: (means[a], a)) if means else None
            selected = None if best is None else candidates[best]
            solved[k, key] = dict(selected=selected)
            # The scene-wise reference can use different future actions even
            # when the later information keys collide. Keep that distinction.
            individual = []
            for i in ids:
                passing = [s for s in remaining if admitted[i, s] and passed[i, s]]
                individual.append(bool(passing))
            rows.append(
                dict(
                    phase_tick=tick,
                    information_key=key,
                    encounter_indices=ids,
                    legal_actions=legal,
                    complete=complete,
                    acceptable_success_actions=sorted(candidates),
                    optimal_time_actions=[a for a in sorted(means) if means[a] == means[best]],
                    teacher_action=best,
                    mean_passage_time_s_by_action=means,
                    selected_schedule_by_encounter=(
                        {} if selected is None else {i: s for i, (s, _) in selected.items()}
                    ),
                    individually_solvable=individual,
                    information_conflict=bool(complete and all(individual) and best is None),
                )
            )
    return sorted(rows, key=lambda r: (r["phase_tick"], r["encounter_indices"]))
