"""Passage-first continuation targets on a finite observation decision tree.

The universal-passage teacher detects incompatible information groups. This
extension also returns their best attainable common policy: maximize the number
of measured passages, then minimize the sum of successful passage times. The
empirical encounter distribution is uniform; no population safety claim follows.
"""

import math

from motion2scene_observation_teacher import observation_consistent_teacher
import numpy as np


def passage_first_information_teacher(keys, phase_ticks, entry_ticks, passed, times, admitted):
    """Return set-valued lexicographic targets and realizable WAIT continuations.

    All remaining branches must be measured. Unknown outcomes withhold targets.
    Successful-time comparisons occur only after passage counts tie, hence their
    denominators are equal. An all-failing group has a finite canonical action
    but no passing supervision; its zero capability remains explicit.
    """
    universal = observation_consistent_teacher(
        keys, phase_ticks, entry_ticks, passed, times, admitted
    )
    keys, passed, times = np.asarray(keys), np.asarray(passed), np.asarray(times, float)
    phases = list(phase_ticks)
    solved, reports = {}, []
    for row in reversed(universal):
        tick, key, ids = row["phase_tick"], row["information_key"], row["encounter_indices"]
        k = phases.index(tick)
        candidates = {}
        if row["complete"]:
            for action in row["legal_actions"]:
                if action or k == len(phases) - 1:
                    candidates[action] = {i: action for i in ids}
                else:
                    children = [solved[phases[k + 1], str(keys[i, k + 1])] for i in ids]
                    if any(child is None for child in children):
                        raise ValueError("complete parent must have complete successor groups")
                    candidates[action] = {
                        i: child[i] for i, child in zip(ids, children, strict=True)
                    }
        values = {}
        for action, selected in candidates.items():
            successful = [i for i, schedule in selected.items() if passed[i, schedule]]
            count = len(successful)
            cost = math.fsum(float(times[i, selected[i]]) for i in successful)
            values[action] = dict(
                passage_count=count,
                successful_time_sum_s=cost,
                mean_successful_time_s=cost / count if count else None,
            )

        def rank(a):
            return (-values[a]["passage_count"], values[a]["successful_time_sum_s"])

        best = min(values, key=lambda a: (*rank(a), a)) if values else None
        maximum = values[best]["passage_count"] if best is not None else None
        selected = candidates[best] if best is not None else None
        solved[tick, key] = selected
        reports.append(
            dict(
                phase_tick=tick,
                information_key=key,
                encounter_indices=ids,
                legal_actions=row["legal_actions"],
                complete=row["complete"],
                universal_success_actions=row["acceptable_success_actions"],
                individual_schedule_capability=sum(row["individually_solvable"]),
                maximum_causal_passages=maximum,
                action_values=values,
                passage_optimal_actions=[
                    a for a in sorted(values) if values[a]["passage_count"] == maximum
                ],
                lexicographic_optimal_actions=[a for a in sorted(values) if rank(a) == rank(best)],
                teacher_action=best,
                selected_schedule_by_encounter=selected or {},
                has_passing_supervision=bool(maximum is not None and maximum > 0),
                information_gap=(
                    sum(row["individually_solvable"]) - maximum if maximum is not None else None
                ),
            )
        )
    return sorted(reports, key=lambda r: (r["phase_tick"], r["encounter_indices"]))


def evaluate_finite_policy(rows, keys, phase_ticks, passed, times):
    """Follow shared decisions to complete measured schedules, never scene IDs."""
    lookup = {(r["phase_tick"], r["information_key"]): r for r in rows}
    schedules = []
    for history in keys:
        selected = 0
        for tick, key in zip(phase_ticks, history, strict=True):
            row = lookup[tick, str(key)]
            if (
                not row["complete"]
                or row["teacher_action"] is None
                or row["teacher_action"] not in row["legal_actions"]
            ):
                raise ValueError("a visited information group lacks complete supervision")
            selected = row["teacher_action"]
            if selected:
                break
        schedules.append(selected)
    successes = [i for i, action in enumerate(schedules) if passed[i, action]]
    cost = math.fsum(float(times[i, schedules[i]]) for i in successes)
    return dict(
        selected_schedules=schedules,
        passage_count=len(successes),
        successful_time_sum_s=cost,
        mean_successful_time_s=cost / len(successes) if successes else None,
    )
