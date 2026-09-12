#!/usr/bin/env python3
"""Compare cue deadlines with finite action-compatible continuation supervision.

This development calculation never declares a physical obstacle irrelevant.
The common-success criterion is conditional on the supplied information groups
and measured finite schedule table, not a guarantee for unseen scenes or noise.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_build_timed_replay import audit_teachers  # noqa: E402
from motion2scene_decision_study import complete_schedules, feature_keys, read_checked  # noqa: E402
from motion2scene_information_ablation import mask_scene_features  # noqa: E402
from motion2scene_observation_teacher import observation_consistent_teacher  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    DEFAULT_RULE,
    best_complete_teacher,
    deadline_eligibility,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    schedule_layout,
)


def action_compatible_gates(keys, phases, entries, passed, times, admitted, original_actions):
    """A cue is unnecessary only in the finite sense of a common successful action.

    WAIT must possess a shared successful policy in all subsequent information
    groups. Eligibility does not authorize retaining an incompatible scene-wise
    target: the returned common action set and continuation must replace it.
    Original-action compatibility checks only set membership; even a compatible
    WAIT requires recomputed continuation costs, not reused scene-wise targets.
    """
    keys = np.asarray(keys)
    original = np.asarray(original_actions, dtype=object)
    if original.shape != keys.shape:
        raise ValueError("one original target action per encounter and phase required")
    teacher = observation_consistent_teacher(keys, phases, entries, passed, times, admitted)
    result = []
    for group in teacher:
        k = list(phases).index(group["phase_tick"])
        actions = group["acceptable_success_actions"]
        for i in group["encounter_indices"]:
            old = original[i, k]
            if old is not None and (type(old) is not int or old not in group["legal_actions"]):
                raise ValueError("original target must be a legal action or unavailable")
            eligible = bool(group["complete"] and actions)
            compatible = bool(eligible and old in actions)
            result.append(
                dict(
                    encounter_index=i,
                    phase_tick=group["phase_tick"],
                    information_group_size=len(group["encounter_indices"]),
                    complete=group["complete"],
                    eligible=eligible,
                    acceptable_actions=actions,
                    original_action_compatible=compatible,
                    requires_action_retargeting=bool(eligible and not compatible),
                    common_teacher_action=group["teacher_action"],
                    selected_continuation=group["selected_schedule_by_encounter"].get(i),
                    information_conflict=group["information_conflict"],
                )
            )
    return sorted(result, key=lambda r: (r["encounter_index"], r["phase_tick"]))


def compare(groups, phases, entries, reveal):
    features = np.asarray([[t["features"] for t in g["targets"]] for g in groups])
    names = groups[0]["targets"][0]["feature_names"]
    masked = mask_scene_features(features, names, phases, reveal)
    keys = feature_keys(masked)
    passed, times, admitted = complete_schedules(groups, dict(enumerate(entries)))
    old_actions = [[t["teacher_action"] for t in g["targets"]] for g in groups]
    rows = action_compatible_gates(keys, phases, entries, passed, times, admitted, old_actions)
    for row in rows:
        g = groups[row["encounter_index"]]
        tick = row["phase_tick"]
        target = next(t for t in g["targets"] if t["phase_tick"] == tick)
        old_teacher = best_complete_teacher(target)
        continuation = None if old_teacher is None else old_teacher["continuation_option_index"]
        entry = None if continuation is None else entries[continuation]
        deadline = deadline_eligibility(
            g["neutral_visibility"], g["scene"]["beam_collision_enabled"], tick, entry, DEFAULT_RULE
        )
        # A captured surface receipt is not usable while corridor features are masked.
        cue_usable = not any(g["scene"]["beam_collision_enabled"]) or (
            reveal is not None and tick >= reveal
        )
        cue_gate = bool(deadline["current_eligible"] and cue_usable)
        row.update(
            scene_id=g["scene_id"],
            all_beam_cue_eligible=cue_gate,
            newly_eligible=bool(row["eligible"] and not cue_gate),
            cue_eligible_without_common_success=bool(cue_gate and not row["eligible"]),
            original_selected_continuation_matches_common_teacher=bool(
                row["eligible"] and continuation == row["selected_continuation"]
            ),
        )
    return dict(
        reveal_tick=reveal,
        assigned_decisions=len(rows),
        rows=rows,
        counts={
            key: sum(r[key] for r in rows)
            for key in (
                "eligible",
                "all_beam_cue_eligible",
                "newly_eligible",
                "cue_eligible_without_common_success",
                "requires_action_retargeting",
                "information_conflict",
            )
        },
    )


def run(source, out):
    original_ref = artifact(source / "result.json")
    original = read_checked(original_ref)
    bank, _, groups, _ = audit_teachers(Path(original["teachers"]["path"]))
    phases, _, layout = schedule_layout(bank)
    for group in groups:
        if group["manifest"]["split"] != "development":
            raise ValueError("development groups only")
        group["targets"].sort(key=lambda t: t["phase_tick"])
        if [t["phase_tick"] for t in group["targets"]] != phases.tolist():
            raise ValueError("complete ordered neutral decision histories required")
    entries = [layout[i] for i in range(len(layout))]
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            source=original_ref,
            implementation=[
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_observation_teacher.py"),
                artifact(ROOT / "scripts/research/motion2scene_information_ablation.py"),
            ],
            information_rule="exact equality of available student vectors; partitions must refine",
            reveals=phases.tolist() + [None],
            objective="common successful continuation, then minimum group mean measured passage time",
            scope="finite development gate/target comparison; no new fit, replay or physical rollout",
        ),
    )
    return write_new(
        out / "result.json",
        dict(
            experiment=experiment,
            new_physics_steps=0,
            conditions=[compare(groups, phases, entries, r) for r in [*phases.tolist(), None]],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out)))
