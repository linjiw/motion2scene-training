"""Offline baseline ranking from actual finite branches and causal neutral prefixes."""

import numpy as np

from .motion2scene_schedule_script import DEFAULT_CONFIG, choose_sensor_schedule
from .motion2scene_timed_schedule_policy import expected_feature_names


def context_oracle(bank, group, recorded_rows):
    """Keep all seven branches; an unknown branch makes the relative-cost oracle incomplete."""
    rows = {row["forced_option_id"]: row for row in recorded_rows}
    assessments = {row["option_id"]: row for row in group["branch_assessments"]}
    if (
        len(rows) != len(recorded_rows)
        or len(assessments) != len(group["branch_assessments"])
        or set(rows) != set(bank.option_ids)
        or set(assessments) != set(bank.option_ids)
    ):
        raise ValueError("one actual assessment per assigned complete schedule required")
    branches = []
    for option in bank.option_ids:
        assessment, row = assessments[option], rows[option]
        known = assessment["task_outcome_admitted"] is True and assessment["task_outcome"] in (
            "pass",
            "failure",
        )
        passed = assessment["task_outcome"] == "pass" if known else None
        measured_time = row["costs"]["passage_time_s"] if known and passed else None
        if known and (type(row["pass"]) is not bool or row["pass"] != passed):
            raise ValueError("stored branch pass differs from its independently audited outcome")
        if passed and (
            measured_time is None or not np.isfinite(measured_time) or measured_time < 0
        ):
            raise ValueError("successful branch needs finite measured passage time")
        branches.append(
            dict(
                option_id=option,
                known=known,
                passed=passed,
                passage_time_s=measured_time,
                physics_steps=assessment["physics_steps"],
                outcome=assessment.get("outcome"),
            )
        )
    known_panel = all(row["known"] for row in branches)
    successful = [row["passage_time_s"] for row in branches if row["known"] and row["passed"]]
    for branch in branches:
        branch["relative_time_regret"] = (
            (branch["passage_time_s"] - min(successful)) / max(max(successful), 1e-12)
            if known_panel and branch["passed"]
            else 1.0 if known_panel else None
        )
    return dict(
        complete_known_panel=known_panel,
        branches=branches,
        unknown_option_ids=[row["option_id"] for row in branches if not row["known"]],
    )


def prefix_receipt(group, option_id, tick):
    rows = [
        row
        for row in group["prefix_comparisons"]
        if row["option_id"] == option_id and row["phase_tick"] == tick
    ]
    if len(rows) != 1 or rows[0]["matched"] is not True:
        return None
    prefix = rows[0].get("prefix")
    if prefix is None or prefix.get("exact_match") is not True or prefix.get("frames") != tick:
        return None
    return rows[0]


def terminal_before(neutral, tick):
    """Global force maxima alone cannot time a terminal event before missing sensing."""
    if not neutral["known"] or neutral["passed"] or neutral.get("outcome") is None:
        return None
    for event in neutral["outcome"]["physical_events"]:
        kind = event["kind"]
        indices = []
        if kind == "recorded_fall_or_upright_threshold_failure":
            indices = [event.get("first_index")]
        elif kind == "recorded_physical_reference_reset":
            indices = event.get("indices", [])
        elif kind in ("recorded_transition_refusal", "recorded_state_mutating_transition"):
            indices = [event.get("observation_index")]
        if any(type(index) is int and 0 <= index + 1 < tick for index in indices):
            return event
    return None


def script_context(bank, group, oracle, settings):
    targets = {row["phase_tick"]: row for row in group["targets"]}
    phases = sorted({row["entry_tick"] for row in bank.request["options"]})
    if len(targets) != len(group["targets"]) or any(tick not in phases for tick in targets):
        raise ValueError("unique registered neutral decision phases required")
    branches = {row["option_id"]: row for row in oracle["branches"]}
    trace, chosen, proof, unknown, terminal = [], "neutral", None, None, None
    for tick in phases:
        target = targets.get(tick)
        if target is None or target.get("available") is False or target.get("features") is None:
            terminal = terminal_before(branches["neutral"], tick)
            if terminal is None:
                unknown = (
                    "required neutral phase is unavailable without an earlier timed terminal event"
                )
            break
        if tuple(target["feature_names"]) != expected_feature_names(len(bank.option_ids)):
            raise ValueError("exact named sensor/state schema required")
        selected, diagnostic = choose_sensor_schedule(
            target["feature_names"],
            target["features"],
            np.asarray(target["legal_mask"]),
            tick,
            bank,
            settings,
        )
        trace.append(
            dict(
                phase_tick=tick,
                selected_option_id=selected,
                recorded_history_sha256=target["recorded_history_sha256"],
                diagnostic=diagnostic,
            )
        )
        proof = prefix_receipt(group, selected, tick)
        if proof is None:
            unknown = "selected branch lacks the exact actual pre-entry neutral prefix"
            chosen = selected
            break
        if selected != "neutral":
            chosen = selected
            option = next(o for o in bank.request["options"] if o["option_id"] == selected)
            if option["entry_tick"] != tick:
                raise ValueError("script selected a schedule outside its actual entry tick")
            break  # Never use forced-neutral observations after a commitment.
    branch = branches[chosen]
    known = bool(oracle["complete_known_panel"] and unknown is None and branch["known"])
    return dict(
        scene_id=group["scene_id"],
        known=known,
        selected_option_id=chosen,
        passed=branch["passed"] if known else None,
        relative_time_regret=branch["relative_time_regret"] if known else None,
        passage_time_s=branch["passage_time_s"] if known else None,
        decision_trace=trace,
        chosen_prefix=proof,
        earlier_neutral_terminal_event=terminal,
        unknown_reason=unknown
        or (None if oracle["complete_known_panel"] else "incomplete seven-branch context oracle"),
    )


def constant_context(group, oracle, option_id):
    matches = [row for row in oracle["branches"] if row["option_id"] == option_id]
    if len(matches) != 1:
        raise ValueError("constant must name one verified complete schedule")
    row = matches[0]
    known = bool(oracle["complete_known_panel"] and row["known"])
    return dict(
        scene_id=group["scene_id"],
        selected_option_id=option_id,
        known=known,
        passed=row["passed"] if known else None,
        passage_time_s=row["passage_time_s"] if known else None,
        relative_time_regret=row["relative_time_regret"] if known else None,
        unknown_reason=None if known else "incomplete seven-branch context oracle",
    )


def summarize(contexts, expected_contexts):
    complete = len(contexts) == expected_contexts and all(row["known"] for row in contexts)
    if complete and any(
        type(row["passed"]) is not bool
        or row["relative_time_regret"] is None
        or not np.isfinite(row["relative_time_regret"])
        or not 0 <= row["relative_time_regret"] <= 1
        for row in contexts
    ):
        raise ValueError("known contexts require boolean outcomes and finite physical regret")
    return dict(
        complete_known_panel=complete,
        assigned_contexts=expected_contexts,
        known_contexts=sum(row["known"] for row in contexts),
        selected_failure_count=sum(not row["passed"] for row in contexts) if complete else None,
        mean_relative_time_regret=(
            float(np.mean([row["relative_time_regret"] for row in contexts])) if complete else None
        ),
        contexts=contexts,
    )


def rank_baselines(bank, groups, oracles, configurations):
    if len(groups) != len(oracles) or not groups or not configurations:
        raise ValueError("common nonempty context panel and registered settings required")
    scripts = []
    for index, settings in enumerate(configurations):
        contexts = [
            script_context(bank, group, oracle, settings)
            for group, oracle in zip(groups, oracles, strict=True)
        ]
        scripts.append(
            dict(setting_index=index, settings=settings, **summarize(contexts, len(groups)))
        )
    constants = [
        dict(
            option_id=option,
            option_index=index,
            **summarize(
                [
                    constant_context(group, oracle, option)
                    for group, oracle in zip(groups, oracles, strict=True)
                ],
                len(groups),
            )
        )
        for index, option in enumerate(bank.option_ids)
    ]
    eligible_scripts = [row for row in scripts if row["complete_known_panel"]]
    eligible_constants = [row for row in constants if row["complete_known_panel"]]
    selected_script = (
        min(
            eligible_scripts,
            key=lambda r: (
                r["selected_failure_count"],
                r["mean_relative_time_regret"],
                r["settings"] != DEFAULT_CONFIG,
                r["setting_index"],
            ),
        )
        if eligible_scripts
        else None
    )
    selected_constant = (
        min(
            eligible_constants,
            key=lambda r: (
                r["selected_failure_count"],
                r["mean_relative_time_regret"],
                r["option_index"],
            ),
        )
        if eligible_constants
        else None
    )
    return dict(
        script_candidates=scripts,
        constant_candidates=constants,
        selected_setting_index=(
            None if selected_script is None else selected_script["setting_index"]
        ),
        preferred_constant_option_id=(
            None if selected_constant is None else selected_constant["option_id"]
        ),
    )
