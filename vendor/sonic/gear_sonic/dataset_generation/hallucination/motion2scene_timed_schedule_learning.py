"""Verified finite-schedule search and phase value imitation for timed registries."""

from dataclasses import dataclass

import numpy as np

from .motion2scene_multi_option_imitation import physical_regret
from .motion2scene_timed_options import definition_digest
from .motion2scene_value_imitation import fit_value_policy

SCHEMA = "motion2scene_timed_schedule_value_v1"


def schedule_layout(bank):
    """Derive decision phases and schedule identities from the admitted registry."""
    if not bank.online_verified:
        raise ValueError("physical online qualification required")
    ids = bank.option_ids
    options = bank.request["options"]
    if bank.request["max_entries_per_episode"] != 1 or ids[0] != "neutral":
        raise ValueError("single-entry finite schedules with a neutral continuation required")
    entries = {0: None}
    entries.update({ids.index(o["option_id"]): o["entry_tick"] for o in options})
    phases = np.array(sorted(set(entries.values()) - {None}), dtype=int)
    if set(entries) != set(range(len(ids))) or not len(phases):
        raise ValueError("every nonneutral action must name a complete schedule")
    mask = np.array(
        [[i == 0 or entries[i] == t for i in range(len(ids))] for t in phases], dtype=bool
    )
    return phases, mask, entries


@dataclass(frozen=True)
class TimedScheduledOutcome:
    branch_id: str
    option_index: int
    physics_seed: int
    passed: bool
    passage_time_s: float | None
    admitted: bool
    prefix_hash_by_tick: dict[int, str]
    physics_steps: int

    def __post_init__(self):
        if (
            not self.branch_id
            or type(self.option_index) is not int
            or self.option_index < 0
            or type(self.physics_seed) is not int
            or type(self.passed) is not bool
            or type(self.admitted) is not bool
            or type(self.physics_steps) is not int
            or self.physics_steps < 1
            or not self.prefix_hash_by_tick
            or any(
                type(t) is not int or t <= 0 or not h for t, h in self.prefix_hash_by_tick.items()
            )
        ):
            raise ValueError("explicit physical schedule identity and measured history required")
        if self.passed and (
            self.passage_time_s is None
            or not np.isfinite(self.passage_time_s)
            or self.passage_time_s < 0
        ):
            raise ValueError("passing physical schedule requires measured passage time")


def timed_schedule_teacher(bank, branches, tick, prefix_hash, physics_seed, legality):
    """Waiting targets the best verified future schedule, not forced walking."""
    phases, qualified, entries = schedule_layout(bank)
    matches = np.flatnonzero(phases == tick)
    legal = np.asarray(legality)
    branches = tuple(branches)
    if (
        len(matches) != 1
        or not prefix_hash
        or legal.dtype.kind != "b"
        or legal.shape != (len(entries),)
        or not legal[0]
        or np.any(legal & ~qualified[matches[0]])
        or len({b.branch_id for b in branches}) != len(branches)
        or len({b.option_index for b in branches}) != len(branches)
    ):
        raise ValueError("registered neutral decision and unique actual schedules required")
    expected = [set() for _ in entries]
    for option, entry in entries.items():
        if entry is not None and entry < tick:
            continue
        immediate = option if entry == tick else 0
        if legal[immediate]:
            expected[immediate].add(option)
    candidates = [[] for _ in entries]
    for branch in branches:
        if branch.option_index not in entries:
            raise ValueError("branch option is outside the verified registry")
        entry = entries[branch.option_index]
        if (
            not branch.admitted
            or branch.physics_seed != physics_seed
            or branch.prefix_hash_by_tick.get(tick) != prefix_hash
            or (entry is not None and entry < tick)
        ):
            continue
        immediate = branch.option_index if entry == tick else 0
        if legal[immediate]:
            candidates[immediate].append(branch)
    chosen = [
        (
            min(
                group,
                key=lambda b: (not b.passed, b.passage_time_s if b.passed else 0, b.branch_id),
            )
            if group
            else None
        )
        for group in candidates
    ]
    admitted = [
        bool(expected[i]) and expected[i] == {b.option_index for b in candidates[i]}
        for i in entries
    ]
    passed = [b is not None and b.passed for b in chosen]
    times = [b.passage_time_s if b is not None and b.passed else None for b in chosen]
    complete = all(admitted[i] for i in np.flatnonzero(legal))
    feasible = [i for i in entries if legal[i] and admitted[i] and passed[i]]
    action = min(feasible, key=lambda i: (times[i], i)) if complete and feasible else None
    return dict(
        phase_tick=int(tick),
        teacher_action=action,
        pass_labels=passed,
        passage_time_s=times,
        admitted=admitted,
        legal_mask=legal.tolist(),
        complete_legal_action_table=complete,
        expected_continuation_counts=[len(s) for s in expected],
        admitted_continuation_counts=[len(g) for g in candidates],
        continuation_branch_ids=[None if b is None else b.branch_id for b in chosen],
        continuation_option_indices=[None if b is None else b.option_index for b in chosen],
        waiting_has_future_adaptation=chosen[0] is not None and chosen[0].option_index != 0,
        scope="finite verified full schedules; actual repeated student decisions still require execution",
    )


def fit_timed_schedule_policy(
    bank,
    features,
    feature_names,
    phase_ticks,
    passed,
    passage_time_s,
    admitted,
    legality,
    *,
    sample_weights=None,
    l2=1e-6,
    allow_measured_tie_initialization=False,
):
    """Fit consequential heads; optionally initialize an otherwise empty phase.

    A complete all-success equal-time table supplies genuine zero regret values.
    This fallback never adds tied rows to a phase with consequential supervision.
    Missing or all-failed tables supply no values; the runtime remains unchanged.
    """
    if type(allow_measured_tie_initialization) is not bool:
        raise ValueError("explicit boolean measured-tie initialization setting required")
    phases, qualified, _ = schedule_layout(bank)
    names = tuple(feature_names)
    x = np.asarray(features, dtype=float)
    ticks = np.asarray(phase_ticks)
    legal = np.asarray(legality)
    n_options = len(bank.option_ids)
    if (
        x.ndim != 2
        or x.shape[1] != 100 + 2 * n_options
        or x.shape[1] != len(names)
        or len(set(names)) != len(names)
        or not len(x)
        or not np.isfinite(x).all()
        or ticks.shape != (len(x),)
        or ticks.dtype.kind not in "iu"
        or legal.shape != (len(x), n_options)
        or legal.dtype.kind != "b"
        or not legal[:, 0].all()
        or not np.isfinite(l2)
        or l2 < 0
    ):
        raise ValueError("finite named sensor features and aligned neutral-phase masks required")
    phase_indices = []
    for tick in ticks:
        index = np.flatnonzero(phases == tick)
        if len(index) != 1:
            raise ValueError("unregistered training phase")
        phase_indices.append(int(index[0]))
    phase_indices = np.asarray(phase_indices)
    if np.any(legal & ~qualified[phase_indices]):
        raise ValueError("training legality includes an unqualified schedule/phase")
    checks = {"phase_s": ticks / 50, "active_skill": np.zeros(len(x))}
    checks.update({f"active_option_{i}": np.full(len(x), float(i == 0)) for i in range(n_options)})
    checks.update({f"option_{i}_legal": legal[:, i] for i in range(n_options)})
    for key, expected in checks.items():
        if key not in names or not np.allclose(x[:, names.index(key)], expected, rtol=0, atol=1e-7):
            raise ValueError(f"feature {key} disagrees with actual decision phase/state")
    regret, supervised = physical_regret(passed, passage_time_s, admitted, legal)
    passed, admitted = np.asarray(passed), np.asarray(admitted)
    times = np.asarray(passage_time_s, dtype=float)
    replay = np.ones(len(x)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if replay.shape != (len(x),) or not np.isfinite(replay).all() or (replay <= 0).any():
        raise ValueError("positive finite replay weights required")
    model = dict(
        schema_version=np.array(SCHEMA),
        feature_names=np.asarray(names),
        option_ids=np.asarray(bank.option_ids),
        request_digest=np.array(definition_digest(bank.request)),
        classes=np.arange(n_options),
        phase_ticks=phases,
        qualified_mask=qualified,
        trained_mask=np.zeros_like(qualified),
        mean=np.zeros((len(phases), x.shape[1])),
        std=np.ones((len(phases), x.shape[1])),
        weights=np.zeros((len(phases), x.shape[1], n_options)),
        bias=np.zeros((len(phases), n_options)),
        l2=np.array(l2),
    )
    # physical_regret already validates successful costs. Exact ties are tested
    # only when every legal action succeeds and has a complete physical table.
    ties = np.zeros(len(x), dtype=bool)
    if allow_measured_tie_initialization:
        complete_success = (admitted | ~legal).all(1) & (passed | ~legal).all(1)
        complete_success &= legal.sum(1) >= 2
        for row in np.flatnonzero(complete_success):
            ties[row] = np.ptp(times[row, legal[row]]) == 0
    fitted_supervision = supervised.copy()
    reports = []
    initialized_phases = []
    for phase in range(len(phases)):
        rows = np.flatnonzero((phase_indices == phase) & supervised)
        initialized = False
        if not len(rows) and allow_measured_tie_initialization:
            rows = np.flatnonzero((phase_indices == phase) & ties)
            initialized = bool(len(rows))
        if not len(rows):
            raise ValueError("each phase requires complete consequential physical teacher targets")
        columns = np.flatnonzero(legal[rows].any(axis=0))
        if len(columns) < 2:
            raise ValueError("phase needs measured alternatives")
        select = np.ix_(rows, columns)
        if initialized:
            # The exact ridge solution for all-zero observed legal value targets:
            # W=0 and b=0 minimize weighted MSE + lambda*||W||^2 for every lambda>=0.
            head = dict(
                mean=x[rows].mean(0),
                std=np.maximum(x[rows].std(0), 0.05),
                weights=np.zeros((x.shape[1], len(columns))),
                bias=np.zeros(len(columns)),
            )
            local_legal = legal[select]
            report = dict(
                method="exact zero-regret ridge solution from measured complete ties",
                l2=l2,
                recorded_decisions=len(rows),
                complete_consequential_decisions=0,
                excluded_decision_indices=[],
                option_fits=[
                    dict(
                        option_index=int(j),
                        target_count=int(local_legal[:, j].sum()),
                        recorded_decision_indices=np.flatnonzero(local_legal[:, j]).tolist(),
                        weighted_mean_squared_error=0.0,
                        regularized_objective=0.0,
                    )
                    for j in range(len(columns))
                ],
                fitted_decisions=[
                    dict(
                        recorded_decision_index=int(i),
                        action=int(np.flatnonzero(mask)[0]),
                        measured_regret=0.0,
                        predicted_regret=[0.0] * len(columns),
                        legal_mask=mask.tolist(),
                    )
                    for i, mask in enumerate(local_legal)
                ],
                scope="exact observed indifference; no scene-independent feasibility guarantee",
            )
            fitted_supervision[rows] = True
            initialized_phases.append(int(phases[phase]))
        else:
            head, report = fit_value_policy(
                x[rows],
                names,
                [bank.option_ids[i] for i in columns],
                passed[select],
                times[select],
                admitted[select],
                legal[select],
                l2=l2,
                sample_weights=replay[rows],
            )
        if report["excluded_decision_indices"]:
            raise ValueError("phase projection altered physical target completeness")
        for key in ("mean", "std"):
            model[key][phase] = head[key]
        model["weights"][phase][:, columns] = head["weights"]
        model["bias"][phase, columns] = head["bias"]
        model["trained_mask"][phase, columns] = True
        values = ((x[rows] - head["mean"]) / head["std"]) @ model["weights"][phase] + model["bias"][
            phase
        ]
        actions = np.where(legal[rows], values, -np.inf).argmax(axis=1)
        report.update(
            measured_tie_initialization=initialized,
            phase_tick=int(phases[phase]),
            global_row_indices=rows.tolist(),
            global_option_indices=columns.tolist(),
            selected_global_actions=actions.tolist(),
            measured_selected_regret=regret[rows, actions].tolist(),
        )
        reports.append(report)
    return model, dict(
        phase_fits=reports,
        excluded_decision_indices=np.flatnonzero(~fitted_supervision).tolist(),
        recorded_decisions=len(x),
        supervised_decisions=int(fitted_supervision.sum()),
        complete_consequential_decisions=int(supervised.sum()),
        measured_tie_initialization_decisions=int((fitted_supervision & ~supervised).sum()),
        initialized_phase_ticks=initialized_phases,
        allow_measured_tie_initialization=allow_measured_tie_initialization,
        scope="offline fit to complete finite physical schedules; no new physics or generalization evidence",
    )
