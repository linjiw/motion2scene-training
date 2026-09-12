"""Only synthetic fixtures; this suite never reads primary experiment data."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination import (
    motion2scene_timed_schedule_learning as learning,
    motion2scene_timed_schedule_policy as runtime,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_value_imitation import (
    fit_value_policy,
)

ROOT = Path(__file__).resolve().parents[2]


def data():
    bank = SimpleNamespace(
        online_verified=True,
        option_ids=("neutral", "early", "middle", "late"),
        request={
            "max_entries_per_episode": 1,
            "options": [
                {"option_id": name, "entry_tick": tick}
                for name, tick in [("early", 15), ("middle", 50), ("late", 70)]
            ],
        },
    )
    ticks, legal, _ = learning.schedule_layout(bank)
    names = runtime.expected_feature_names(4)
    x = np.zeros((3, len(names)))
    passed = legal.copy()
    times = np.where(legal, 5.0, np.nan)
    passed[0, 1] = False
    times[0, 1] = np.nan
    times[2, 3] = 5.5
    for i, tick in enumerate(ticks):
        x[i, 0] = i
        x[i, names.index("phase_s")] = tick / 50
        x[i, names.index("active_option_0")] = 1
        x[i, -4:] = legal[i]
    return [bank, x, names, ticks, passed, times, np.ones_like(legal), legal]


def fit(d, module=learning, **kwargs):
    return module.fit_timed_schedule_policy(*d, l2=10, **kwargs)


def test_old_default_still_fails_for_phase_with_only_measured_indifference():
    with pytest.raises(ValueError, match="each phase requires complete consequential"):
        fit(data())


def test_only_empty_consequential_phase_is_initialized_and_existing_runtime_accepts():
    d = data()
    model, report = fit(d, allow_measured_tie_initialization=True)
    runtime.validate_schedule_policy(model, d[0])
    assert report["initialized_phase_ticks"] == [50]
    assert report["complete_consequential_decisions"] == 2
    assert report["supervised_decisions"] == 3
    assert report["measured_tie_initialization_decisions"] == 1
    assert report["excluded_decision_indices"] == []
    head = report["phase_fits"][1]
    assert head["measured_tie_initialization"]
    assert head["global_row_indices"] == [1]
    assert head["global_option_indices"] == [0, 2]
    assert head["measured_selected_regret"] == [0.0]
    np.testing.assert_array_equal(model["weights"][1], 0)
    np.testing.assert_array_equal(model["bias"][1], 0)
    np.testing.assert_array_equal(model["mean"][1], d[1][1])
    np.testing.assert_array_equal(model["std"][1], 0.05)
    choice, _ = runtime.choose_schedule_option(
        "learned", d[2], d[1][1], d[-1][1], 0, 50, None, d[0], policy=model
    )
    assert choice == "neutral"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_zero_or_invalid_weight_cannot_authorize_measured_initialization(bad):
    with pytest.raises(ValueError, match="positive finite replay weights"):
        fit(data(), allow_measured_tie_initialization=True, sample_weights=[1.0, bad, 1.0])


def test_all_zero_weights_fail():
    with pytest.raises(ValueError, match="positive finite replay weights"):
        fit(data(), allow_measured_tie_initialization=True, sample_weights=[0.0, 0.0, 0.0])


def append_row(d, row):
    for i in (1, 3, 4, 5, 6, 7):
        d[i] = np.concatenate((d[i], d[i][row : row + 1]), axis=0)
    return d


def test_positive_nonuniform_weights_give_exact_zero_solution_and_tied_row_normalization():
    d = append_row(data(), 1)
    d[1][-1, 0] = 5.0
    model, report = fit(d, allow_measured_tie_initialization=True, sample_weights=[1, 2, 1, 7])
    assert report["measured_tie_initialization_decisions"] == 2
    np.testing.assert_array_equal(model["weights"][1], 0)
    np.testing.assert_array_equal(model["bias"][1], 0)
    np.testing.assert_array_equal(model["mean"][1], d[1][[1, 3]].mean(0))
    np.testing.assert_array_equal(model["std"][1], np.maximum(d[1][[1, 3]].std(0), 0.05))
    assert all(o["target_count"] == 2 for o in report["phase_fits"][1]["option_fits"])


def test_mixed_ties_and_signal_preserve_every_already_supervised_phase_array_exactly():
    d = append_row(data(), 1)
    d[5][-1, 2] = 5.5
    d[1][-1, 0] = 9
    original, original_report = fit(d, sample_weights=[1, 10, 3, 4])
    current, current_report = fit(
        d, allow_measured_tie_initialization=True, sample_weights=[1, 10, 3, 4]
    )
    assert current_report["initialized_phase_ticks"] == []
    assert (
        current_report["excluded_decision_indices"]
        == original_report["excluded_decision_indices"]
        == [1]
    )
    for key in original:
        np.testing.assert_array_equal(original[key], current[key])

    # Independent pre-existing solver API checks the numerical contract without
    # importing historical source or using any external experiment artifact.
    for phase_index, (tick, rows) in enumerate(((15, [0]), (50, [3]), (70, [2]))):
        columns = np.flatnonzero(d[-1][rows].any(axis=0))
        select = np.ix_(rows, columns)
        head, _ = fit_value_policy(
            d[1][rows],
            d[2],
            [d[0].option_ids[i] for i in columns],
            d[4][select],
            d[5][select],
            d[6][select],
            d[7][select],
            l2=10,
            sample_weights=np.array([1, 10, 3, 4])[rows],
        )
        assert current_report["phase_fits"][phase_index]["phase_tick"] == tick
        for key in ("mean", "std"):
            np.testing.assert_array_equal(current[key][phase_index], head[key])
        np.testing.assert_array_equal(current["weights"][phase_index][:, columns], head["weights"])
        np.testing.assert_array_equal(current["bias"][phase_index, columns], head["bias"])


@pytest.mark.parametrize("kind", ["missing", "all_fail", "single_legal"])
def test_missing_all_failed_and_single_choice_tables_do_not_initialize(kind):
    d = data()
    if kind == "missing":
        d[6][1, 2] = False
    elif kind == "all_fail":
        d[4][1] = False
        d[5][1] = np.nan
    else:
        d[7][1, 2] = False
        d[1][1, -4:] = d[7][1]
    with pytest.raises(ValueError, match="each phase requires complete consequential"):
        fit(d, allow_measured_tie_initialization=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1])
def test_invalid_success_time_remains_rejected(bad):
    d = data()
    d[5][1, 2] = bad
    with pytest.raises(
        ValueError, match="verified successful options require measured passage times"
    ):
        fit(d, allow_measured_tie_initialization=True)


def test_near_equal_cost_is_consequential_not_zeroed():
    d = data()
    d[5][1, 2] = 5.0001
    model, report = fit(d, allow_measured_tie_initialization=True)
    assert report["initialized_phase_ticks"] == []
    assert model["bias"][1, 2] < 0


def outcomes(bank):
    return [
        learning.TimedScheduledOutcome(
            str(i), i, 991, True, 5.0, True, {15: "prefix", 50: "prefix", 70: "prefix"}, 1192
        )
        for i in range(4)
    ]


def test_teacher_requires_all_future_branches_even_when_present_wait_and_entry_tie():
    d = data()
    bank = d[0]
    branches = outcomes(bank)
    good = learning.timed_schedule_teacher(bank, branches, 50, "prefix", 991, d[7][1])
    assert good["complete_legal_action_table"] and good["teacher_action"] == 0
    assert good["expected_continuation_counts"][0] == 2
    for subset in (
        branches[:-1],
        branches[:-1]
        + [
            learning.TimedScheduledOutcome(
                "different", 3, 991, True, 5.0, True, {50: "different"}, 1192
            )
        ],
    ):
        target = learning.timed_schedule_teacher(bank, subset, 50, "prefix", 991, d[7][1])
        assert not target["complete_legal_action_table"]
        assert not target["admitted"][0]
        assert target["teacher_action"] is None
        d[4][1] = target["pass_labels"]
        d[5][1] = np.asarray(target["passage_time_s"], dtype=float)
        d[6][1] = target["admitted"]
        with pytest.raises(ValueError, match="each phase requires"):
            fit(d, allow_measured_tie_initialization=True)


def test_untrained_default_is_not_a_supported_runtime_substitute():
    d = data()
    model, _ = fit(d, allow_measured_tie_initialization=True)
    model["trained_mask"][1] = False
    with pytest.raises(ValueError, match="legal trained heads"):
        runtime.validate_schedule_policy(model, d[0])


def test_cli_boolean_is_opt_in_and_bound_to_training_registration_and_fitter():
    path = ROOT / "scripts/research/motion2scene_train_timed_schedules.py"
    tree = ast.parse(path.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    idx = [a.arg for a in function.args.kwonlyargs].index("allow_measured_tie_initialization")
    assert (
        isinstance(function.args.kw_defaults[idx], ast.Constant)
        and function.args.kw_defaults[idx].value is False
    )
    for name in ("fit_timed_schedule_policy", "dict"):
        assert any(
            isinstance(n.func, ast.Name)
            and n.func.id == name
            and any(k.arg == "allow_measured_tie_initialization" for k in n.keywords)
            for n in calls
        )
    assert "--allow-measured-tie-initialization" in path.read_text()
