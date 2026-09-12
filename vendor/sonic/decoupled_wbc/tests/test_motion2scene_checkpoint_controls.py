from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import motion2scene_checkpoint_controls as controls
import motion2scene_watch_checkpoint_controls as watcher
from test_motion2scene_timed_schedule_curriculum import student, target
from test_motion2scene_timed_schedule_learning import data


def fixture(ties=False):
    bank, names, x, ticks, passed, times, legal = data()
    if ties:
        passed = legal.copy()
        times = np.where(legal, 3.0, np.nan)
    groups, records = [], []
    for group in range(2):
        targets = []
        for i in range(group, 6, 2):
            targets.append(
                dict(
                    feature_names=list(names),
                    features=x[i].tolist(),
                    phase_tick=int(ticks[i]),
                    pass_labels=passed[i].tolist(),
                    complete_legal_action_table=True,
                    teacher_action=int(np.flatnonzero(passed[i] & legal[i])[0]),
                    admitted=[True] * 4,
                    passage_time_s=[float(v) if np.isfinite(v) else None for v in times[i]],
                    legal_mask=legal[i].tolist(),
                )
            )
            records.append(
                dict(
                    encounter_id=str(group),
                    phase_tick=int(ticks[i]),
                    supervision_available=True,
                    coverage_key=(group, int(ticks[i])),
                    gap=float(group),
                )
            )
        groups.append(dict(targets=targets))
    return bank, groups, records


def test_all_seven_controls_produce_compatible_models_and_exact_fixed_refits():
    bank, groups, records = fixture()
    fitted = controls.fit_controls(bank, groups, records, [r["gap"] for r in records])
    assert set(fitted) == set(controls.MODES)
    for mode, result in fitted.items():
        assert result["model"]["weights"].shape == (3, 108, 4)
        assert len(result["weights"]) == 3
        assert all(np.isfinite(step["weights"]).all() for step in result["weights"])
        if "error" not in mode:
            assert result["weights"][0]["weights"] == result["weights"][-1]["weights"]


def test_measured_tie_initialization_has_finite_zero_residual_not_invented_failure():
    bank, groups, records = fixture(ties=True)
    fitted = controls.fit_controls(bank, groups, records, [None] * len(records))
    current = fitted["current_error"]
    assert current["weights"][0]["supervised_error"] == [0.0] * 6
    assert current["weights"][-1]["weights"] == fitted["uniform_coverage"]["weights"][-1]["weights"]
    assert current["fit"]["measured_tie_initialization_decisions"] == 6


def test_ungated_control_keeps_history_and_outcome_requirements(monkeypatch):
    row = target()
    actual = student(row, False, None)
    actual.update(
        manifest=dict(deps="same"),
        phase_records={
            15: dict(
                features=row["features"],
                legal_mask=row["legal_mask"],
                recorded_history_sha256="same",
                neutral_phase_available=True,
            )
        },
    )
    monkeypatch.setattr(
        controls,
        "bound",
        lambda ref: {"manifest": "manifest"} if ref == "collection" else {"deps": "same"},
    )
    monkeypatch.setattr(controls.replay, "dependency_identities", lambda manifest: manifest["deps"])
    groups = [dict(collection="collection", targets=[row])]
    assert controls.ungated_scores(groups, [actual])[0]["gap"] == 1
    actual["phase_records"][15]["recorded_history_sha256"] = "different"
    assert controls.ungated_scores(groups, [actual])[0]["gap"] is None
    assert controls.ungated_scores(groups, [None])[0]["gap"] is None


def test_worker_only_fits_completed_fixed_checkpoints(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    controls.write_new(
        plan,
        dict(
            schema="motion2scene_expanded_acquisition_v1",
            runs=[dict(run_id="seed_observation", arm="observation_curriculum")],
        ),
    )
    validation = tmp_path / "validation"
    validation.mkdir()
    controls.write_new(validation / "result.json", {})
    complete = tmp_path / "M2.json"
    controls.write_new(complete, {})
    monkeypatch.setattr(
        watcher,
        "checkpoint_slots",
        lambda plan, name, budget: [
            dict(model=complete if budget == 2 else tmp_path / f"missing_{budget}.json")
        ],
    )
    launches = []

    def fake_fit(plan, name, budget, validation, out):
        launches.append((name, budget))
        return dict(fixture=True)

    monkeypatch.setattr(watcher, "run", fake_fit)

    def stop_waiting(seconds):
        assert seconds == 45
        raise InterruptedError("test ends at the first prerequisite wait")

    monkeypatch.setattr(watcher.time, "sleep", stop_waiting)
    with pytest.raises(InterruptedError):
        watcher.work(plan, validation, tmp_path / "controls")
    assert launches == [("seed_observation", 2)]
