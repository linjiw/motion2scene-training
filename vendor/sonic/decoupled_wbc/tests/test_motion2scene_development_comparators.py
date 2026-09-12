"""Preserve observed failures and unknown tails in the comparator readout."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_summarize_development_comparators as comparators


def row(complete, kinds=(), contact_free=True, fall=False):
    return dict(
        measurement_admitted=complete,
        outcome=dict(physical_events=[dict(kind=k) for k in kinds]),
        contact_audit=dict(
            complete_synchronized_streams=True,
            no_undesired_measured_contact=contact_free,
        ),
        passage=dict(fall_anywhere_in_first_episode=fall) if complete else None,
        costs=dict(whole_episode_time_s=5.96 if complete else None),
    )


def test_partial_observed_contact_is_failure_but_unobserved_later_fall_is_unknown():
    measured = comparators.physical_events(row(False, ["verified_environment_contact"], False))
    assert measured["undesired_environment_contact"] == "observed"
    assert measured["fall_or_upright_threshold_failure"] == "unknown"
    assert measured["recorded_episode_duration_s"] is None


def test_partial_contact_free_prefix_cannot_establish_full_episode_absence():
    measured = comparators.physical_events(row(False))
    assert measured["undesired_environment_contact"] == "unknown"
    assert measured["fall_or_upright_threshold_failure"] == "unknown"


def test_partial_measured_fall_is_kept():
    measured = comparators.physical_events(
        row(False, ["recorded_fall_or_upright_threshold_failure"])
    )
    assert measured["fall_or_upright_threshold_failure"] == "observed"


def test_complete_contact_and_fall_audits_support_absence():
    measured = comparators.physical_events(row(True))
    assert measured["undesired_environment_contact"] == "not_observed"
    assert measured["fall_or_upright_threshold_failure"] == "not_observed"
    assert measured["recorded_episode_duration_s"] == 5.96


def test_complete_fall_later_than_passage_is_retained():
    measured = comparators.physical_events(row(True, fall=True))
    assert measured["fall_or_upright_threshold_failure"] == "observed"


@pytest.mark.parametrize("fall,expected", [(False, "not_observed"), (True, "observed")])
def test_single_beam_fall_field_covers_whole_first_episode(fall, expected):
    value = row(True)
    value["passage"] = dict(fall_observed=fall)
    assert comparators.physical_events(value)["fall_or_upright_threshold_failure"] == expected


def test_course_fall_through_passage_cannot_establish_absence_in_recovery():
    value = row(True)
    value["passage"] = dict(beam_count=2, fall_observed=False)
    assert comparators.physical_events(value)["fall_or_upright_threshold_failure"] == "unknown"


def test_course_full_episode_event_overrides_earlier_passage_absence():
    value = row(True)
    value["passage"] = dict(beam_count=2, fall_observed=False, fall_anywhere_in_first_episode=True)
    assert comparators.physical_events(value)["fall_or_upright_threshold_failure"] == "observed"


def test_incomplete_block_cannot_produce_comparator_report(tmp_path, monkeypatch):
    expected = [dict(mode="forced", assignment_id=f"episode_{i}") for i in range(48)]
    declaration = dict(path="readiness", sha256="bound")
    monkeypatch.setattr(
        comparators.ready, "verify", lambda _: (declaration, dict(assignments=expected))
    )
    value = dict(
        readiness=declaration,
        assignments=[dict(assignment_id=r["assignment_id"]) for r in expected[:-1]],
        complete_panel=False,
        assigned_panel_episodes=138,
    )
    (tmp_path / "comparators_complete.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete original 48"):
        comparators.read_completed(tmp_path)


def test_reordered_block_cannot_select_a_different_subset(tmp_path, monkeypatch):
    expected = [dict(mode="forced", assignment_id=f"episode_{i}") for i in range(48)]
    declaration = dict(path="readiness", sha256="bound")
    monkeypatch.setattr(
        comparators.ready, "verify", lambda _: (declaration, dict(assignments=expected))
    )
    value = dict(
        readiness=declaration,
        assignments=[dict(assignment_id=r["assignment_id"]) for r in reversed(expected)],
        complete_panel=False,
        assigned_panel_episodes=138,
    )
    (tmp_path / "comparators_complete.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identities or ordering"):
        comparators.read_completed(tmp_path)
