"""Synthetic audit checks; these fixtures are not physical evidence."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_audit_complementary_capability import (  # noqa: E402
    all_capsule_clearance,
    beam_force_trace,
    sensor_comparisons,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402


def test_full_capsule_forecast_preserves_true_clearance_beyond_proposal_cap():
    beam = dict(
        center_xy_m=[0, 0], yaw_rad=0, underside_m=1, thickness_m=0.2, length_m=1, width_m=1
    )
    points = np.zeros((1, 1, 3))
    capsules = points, points, np.asarray([0.1]), ["body"]
    assert all_capsule_clearance(capsules, beam, np.zeros(4)) == pytest.approx(0.9)
    assert all_capsule_clearance(capsules, beam, np.array([0, 0, -0.5, 0])) == pytest.approx(0.4)


def test_force_audit_maps_each_subject_own_counterpart_column(tmp_path):
    force = np.zeros((1192, 2, 2, 3))
    force[5, 0, 1, 0] = 12
    force[7, 1, 0, 0] = -20
    # A huge unrelated counterpart force cannot become beam collision.
    force[3, 0, 0, 0] = 1000
    path = tmp_path / "pairs.npz"
    np.savez(
        path,
        force_w=force,
        physics_steps=np.arange(1, 1193),
        physics_dt_s=0.005,
        body_names=["a", "b"],
    )
    mapping = dict(
        pair_subject_body_names=["a", "b"],
        sensors={
            "a": dict(filter_paths=["floor", "beam"]),
            "b": dict(filter_paths=["beam", "floor"]),
        },
    )
    metadata = tmp_path / "mapping.json"
    metadata.write_text(json.dumps(mapping))
    row = dict(environment_mapping=artifact(metadata), environment_pairs=artifact(path))
    _, trace, report = beam_force_trace(row, "beam")
    assert trace.max() == 20 and report["contacting_bodies"] == ["a", "b"]
    assert report["first_beam_force_above_1n_s"] == pytest.approx(0.03)
    assert report["physics_steps_above_1n"] == 2
    mapping["sensors"]["b"]["filter_paths"] = ["other", "floor"]
    metadata.write_text(json.dumps(mapping))
    row["environment_mapping"] = artifact(metadata)
    with pytest.raises(ValueError, match="measured beam counterpart"):
        beam_force_trace(row, "beam")


def test_proprioceptive_difference_is_not_called_sensor_distinguishability():
    names = ["corridor_0_0.75_upper_hit", "root_lin_vel_w_0"]
    target = dict(phase_tick=15, feature_names=names, features=[0, 0])
    original = dict(phase_tick=15, feature_names=names, features=[0, 1])
    rows = sensor_comparisons(dict(targets=[target]), [dict(scene_id="other", targets=[original])])
    comparison = rows[0]["comparisons"][0]
    assert comparison["exact_sensor28_alias"] and not comparison["exact_full114_alias"]
    assert comparison["changed_sensor_feature_count"] == 0
