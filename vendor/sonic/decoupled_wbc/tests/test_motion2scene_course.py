import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_course import (
    COURSE_SCHEMA,
    audit_imported_beams,
    author_course,
    reference_frame_budget,
    score_course,
    synchronize_course_forces,
    validate_course_definition,
)


def beams():
    return [
        {
            "center_xy_m": [x, 0],
            "yaw_rad": 0,
            "length_m": 0.2,
            "width_m": 1.2,
            "thickness_m": 0.1,
            "underside_m": 1.25,
        }
        for x in (0.3, 0.8)
    ]


def payload():
    t = np.arange(101) / 50
    pos = np.c_[t, t * 0, t * 0 + 0.8]
    return {
        "motion_time_s": t,
        "fps": 50,
        "root_pos_w": pos,
        "projected_gravity_b": np.tile([0, 0, -1], (101, 1)),
        "body_pos_w": pos[:, None],
    }


def test_earlier_beam_contacts_remain_scored_until_course_end():
    p = payload()
    forces = np.zeros((101, 2, 1, 3))
    assert score_course(p, forces, beams(), commands_valid=True, timeout_s=3)["pass"]
    # Contact on beam0 after its own completion still invalidates the course.
    forces[50, 0, 0, 0] = 2
    result = score_course(p, forces, beams(), commands_valid=True, timeout_s=3)
    assert not result["pass"] and result["max_force_by_beam_n"] == [2, 0]


def test_nonfinite_course_force_cannot_pass():
    forces = np.zeros((101, 2, 1, 3))
    forces[50, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="complete course forces"):
        score_course(payload(), forces, beams(), commands_valid=True, timeout_s=3)


def test_reference_budget_matches_exclusive_endpoint_loader():
    budget = reference_frame_budget(120, 30)
    assert budget["frames"] == 199
    assert budget["duration_s"] == 3.96
    assert reference_frame_budget(120, 50)["frames"] == 120
    assert reference_frame_budget(121, 30)["frames"] == 200


def test_authored_beams_preserve_actual_lengths():
    template = '#usda 1.0\ndef Xform "World" {\n    def Cube "CounterfactualBeam" { }\n}\n'
    definitions = beams()
    definitions[1]["length_m"] = 0.7
    scene = author_course(template, definitions)
    assert 'def Cube "CounterfactualBeam_01"' in scene
    assert "xformOp:scale = (0.7, 1.2, 0.1)" in scene


def test_template_tail_is_never_silently_deleted():
    template = '#usda 1.0\ndef Xform "World" {\n    def Cube "CounterfactualBeam" { }\n    def Cube "Keep" {}\n}\n'
    with pytest.raises(ValueError, match="final prim"):
        author_course(template, beams())


def test_incomplete_early_beam_keeps_late_contacts_and_no_course_completion():
    definitions = beams()
    definitions[0]["center_xy_m"][0] = 3
    force = np.zeros((101, 2, 1, 3))
    force[90, 0, 0, 0] = 7
    result = score_course(payload(), force, definitions, commands_valid=True, timeout_s=3)
    assert result["beam_finish_frames_exclusive"][0] is None
    assert result["beam_finish_frames_exclusive"][1] is not None
    assert result["passage_finish_frame_exclusive"] is None
    assert result["max_force_by_beam_n"] == [7, 0]
    assert "course_beam_contact" in result["failure_reasons"]


def test_second_beam_not_reached_before_reference_end_is_explicit_failure():
    definitions = beams()
    definitions[1]["center_xy_m"][0] = 3
    result = score_course(
        payload(),
        np.zeros((101, 2, 1, 3)),
        definitions,
        commands_valid=True,
        timeout_s=3,
        reference_frames=101,
        final_reference_phase_s=2,
    )
    assert result["completed_beams"] == 1 and not result["pass"]
    assert result["reference_horizon_reached"]
    assert "reference_horizon_reached_before_completion" in result["failure_reasons"]


def test_starting_beyond_constraint_is_not_a_traversal():
    definitions = beams()
    definitions[0]["center_xy_m"][0] = -0.3
    result = score_course(
        payload(), np.zeros((101, 2, 1, 3)), definitions, commands_valid=True, timeout_s=3
    )
    assert not result["pass"]
    assert "invalid_initial_approach" in result["failure_reasons"]


def test_every_beam_retains_its_own_physics_peak():
    force = np.zeros((8, 2, 1, 3))
    force[0, 0, 0, 2] = 10
    force[2, 1, 0, 1] = -12
    packet = {
        "physics_force_w": force,
        "control_force_w": force[[3, 7]].copy(),
        "physics_steps": np.arange(1, 9),
        "control_steps": np.array([4, 8]),
        "physics_dt": 0.005,
    }
    peaks, error = synchronize_course_forces(packet)
    assert error == 0 and peaks.shape == (2, 2, 1, 3)
    assert np.array_equal(peaks[0, 0, 0], [0, 0, 10])
    assert np.array_equal(peaks[0, 1, 0], [0, -12, 0])
    packet["control_force_w"][0, 1, 0, 0] = 1
    with pytest.raises(ValueError, match="cached/direct"):
        synchronize_course_forces(packet)


def test_native_geometry_checks_actual_dimensions_not_just_prim_presence():
    definitions = beams()
    rows = []
    for i, beam in enumerate(definitions):
        matrix = np.diag([beam["length_m"], beam["width_m"], beam["thickness_m"], 1.0])
        matrix[3, :3] = [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
        rows.append(
            {
                "path": "/World/ground/terrain/CounterfactualBeam" + ("_01" if i else ""),
                "type": "Cube",
                "collision": True,
                "rigid_body": True,
                "attributes": {"size": "1.0"},
                "local_to_world_at_capture_start": matrix.tolist(),
            }
        )
    assert len(audit_imported_beams(rows, definitions)) == 2
    rows[1]["local_to_world_at_capture_start"][0][0] = 0.7
    with pytest.raises(ValueError, match="dimensions differ"):
        audit_imported_beams(rows, definitions)


def test_definition_rejects_evaluation_and_reference_extension():
    definition = {
        "schema": COURSE_SCHEMA,
        "split": "development",
        "course_id": "fresh_smoke",
        "motion_mode": "one_authored_reference_pass",
        "beams": beams(),
        "reference_budget": reference_frame_budget(120, 30),
        "timeout_s": 3.96,
    }
    validate_course_definition(definition)
    definition["split"] = "evaluation"
    with pytest.raises(ValueError, match="fresh-development"):
        validate_course_definition(definition)
    definition["split"] = "development"
    definition["timeout_s"] = 5
    with pytest.raises(ValueError, match="existing reference"):
        validate_course_definition(definition)
