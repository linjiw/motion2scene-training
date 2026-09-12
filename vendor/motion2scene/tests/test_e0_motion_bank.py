from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/e0_motion_bank.py"


def _module():
    spec = importlib.util.spec_from_file_location("e0_motion_bank", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_results_keep_conjunctive_misses_visible() -> None:
    module = _module()
    rows = [
        {
            "body_mode": "walk",
            "turn": "straight",
            "outcome": "accepted",
            "external_contact_force_n": 0.0,
            "heading_change_rad": 0.0,
            "root_height_min_m": 0.70,
            "semantic": None,
            "status": "completed",
            "fps": 50.0,
        },
        {
            "body_mode": "duck_under",
            "turn": "gentle_right",
            "outcome": "accepted",
            "external_contact_force_n": 0.0,
            "heading_change_rad": -1.0,
            "root_height_min_m": 0.67,
            "semantic": {"satisfied": True, "measurements": {"duck_drop_m": 0.11}},
            "status": "completed",
            "fps": 50.0,
        },
        {
            "body_mode": "walk",
            "turn": "gentle_left",
            "outcome": "rejected",
            "external_contact_force_n": 0.0,
            "heading_change_rad": 1.0,
            "root_height_min_m": 0.75,
            "semantic": None,
            "status": "completed",
            "fps": 50.0,
        },
        {
            "body_mode": "arm_tuck",
            "turn": "straight",
            "outcome": "rejected",
            "external_contact_force_n": 0.0,
            "heading_change_rad": 0.0,
            "root_height_min_m": 0.68,
            "semantic": {"satisfied": True, "measurements": {"width_reduction_m": 0.20}},
            "status": "completed",
            "fps": 50.0,
        },
        {
            "body_mode": "shoulder_turn",
            "turn": "gentle_left",
            "outcome": "accepted",
            "external_contact_force_n": 0.0,
            "heading_change_rad": 1.0,
            "root_height_min_m": 0.72,
            "semantic": {"satisfied": True, "measurements": {"width_reduction_m": 0.09}},
            "status": "completed",
            "fps": 50.0,
        },
    ]
    diversity = {"pooled_rank": 4.6, "between_episode_rank": 2.75}

    result = module.prediction_results(rows, diversity, "completed")

    assert result["P1_q3_acceptance_and_zero_external"]["passed"] is False
    assert result["P2_route_heading_signs"]["passed"] is True
    assert result["P3_duck_depth_and_walk_contrast"]["passed"] is False
    assert result["P4_narrowing_semantics"]["passed"] is True
    assert result["P5_action_token_diversity"]["passed"] is False
    assert result["P6_infrastructure_contract"]["passed"] is True
