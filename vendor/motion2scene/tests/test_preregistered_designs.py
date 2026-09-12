from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_shared_seed_v2_is_a_complete_paired_factorial() -> None:
    config = json.loads((ROOT / "configs/e0_kimodo_shared_seed_factorial_v2.json").read_text())
    prompt_path = ROOT / config["prompt_file"]
    prompts = [
        line for line in prompt_path.read_text().splitlines() if line and not line.startswith("#")
    ]
    cells = config["prompt_cells"]
    seeds = config["design"]["generation_seeds"]

    assert sha256(prompt_path) == config["prompt_file_sha256"]
    assert len(prompts) == len(set(prompts)) == 18
    assert len(cells) == 18
    assert len(seeds) == len(set(seeds)) == 8
    assert config["design"]["registered_references"] == 144
    assert config["design"]["independent_experimental_unit"] == "generation_seed"
    assert sorted(cell["prompt_index"] for cell in cells) == list(range(18))
    assert config["generator"]["seed_design"] == "shared_across_prompts"


def test_pilot_is_never_a_confirmatory_test_set() -> None:
    config = json.loads((ROOT / "configs/e0_kimodo_shared_seed_factorial_v2.json").read_text())
    development = config["development_corpus"]
    assert development["permanent_role"] == "pilot_characterization_v1"
    assert "final test set" in development["forbidden_uses"]
    assert "paired S0-S4 label" in development["forbidden_uses"]


def test_v2_freezes_functional_not_root_or_world_axis_metrics() -> None:
    config = json.loads((ROOT / "configs/e0_kimodo_shared_seed_factorial_v2.json").read_text())
    effects = config["semantic_protocol"]["minimum_effects"]
    assert "duck_whole_body_top_reduction_m" in effects
    assert "arm_tuck_route_normal_width_reduction_m" in effects
    assert config["semantic_protocol"]["null_motion"] == "same_generation_seed_same_route_walk"


def test_q4_protocol_is_three_seed_and_main_bank_is_strict() -> None:
    protocol = json.loads((ROOT / "experiments/registrations/E0_Q4_PROTOCOL_V1.json").read_text())
    assert protocol["registered_candidate_count"] == 0
    assert len(protocol["physics_seeds"]) == len(set(protocol["physics_seeds"])) == 3
    assert protocol["main_bank_admission"] == "q4_strict_only"
    assert protocol["candidate_admission"]["achieved_semantic_status"].startswith(
        "controller_retained_S4"
    )


def test_route_retention_validation_uses_fresh_neutral_seeds() -> None:
    design = json.loads((ROOT / "configs/e1_route_retention_heldout_v1.json").read_text())
    generation = design["generation"]
    prompt_path = ROOT / generation["prompt_file"]

    assert sha256(prompt_path) == generation["prompt_file_sha256"]
    assert generation["seeds"] == list(range(42001, 42009))
    assert not set(generation["seeds"]) & set(range(41001, 41009))
    assert generation["registered_reference_count"] == 8
    assert design["analysis_role"] == "heldout_instrument_validation"
    assert sha256(ROOT / design["analysis_implementation"]["path"]) == design[
        "analysis_implementation"
    ]["sha256"]
    assert sha256(ROOT / design["registered_predictions"]["path"]) == design[
        "registered_predictions"
    ]["sha256"]
    assert not design["relative_retention_policy"][
        "cumulative_absolute_curvature_is_a_hard_gate"
    ]
    assert "original v2" in design["promotion_limits"][0]
