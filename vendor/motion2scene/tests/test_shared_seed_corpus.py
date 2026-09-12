from __future__ import annotations

import json
from pathlib import Path

import pytest

from motion2scene.motion.shared_seed_corpus import (
    load_registered_references,
    matched_walks,
    prompt_index_from_name,
)


def design() -> dict:
    return {
        "prompt_design_version": "v2",
        "design": {"generation_seeds": [11, 12], "registered_references": 4},
        "prompt_cells": [
            {
                "prompt_index": 0,
                "prompt_cell_id": "walk_straight",
                "body_mode": "walk",
                "route": "straight",
            },
            {
                "prompt_index": 1,
                "prompt_cell_id": "duck_straight",
                "body_mode": "duck_under",
                "route": "straight",
            },
        ],
    }


def write_reference(root: Path, index: int, seed: int) -> None:
    stem = f"{index:03d}_motion_s{seed}"
    (root / f"{stem}.csv").write_text("0\n", encoding="utf-8")
    (root / f"{stem}.json").write_text(
        json.dumps(
            {
                "csv": f"{stem}.csv",
                "generation_seed": seed,
                "prompt_cell_id": f"prompt-{index:03d}",
                "matched_seed_group_id": f"v2:seed:{seed}",
                "prompt_design_version": "v2",
            }
        ),
        encoding="utf-8",
    )


def test_loads_complete_matrix_and_pairs_null_walks(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design()), encoding="utf-8")
    for index in range(2):
        for seed in (11, 12):
            write_reference(tmp_path, index, seed)

    _, references = load_registered_references(tmp_path, design_path)
    walks = matched_walks(references)

    assert len(references) == 4
    assert set(walks) == {(11, "straight"), (12, "straight")}
    assert {item.cell.prompt_cell_id for item in references if item.cell.prompt_index == 1} == {
        "duck_straight"
    }


def test_missing_registered_cell_fails_closed(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design()), encoding="utf-8")
    for index, seed in ((0, 11), (0, 12), (1, 11)):
        write_reference(tmp_path, index, seed)

    with pytest.raises(ValueError, match="does not match registration"):
        load_registered_references(tmp_path, design_path)


def test_prompt_index_requires_canonical_prefix() -> None:
    with pytest.raises(ValueError, match="three-digit"):
        prompt_index_from_name(Path("3_motion.csv"))
