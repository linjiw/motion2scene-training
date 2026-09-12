"""Fail-closed indexing and pairing for a shared-seed Kimodo corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptCell:
    """One registered cell in the body-mode by route prompt design."""

    prompt_index: int
    prompt_cell_id: str
    body_mode: str
    route: str


@dataclass(frozen=True)
class GeneratedReference:
    """A generated CSV bound to its immutable sidecar and registered cell."""

    csv_path: Path
    sidecar_path: Path
    generation_seed: int
    generated_prompt_cell_id: str
    matched_seed_group_id: str
    prompt_design_version: str
    cell: PromptCell

    @property
    def motion_id(self) -> str:
        return self.csv_path.stem

    @property
    def pair_key(self) -> tuple[int, str]:
        return self.generation_seed, self.cell.route


def prompt_index_from_name(path: Path) -> int:
    """Read the fixed-width prompt index from a generated artifact name."""

    prefix = path.stem.split("_", 1)[0]
    if len(prefix) != 3 or not prefix.isdigit():
        raise ValueError(f"artifact does not begin with a three-digit prompt index: {path.name}")
    return int(prefix)


def load_registered_references(
    motions_dir: Path,
    design_path: Path,
) -> tuple[dict, tuple[GeneratedReference, ...]]:
    """Load and validate a complete registered generation matrix.

    The generated sidecar uses a stable numeric ID such as ``prompt-003`` while the design
    gives that cell its semantic ID, such as ``duck_under_straight``. Both are retained; the
    numeric filename index is the join key and disagreement in prompt text or design version
    is fatal.
    """

    design = json.loads(design_path.read_text(encoding="utf-8"))
    cells = {
        int(item["prompt_index"]): PromptCell(
            prompt_index=int(item["prompt_index"]),
            prompt_cell_id=str(item["prompt_cell_id"]),
            body_mode=str(item["body_mode"]),
            route=str(item["route"]),
        )
        for item in design["prompt_cells"]
    }
    expected_seeds = {int(value) for value in design["design"]["generation_seeds"]}
    expected_version = str(design["prompt_design_version"])
    registered_count = int(design["design"]["registered_references"])

    references: list[GeneratedReference] = []
    seen: set[tuple[int, int]] = set()
    for csv_path in sorted(motions_dir.glob("*.csv")):
        index = prompt_index_from_name(csv_path)
        if index not in cells:
            raise ValueError(f"unregistered prompt index {index} in {csv_path.name}")
        sidecar_path = csv_path.with_suffix(".json")
        if not sidecar_path.exists():
            raise ValueError(f"generated CSV is missing its sidecar: {csv_path.name}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        seed = int(sidecar["generation_seed"])
        key = (index, seed)
        if key in seen:
            raise ValueError(f"duplicate prompt/seed cell: {key}")
        seen.add(key)
        if seed not in expected_seeds:
            raise ValueError(f"unregistered generation seed {seed} in {sidecar_path.name}")
        if sidecar["prompt_design_version"] != expected_version:
            raise ValueError(f"prompt design version mismatch in {sidecar_path.name}")
        if sidecar["csv"] != csv_path.name:
            raise ValueError(f"sidecar CSV identity mismatch in {sidecar_path.name}")
        references.append(
            GeneratedReference(
                csv_path=csv_path,
                sidecar_path=sidecar_path,
                generation_seed=seed,
                generated_prompt_cell_id=str(sidecar["prompt_cell_id"]),
                matched_seed_group_id=str(sidecar["matched_seed_group_id"]),
                prompt_design_version=str(sidecar["prompt_design_version"]),
                cell=cells[index],
            )
        )

    expected = {(index, seed) for index in cells for seed in expected_seeds}
    missing = expected - seen
    extra = seen - expected
    if len(references) != registered_count or missing or extra:
        raise ValueError(
            "generated matrix does not match registration: "
            f"observed={len(references)} registered={registered_count} "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return design, tuple(references)


def matched_walks(
    references: tuple[GeneratedReference, ...],
) -> dict[tuple[int, str], GeneratedReference]:
    """Return the unique same-seed, same-route null motion for every pair key."""

    result: dict[tuple[int, str], GeneratedReference] = {}
    for reference in references:
        if reference.cell.body_mode != "walk":
            continue
        if reference.pair_key in result:
            raise ValueError(f"multiple matched walks for {reference.pair_key}")
        result[reference.pair_key] = reference
    expected_keys = {reference.pair_key for reference in references}
    missing = expected_keys - set(result)
    if missing:
        raise ValueError(f"missing same-seed same-route walk(s): {sorted(missing)}")
    return result
