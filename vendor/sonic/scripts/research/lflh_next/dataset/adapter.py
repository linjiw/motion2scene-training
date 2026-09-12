"""Read geometry records and explicitly guard the physical-teacher boundary."""

import json
from pathlib import Path

import numpy as np


def load_reference_scene(dataset, scene_id, require_executed_teacher=True):
    root = Path(dataset).resolve()
    if Path(scene_id).name != scene_id or not scene_id:
        raise ValueError("Expected a scene identifier, not a path")
    with (root / "scenes" / f"{scene_id}.json").open() as f:
        record = json.load(f)
    if require_executed_teacher:
        labels = record.get("physical_labels") or {}
        if not record.get("teacher_eligible") or labels.get("teacher_actions") is None:
            raise ValueError(
                "Executed teacher labels unavailable: this is a kinematic reference-scene record"
            )
    path = (root / "scenes" / record["reference"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Reference path escaped dataset root")
    with np.load(path, allow_pickle=False) as arrays:
        reference = {k: arrays[k] for k in arrays.files}
    return record, reference
