"""Load an arm-specific ridge policy while binding its full physical motion bank."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    checked_artifact,
    definition_digest,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    load_schedule_policy,
)
from scripts.research.motion2scene_extension_teaching import arm_view


def read(ref):
    return json.loads(checked_artifact(ref).read_text())


def load_extension_policy(path, sha256, bank):
    result_ref = dict(path=str(path), sha256=sha256)
    result = read(result_ref)
    study = read(result["study"])
    prepared = read(study["prepared"])
    registry = read(result["registry"])
    arm, analysis = result["arm"], result["analysis"]
    view, indices = arm_view(bank, arm)
    fold = analysis["fold"]
    if (
        study["schema"] != "motion2scene_extension_equivalent_teaching_v1"
        or result["registry"] != study["registry"]
        or definition_digest(registry["request"]) != definition_digest(bank.request)
        or arm not in study["arms"]
        or analysis["arm"] != arm
        or fold not in study["folds"]
        or len(fold["training_task_ids"]) != 8
        or len(set(fold["training_task_ids"])) != 8
        or len(fold["evaluation_task_ids"]) != 4
        or len(set(fold["evaluation_task_ids"])) != 4
        or set(fold["training_task_ids"]) & set(fold["evaluation_task_ids"])
        or analysis["original_option_indices"] != indices.tolist()
        or analysis["assigned_training_branches"] != 8 * len(view.option_ids)
        or result["logical_request_digest"] != definition_digest(view.request)
        or study["l2"] != 10.0
    ):
        raise ValueError("arm policy, training fold or full physical bank binding differs")
    assignments = {t["task_id"]: t for t in prepared["tasks"]}
    expected_paths = [
        str(Path(assignments[task]["collection"]["path"]).parent / "result.json")
        for task in fold["training_task_ids"]
    ]
    if [r["path"] for r in analysis["training_collections"]] != expected_paths:
        raise ValueError("policy training collections differ from its assigned training-only fold")
    policy = result["policy"]
    model = load_schedule_policy(policy["path"], policy["sha256"], view)
    if float(model["l2"]) != 10.0 or not np.array_equal(
        model["trained_mask"], model["qualified_mask"]
    ):
        raise ValueError("common penalty and every available arm head required")
    return model, result
