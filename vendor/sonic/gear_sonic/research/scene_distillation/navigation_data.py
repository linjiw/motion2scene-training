"""A task-success view of executed stopping episodes, distinct from motor prefixes."""

import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.motor_training import current_frame_extension
from gear_sonic.research.scene_distillation.navigation_localization import episode_localization
from gear_sonic.research.scene_distillation.tasks import validate_task


def read_bound(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Changed navigation evidence: {path}")
    return json.loads(Path(path).read_text())


def load_successful_tasks(config):
    """Keep chronological rows and target masks; never promote generic prefix eligibility.

    This adapter is intentionally restricted to the recorded stopping collector's
    contiguous, 50Hz episodes. It is not an importer for arbitrary legacy queries.
    Ancestry and motion IDs stay in metadata and never enter actor tensors.
    """
    manifest = read_bound(config["dataset_manifest"], config["dataset_manifest_sha256"])
    if (
        manifest.get("schema") != "bfm_executed_foundation_v1"
        or manifest.get("context_schema") != "complete_known_map_goal_v1"
        or manifest.get("teacher_sha256") != config["teacher_sha256"]
    ):
        raise ValueError("Unbound task-context manifest")
    catalog = read_bound(config["ancestry_catalog"], config["ancestry_catalog_sha256"])
    ancestry = {e["id"]: e for e in catalog}
    admitted = {}
    for parent in manifest["parents"]:
        collection = read_bound(parent["path"], parent["sha256"])
        if (
            collection.get("scene_task_success") is not True
            or collection.get("teacher_sha256") != config["teacher_sha256"]
        ):
            raise ValueError("Navigation positives require independent task success")
        for e in collection["episodes"]:
            admitted[e["path"]] = e
    episodes = []
    seen = set()
    for e in manifest["episodes"]:
        if e != admitted.get(e["path"]) or not e["eligible"] or e["split"] != "train":
            raise ValueError("Aggregate does not match successful train episode evidence")
        if e["path"] in seen:
            raise ValueError("Duplicate task episode")
        seen.add(e["path"])
        motion = ancestry[e["motion_id"]]
        if motion["split"] != "train" or not motion.get("group"):
            raise ValueError("Missing train ancestry")
        score = read_bound(e["task_success_evidence"]["path"], e["task_success_evidence"]["sha256"])
        task = validate_task(read_bound(e["task_path"], e["task_sha256"]))
        if (
            score.get("state") != "complete"
            or score.get("navigation_success") is not True
            or score.get("teacher_mode") is not True
            or score.get("teacher_sha256") != config["teacher_sha256"]
            or score.get("task_sha256") != e["task_sha256"]
            or score.get("control_dt") != 0.02
            or score.get("contact_dt") != 0.005
            or task["motion_id"] != e["motion_id"]
            or task["split"] != "train"
        ):
            raise ValueError("Task success is missing or bound to different execution")
        if sha(e["path"]) != e["sha256"]:
            raise ValueError("Changed task shard")
        with np.load(e["path"], allow_pickle=False) as stored:
            arrays = {k: torch.from_numpy(stored[k].copy()) for k in stored.files}
        n = e["rows"]
        widths = dict(
            proprio=(930,),
            controls=(79,),
            control_mask=(79,),
            teacher_tokens=(64,),
            teacher_actions=(29,),
            future_reference=(640,),
            navigation_context=(10,),
            obstacles_body=(5, 15),
            obstacle_mask=(5,),
            query_mask=(),
        )
        if n != score["control_steps"] or n < 1:
            raise ValueError("Episode length differs from physical receipt")
        for k, shape in widths.items():
            if arrays[k].shape != (n, *shape) or not torch.isfinite(arrays[k]).all():
                raise ValueError(f"Invalid task array {k}")
        for k in ("query_mask", "control_mask", "obstacle_mask"):
            if arrays[k].dtype != torch.bool:
                raise ValueError(f"Nonboolean mask {k}")
        if int(arrays["query_mask"].sum()) != e["supported_query_rows"]:
            raise ValueError("Target support count mismatch")
        if config.get("localization", False):
            reconstructed = torch.from_numpy(episode_localization(arrays, task["goal_xyz"]))
            if "localization" in arrays and not torch.equal(reconstructed, arrays["localization"]):
                raise ValueError("Stored localization differs from causal reconstruction")
            arrays["localization"] = reconstructed
        extra = current_frame_extension(arrays["future_reference"], arrays["controls"])
        arrays["controls"] = torch.cat([arrays["controls"], extra], -1)
        # Row indices remain intact even if a future view masks an interior target.
        arrays["decision_index"] = torch.arange(n)
        episodes.append(
            dict(
                arrays=arrays,
                motion_id=e["motion_id"],
                ancestry=motion["group"],
                task_path=e["task_path"],
                shard_sha256=e["sha256"],
            )
        )
    if not episodes:
        raise ValueError("No successful navigation episodes")
    return episodes
