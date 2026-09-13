"""Admit only motor-executed suffixes with independently recomputed task support."""

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.direct_context import score_navigation_task
from gear_sonic.research.scene_distillation.navigation_continuation import (
    continuation_commands,
    continuation_outcomes,
    validate_continuation,
)
from gear_sonic.research.scene_distillation.navigation_data import read_bound
from gear_sonic.research.scene_distillation.navigation_localization import episode_localization
from gear_sonic.research.scene_distillation.navigation_recovery import suffix_support
from gear_sonic.research.scene_distillation.tasks import validate_task


def load_motor_recoveries(config):
    fresh_behavior = config.get("fresh_recovery_behavior_sha256")
    if fresh_behavior is not None and (
        not isinstance(fresh_behavior, str)
        or len(fresh_behavior) != 64
        or any(c not in "0123456789abcdef" for c in fresh_behavior)
    ):
        raise ValueError("Invalid fresh recovery behavior hash")
    manifest = read_bound(config["dataset_manifest"], config["dataset_manifest_sha256"])
    if manifest.get("schema") != "motor_navigation_recovery_manifest_v1":
        raise ValueError("Expected executed motor recovery manifest")
    catalog = read_bound(config["ancestry_catalog"], config["ancestry_catalog_sha256"])
    ancestry = {e["id"]: e for e in catalog}
    tasks = read_bound(config["task_manifest"], config["task_manifest_sha256"])
    allowed = {e["task_sha256"] for e in tasks["tasks"]}
    episodes, seen = [], set()
    for parent in manifest["parents"]:
        receipt = read_bound(parent["path"], parent["sha256"])
        if (
            receipt.get("schema") != "executed_motor_navigation_recovery_v1"
            or receipt.get("motor_sha256") != config["motor_sha256"]
            or receipt.get("teacher_sha256") != config["teacher_sha256"]
            or receipt["task"]["sha256"] not in allowed
            or receipt.get("reference_phase_switch") is not False
        ):
            raise ValueError("Recovery source, motor or original task contract mismatch")
        task = validate_task(read_bound(receipt["task"]["path"], receipt["task"]["sha256"]))
        source = ancestry[task["motion_id"]]
        if task["split"] != "train" or source["split"] != "train" or not source.get("group"):
            raise ValueError("Recovery lacks training ancestry")
        score = read_bound(receipt["score"]["path"], receipt["score"]["sha256"])
        if (
            score.get("state") != "complete"
            or score.get("teacher_mode") is not False
            or score.get("teacher_sha256") != config["teacher_sha256"]
            or score.get("student_sha256") != receipt["behavior_checkpoint"]["sha256"]
            or score.get("task_sha256") != receipt["task"]["sha256"]
            or score.get("control_dt") != 0.02
            or score.get("contact_dt") != 0.005
        ):
            raise ValueError("Recovery score is not bound to its behavior execution")
        switch = receipt["takeover_tick"]
        if not isinstance(switch, int) or not 0 <= switch < task["deadline_ticks"]:
            raise ValueError("Invalid recovery switch")
        expected_source = (
            "motor_demonstration" if switch == 0 else "navigation_prefix_motor_recovery"
        )
        profiles = (
            ("motion_full_current_v2",)
            if switch == 0
            else ("nav_goal_map_v1", "nav_goal_map_localization_v2")
        )
        if receipt.get("source") != expected_source or score.get("actor_profile") not in profiles:
            raise ValueError("Recovery behavior profile differs from its source")
        if switch == 0 and receipt["behavior_checkpoint"]["sha256"] != config["motor_sha256"]:
            raise ValueError("Motor demonstration uses a different motor")
        for binding in [receipt["shard"], receipt["behavior_checkpoint"], parent["trace"]]:
            if sha(binding["path"]) != binding["sha256"]:
                raise ValueError("Changed recovery shard, trace or behavior policy")
        if receipt["shard"]["path"] in seen:
            raise ValueError("Duplicate recovery shard")
        seen.add(receipt["shard"]["path"])
        with np.load(parent["trace"]["path"], allow_pickle=False) as trace:
            actual_fall = bool((trace["root_xyz"][:, 2] < 0.25).any())
            actual = score_navigation_task(
                task, trace["root_xyz"], trace["speed"], trace["undesired_force"], fell=actual_fall
            )
            if any(score.get(key) != value for key, value in actual.items()):
                raise ValueError("Recovery whole-task score differs from physics")
            mask, suffix = suffix_support(task, trace, switch, actual_fall)
            previous_post_positions = trace["root_xyz"][:-1].copy()
            if "continuation_outcomes" in receipt and receipt[
                "continuation_outcomes"
            ] != continuation_outcomes(task, trace, min(switch, len(mask))):
                raise ValueError("Continuation support labels disagree with execution")
        if suffix != receipt["suffix_score"] or bool(mask.any()) != receipt["supported"]:
            raise ValueError("Recovery success disagrees with physical suffix")
        with np.load(receipt["shard"]["path"], allow_pickle=False) as stored:
            arrays = {k: torch.from_numpy(stored[k].copy()) for k in stored.files}
        n = receipt["rows"]
        shapes = dict(
            proprio=(930,),
            controls=(114,),
            control_mask=(114,),
            actions=(29,),
            motor_actions=(29,),
            teacher_actions=(29,),
            teacher_tokens=(64,),
            navigation_context=(10,),
            obstacles_body=(5, 15),
            obstacle_mask=(5,),
            query_mask=(),
            learner_query_mask=(),
        )
        if n != score["control_steps"] or n != len(mask) or n > task["deadline_ticks"]:
            raise ValueError("Recovery length differs from original task execution")
        for key, shape in shapes.items():
            if arrays[key].shape != (n, *shape) or not torch.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid recovery array {key}")
        for key in ["control_mask", "obstacle_mask", "query_mask", "learner_query_mask"]:
            if arrays[key].dtype != torch.bool:
                raise ValueError("Nonboolean recovery mask")
        expected_mask = torch.ones_like(arrays["control_mask"])
        expected_mask[:, 7] = False
        if not torch.equal(arrays["control_mask"], expected_mask):
            raise ValueError("Recovery requires the complete 114D current command")
        if (
            not np.array_equal(arrays["query_mask"].numpy(), mask)
            or int(mask.sum()) != receipt["supported_rows"]
        ):
            raise ValueError("Recovery target mask admits unqualified rows")
        entry = np.zeros(n, dtype=bool)
        switch = receipt["takeover_tick"]
        if switch > 0 and mask.any():
            entry[switch] = True
        if (
            not np.array_equal(arrays["learner_query_mask"].numpy(), entry)
            or int(entry.sum()) != receipt["learner_query_rows"]
        ):
            raise ValueError("Recovery tails cannot be counted as new learner queries")
        if not torch.equal(arrays["actions"][mask], arrays["motor_actions"][mask]):
            raise ValueError("Supported actions were not executed by the motor")
        if not np.allclose(
            arrays["measured_root_xyz"][1:].numpy(), previous_post_positions, atol=1e-6, rtol=0
        ):
            raise ValueError("Recovery pre-action pose does not follow physical history")
        reconstructed = torch.from_numpy(episode_localization(arrays, task["goal_xyz"]))
        if not torch.equal(reconstructed, arrays["localization"]):
            raise ValueError("Recovery localization differs from causal pose history")
        provider = receipt.get("continuation")
        if provider is not None:
            validate_continuation(provider)
            for key, shape in {
                "nominal_controls": (n, 114),
                "reference_anchor_xyz": (n, 3),
            }.items():
                if (
                    key not in arrays
                    or arrays[key].shape != shape
                    or not torch.isfinite(arrays[key]).all()
                ):
                    raise ValueError("Missing continuation reconstruction evidence")
            for tick in range(n):
                nominal = arrays["nominal_controls"][tick].numpy()
                expected = (
                    nominal
                    if tick < switch
                    else continuation_commands(
                        nominal,
                        arrays["measured_root_xyz"][tick].numpy(),
                        arrays["measured_root_wxyz"][tick].numpy(),
                        arrays["reference_anchor_xyz"][tick].numpy(),
                        task["goal_xyz"],
                        reconstructed[tick].numpy(),
                        provider,
                    )
                )
                if not np.array_equal(expected, arrays["controls"][tick].numpy()):
                    raise ValueError("Stored command differs from declared continuation")
        if mask.any():
            arrays["decision_index"] = torch.arange(n)
            episodes.append(
                dict(
                    arrays=arrays,
                    motion_id=task["motion_id"],
                    ancestry=source["group"],
                    role=(
                        "recovery"
                        if switch > 0
                        and (
                            fresh_behavior is None
                            or receipt["behavior_checkpoint"]["sha256"] == fresh_behavior
                        )
                        else "replay"
                    ),
                    task_path=receipt["task"]["path"],
                    shard_sha256=receipt["shard"]["sha256"],
                )
            )
    if not episodes:
        raise ValueError("No supported motor continuations")
    return episodes
