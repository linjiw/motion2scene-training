"""Separate actor inputs, teacher-only information, and outcome/provenance evidence."""

import hashlib
import json
import math

import numpy as np

from gear_sonic.research.scene_distillation.policy import PUBLIC_SHAPES


def public_observation_sha256(observation):
    if set(observation) != set(PUBLIC_SHAPES):
        raise ValueError(
            "Unknown actor fields: route, phase, motion ID and future reference are forbidden"
        )
    digest = hashlib.sha256()
    for key in sorted(PUBLIC_SHAPES):
        array = np.asarray(observation[key])
        if array.shape != PUBLIC_SHAPES[key]:
            raise ValueError(f"Invalid actor field {key}")
        if key == "obstacle_mask":
            if array.dtype != np.bool_:
                raise ValueError("Obstacle validity must be boolean")
            array = array.astype(np.uint8)
        else:
            array = array.astype("<f4")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite actor observation {key}")
        digest.update(key.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_query_record(record, *, allowed_motion_ids, teacher_sha256):
    """Consistency checks on a recorded teacher query; hashes alone are not authentication.

    DAgger labels must be queried at the student's state, including its history.
    A teacher's action from an earlier nominal trajectory is rejected.
    """
    if record["motion_id"] not in allowed_motion_ids or record["split"] != "train":
        raise ValueError("Development or unknown motion cannot train the student")
    if record["teacher_checkpoint_sha256"] != teacher_sha256:
        raise ValueError("Teacher checkpoint changed")
    if record["sample_origin"] not in ("teacher_rollout", "student_rollout"):
        raise ValueError("Unknown physical state source")
    expected_label = (
        "queried_teacher_on_student_state"
        if record["sample_origin"] == "student_rollout"
        else "executed_teacher"
    )
    if record["label_source"] != expected_label:
        raise ValueError("Teacher labels do not match the state collection mode")
    digest = public_observation_sha256(record["actor_observation"])
    if digest != record["teacher_query_observation_sha256"]:
        raise ValueError("Teacher was queried using a different observation/history")
    for key in ("state_sha256", "scene_sha256", "goal_sha256"):
        if record[key] != record["teacher_query_" + key]:
            raise ValueError(f"Teacher query {key} does not match the student's state/task")
    for key in ("decision_time_s", "observation_time_s", "teacher_query_state_time_s"):
        if not math.isfinite(record[key]):
            raise ValueError("Invalid simulator timestamp")
    if record["observation_time_s"] > record["decision_time_s"]:
        raise ValueError("Student observation is from the future")
    if record["teacher_query_state_time_s"] != record["decision_time_s"]:
        raise ValueError("Teacher label comes from a different simulator instant")
    if record["teacher_qualification"] != "scene_teacher_qualified":
        raise ValueError("A tracking-only teacher is not yet a qualified scene teacher")
    if not record.get("qualification_receipt_sha256") or not record.get(
        "state_snapshot_receipt_sha256"
    ):
        raise ValueError("Missing bound qualification or state-snapshot receipt")
    return digest


def audit_scene_pairing(scenes):
    """Count paired proposals without turning kinematic proxy flags into dynamic labels."""
    grouped = {}
    for scene in scenes:
        grouped.setdefault(scene["motion_id"], []).append(scene)
    nested = 0
    for pair in grouped.values():
        if len(pair) == 2:
            ordered = sorted(pair, key=lambda x: x["requested_obstacles"])
            nested += set(ordered[0]["cells"]).issubset(ordered[1]["cells"])
    return {
        "motions": len(grouped),
        "scenes": len(scenes),
        "nested_scene_pairs": nested,
        "teacher_eligible_scenes": sum(s["teacher_eligible"] for s in scenes),
        "executed_scene_teacher_labels": sum(s["teacher_actions_present"] for s in scenes),
        "validated_obstacle_switch_pairs": None,
        "switch_pair_status": "Requires matched state/goals, changed scene, and different qualified responses",
    }
