"""Fail-closed contract for future executed teacher-label shards."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha


def load_teacher_shard(path, metadata_path, *, allowed_motion_ids, decoder_sha256):
    metadata = json.loads(Path(metadata_path).read_text())
    required = {
        "label_source": "executed_teacher",
        "state_source": "measured_simulator_state",
        "route_source": "provided_command",
        "token_representation": "post_fsq_64",
        "decoder_sha256": decoder_sha256,
        "split": "train",
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"Teacher shard requires {key}={value!r}")
    if sha(path) != metadata.get("shard_sha256"):
        raise ValueError("Teacher shard hash mismatch")
    receipts = metadata.get("execution_receipts", [])
    if not receipts:
        raise ValueError("Executed teacher receipts are required; reference poses are not labels")
    for receipt in receipts:
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Execution receipt hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    expected_shapes = {
        "proprio": (930,),
        "start_goal_body": (6,),
        "teacher_tokens": (64,),
        "teacher_action_mean": (29,),
        "obstacles_body": (5, 15),
        "obstacle_mask": (5,),
        "route_body": (16, 3),
        "route_mask": (16,),
        "motion_id": (),
        "decision_time_s": (),
        "observation_time_s": (),
    }
    if "proprio" not in arrays:
        raise ValueError("Missing measured proprioception")
    size = len(arrays["proprio"])
    if size < 1:
        raise ValueError("Empty teacher shard")
    for key, shape in expected_shapes.items():
        if key not in arrays or arrays[key].shape != (size, *shape):
            raise ValueError(f"Invalid teacher field {key}")
        if arrays[key].dtype.kind in "fc" and not np.isfinite(arrays[key]).all():
            raise ValueError(f"Non-finite teacher field {key}")
    if not set(arrays["motion_id"].tolist()) <= set(allowed_motion_ids):
        raise ValueError("Teacher shard contains development or unknown motion IDs")
    if np.any(arrays["observation_time_s"] > arrays["decision_time_s"]):
        raise ValueError("Future observations cannot enter the student")
    if not np.allclose(
        arrays["teacher_tokens"] * 16, np.round(arrays["teacher_tokens"] * 16), atol=1e-5
    ):
        raise ValueError("Teacher token representation is not the release FSQ lattice")
    if np.any(arrays["teacher_tokens"] < -1) or np.any(arrays["teacher_tokens"] > 15 / 16):
        raise ValueError("Teacher tokens exceed the release FSQ bounds")
    return arrays
