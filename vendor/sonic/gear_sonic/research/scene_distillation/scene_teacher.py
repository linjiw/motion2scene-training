"""Bound scene continuations and same-state expert labels; never infer labels from geometry."""

import json
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import sha


def verify_scene_receipt(binding, teacher_sha256, task_sha256):
    """Verify physical evidence and explicit contact/arrival checks for an exact task."""
    path = Path(binding["path"])
    if sha(path) != binding["sha256"]:
        raise ValueError("Scene qualification receipt changed")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("state") != "complete"
        or receipt.get("teacher_sha256") != teacher_sha256
        or receipt.get("task_sha256") != task_sha256
    ):
        raise ValueError("Scene qualification task/teacher mismatch")
    required = ("whole_motion", "environment_contacts", "goal_reached", "terminal_hold")
    if any(receipt.get("checks", {}).get(key) is not True for key in required):
        raise ValueError("Scene qualification lacks complete physical checks")
    evidence = receipt.get("evidence", [])
    if not evidence:
        raise ValueError("Scene qualification has no executed evidence")
    for item in evidence:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Scene physical evidence changed")
    if receipt.get("control_dt") != 0.02 or receipt.get("contact_dt") != 0.005:
        raise ValueError("Scene qualification timing mismatch")
    return receipt


class QualifiedSceneRegistry:
    """Select only an explicitly executed continuation for an exactly bound scene/goal.

    This supplies a finite task expert, not a general route planner. Changed scene
    or changed goal must have its own qualified task entry. Missing entries stay missing.
    """

    def __init__(self, entries, teacher_sha256):
        self.teacher_sha256 = teacher_sha256
        self.entries = {}
        self.requests = {}
        for entry in entries:
            task_sha = sha(entry["task_path"])
            verify_scene_receipt(entry["qualification"], teacher_sha256, task_sha)
            self.entries.setdefault(task_sha, []).append(entry)
            task = json.loads(Path(entry["task_path"]).read_text())
            if task.get("schema") == "bfm_known_map_navigation_task_v1":
                from gear_sonic.research.scene_distillation.tasks import navigation_request_sha256

                self.requests.setdefault(navigation_request_sha256(task), []).append(entry)

    def select(self, task_path):
        candidates = self.entries.get(sha(task_path), [])
        if not candidates:
            raise ValueError("No physically qualified continuation for this scene and goal")
        # Fixed choice per episode prevents averaging or switching incompatible routes.
        return sorted(candidates, key=lambda item: item["continuation_id"])[0]

    def select_request(self, task):
        """Choose among physically qualified continuations for the same public request.

        The caller must load the returned task's bound native reference before
        constructing the runtime; reference switching during an episode is unsupported.
        """
        from gear_sonic.research.scene_distillation.tasks import navigation_request_sha256

        candidates = self.requests.get(navigation_request_sha256(task), [])
        if not candidates:
            raise ValueError(
                "No physically qualified continuation for the public navigation request"
            )
        return sorted(candidates, key=lambda item: item["continuation_id"])[0]
