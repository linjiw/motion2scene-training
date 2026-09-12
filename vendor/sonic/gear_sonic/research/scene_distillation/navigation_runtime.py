"""Native evaluation-entry callback for bounded scene DAgger and residual PPO stages."""

import json
import os
from pathlib import Path
import sys

import torch

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.commands import MaskedMotionFoundation
from gear_sonic.research.scene_distillation.navigation import FrozenFoundationNavigator
from gear_sonic.research.scene_distillation.online import (
    NativeNavigationRuntime,
    collect_navigation_episode,
)
from gear_sonic.research.scene_distillation.residual_rl import (
    BoundedResidualActorCritic,
    train_live_residual,
)
from gear_sonic.research.scene_distillation.scene_teacher import QualifiedSceneRegistry
from gear_sonic.research.scene_distillation.train import make_decoder


def build_navigator(config, device):
    if sha(config["teacher_checkpoint"]) != config["teacher_sha256"]:
        raise ValueError("Teacher changed")
    if sha(config["foundation_checkpoint"]) != config["foundation_sha256"]:
        raise ValueError("Foundation changed")
    saved = torch.load(config["foundation_checkpoint"], map_location=device, weights_only=False)
    if saved["stage"] != "foundation" or saved["teacher_sha256"] != config["teacher_sha256"]:
        raise ValueError("Foundation provenance mismatch")
    foundation = MaskedMotionFoundation().to(device)
    foundation.load_state_dict(saved["model"], strict=True)
    decoder = make_decoder(config["teacher_checkpoint"], device)
    model = FrozenFoundationNavigator(
        foundation,
        decoder,
        command_lower=config["command_lower"],
        command_upper=config["command_upper"],
        action_residual_limit=config["action_residual_limit"],
    ).to(device)
    if config.get("navigator_checkpoint"):
        if sha(config["navigator_checkpoint"]) != config["navigator_sha256"]:
            raise ValueError("Navigator changed")
        saved = torch.load(config["navigator_checkpoint"], map_location=device, weights_only=False)
        if (
            saved["stage"] != "navigation"
            or saved["teacher_sha256"] != config["teacher_sha256"]
            or saved["config"]["foundation_sha256"] != config["foundation_sha256"]
        ):
            raise ValueError("Navigator provenance mismatch")
        model.load_state_dict(saved["model"], strict=True)
    return model.eval()


def require_public_prior_qualification(binding, foundation_sha256):
    if sha(binding["path"]) != binding["sha256"]:
        raise ValueError("Foundation qualification changed")
    receipt = json.loads(Path(binding["path"]).read_text())
    if (
        receipt.get("state") != "complete"
        or receipt.get("foundation_sha256") != foundation_sha256
        or receipt.get("public_prior_only") is not True
        or receipt.get("navigation_commands_qualified") is not True
    ):
        raise ValueError("Foundation has not passed public-prior command qualification")
    evidence = receipt.get("evidence", [])
    if not evidence or any(sha(x["path"]) != x["sha256"] for x in evidence):
        raise ValueError("Missing/changed physical foundation qualification evidence")


class NavigationStageCallback:
    """Use with eval_callbacks=[im_eval] and the im_eval target override.

    Configuration explicitly binds task, scene qualification, foundation and teacher.
    It cannot turn a tracking-only qualification into scene expert evidence.
    """

    def __init__(self, stage_config, **kwargs):
        self.config = json.loads(Path(stage_config).read_text())

    def on_step_end(self, args, state, control, **kwargs):
        c = self.config
        if c["stage"] not in ("collect_navigation", "residual_ppo"):
            raise ValueError("Unknown native scene training stage")
        registry = QualifiedSceneRegistry(c["scene_registry"], c["teacher_sha256"])
        continuation = registry.select(c["task_path"])
        if c["stage"] == "residual_ppo" or c["teacher_probability"] < 1:
            require_public_prior_qualification(
                c["foundation_qualification"], c["foundation_sha256"]
            )
        env = kwargs["env"]
        model = build_navigator(c, env.device)
        runtime = NativeNavigationRuntime(
            env,
            kwargs["model"].policy,
            c["teacher_sha256"],
            c["task_path"],
            continuation["qualification"],
        )
        try:
            if c["stage"] == "collect_navigation":
                collect_navigation_episode(
                    runtime,
                    model,
                    c["output"],
                    max_steps=c["max_steps"],
                    teacher_probability=c["teacher_probability"],
                    seed=c["seed"],
                )
            else:
                policy = BoundedResidualActorCritic(c["action_residual_limit"]).to(env.device)
                if model.action_residual_head is not None:
                    policy.actor.load_state_dict(model.action_residual_head.state_dict())
                train_live_residual(
                    runtime,
                    model,
                    policy,
                    c["output"],
                    iterations=c["iterations"],
                    horizon=c["horizon"],
                    learning_rate=c["learning_rate"],
                    reward_profile=c["reward_profile"],
                )
        finally:
            runtime.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
