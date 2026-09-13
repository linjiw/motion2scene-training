"""Native tracking evaluation and exploratory DAgger for direct-action students."""

import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.action_flow import CONTEXT_SHAPES, ActionStudent
from gear_sonic.research.scene_distillation.collect import (
    PrefixFoundationCollectionCallback,
    native_commands,
)
from gear_sonic.research.scene_distillation.direct_context import task_context
from gear_sonic.research.scene_distillation.evaluate_commands import CommandEvaluationCallback
from gear_sonic.research.scene_distillation.train_action_flow import profile_mask


def load_action_student(path, teacher_sha256, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["stage"] == "context_token":
        from gear_sonic.research.scene_distillation.context_token import ContextTokenPolicy

        if saved["teacher_sha256"] != teacher_sha256:
            raise ValueError("Context BFM teacher mismatch")
        policy = ContextTokenPolicy(saved["config"], device)
        policy.foundation.load_state_dict(saved["model"], strict=True)
        return policy.eval()
    if saved["stage"] != "action_distillation" or saved["teacher_sha256"] != teacher_sha256:
        raise ValueError("Action checkpoint/teacher mismatch")
    if sha(saved["config"]["teacher_checkpoint"]) != teacher_sha256:
        raise ValueError("Teacher changed")
    model = ActionStudent(**saved["config"]["model"]).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    return model


def load_context_catalog(path, env):
    catalog = json.loads(Path(path).read_text())
    if catalog.get("schema") != "direct_context_catalog_v1":
        raise ValueError("Unknown context catalog")
    tasks = catalog["tasks"]
    if env.config.terrain_type == "plane":
        if catalog["terrain"] != "plane" or any(t["obstacles"] for t in tasks.values()):
            raise ValueError("Plane context cannot claim obstacle execution")
    else:
        if env.num_envs != 1 or len(tasks) != 1:
            raise ValueError("Exact scene catalog requires one environment and task")
        task = next(iter(tasks.values()))
        if sha(env.config.scene_usd_path) != task["scene_usd_sha256"]:
            raise ValueError("Context scene differs from loaded collision asset")
    keys = list(env._motion_lib.curr_motion_keys)
    for key in keys:
        task = tasks[key[-5:]]
        motion = Path(env._motion_lib.m_cfg.motion_file) / (key + ".pkl")
        if sha(motion) != task["native_motion"]["sha256"]:
            raise ValueError("Context/reference pairing changed")
        if task.get("causal_pair_role") == "changed_goal_requires_new_expert_continuation":
            raise ValueError("Changed goal has no matching expert action labels")
    return tasks


def native_context(env, tasks):
    keys = list(env._motion_lib.curr_motion_keys)
    command = env.motion_command
    root = (command.robot_anchor_pos_w - env.env.scene.env_origins).detach().cpu().numpy()
    quat = command.robot_anchor_quat_w.detach().cpu().numpy()
    rows = [task_context(tasks[key[-5:]], root[i], quat[i]) for i, key in enumerate(keys)]
    return {
        k: torch.as_tensor(np.stack([r[k] for r in rows]), device=env.device)
        for k in CONTEXT_SHAPES
    }


class ContextPrefixCollectionCallback(PrefixFoundationCollectionCallback):
    """Executed, locally supported teacher rows plus actual known-map task labels.

    These are exploratory paired labels, not a replacement for scene qualification.
    The catalog must bind the actually loaded motion and collision scene.
    """

    def _pre_evaluate_policy(self, reset_env=True):
        super()._pre_evaluate_policy(reset_env)
        path = self.lock["context_catalog"]
        if sha(path) != self.lock["context_catalog_sha256"]:
            raise ValueError("Context catalog changed")
        self.context_tasks = load_context_catalog(path, self.env)

    def _pre_eval_env_step(self, actor_state):
        actor_state = super()._pre_eval_env_step(actor_state)
        context = native_context(self.env, self.context_tasks)
        self.rows[-1].update({k: v.detach().cpu().numpy().copy() for k, v in context.items()})
        command = self.env.motion_command
        self.rows[-1]["measured_root_xyz"] = (
            (command.robot_anchor_pos_w - self.env.env.scene.env_origins)
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        self.rows[-1]["measured_root_wxyz"] = (
            command.robot_anchor_quat_w.detach().cpu().numpy().copy()
        )
        return actor_state

    def _post_evaluate_policy(self, eval_res):
        result = super()._post_evaluate_policy(eval_res)
        path = Path(self.output_dir) / "collection.json"
        manifest = json.loads(path.read_text())
        manifest.update(
            context_schema="complete_known_map_goal_v1",
            scene_labels=True,
            scene_task_qualified=False,
            context_catalog={
                "path": self.lock["context_catalog"],
                "sha256": self.lock["context_catalog_sha256"],
            },
        )
        manifest["parents"] = [manifest["context_catalog"]]
        path.write_text(json.dumps(manifest, indent=2))
        return result


class ActionDaggerCollectionCallback(ContextPrefixCollectionCallback):
    def _pre_evaluate_policy(self, reset_env=True):
        super()._pre_evaluate_policy(reset_env)
        if sha(self.lock["student_checkpoint"]) != self.lock["student_sha256"]:
            raise ValueError("Student changed")
        self.driver = load_action_student(
            self.lock["student_checkpoint"], self.lock["teacher_sha256"], self.env.device
        )
        self.intervention_rng = torch.Generator(device=self.env.device).manual_seed(
            self.lock["intervention_seed"]
        )
        self.teacher_probability = self.lock["teacher_probability"]
        if not 0 <= self.teacher_probability <= 1:
            raise ValueError("Invalid intervention probability")
        self.driver_noise = torch.zeros(self.env.num_envs, 29, device=self.env.device)
        if self.lock.get("noise_mode", "zero") == "episode":
            self.driver_noise.normal_(generator=self.intervention_rng)
        elif self.lock.get("noise_mode", "zero") != "zero":
            raise ValueError("Invalid flow noise mode")

    def _pre_eval_env_step(self, actor_state):
        actor_state = super()._pre_eval_env_step(actor_state)
        row = self.rows[-1]
        batch = {
            k: torch.as_tensor(row[k], device=self.env.device)
            for k in ["proprio", "controls", "control_mask"]
        }
        context = None
        if self.driver.condition.context_enabled:
            context = {k: torch.as_tensor(row[k], device=self.env.device) for k in CONTEXT_SHAPES}
        mask = profile_mask(batch["control_mask"], self.lock["command_profile"])
        with torch.no_grad():
            student = self.driver.actions(
                batch["proprio"], batch["controls"], mask, context=context, noise=self.driver_noise
            )["actions"]
        intervention = (
            torch.rand(self.env.num_envs, device=self.env.device, generator=self.intervention_rng)
            < self.teacher_probability
        )
        row["teacher_intervention"] = intervention.cpu().numpy()
        actor_state["actions"] = torch.where(intervention[:, None], actor_state["actions"], student)
        return actor_state

    def _post_evaluate_policy(self, eval_res):
        result = super()._post_evaluate_policy(eval_res)
        path = Path(self.output_dir) / "collection.json"
        manifest = json.loads(path.read_text())
        for e in manifest["episodes"]:
            e["source"] = "queried_native_teacher_on_student_state"
            e["whole_motion_eligible"] = False
        manifest.update(
            source="queried_native_teacher_on_student_state",
            query_qualification="contiguous_local_support_not_recovery_qualified",
            teacher_probability=self.teacher_probability,
            student_checkpoint_sha256=self.lock["student_sha256"],
            teacher_interventions=sum(int(r["teacher_intervention"].sum()) for r in self.rows),
        )
        path.write_text(json.dumps(manifest, indent=2))
        return result


class ActionEvaluationCallback(CommandEvaluationCallback):
    def __init__(
        self,
        *args,
        student_checkpoint,
        teacher_sha256,
        command_profile,
        context_catalog=None,
        noise_mode="zero",
        flow_steps=None,
        residual_enabled=True,
        **kwargs
    ):
        super().__init__(
            *args,
            student_checkpoint=student_checkpoint,
            teacher_sha256=teacher_sha256,
            command_profile="full",
            **kwargs
        )
        self.command_profile = command_profile
        self.context_catalog = context_catalog
        self.noise_mode, self.flow_steps = noise_mode, flow_steps
        self.residual_enabled = residual_enabled

    def _get_inference_policy(self, device=None):
        device = self.env.device if device is None else device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.student = load_action_student(self.student_checkpoint, self.teacher_sha256, device)
        tasks = (
            load_context_catalog(self.context_catalog, self.env)
            if self.student.condition.context_enabled
            else None
        )
        rng = torch.Generator(device=device).manual_seed(91341)
        noise = torch.zeros(self.env.num_envs, 29, device=device)
        if self.noise_mode == "episode":
            noise.normal_(generator=rng)
        elif self.noise_mode != "zero":
            raise ValueError("Invalid inference noise mode")

        def policy(obs_dict, **kwargs):
            controls, available = native_commands(
                self.env.motion_command, self.env.env.scene.env_origins
            )
            proprio = obs_dict["actor_obs"]
            if self.model.policy.running_mean_std is not None:
                proprio = self.model.policy.running_mean_std(proprio)
            context = native_context(self.env, tasks) if tasks else None
            return self.student.actions(
                proprio,
                controls,
                profile_mask(available, self.command_profile),
                context=context,
                noise=noise,
                steps=self.flow_steps,
                residual_enabled=self.residual_enabled,
            )["actions"]

        return policy
