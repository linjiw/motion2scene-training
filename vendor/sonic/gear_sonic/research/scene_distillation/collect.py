"""Native same-state teacher collection for the bounded motion-foundation pilot."""

import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.research.hindsight_training.qualify import TrackingQualificationCallback
from gear_sonic.research.hindsight_training.runtime import sha, write_new


def native_commands(command, env_origins):
    """Instantaneous target in measured anchor frame; future trajectory is posterior-only."""
    from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

    batch = command.num_envs
    q = command.robot_anchor_quat_w
    target = command.anchor_quat_w
    relative = quat_mul(quat_inv(q), target)
    w, x, y, z = relative.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    controls = q.new_zeros(batch, 79)
    controls[:, 0] = yaw.sin()
    controls[:, 1] = yaw.cos()
    controls[:, 2:5] = quat_apply_inverse(q, command.anchor_lin_vel_w)
    controls[:, 5] = command.anchor_pos_w[:, 2] - env_origins[:, 2]
    controls[:, 6] = quat_apply_inverse(q, command.anchor_ang_vel_w)[:, 2]
    delta = command.body_pos_w - command.robot_anchor_pos_w[:, None]
    controls[:, 8:50] = quat_apply_inverse(
        q[:, None].expand(-1, 14, -1).reshape(-1, 4), delta.reshape(-1, 3)
    ).reshape(batch, 42)
    controls[:, 50:] = command.joint_pos
    available = torch.ones_like(controls, dtype=torch.bool)
    available[:, 7] = False  # Arrival is not supplied by a tracking reference.
    return controls, available


class FoundationCollectionCallback(TrackingQualificationCallback):
    """Preserve all attempts; export eligible executed teacher episodes after evaluation."""

    def __init__(self, *args, collection_lock, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_lock = Path(collection_lock)
        self.lock = json.loads(self.collection_lock.read_text())
        if sha(self.lock["teacher_checkpoint"]) != self.lock["teacher_sha256"]:
            raise ValueError("Teacher changed before collection")
        self.rows = []
        self.captured = {}
        self.handles = []

    def _pre_evaluate_policy(self, reset_env=True):
        if self.lock.get("precision") == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        super()._pre_evaluate_policy(reset_env)
        if self.num_total_env_eval_loops != 1:
            raise ValueError("Pilot collector requires one complete resident motion batch")
        module = self.model.policy.actor_module
        for name, layer in (
            ("future_reference", module.encoders["g1"].module[0]),
            ("decoder_input", module.decoders["g1_dyn"].module[0]),
        ):

            def capture(_module, inputs, name=name):
                self.captured[name] = inputs[0].detach().reshape(self.env.num_envs, -1).clone()

            self.handles.append(layer.register_forward_pre_hook(capture))

    def _pre_eval_env_step(self, actor_state):
        if len(self.rows) >= self.lock["max_control_steps"]:
            raise RuntimeError("Collection control-step ceiling reached; preserve failed attempt")
        self.captured.clear()
        actor_state = super()._pre_eval_env_step(actor_state)
        inputs = self.captured["decoder_input"]
        controls, available = native_commands(
            self.env.motion_command, self.env.env.scene.env_origins
        )
        values = {
            "proprio": inputs[:, 64:],
            "teacher_tokens": inputs[:, :64],
            "teacher_actions": actor_state["actions"],
            "privileged_state": actor_state["obs"]["critic_obs"],
            "future_reference": self.captured["future_reference"],
            "controls": controls,
            "control_mask": available,
        }
        widths = {
            "proprio": 930,
            "teacher_tokens": 64,
            "teacher_actions": 29,
            "privileged_state": 1645,
            "future_reference": 640,
            "controls": 79,
            "control_mask": 79,
        }
        for key, value in values.items():
            if value.shape != (self.env.num_envs, widths[key]) or not torch.isfinite(value).all():
                raise ValueError(f"Native collection shape/finite failure: {key}")
        self.rows.append(
            {key: value.detach().cpu().numpy().copy() for key, value in values.items()}
        )
        return actor_state

    def _post_evaluate_policy(self, eval_res):
        result = super()._post_evaluate_policy(eval_res)
        output = Path(self.output_dir)
        arrays = {key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]}
        metrics = eval_res["all_metrics_dict"]
        episodes = []
        for i, key in enumerate(metrics["motion_keys"]):
            motion_id = key.split("_")[-1]
            if motion_id not in self.lock["train_ids"]:
                raise ValueError("Unknown/development motion in foundation collection")
            ref, pred = self.gt_pos_all[i], self.pred_pos_all[i]
            count = len(ref)
            root_max = float(np.linalg.norm(pred[:, 0, :2] - ref[:, 0, :2], axis=-1).max())
            body_mean = float(np.linalg.norm(pred - ref, axis=-1).mean())
            eligible = (
                not bool(metrics["terminated"][i])
                and root_max <= self.lock["max_root_xy_m"]
                and body_mean <= self.lock["max_body_mean_m"]
            )
            path = output / f"episode-{motion_id}.npz"
            np.savez_compressed(path, **{name: a[:count, i] for name, a in arrays.items()})
            episodes.append(
                {
                    "motion_id": motion_id,
                    "split": "train",
                    "path": str(path),
                    "sha256": sha(path),
                    "rows": count,
                    "eligible": eligible,
                    "terminated": bool(metrics["terminated"][i]),
                    "max_root_xy_m": root_max,
                    "mean_body_error_m": body_mean,
                }
            )
        for handle in self.handles:
            handle.remove()
        write_new(
            output / "collection.json",
            {
                "schema": "bfm_executed_foundation_v1",
                "teacher_sha256": self.lock["teacher_sha256"],
                "lock_sha256": sha(self.collection_lock),
                "episodes": episodes,
                "control_steps": len(self.rows),
                "environment_transitions": len(self.rows) * self.env.num_envs,
                "dt": 0.02,
                "scene_labels": False,
                "source": "executed_native_teacher",
            },
        )
        return result


def supported_prefix(mask, valid_steps, minimum_rows):
    """Censor at the first unsupported decision or native termination, never re-enter."""
    mask = np.asarray(mask, dtype=bool).copy()
    mask[max(0, valid_steps) :] = False
    mask = np.logical_and.accumulate(mask)
    if mask.sum() < minimum_rows:
        mask[:] = False
    return mask


def teacher_support_mask(mask, valid_steps, minimum_rows, whole_motion_eligible):
    """Keep a qualified whole demonstration; use the stricter prefix screen for other attempts."""
    if not whole_motion_eligible:
        return supported_prefix(mask, valid_steps, minimum_rows)
    result = np.arange(len(mask)) < valid_steps
    if result.sum() < minimum_rows:
        result[:] = False
    return result


class PrefixFoundationCollectionCallback(FoundationCollectionCallback):
    """Executed teacher prefixes with local support, distinct from whole-motion success."""

    def _pre_eval_env_step(self, actor_state):
        actor_state = super()._pre_eval_env_step(actor_state)
        command = self.env.motion_command
        root = torch.linalg.vector_norm(
            (command.anchor_pos_w - command.robot_anchor_pos_w)[:, :2], dim=-1
        )
        body = torch.linalg.vector_norm(command.body_pos_w - command.robot_body_pos_w, dim=-1).mean(
            -1
        )
        self.rows[-1]["query_mask"] = (
            ((root <= self.lock["max_root_xy_m"]) & (body <= self.lock["max_body_mean_m"]))
            .cpu()
            .numpy()
        )
        return actor_state

    def _post_evaluate_policy(self, eval_res):
        result = super()._post_evaluate_policy(eval_res)
        path = Path(self.output_dir) / "collection.json"
        manifest = json.loads(path.read_text())
        metrics = eval_res["all_metrics_dict"]
        expected = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids).cpu().tolist()
        for i, episode in enumerate(manifest["episodes"]):
            shard = Path(episode["path"])
            with np.load(shard) as data:
                arrays = {k: data[k].copy() for k in data.files}
            valid = int(round(metrics["progress"][i] * expected[i]))
            arrays["query_mask"] = teacher_support_mask(
                arrays["query_mask"], valid, self.lock["minimum_prefix_rows"], episode["eligible"]
            )
            np.savez_compressed(shard, **arrays)
            episode.update(
                whole_motion_eligible=episode["eligible"],
                source=(
                    "executed_native_teacher"
                    if episode["eligible"]
                    else "executed_native_teacher_prefix"
                ),
                eligible=bool(arrays["query_mask"].any()),
                supported_query_rows=int(arrays["query_mask"].sum()),
                sha256=sha(shard),
            )
        manifest.update(
            source="executed_native_teacher_prefix",
            query_qualification="contiguous_local_tracking_support_not_whole_motion_qualification",
            precision=self.lock.get("precision", "native_default"),
        )
        path.write_text(json.dumps(manifest, indent=2))
        return result


class DaggerFoundationCollectionCallback(FoundationCollectionCallback):
    """Exploratory same-state teacher queries under a public-prior student driver.

    A numerical support screen is not a physical recovery qualification. The
    manifest labels this distinction and fitting requires an explicit experiment flag.
    """

    def _pre_evaluate_policy(self, reset_env=True):
        super()._pre_evaluate_policy(reset_env)
        from gear_sonic.research.hindsight_training.student import FrozenSonicDecoder
        from gear_sonic.research.scene_distillation.commands import PROFILES, build_foundation

        path = self.lock["student_checkpoint"]
        if sha(path) != self.lock["student_sha256"]:
            raise ValueError("Student driver checkpoint changed")
        saved = torch.load(path, map_location=self.env.device, weights_only=False)
        if saved["teacher_sha256"] != self.lock["teacher_sha256"] or saved["stage"] != "foundation":
            raise ValueError("Student driver has a different teacher")
        self.driver = build_foundation(saved["config"]).to(self.env.device).eval()
        self.driver.load_state_dict(saved["model"], strict=True)
        self.driver_decoder = (
            FrozenSonicDecoder(self.model.policy.state_dict()).to(self.env.device).eval()
        )
        self.profile = PROFILES[self.lock["command_profile"]]
        self.teacher_probability = self.lock.get("teacher_probability", 0.0)
        if not 0 <= self.teacher_probability <= 1:
            raise ValueError("DAgger teacher intervention probability must be in [0, 1]")
        self.intervention_rng = torch.Generator(device=self.env.device)
        self.intervention_rng.manual_seed(self.lock.get("intervention_seed", 91220))

    def _pre_eval_env_step(self, actor_state):
        actor_state = super()._pre_eval_env_step(actor_state)
        row = self.rows[-1]
        command = self.env.motion_command
        root = torch.linalg.vector_norm(
            (command.anchor_pos_w - command.robot_anchor_pos_w)[:, :2], dim=-1
        )
        body = torch.linalg.vector_norm(command.body_pos_w - command.robot_body_pos_w, dim=-1).mean(
            -1
        )
        row["query_mask"] = (
            ((root <= self.lock["max_root_xy_m"]) & (body <= self.lock["max_body_mean_m"]))
            .cpu()
            .numpy()
        )
        controls, mask = native_commands(command, self.env.env.scene.env_origins)
        selected = torch.zeros_like(mask)
        selected[:, list(self.profile)] = True
        proprio = self.captured["decoder_input"][:, 64:]
        result = self.driver.prior_step(proprio, controls, mask & selected)
        student_actions = self.driver_decoder(result["tokens"], proprio)
        intervention = (
            torch.rand(self.env.num_envs, device=self.env.device, generator=self.intervention_rng)
            < self.teacher_probability
        )
        row["teacher_intervention"] = intervention.cpu().numpy()
        actor_state["actions"] = torch.where(
            intervention[:, None], actor_state["actions"], student_actions
        )
        return actor_state

    def _post_evaluate_policy(self, eval_res):
        result = super()._post_evaluate_policy(eval_res)
        path = Path(self.output_dir) / "collection.json"
        manifest = json.loads(path.read_text())
        metrics = eval_res["all_metrics_dict"]
        expected = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids).cpu().tolist()
        for i, episode in enumerate(manifest["episodes"]):
            shard = Path(episode["path"])
            with np.load(shard) as f:
                arrays = {k: f[k].copy() for k in f.files}
            valid = min(len(arrays["query_mask"]), int(round(metrics["progress"][i] * expected[i])))
            arrays["query_mask"][valid:] = False
            np.savez_compressed(shard, **arrays)
            episode.update(
                sha256=sha(shard),
                eligible=bool(arrays["query_mask"].sum() >= 2),
                supported_query_rows=int(arrays["query_mask"].sum()),
            )
        manifest.update(
            source="queried_native_teacher_on_student_state",
            query_qualification="numerical_support_screen_only_not_recovery_qualified",
            student_checkpoint_sha256=self.lock["student_sha256"],
            teacher_probability=self.teacher_probability,
            teacher_interventions=int(sum(r["teacher_intervention"].sum() for r in self.rows)),
        )
        path.write_text(json.dumps(manifest, indent=2))
        return result
