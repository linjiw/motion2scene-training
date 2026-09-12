"""Single-environment native DAgger collector for already-qualified scene tasks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.contracts import public_observation_sha256
from gear_sonic.research.scene_distillation.observations import navigation_observation
from gear_sonic.research.scene_distillation.scene_teacher import verify_scene_receipt


def state_digest(env):
    robot = env.env.scene["robot"]
    digest = hashlib.sha256()
    for x in (robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel):
        digest.update(x.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class NativeNavigationRuntime:
    """Adapter around an initialized native SONIC environment and frozen teacher.

    Caller constructs the environment with the bound USD and reference motion.
    No simulator reset occurs between observing, querying, and acting.
    """

    def __init__(self, env, teacher, teacher_sha256, task, qualification):
        if env.num_envs != 1:
            raise ValueError("Scene USD runtime is explicitly single-environment")
        self.task_path = Path(task)
        self.task = json.loads(self.task_path.read_text())
        self.reference = None
        if self.task.get("schema") == "bfm_known_map_navigation_task_v1":
            from gear_sonic.research.scene_distillation.tasks import validate_task

            validate_task(self.task)
            with np.load(self.task["reference"]["path"], allow_pickle=False) as data:
                self.reference = (data["time_s"].copy(), data["qpos"][:, :3].copy())
        if self.task.get("split") != "train":
            raise ValueError("Scene collection requires an explicitly training-split task")
        self.qualification = verify_scene_receipt(qualification, teacher_sha256, sha(task))
        self.env = env
        self.teacher = teacher.eval()
        self.teacher_sha256 = teacher_sha256
        self.binding = qualification
        scene_path = Path(env.config.scene_usd_path)
        if sha(scene_path) != self.task["scene_usd_sha256"]:
            raise ValueError("Loaded collision scene does not match qualified task")
        self.observation = None
        self.tick = 0
        self.teacher.eval_mode()
        if self.teacher.running_mean_std is not None:
            raise ValueError("This scene adapter requires native unnormalized actor history")
        self._original_compute = env.env.termination_manager.compute
        self._measurement = None
        self._arrival_ticks = 0

        def compute(*args, **kwargs):
            result = self._original_compute(*args, **kwargs)
            robot = env.env.scene["robot"]
            force = env.env.scene["contact_forces"].data.net_forces_w
            names = env.env.scene["contact_forces"].body_names
            nonfoot = [
                i
                for i, name in enumerate(names)
                if name not in ("left_ankle_roll_link", "right_ankle_roll_link")
            ]
            goal = torch.as_tensor(self.task["goal_xyz"], device=env.device)
            self._measurement = {
                "distance": torch.linalg.vector_norm(
                    env.motion_command.robot_anchor_pos_w - goal, dim=-1
                ),
                "speed": torch.linalg.vector_norm(
                    robot.data.body_lin_vel_w[:, env.motion_command.robot_anchor_body_index], dim=-1
                ),
                "nonfoot_force": torch.linalg.vector_norm(force[:, nonfoot], dim=-1).amax(-1),
            }
            self._measurement = {k: v.detach().clone() for k, v in self._measurement.items()}
            return result

        env.env.termination_manager.compute = compute

    def close(self):
        self.env.env.termination_manager.compute = self._original_compute

    def reset(self):
        self.env.set_is_evaluating(True)
        self.observation = self.env.reset_all()
        self.teacher.init_rollout()
        self.tick = 0
        self._arrival_ticks = 0
        keys = list(self.env._motion_lib.curr_motion_keys)
        from gear_sonic.research.scene_distillation.tasks import verify_native_motion

        verify_native_motion(self.task, self.env._motion_lib)
        if keys != ["hindsight_" + self.task["motion_id"]]:
            raise ValueError("Loaded reference does not match qualified continuation")
        return self.public_observation()

    def public_observation(self):
        command = self.env.motion_command
        o = navigation_observation(
            proprio=self.observation["actor_obs"][0].detach().cpu().numpy(),
            root_xyz=command.robot_anchor_pos_w[0].detach().cpu().numpy(),
            root_wxyz=command.robot_anchor_quat_w[0].detach().cpu().numpy(),
            start_xyz=self.task["start_xyz"],
            goal_xyz=self.task["goal_xyz"],
            obstacles=self.task["obstacles"],
        )
        return {k: torch.as_tensor(v, device=self.env.device)[None] for k, v in o.items()}

    @torch.no_grad()
    def query(self, *, nominal_teacher_state=False):
        before = state_digest(self.env)
        action = self.teacher.act_inference(obs_dict=self.observation, skip_episode_attnmask=True)
        if state_digest(self.env) != before:
            raise ValueError("Teacher query changed the simulator state")
        # A scene-qualified nominal rollout is not an arbitrary recovery expert.
        # Only a separately qualified recovery envelope can authorize off-reference queries.
        command = self.env.motion_command
        error = float(torch.linalg.vector_norm(command.anchor_pos_w - command.robot_anchor_pos_w))
        envelope = self.qualification.get("query_support", {})
        threshold = envelope.get("max_anchor_error_m")
        eligible = nominal_teacher_state or (
            envelope.get("recovery_queries_qualified") is True
            and threshold is not None
            and error <= threshold
        )
        return (
            action,
            eligible,
            {
                "tick": self.tick,
                "state_sha256": before,
                "anchor_error_m": error,
                "qualification_sha256": self.binding["sha256"],
                "observation_sha256": public_observation_sha256(
                    {k: v[0].detach().cpu().numpy() for k, v in self.public_observation().items()}
                ),
                "task_sha256": sha(self.task_path),
                "teacher_sha256": self.teacher_sha256,
                "qualified": bool(eligible),
            },
        )

    def training_labels(self):
        """Privileged reference supervision is recorded separately from actor inputs."""
        if self.reference is None:
            return {}
        from gear_sonic.research.scene_distillation.collect import native_commands
        from gear_sonic.research.scene_distillation.tasks import trajectory_labels

        command = self.env.motion_command
        phase = float((command.time_steps + command.motion_start_time_steps)[0]) * 0.02
        labels = trajectory_labels(
            *self.reference,
            phase,
            command.robot_anchor_pos_w[0].detach().cpu().numpy(),
            command.robot_anchor_quat_w[0].detach().cpu().numpy(),
        )
        controls, mask = native_commands(command, self.env.env.scene.env_origins)
        labels.update(
            label_controls=controls[0].detach().cpu().numpy(),
            label_control_mask=mask[0].detach().cpu().numpy(),
            label_reference_time_s=np.asarray(phase, dtype=np.float32),
        )
        return labels

    def step(self, action):
        goal = torch.as_tensor(self.task["goal_xyz"], device=self.env.device)
        before = torch.linalg.vector_norm(
            self.env.motion_command.robot_anchor_pos_w - goal, dim=-1
        ).clone()
        self._measurement = None
        self.observation, reward, done, extras = self.env.step({"actions": action})
        self.tick += 1
        if self.task.get("reward_profile") == "known_goal_progress_v1":
            if self._measurement is None:
                raise ValueError("No pre-reset physical navigation measurement")
            m = self._measurement
            arrival = bool((m["distance"] <= 0.25).all() and (m["speed"] <= 0.1).all())
            self._arrival_ticks = self._arrival_ticks + 1 if arrival else 0
            stabilized = self._arrival_ticks >= 50
            # Hypothesis reward at 50 Hz; net nonfoot force includes self-contact.
            # It does not replace 200 Hz contact-qualified evaluation.
            reward = (
                10 * (before - m["distance"])
                - 0.01
                - 0.1 * (m["nonfoot_force"] > 10).float()
                - 5 * (done.reshape(-1).bool() & ~extras["time_outs"].reshape(-1).bool()).float()
                + 10 * float(stabilized)
            )
            extras["navigation_measurement"] = m
            extras["goal_stabilized"] = stabilized
            if stabilized:
                done = torch.ones_like(done)
        if self.task.get("deadline_ticks") and self.tick >= self.task["deadline_ticks"]:
            done = torch.ones_like(done)
            extras["navigation_deadline"] = True
        return self.public_observation(), reward, done, extras


def collect_navigation_episode(runtime, navigator, output, *, max_steps, teacher_probability, seed):
    """Public-prior student rollout, same-state queries, retained interventions/failures.

    The frozen qualified reference is fixed throughout this episode. This function
    cannot manufacture changed-scene/goal supervision or certify its own queries.
    """
    if not 1 <= max_steps <= 3000 or not 0 <= teacher_probability <= 1:
        raise ValueError("Invalid DAgger collection budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    observation = runtime.reset()
    hidden = None
    rows = []
    queries = []
    status = "censored"
    done = False
    nominal_teacher_state = True
    completed_steps = 0
    try:
        for tick in range(max_steps):
            with torch.no_grad():
                teacher, qualified, query = runtime.query(
                    nominal_teacher_state=nominal_teacher_state
                )
                result = navigator.forward_step(observation, hidden)
                hidden = result["hidden"]
                intervention = bool(rng.random() < teacher_probability)
                action = teacher if intervention else result["actions"]
                nominal_teacher_state = nominal_teacher_state and intervention
            row = {k: v[0].detach().cpu().numpy().copy() for k, v in observation.items()}
            if hasattr(runtime, "training_labels"):
                row.update(runtime.training_labels())
            row.update(
                teacher_actions=(
                    teacher[0].detach().cpu().numpy()
                    if qualified
                    else np.full(29, np.nan, dtype=np.float32)
                ),
                qualified_mask=np.asarray(qualified),
                intervention=np.asarray(intervention),
            )
            rows.append(row)
            queries.append(query)
            observation, reward, dones, extras = runtime.step(action)
            completed_steps += 1
            rows[-1]["reward"] = np.asarray(float(reward.reshape(-1)[0]), dtype=np.float32)
            done = bool(dones.any())
            if done:
                status = "episode_ended"
                break
        path = output / "episode.npz"
        np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
        write_new(output / "queries.json", queries)
        write_new(
            output / "collection.json",
            {
                "schema": "bfm_executed_navigation_v1",
                "teacher_sha256": runtime.teacher_sha256,
                "episodes": [
                    {
                        "motion_id": runtime.task["motion_id"],
                        "split": "train",
                        "path": str(path),
                        "sha256": sha(path),
                        "rows": len(rows),
                        "eligible": any(bool(r["qualified_mask"]) for r in rows),
                        "qualification": runtime.binding,
                        "task_sha256": sha(runtime.task_path),
                        "task_path": str(runtime.task_path),
                        "queries": {
                            "path": str(output / "queries.json"),
                            "sha256": sha(output / "queries.json"),
                        },
                    }
                ],
            },
        )
    except Exception:
        status = "failed"
        raise
    finally:
        if status == "failed":
            write_new(output / "partial-queries.json", queries)
            if rows:
                shared = set.intersection(*(set(row) for row in rows))
                np.savez_compressed(
                    output / "partial-episode.npz",
                    **{k: np.stack([row[k] for row in rows]) for k in shared},
                )
        write_new(
            output / "receipt.json",
            {
                "state": status,
                "environment_transitions": completed_steps,
                "attempted_decisions": len(rows),
                "native_completed_step_returns": runtime.tick,
                "qualified_queries": sum(bool(r["qualified_mask"]) for r in rows),
                "seed": seed,
                "teacher_probability": teacher_probability,
                "navigation_success": None,
                "episode_done": done,
                "queries_are_same_state": True,
            },
        )
