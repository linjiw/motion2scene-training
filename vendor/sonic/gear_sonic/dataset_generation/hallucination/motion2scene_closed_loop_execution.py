"""Opt-in repeated sensor decisions within the qualified single-encounter phases.

This runtime retains the frozen tracker and its two same-phase references. It
does not provide arbitrary-duration options, stopping, steering or course reset.
"""

from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.utils import configclass
import numpy as np
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand

from .motion2scene_action_contract import command_permission, requested_skill
from .motion2scene_closed_loop_policy import (
    POLICY_SCHEMA,
    choose_history_skill,
    history_policy_features,
    legal_skill_mask,
    load_history_policy,
)
from .motion2scene_observation_delay import ObservationDelay
from .motion2scene_observation_history import (
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
    capture_ray_fan,
)
from .motion2scene_overhang_eval_execution import (
    EvaluationBankCommand,
    EvaluationBankEnvCfg,
    EvaluationBankRecorder,
    EvaluationBankRecorderCfg,
)


class ClosedLoopCommand(EvaluationBankCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.observation_delay = ObservationDelay(cfg.observation_delay_s)
        self.observation_history = FloorCeilingHistory(
            HistoryGrid(max_age_s=cfg.history_max_age_s, max_frames=cfg.history_max_frames)
        )
        self.history_policy = (
            load_history_policy(cfg.learned_policy_path, cfg.learned_policy_sha256)
            if cfg.closed_loop_mode == "learned"
            else None
        )
        self.closed_loop_rows = []
        self.closed_loop_feature_names = None
        self.forced_decision_issued = False

    def _update_command(self):
        from omni.physx import get_physx_scene_query_interface

        if not self._bank_verified:
            self._verify_bank()
        TrackingCommand._update_command(self)
        if not self._env.sim.cfg.enable_scene_query_support:
            raise RuntimeError("scene queries disabled")
        data = self.robot.data
        root = data.root_pos_w[0].detach().cpu().numpy()
        quat = data.root_quat_w[0].detach().cpu().numpy()
        phase_s = round(float((self.time_steps + self.motion_start_time_steps)[0]) / 50, 8)
        elapsed_s = len(self.closed_loop_rows) / 50
        measurements = capture_ray_fan(root, quat, get_physx_scene_query_interface())
        delivered = self.observation_delay.push(
            {"measurements": [asdict(ray) for ray in measurements], "phase_s": phase_s}
        )
        if delivered is not None:
            self.observation_history.push(
                [SensorRay(**ray) for ray in delivered["measurements"]],
                delivered["capture_elapsed_s"],
                delivered_time_s=elapsed_s,
            )
        observation = self.observation_history.snapshot(root, quat, elapsed_s)
        current_lib = self.motion_lib
        before_joint_reference = self.joint_pos.clone()
        before_root_reference = self.anchor_pos_w.clone()
        before_state = data.root_state_w.clone()
        before_joints = data.joint_pos.clone()
        before_clock = self.time_steps.clone()
        try:
            self.motion_lib = self._interface_libraries[1 - self.interface_active]
            joint_jump = float((self.joint_pos - before_joint_reference).abs().max())
            root_jump = float(torch.linalg.vector_norm(self.anchor_pos_w - before_root_reference))
        finally:
            self.motion_lib = current_lib
        active_before = self.interface_active
        legality = legal_skill_mask(active_before, phase_s, joint_jump, root_jump)
        state = {
            name: getattr(data, source)[0].detach().cpu().tolist()
            for name, source in (
                ("projected_gravity_b", "projected_gravity_b"),
                ("root_lin_vel_w", "root_lin_vel_w"),
                ("root_ang_vel_w", "root_ang_vel_w"),
                ("dof_pos", "joint_pos"),
                ("dof_vel", "joint_vel"),
            )
        }
        age_s = (
            elapsed_s - delivered["capture_elapsed_s"]
            if delivered is not None
            else self.cfg.observation_delay_s + elapsed_s
        )
        names, features = history_policy_features(
            observation,
            self.observation_history.grid,
            state,
            phase_s,
            active_before,
            legality,
            max(0, age_s),
        )
        if self.closed_loop_feature_names is None:
            self.closed_loop_feature_names = names
        elif self.closed_loop_feature_names != names:
            raise RuntimeError("policy feature schema changed during execution")
        if self.cfg.closed_loop_mode == "forced":
            requested, self.forced_decision_issued = requested_skill(
                self.cfg.encounter_action,
                self.cfg.forced_entry_time_s,
                phase_s,
                active_before,
                self.forced_decision_issued,
            )
            logits = None
        else:
            requested, logits = choose_history_skill(
                self.cfg.closed_loop_mode,
                names,
                features,
                legality,
                active_before,
                phase_s,
                self.history_policy,
            )
        allowed, reasons = command_permission(
            requested, active_before, phase_s, joint_jump, root_jump
        )
        attempted = requested != active_before
        if attempted and allowed:
            self.motion_lib = self._interface_libraries[requested]
            self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
            self.interface_active = requested
            self.interface_switches.append(
                {
                    "time_s": phase_s,
                    "from": active_before,
                    "to": requested,
                    "joint_reference_jump_rad": joint_jump,
                    "root_reference_jump_m": root_jump,
                    "root_state_unchanged": bool(torch.equal(before_state, data.root_state_w)),
                    "joint_state_unchanged": bool(torch.equal(before_joints, data.joint_pos)),
                    "clock_unchanged": bool(torch.equal(before_clock, self.time_steps)),
                }
            )
        row = {
            "time_s": phase_s,
            "capture_elapsed_s": elapsed_s,
            "observation_age_s": age_s,
            "delivered_capture_elapsed_s": (
                None if delivered is None else delivered["capture_elapsed_s"]
            ),
            "features": features.tolist(),
            "state": state,
            "root_pos_w": root.tolist(),
            "root_quat_w": quat.tolist(),
            "active_before": active_before,
            "active": self.interface_active,
            "legal_mask": legality.tolist(),
            "policy_logits": logits,
            "measurements": [asdict(ray) for ray in measurements],
            "floor_observed_cells": int(observation["floor_observed"].sum()),
            "ceiling_observed_cells": int(observation["ceiling_observed"].sum()),
            "occupied_voxels": int(observation["occupied"].sum()),
            "transition": {
                "requested": requested,
                "attempted": attempted,
                "allowed": allowed,
                "reasons": reasons,
                "joint_jump_rad": joint_jump,
                "root_jump_m": root_jump,
                "root_state_unchanged": bool(torch.equal(before_state, data.root_state_w)),
                "joint_state_unchanged": bool(torch.equal(before_joints, data.joint_pos)),
                "clock_unchanged": bool(torch.equal(before_clock, self.time_steps)),
            },
        }
        self.closed_loop_rows.append(row)
        self.interface_rows.append(row)


class ClosedLoopEnvCfg(EvaluationBankEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        command = self.commands.motion
        command.class_type = ClosedLoopCommand
        command.closed_loop_mode = str(config.get("closed_loop_mode", "scripted"))
        if command.closed_loop_mode not in (
            "scripted",
            "learned",
            "always_walk",
            "always_adapt",
            "forced",
        ):
            raise ValueError("unknown closed-loop mode")
        command.observation_delay_s = float(config.get("observation_delay_s", 0))
        command.history_max_age_s = float(config.get("history_max_age_s", 0.5))
        command.history_max_frames = int(config.get("history_max_frames", 26))
        command.learned_policy_path = str(config.get("learned_policy_path", ""))
        command.learned_policy_sha256 = str(config.get("learned_policy_sha256", ""))
        command.forced_entry_time_s = float(config.get("forced_entry_time_s", 0.3))
        command.encounter_action = int(config.get("encounter_action", 0))
        if command.closed_loop_mode == "forced":
            requested_skill(command.encounter_action, command.forced_entry_time_s, 0, 0, False)


class ClosedLoopRecorder(EvaluationBankRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._history_torques = []
        self._history_velocities = []

    def record_post_physics_decimation_step(self):
        result = super().record_post_physics_decimation_step()
        robot = self.env.scene["robot"]
        self._history_torques.append(robot.data.applied_torque[0].detach().cpu().numpy().copy())
        self._history_velocities.append(
            robot.root_physx_view.get_dof_velocities()[0].detach().cpu().numpy().copy()
        )
        return result

    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_closed_loop_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        robot = self.env.scene["robot"]
        measured_torque = not any(
            getattr(actuator, "is_implicit_model", True) for actuator in robot.actuators.values()
        )
        arrays = {
            "velocity_rad_s": np.asarray(self._history_velocities),
            "physics_steps": np.asarray(self._physics_steps),
            "physics_dt_s": self.env.physics_dt,
            "joint_names": np.array(robot.joint_names),
            "measured_torque_available": measured_torque,
            "torque_nm" if measured_torque else "estimated_torque_nm": np.asarray(
                self._history_torques
            ),
        }
        np.savez_compressed(Path(self.save_dir, "option_mechanical_work.npz"), **arrays)
        rows = command.closed_loop_rows
        np.savez_compressed(
            Path(self.save_dir, "closed_loop_features.npz"),
            schema_version=POLICY_SCHEMA,
            feature_names=np.array(command.closed_loop_feature_names),
            features=np.asarray([row["features"] for row in rows], dtype=np.float32),
            phase_s=np.array([row["time_s"] for row in rows]),
            capture_elapsed_s=np.array([row["capture_elapsed_s"] for row in rows]),
            active_before=np.array([row["active_before"] for row in rows]),
            active_after=np.array([row["active"] for row in rows]),
            requested=np.array([row["transition"]["requested"] for row in rows]),
            legal_mask=np.array([row["legal_mask"] for row in rows]),
            joint_names=np.array(robot.joint_names),
        )
        path = Path(self.save_dir, "reactive_interface.json")
        document = json.loads(path.read_text())
        document.update(
            {
                "mode": command.cfg.closed_loop_mode,
                "feature_schema": POLICY_SCHEMA,
                "feature_names": list(command.closed_loop_feature_names),
                "sensor": (
                    "65 nearest-hit PhysX rays at one body-mounted pose, 50 Hz; "
                    "ideal collision-query normals when available"
                ),
                "history_grid": asdict(command.observation_history.grid),
                "rule_inputs": "delivered sensor history, measured robot state, clock, and legal skill mask",
                "policy_sha256": command.cfg.learned_policy_sha256 or None,
                "measured_torque_available": measured_torque,
                "limitations": (
                    "one encounter, two same-phase references, existing entry/return phase guards; "
                    "ideal range sensor, finite static-scene memory; refusal retains active motion; "
                    "implicit PD estimates are not measured mechanical work or battery energy"
                ),
            }
        )
        path.write_text(json.dumps(document, indent=2))
        self._closed_loop_saved = True


@configclass
class ClosedLoopRecorderCfg(EvaluationBankRecorderCfg):
    class_type = ClosedLoopRecorder
