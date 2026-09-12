"""Opt-in online selection among physically qualified same-phase SONIC references.

The tracker is unchanged. An episode enters at most one non-neutral reference,
maintains its authored phase, and returns through the existing return window.
"""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import random

from isaaclab.utils import configclass
import numpy as np
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

from .motion2scene_closed_loop_execution import (
    ClosedLoopCommand,
    ClosedLoopEnvCfg,
    ClosedLoopRecorder,
    ClosedLoopRecorderCfg,
)
from .motion2scene_closed_loop_policy import history_policy_features
from .motion2scene_multi_option_policy import (
    SCHEMA,
    choose_option,
    extend_option_features,
    legal_option_mask,
    load_multi_policy,
    load_option_registry,
)
from .motion2scene_observation_history import SensorRay, capture_ray_fan


class MultiOptionCommand(ClosedLoopCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.option_registry = load_option_registry(
            cfg.option_registry_path, cfg.option_registry_sha256
        )
        entries = self.option_registry["references"]
        for lib, entry in zip(self._interface_libraries, entries[:2], strict=True):
            if Path(lib.m_cfg.motion_file).resolve() != Path(entry["motion"]["path"]).resolve():
                raise ValueError("primary/first alternate do not match qualified registry order")
        # Additional deterministic evaluation libraries must not alter reset or
        # frozen policy random streams relative to the qualified binary interface.
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            for entry in entries[2:]:
                alternate_cfg = copy.deepcopy(self.motion_lib.m_cfg)
                alternate_cfg.motion_file = entry["motion"]["path"]
                alternate_cfg.filter_motion_keys = None
                lib = MotionLibRobot(alternate_cfg, self.num_envs, self.device)
                lib.load_motions_for_evaluation(start_idx=0)
                if lib._num_motions != 1 or not torch.equal(
                    self.motion_lib.get_time_step_total(self.motion_ids),
                    lib.get_time_step_total(self.motion_ids),
                ):
                    raise ValueError("qualified alternatives require one equal-duration clip")
                self._interface_libraries.append(lib)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(gpu_state)
        self.option_names = [entry["name"] for entry in entries]
        self.option_entry_times = [entry["qualified_entry_times_s"] for entry in entries]
        self.multi_policy = (
            load_multi_policy(cfg.multi_policy_path, cfg.multi_policy_sha256, self.option_names)
            if cfg.multi_option_mode == "learned"
            else None
        )
        if not 0 <= cfg.preferred_option_index < len(entries):
            raise ValueError("preferred option index outside qualified registry")

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
        phase = round(float((self.time_steps + self.motion_start_time_steps)[0]) / 50, 8)
        elapsed = len(self.closed_loop_rows) / 50
        measurements = capture_ray_fan(root, quat, get_physx_scene_query_interface())
        delivered = self.observation_delay.push(
            {"measurements": [asdict(r) for r in measurements], "phase_s": phase}
        )
        if delivered is not None:
            self.observation_history.push(
                [SensorRay(**r) for r in delivered["measurements"]],
                delivered["capture_elapsed_s"],
                delivered_time_s=elapsed,
            )
        observation = self.observation_history.snapshot(root, quat, elapsed)
        current = self.motion_lib
        joint_reference, root_reference = self.joint_pos.clone(), self.anchor_pos_w.clone()
        before_state, before_joints, before_clock = (
            data.root_state_w.clone(),
            data.joint_pos.clone(),
            self.time_steps.clone(),
        )
        joint_jumps, root_jumps = [], []
        try:
            for lib in self._interface_libraries:
                self.motion_lib = lib
                joint_jumps.append(float((self.joint_pos - joint_reference).abs().max()))
                root_jumps.append(
                    float(torch.linalg.vector_norm(self.anchor_pos_w - root_reference))
                )
        finally:
            self.motion_lib = current
        active = self.interface_active
        legality = legal_option_mask(
            active, phase, joint_jumps, root_jumps, self.option_entry_times
        )
        state = {
            name: getattr(data, name)[0].detach().cpu().tolist()
            for name in ("projected_gravity_b", "root_lin_vel_w", "root_ang_vel_w")
        }
        state.update(
            dof_pos=data.joint_pos[0].detach().cpu().tolist(),
            dof_vel=data.joint_vel[0].detach().cpu().tolist(),
        )
        age = (
            elapsed - delivered["capture_elapsed_s"]
            if delivered is not None
            else self.cfg.observation_delay_s + elapsed
        )
        base_names, base_features = history_policy_features(
            observation,
            self.observation_history.grid,
            state,
            phase,
            int(active != 0),
            [legality[0], bool(legality[1:].any())],
            max(0, age),
        )
        names, features = extend_option_features(base_names, base_features, active, legality)
        if self.closed_loop_feature_names is None:
            self.closed_loop_feature_names = names
        elif self.closed_loop_feature_names != names:
            raise RuntimeError("multi-option feature schema changed")
        if self.cfg.multi_option_mode == "forced":
            requested = active
            if active and phase >= 3.3:
                requested = 0
            elif not self.forced_decision_issued and phase >= self.cfg.forced_entry_time_s:
                if abs(phase - self.cfg.forced_entry_time_s) < 1e-8:
                    requested = self.cfg.preferred_option_index
                self.forced_decision_issued = True
            logits = None
        else:
            requested, logits = choose_option(
                self.cfg.multi_option_mode,
                names,
                features,
                legality,
                active,
                phase,
                self.cfg.preferred_option_index,
                self.multi_policy,
            )
        allowed = bool(legality[requested])
        attempted = requested != active
        if attempted and allowed:
            self.motion_lib = self._interface_libraries[requested]
            self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
            self.interface_active = requested
        unchanged = {
            "root_state_unchanged": bool(torch.equal(before_state, data.root_state_w)),
            "joint_state_unchanged": bool(torch.equal(before_joints, data.joint_pos)),
            "clock_unchanged": bool(torch.equal(before_clock, self.time_steps)),
        }
        if attempted and allowed:
            self.interface_switches.append(
                {
                    "time_s": phase,
                    "from": active,
                    "to": requested,
                    "joint_reference_jump_rad": joint_jumps[requested],
                    "root_reference_jump_m": root_jumps[requested],
                    **unchanged,
                }
            )
        row = {
            "time_s": phase,
            "capture_elapsed_s": elapsed,
            "observation_age_s": age,
            "delivered_capture_elapsed_s": (
                None if delivered is None else delivered["capture_elapsed_s"]
            ),
            "features": features.tolist(),
            "state": state,
            "root_pos_w": root.tolist(),
            "root_quat_w": quat.tolist(),
            "active_before": active,
            "active": self.interface_active,
            "legal_mask": legality.tolist(),
            "policy_logits": logits,
            "measurements": [asdict(ray) for ray in measurements],
            "floor_observed_cells": int(observation["floor_observed"].sum()),
            "ceiling_observed_cells": int(observation["ceiling_observed"].sum()),
            "occupied_voxels": int(observation["occupied"].sum()),
            "option_joint_jumps_rad": joint_jumps,
            "option_root_jumps_m": root_jumps,
            "transition": {
                "requested": requested,
                "attempted": attempted,
                "allowed": allowed,
                "reasons": [] if allowed else ["phase_jump_or_unqualified_option_transition"],
                "joint_jump_rad": joint_jumps[requested],
                "root_jump_m": root_jumps[requested],
                **unchanged,
            },
        }
        self.closed_loop_rows.append(row)
        self.interface_rows.append(row)


class MultiOptionEnvCfg(ClosedLoopEnvCfg):
    def __init__(self, config, **kwargs):
        base = dict(config)
        base.update(closed_loop_mode="forced", encounter_action=0)
        super().__init__(base, **kwargs)
        command = self.commands.motion
        command.class_type = MultiOptionCommand
        command.option_registry_path = str(config["option_registry_path"])
        command.option_registry_sha256 = str(config["option_registry_sha256"])
        command.multi_option_mode = str(config.get("multi_option_mode", "scripted"))
        if command.multi_option_mode not in (
            "forced",
            "scripted",
            "always_walk",
            "always_adapt",
            "learned",
        ):
            raise ValueError("unknown multi-option mode")
        command.preferred_option_index = int(config.get("preferred_option_index", 1))
        command.multi_policy_path = str(config.get("multi_policy_path", ""))
        command.multi_policy_sha256 = str(config.get("multi_policy_sha256", ""))


class MultiOptionRecorder(ClosedLoopRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_multi_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir, "closed_loop_features.npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays.update(
            schema_version=np.array(SCHEMA),
            option_ids=np.array(command.option_names),
            classes=np.arange(len(command.option_names)),
        )
        np.savez_compressed(path, **arrays)
        path = Path(self.save_dir, "reactive_interface.json")
        document = json.loads(path.read_text())
        document.update(
            mode=command.cfg.multi_option_mode,
            feature_schema=SCHEMA,
            sensor="65 nearest-hit PhysX rays at one body-mounted pose, 50 Hz; recorded hit normals",
            option_names=command.option_names,
            option_registry_sha256=command.cfg.option_registry_sha256,
            policy_sha256=command.cfg.multi_policy_sha256 or None,
            limitations=(
                "one encounter; qualified 0->k entry ticks and k->0 return at 3.3–3.5 seconds; "
                "no k->j switches, arbitrary duration, stopping or steering"
            ),
        )
        path.write_text(json.dumps(document, indent=2))
        self._multi_saved = True


@configclass
class MultiOptionRecorderCfg(ClosedLoopRecorderCfg):
    class_type = MultiOptionRecorder
