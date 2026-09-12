"""Experimental forced finite schedules; never an online qualified option bank."""

import copy
import json
from pathlib import Path
import pickle
import random

from isaaclab.utils import configclass
import numpy as np
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

from .motion2scene_long_reference_execution import LongReferenceCommand, LongReferenceEnvCfg
from .motion2scene_option_execution import OptionRecorder, OptionRecorderCfg
from .motion2scene_overhang_observer import observe_overhang
from .motion2scene_timed_options import (
    TimedOptionState,
    apply_timed_request,
    assert_loaded_bank,
    checked_artifact,
    definition_digest,
    guard_before_reference_advance,
    validate_request,
)


class LongScheduleCommand(LongReferenceCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        request_path = checked_artifact(
            {"path": cfg.qualification_request_path, "sha256": cfg.qualification_request_sha256}
        )
        self.timed_bank = validate_request(json.loads(request_path.read_text()))
        self.timed_state = TimedOptionState()
        self.forced_option = cfg.forced_option_id
        if self.forced_option not in self.timed_bank.option_ids:
            raise ValueError("forced option is absent from the experimental request")
        references = self.timed_bank.request["references"]
        self.reference_ids = [reference["reference_id"] for reference in references]
        for reference in references:
            checked_artifact(reference["motion"])
        checked_artifact(self.timed_bank.request["controller"])
        for library, reference in zip(self._interface_libraries, references[:2], strict=True):
            if Path(library.m_cfg.motion_file).resolve() != Path(reference["motion"]["path"]):
                raise ValueError("initial references differ from the complete experimental bank")
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            for reference in references[2:]:
                config = copy.deepcopy(self.motion_lib.m_cfg)
                config.motion_file = reference["motion"]["path"]
                config.filter_motion_keys = None
                library = MotionLibRobot(config, self.num_envs, self.device)
                library.load_motions_for_evaluation(start_idx=0)
                if library._num_motions != 1:
                    raise ValueError("exactly one clip per experimental reference required")
                self._interface_libraries.append(library)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(gpu_state)
        self.option_library_indices = [0] + [
            self.reference_ids.index(option["reference_id"])
            for option in self.timed_bank.request["options"]
        ]
        self.forced_schedule = next(
            (
                option
                for option in self.timed_bank.request["options"]
                if option["option_id"] == self.forced_option
            ),
            None,
        )
        self.capture_entry_tick = min(
            option["entry_tick"] for option in self.timed_bank.request["options"]
        )

    def _capture_state(self, tick, packet):
        data = self.robot.data
        return {
            "phase_s": tick / 50,
            "tick": tick,
            "forced_option_id": self.forced_option,
            "active_before": self.timed_state.active,
            "root_pos_w": data.root_pos_w[0].detach().cpu().tolist(),
            "root_quat_w": data.root_quat_w[0].detach().cpu().tolist(),
            "joint_names": list(self.robot.joint_names),
            "state": {
                "dof_pos": data.joint_pos[0].detach().cpu().tolist(),
                "dof_vel": data.joint_vel[0].detach().cpu().tolist(),
                "projected_gravity_b": data.projected_gravity_b[0].detach().cpu().tolist(),
                "root_lin_vel_w": data.root_lin_vel_w[0].detach().cpu().tolist(),
                "root_ang_vel_w": data.root_ang_vel_w[0].detach().cpu().tolist(),
            },
            "packet": packet,
            "physics_step": int(self._env._sim_step_counter),
        }

    def _update_command(self):
        try:
            self._advance_forced_schedule()
        except Exception as error:
            for recorder in self._env.recorder_manager._terms.values():
                if isinstance(recorder, LongScheduleRecorder):
                    recorder.preserve_abort(repr(error))
            raise

    def _observe_schedule(self, tick, scene_query):
        occupied, origin, rays = observe_overhang(
            self.robot.data.root_pos_w[0].detach().cpu().numpy(),
            self.robot.data.root_quat_w[0].detach().cpu().numpy(),
            scene_query,
        )
        packet = self.observation_delay.push(
            {"time_s": tick / 50, "occupied": occupied, "origin": origin, "rays": rays}
        )
        return occupied, origin, rays, packet

    def _requested_schedule(self, tick, joint_jumps, root_jumps, packet):
        requested = self.timed_state.active
        if self.forced_schedule is not None:
            if tick == self.forced_schedule["entry_tick"]:
                requested = self.forced_option
            elif tick == self.forced_schedule["return_tick"]:
                requested = "neutral"
        return requested

    def _advance_forced_schedule(self):
        from omni.physx import get_physx_scene_query_interface

        if not self._bank_verified:
            self._verify_bank()
            counts = [
                int(library.get_time_step_total(self.motion_ids)[0])
                for library in self._interface_libraries
            ]
            assert_loaded_bank(self.timed_bank, self.reference_ids, counts, 50)
            self.bank_audit["same_neutral_xy_by_reference_m"] = [
                float(np.max(np.abs(root[:, :2] - self.bank_arrays["root_xyz"][0, :, :2])))
                for root in self.bank_arrays["root_xyz"]
            ]
        current = int((self.time_steps + self.motion_start_time_steps)[0])
        guard_before_reference_advance(current, self.timed_bank.frame_count)
        TrackingCommand._update_command(self)
        tick = int((self.time_steps + self.motion_start_time_steps)[0])
        if not self._env.sim.cfg.enable_scene_query_support:
            raise RuntimeError("scene queries disabled")
        occupied, origin, rays, packet = self._observe_schedule(
            tick, get_physx_scene_query_interface()
        )
        root_state = self.robot.data.root_state_w.clone()
        joint_state = self.robot.data.joint_pos.clone()
        clock = self.time_steps.clone()
        current_library = self.motion_lib
        joint_reference, root_reference = self.joint_pos.clone(), self.anchor_pos_w.clone()
        joint_jumps, root_jumps = [], []
        try:
            for index in self.option_library_indices:
                self.motion_lib = self._interface_libraries[index]
                joint_jumps.append(float((self.joint_pos - joint_reference).abs().max()))
                root_jumps.append(
                    float(torch.linalg.vector_norm(self.anchor_pos_w - root_reference))
                )
        finally:
            self.motion_lib = current_library
        requested = self._requested_schedule(tick, joint_jumps, root_jumps, packet)
        if tick == self.capture_entry_tick and self.decision_capture is None:
            self.decision_capture = self._capture_state(tick, packet)
        old = self.timed_state.active
        state, transition = apply_timed_request(
            self.timed_bank,
            self.timed_state,
            tick,
            requested,
            joint_jumps,
            root_jumps,
            qualification_only=True,
        )
        index = self.timed_bank.option_ids.index(requested)
        transition.update(
            attempted=requested != old,
            joint_jump_rad=joint_jumps[index],
            root_jump_m=root_jumps[index],
        )
        if state.active != old:
            active_index = self.timed_bank.option_ids.index(state.active)
            self.motion_lib = self._interface_libraries[self.option_library_indices[active_index]]
            self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
            self.interface_active = active_index
        unchanged = {
            "root_state_unchanged": bool(torch.equal(root_state, self.robot.data.root_state_w)),
            "joint_state_unchanged": bool(torch.equal(joint_state, self.robot.data.joint_pos)),
            "clock_unchanged": bool(torch.equal(clock, self.time_steps)),
        }
        if not all(unchanged.values()):
            raise RuntimeError("schedule switch changed physical state or reference clock")
        transition.update(unchanged)
        if state.active != old:
            self.interface_switches.append(
                {
                    "tick": tick,
                    "time_s": tick / 50,
                    "from": old,
                    "to": state.active,
                    "joint_jump_rad": joint_jumps[index],
                    "root_jump_m": root_jumps[index],
                    **unchanged,
                }
            )
        self.timed_state = state
        self.interface_rows.append(
            {
                **self._capture_state(tick, packet),
                "active_before": old,
                "time_s": tick / 50,
                "active": self.timed_state.active,
                "origin": origin,
                "rays": rays,
                "occupied": occupied,
                "transition": transition,
            }
        )


class LongScheduleEnvCfg(LongReferenceEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = LongScheduleCommand
        for name in (
            "qualification_request_path",
            "qualification_request_sha256",
            "forced_option_id",
        ):
            setattr(self.commands.motion, name, str(config[name]))


class LongScheduleRecorder(OptionRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._all_body_forces = []
        self._support_forces = {"left": [], "right": []}

    def record_post_physics_decimation_step(self):
        result = super().record_post_physics_decimation_step()
        sensor = self.env.scene.sensors["contact_forces"]
        values = sensor.contact_physx_view.get_net_contact_forces(dt=self.env.physics_dt)
        self._all_body_forces.append(values.detach().cpu().numpy().reshape(-1, 3).copy())
        for side in self._support_forces:
            sensor = self.env.scene.sensors[f"{side}_foot_support_contact"]
            values = sensor.contact_physx_view.get_contact_force_matrix(dt=self.env.physics_dt)
            self._support_forces[side].append(values.detach().cpu().numpy().reshape(-1, 3).copy())
        return result

    def _save_contact_arrays(self, destination):
        np.savez_compressed(
            destination,
            net_force_w=np.asarray(self._all_body_forces),
            left_floor_force_w=np.asarray(self._support_forces["left"]),
            right_floor_force_w=np.asarray(self._support_forces["right"]),
            physics_steps=np.asarray(self._physics_steps),
            control_steps=np.asarray(self._control_steps),
            physics_dt_s=self.env.physics_dt,
            body_names=np.asarray(self.env.scene.sensors["contact_forces"].body_names),
        )

    def preserve_abort(self, reason):
        destination = Path(self.save_dir)
        destination.mkdir(parents=True, exist_ok=True)
        command = self.env.command_manager.get_term("motion")
        self._save_contact_arrays(destination / "aborted_all_body_contacts.npz")
        with (destination / "aborted_raw_recording.pkl").open("wb") as handle:
            pickle.dump(getattr(self, "_frame_data", None), handle)
        np.savez_compressed(
            destination / "aborted_physics.npz",
            beam_force_w=np.asarray(self._physics_forces),
            physics_steps=np.asarray(self._physics_steps),
            estimated_torque_nm=np.asarray(self._option_torques),
            velocity_rad_s=np.asarray(self._option_velocities),
        )
        (destination / "aborted_interface.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "status": "incomplete_failed_schedule",
                    "physical_steps_recorded": len(self._physics_steps),
                    "observations": command.interface_rows,
                    "switches": command.interface_switches,
                },
                indent=2,
            )
            + "\n"
        )

    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_schedule_saved", False):
            return
        self._save_contact_arrays(Path(self.save_dir, "all_body_contacts.npz"))
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir, "reactive_interface.json")
        data = json.loads(path.read_text())
        data.update(
            rule_inputs="explicit experimental whole schedule; exact ticks and current-reference jumps",
            limitations="forced qualification only; no online learned selection or protective stop",
            timed_request_digest=definition_digest(command.timed_bank.request),
            forced_option_id=command.forced_option,
            loaded_reference_ids=command.reference_ids,
            loaded_reference_frames=[command.timed_bank.frame_count] * len(command.reference_ids),
            actual_bank_shape=list(command.bank_arrays["root_xyz"].shape),
            qualification_only=True,
        )
        path.write_text(json.dumps(data, indent=2) + "\n")
        self._schedule_saved = True


@configclass
class LongScheduleRecorderCfg(OptionRecorderCfg):
    class_type = LongScheduleRecorder
