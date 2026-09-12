"""Opt-in multi-beam contact instrumentation with the frozen option controller.

Course geometry is used by simulation and recording only, never by the student.
This adapter does not extend reference duration or permit additional transitions.
"""

import hashlib
import json
from pathlib import Path

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import numpy as np

from .motion2scene_course import audit_imported_beams, beam_prim_name, validate_course_definition
from .motion2scene_multi_option_execution import (
    MultiOptionCommand,
    MultiOptionEnvCfg,
    MultiOptionRecorder,
    MultiOptionRecorderCfg,
)


class CourseCommand(MultiOptionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if any(
            int(lib.get_time_step_total(self.motion_ids)[0]) != cfg.course_reference_frames
            for lib in self._interface_libraries
        ):
            raise ValueError("loaded reference durations differ from the registered course budget")
        self.course_reference_exhaustion_events = []

    def _update_command(self):
        current = int((self.time_steps + self.motion_start_time_steps)[0])
        if current + 1 >= self.cfg.course_reference_frames:
            self.course_reference_exhaustion_events.append(
                {"current_frame": current, "refused_next_frame": current + 1}
            )
            # Abort simulation rather than silently resample or hold the last
            # pose. This is not a qualified protective robot stop.
            raise RuntimeError("course reference exhausted; refusing unqualified clock wrap")
        super()._update_command()


class CourseEnvCfg(MultiOptionEnvCfg):
    def __init__(self, config, **kwargs):
        raw = Path(config["course_definition_path"]).read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != config["course_definition_sha256"]:
            raise ValueError("course definition changed")
        definition = json.loads(raw)
        validate_course_definition(definition)
        super().__init__(config, **kwargs)
        self.course_definition = definition
        self.course_definition_sha256 = config["course_definition_sha256"]
        self.commands.motion.class_type = CourseCommand
        self.commands.motion.course_reference_frames = definition["reference_budget"]["frames"]
        names = list(config["beam_contact_body_names"])
        for index in range(1, len(definition["beams"])):
            setattr(
                self.scene,
                f"course_beam_contact_{index}",
                ContactSensorCfg(
                    prim_path=f"/World/ground/terrain/{beam_prim_name(index)}",
                    filter_prim_paths_expr=[f"/World/envs/env_0/Robot/{name}" for name in names],
                    update_period=0.0,
                    history_length=0,
                    track_air_time=False,
                    force_threshold=0.0,
                ),
            )


class CourseRecorder(MultiOptionRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._course_extra_physics = []
        self._course_extra_control = []

    def _initialize(self):
        super()._initialize()
        inventory = json.loads(Path(self.save_dir, "native_collision_inventory.json").read_text())
        self._course_import_audit = audit_imported_beams(
            inventory["shapes"], self.env.cfg.course_definition["beams"]
        )

    def _extra_sensors(self):
        return [
            self.env.scene.sensors[f"course_beam_contact_{index}"]
            for index in range(1, len(self.env.cfg.course_definition["beams"]))
        ]

    def record_post_physics_decimation_step(self):
        result = super().record_post_physics_decimation_step()
        values = []
        for sensor in self._extra_sensors():
            force = sensor.contact_physx_view.get_contact_force_matrix(dt=self.env.physics_dt)
            # Direct PhysX views flatten sensor/environment axes, unlike
            # timestamp-cached Sensor.data. Preserve every robot-body filter.
            force = force.detach().cpu().numpy().reshape(1, 1, -1, 3)[0, 0].copy()
            if force.shape != self._physics_forces[-1].shape or not np.isfinite(force).all():
                raise ValueError("invalid additional-beam physics contact matrix")
            values.append(force)
        self._course_extra_physics.append(values)
        return result

    def record_post_step(self):
        result = super().record_post_step()
        values = []
        for sensor in self._extra_sensors():
            force = sensor.data.force_matrix_w.detach().cpu().numpy()[0, 0].copy()
            if force.shape != self._beam_frames[-1].shape or not np.isfinite(force).all():
                raise ValueError("invalid additional-beam control contact matrix")
            values.append(force)
        self._course_extra_control.append(values)
        return result

    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_course_saved", False):
            return
        count = len(self.env.cfg.course_definition["beams"])
        physics = np.asarray(self._physics_forces)[:, None]
        control = np.asarray(self._beam_frames)[:, None]
        if count > 1:
            physics = np.concatenate((physics, np.asarray(self._course_extra_physics)), axis=1)
            control = np.concatenate((control, np.asarray(self._course_extra_control)), axis=1)
        if (
            physics.shape[1] != count
            or control.shape[1] != count
            or len(physics) != len(self._physics_steps)
            or len(control) != len(self._control_steps)
        ):
            raise RuntimeError("course contact streams or beam counts do not align")
        np.savez_compressed(
            Path(self.save_dir, "course_contacts.npz"),
            physics_force_w=physics,
            control_force_w=control,
            physics_steps=np.asarray(self._physics_steps),
            control_steps=np.asarray(self._control_steps),
            physics_dt=self.env.physics_dt,
            beam_names=np.asarray([beam_prim_name(i) for i in range(count)]),
            filter_paths=np.asarray(self._beam_sensor.cfg.filter_prim_paths_expr),
        )
        inventory = json.loads(Path(self.save_dir, "native_collision_inventory.json").read_text())
        imported = {row["path"]: row for row in inventory["shapes"]}
        paths = [f"/World/ground/terrain/{beam_prim_name(i)}" for i in range(count)]
        command = self.env.command_manager.get_term("motion")
        Path(self.save_dir, "course_capture.json").write_text(
            json.dumps(
                {
                    "definition": self.env.cfg.course_definition,
                    "definition_sha256": self.env.cfg.course_definition_sha256,
                    "imported_beams": [imported[path] for path in paths],
                    "import_geometry_audit": self._course_import_audit,
                    "reference_budget": self.env.cfg.course_definition["reference_budget"],
                    "reference_exhaustion_events": command.course_reference_exhaustion_events,
                    "final_reference_phase_s": command.closed_loop_rows[-1]["time_s"],
                    "student_inputs": "existing sensor history, robot state, phase and option legality",
                    "scope": (
                        "multiple constraints during one finite authored reference; "
                        "no new duration, protective stop or repeated entry is qualified"
                    ),
                },
                indent=2,
            )
        )
        self._course_saved = True


@configclass
class CourseRecorderCfg(MultiOptionRecorderCfg):
    class_type = CourseRecorder
