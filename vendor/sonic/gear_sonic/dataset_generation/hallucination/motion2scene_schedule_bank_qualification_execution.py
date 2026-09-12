"""Full-bank forced qualification with a matched capture at each schedule's entry."""

import json
from pathlib import Path

from isaaclab.utils import configclass

from .motion2scene_environment_contact_execution import (
    EnvironmentContactEnvCfg,
    EnvironmentContactRecorder,
    EnvironmentContactRecorderCfg,
)
from .motion2scene_long_schedule_execution import LongScheduleCommand


class ScheduleBankQualificationCommand(LongScheduleCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.forced_schedule is not None:
            self.capture_entry_tick = self.forced_schedule["entry_tick"]


class ScheduleBankQualificationEnvCfg(EnvironmentContactEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = ScheduleBankQualificationCommand


class ScheduleBankQualificationRecorder(EnvironmentContactRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_entry_metadata_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir, "reactive_interface.json")
        value = json.loads(path.read_text())
        capture = command.decision_capture
        tick = command.capture_entry_tick
        if capture is None or capture["tick"] != tick:
            raise RuntimeError("missing matched capture at this schedule's actual entry")
        row = value["observations"][tick - 1]
        for key in ("tick", "physics_step", "root_pos_w", "root_quat_w", "state"):
            if capture[key] != row[key]:
                raise RuntimeError(f"entry capture differs from actual synchronized row: {key}")
        value["qualification_entry_capture_tick"] = command.capture_entry_tick
        value["qualification_expected_prefix_frames"] = command.capture_entry_tick
        path.write_text(json.dumps(value, indent=2) + "\n")
        self._entry_metadata_saved = True


@configclass
class ScheduleBankQualificationRecorderCfg(EnvironmentContactRecorderCfg):
    class_type = ScheduleBankQualificationRecorder
