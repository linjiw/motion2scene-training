"""Bounded neutral-reference qualification without changing the frozen tracker."""

import json
from pathlib import Path

from isaaclab.utils import configclass
import joblib
import numpy as np
import torch

from .motion2scene_comparison_execution import ComparisonCommand, ComparisonEnvCfg
from .motion2scene_long_reference_bank import audit_finite_root_route
from .motion2scene_option_execution import OptionRecorder, OptionRecorderCfg


class LongReferenceCommand(ComparisonCommand):
    def _verify_bank(self):
        roots, dofs, errors = [], [], []
        for library in self._interface_libraries:
            n = int(library.get_time_step_total(self.motion_ids)[0])
            ids = torch.zeros(n, dtype=torch.long, device=self.device)
            steps = torch.arange(n, device=self.device)
            root = library.get_root_pos_w(ids, steps).detach().cpu().numpy().copy()
            dof = library.get_dof_pos(ids, steps).detach().cpu().numpy().copy()
            entry = next(iter(joblib.load(library.m_cfg.motion_file).values()))
            errors.append(
                audit_finite_root_route(
                    root,
                    entry["root_trans_offset"],
                    float(entry["fps"]),
                    self.cfg.expected_reference_frames,
                )
            )
            if not np.isfinite(dof).all():
                raise RuntimeError("nonfinite loaded joint references")
            roots.append(root)
            dofs.append(dof)
        self.bank_arrays = {
            "root_xyz": np.asarray(roots),
            "joint_pos": np.asarray(dofs),
            "fps": np.asarray(50.0),
        }
        self.bank_audit = {
            "source_route_max_error_m": errors,
            "alternate_loader": "load_motions_for_evaluation",
            "same_root_xy_max_error_m": float(np.max(np.abs(roots[0][:, :2] - roots[1][:, :2]))),
            "random_stream_restored_after_reload": True,
            "expected_reference_frames": self.cfg.expected_reference_frames,
        }
        self._bank_verified = True

    def _update_command(self):
        if not self._bank_verified:
            self._verify_bank()
            shape = self.bank_arrays["root_xyz"].shape
            if shape != (2, self.cfg.expected_reference_frames, 3):
                raise RuntimeError(f"unexpected neutral reference bank shape: {shape}")
            if self.cfg.encounter_action != 0:
                raise RuntimeError("this qualification only supports neutral execution")
        current = int((self.time_steps + self.motion_start_time_steps)[0])
        if current + 1 >= self.cfg.expected_reference_frames:
            raise RuntimeError("finite neutral reference exhausted; looping is not qualified")
        super()._update_command()
        row = self.interface_rows[-1]
        row["root_pos_w"] = self.robot.data.root_pos_w[0].detach().cpu().tolist()
        row["root_quat_w"] = self.robot.data.root_quat_w[0].detach().cpu().tolist()
        row["physics_step"] = int(self._env._sim_step_counter)


class LongReferenceEnvCfg(ComparisonEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = LongReferenceCommand
        self.commands.motion.expected_reference_frames = int(config["expected_reference_frames"])


class LongReferenceRecorder(OptionRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_long_reference_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir, "reactive_interface.json")
        data = json.loads(path.read_text())
        data["neutral_reference_qualification"] = {
            "expected_reference_frames": command.cfg.expected_reference_frames,
            "observed_bank_shape": list(command.bank_arrays["root_xyz"].shape),
            "clock_guard": "refuse before increment would reach the final excluded bank index",
            "sensor_clock": "post-physics poses; command phase is one 50Hz tick ahead",
            "neutral_only": True,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")
        self._long_reference_saved = True


@configclass
class LongReferenceRecorderCfg(OptionRecorderCfg):
    class_type = LongReferenceRecorder
