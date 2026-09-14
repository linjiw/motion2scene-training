"""Phase-rewind re-entry probe: can the frozen motor resume from a displaced learner state?

At the switch tick the reference clock is rewound (never advanced) to the frame whose
reference pelvis is nearest the measured pelvis in the horizontal plane. The motor then
tracks the nominal reference from that frame. This is a declared reference-phase switch
and therefore a diagnostic only: it emits no motor-recovery receipt and the recovery
loader rejects phase switches, so no row from a probe can become training data without
a separately versioned continuation contract.
"""

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import write_new
from gear_sonic.research.scene_distillation.direct_context import score_navigation_task
from gear_sonic.research.scene_distillation.motor_runtime import (
    current_orientation_observation,
    motor_commands,
)
from gear_sonic.research.scene_distillation.navigation_continuation import continuation_outcomes
from gear_sonic.research.scene_distillation.navigation_takeover import NavigationTakeoverCallback


def rewind_frame(reference_root_xy, robot_xy, current_frame, min_rewind=0):
    """Nearest reference frame at or before the current one; rewind only, never skip ahead."""
    reference_root_xy = np.asarray(reference_root_xy, dtype=np.float64)
    if reference_root_xy.ndim != 2 or reference_root_xy.shape[1] != 2 or current_frame < 0:
        raise ValueError("Invalid reference path or frame")
    last = min(int(current_frame) - int(min_rewind), len(reference_root_xy) - 1)
    if last < 0:
        return 0
    distance = np.linalg.norm(reference_root_xy[: last + 1] - np.asarray(robot_xy)[:2], axis=1)
    return int(distance.argmin())


class ReentryProbeCallback(NavigationTakeoverCallback):
    """Navigation drives to the switch; the motor continues from the rewound reference."""

    def _begin_task(self, env, teacher, task):
        super()._begin_task(env, teacher, task)
        command = env.motion_command
        library = env._motion_lib
        motion_id = command.motion_ids[0:1]
        frames = int(library.get_time_step_total(motion_id)[0].item())
        steps = torch.arange(frames, dtype=torch.long, device=motion_id.device)
        origin = env.env.scene.env_origins[0]
        self.reference_root_xy = (
            (library.get_body_pos_w(motion_id.expand(frames), steps)[:, 0] - origin)[:, :2]
            .cpu()
            .numpy()
        )
        self.rewind = None

    def _student_action(self, student, env, teacher, observation, task, noise):
        tick = self.probe_tick
        if tick == self.config["takeover_tick"] and self.rewind is None:
            command = env.motion_command
            origin = env.env.scene.env_origins[0]
            robot_xy = (command.robot_anchor_pos_w[0] - origin)[:2].cpu().numpy()
            current = int((command.motion_start_time_steps + command.time_steps)[0].item())
            target = rewind_frame(
                self.reference_root_xy, robot_xy, current, self.config.get("min_rewind_frames", 0)
            )
            command.time_steps[:] = target - command.motion_start_time_steps
            self.rewind = dict(
                decision_tick=tick,
                nominal_frame=current,
                rewound_frame=target,
                rewind_frames=current - target,
                reference_xy_error_m=float(
                    np.linalg.norm(self.reference_root_xy[target] - robot_xy)
                ),
                nominal_xy_error_m=float(np.linalg.norm(self.reference_root_xy[current] - robot_xy)),
                rewind_only=True,
            )
        return super()._student_action(student, env, teacher, observation, task, noise)

    def _goal_stop(self, task, roots, speeds, forces, fell):
        # Score the post-switch suffix on its own so a learner-accumulated hold cannot count.
        start = self.config["takeover_tick"]
        if len(roots) <= start:
            return super()._goal_stop(task, roots, speeds, forces, fell)
        post = score_navigation_task(
            task, np.asarray(roots)[start:], np.asarray(speeds)[start:], np.asarray(forces)[start:], fell=fell
        )
        return bool(post["navigation_success"] and (np.asarray(forces) <= 1).all())

    def _complete_task(self, task, score, output):
        with np.load(output / "trace.npz") as trace:
            outcomes = continuation_outcomes(
                task, trace, min(self.config["takeover_tick"], score["control_steps"])
            )
        write_new(
            output / "reentry.json",
            dict(
                diagnostic_only=True,
                provider="phase_rewind_nearest_reference_frame",
                reference_phase_switch=True,
                training_admitted=False,
                unassisted_navigation_result=False,
                takeover_tick=self.config["takeover_tick"],
                rewind=self.rewind,
                outcomes=outcomes,
                behavior_checkpoint=dict(
                    path=self.config["student_checkpoint"], sha256=self.config["student_sha256"]
                ),
                motor_sha256=self.config["motor_sha256"],
            ),
        )
