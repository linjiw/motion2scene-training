"""Write one scene-task stage config for the DirectSceneTaskCallback family.

Modes: teacher (StoppingTeacherCallback), full (FullMotorTaskCallback with a motor
checkpoint), nav (NavigationMotorCallback with a navigation checkpoint), recovery
(MotorRecoveryCollectionCallback: switch 0 = motor demonstration, >0 = learner prefix).
"""

import argparse
import json
import os
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import sha

CALLBACKS = dict(
    teacher="gear_sonic.research.scene_distillation.stopping_teacher.StoppingTeacherCallback",
    full="gear_sonic.research.scene_distillation.navigation_motor_runtime.FullMotorTaskCallback",
    nav="gear_sonic.research.scene_distillation.navigation_motor_runtime.NavigationMotorCallback",
    recovery="gear_sonic.research.scene_distillation.navigation_recovery.MotorRecoveryCollectionCallback",
    reentry="gear_sonic.research.scene_distillation.navigation_reentry.ReentryProbeCallback",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=sorted(CALLBACKS), required=True)
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="task output dir (must not exist)")
    p.add_argument("--student", type=Path, help="motor or navigation checkpoint")
    p.add_argument("--motor", type=Path, help="frozen motor checkpoint (recovery mode)")
    p.add_argument("--takeover-tick", type=int, default=0)
    p.add_argument("--actor-profile", default="nav_goal_map_localization_v2")
    p.add_argument("--collect-to-deadline", action="store_true")
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    packet = Path(os.environ["NAV_PACKET"])
    ids = json.loads((packet / "ids.json").read_text())
    task = json.loads(a.task.read_text())
    c = dict(
        teacher_checkpoint=ids["teacher_checkpoint"],
        teacher_sha256=ids["teacher_sha256"],
        task_path=str(a.task.resolve()),
        output=str(a.output.resolve()),
        max_steps=task["deadline_ticks"],
        teacher_mode=a.mode == "teacher",
        command_profile="context",
    )
    if a.mode == "teacher":
        c["collect_to_deadline"] = a.collect_to_deadline
    if a.mode in ("full", "nav", "recovery", "reentry"):
        c.update(student_checkpoint=str(a.student.resolve()), student_sha256=sha(a.student))
    if a.mode == "full":
        c["actor_profile"] = "motion_full_current_v2"
    if a.mode == "nav":
        c["actor_profile"] = a.actor_profile
    if a.mode == "reentry":
        motor = a.motor or a.student
        c.update(actor_profile=a.actor_profile, takeover_tick=a.takeover_tick,
                 motor_checkpoint=str(motor.resolve()), motor_sha256=sha(motor))
    if a.mode == "recovery":
        motor = a.motor or a.student
        c.update(
            actor_profile="motion_full_current_v2" if a.takeover_tick == 0 else a.actor_profile,
            takeover_tick=a.takeover_tick,
            motor_checkpoint=str(motor.resolve()),
            motor_sha256=sha(motor),
        )
    a.config.write_text(json.dumps(c, indent=2))
    print(CALLBACKS[a.mode])


if __name__ == "__main__":
    main()
