"""Native exact-scene teacher audit with pair-resolved environment contacts at 200 Hz."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.collect import FoundationCollectionCallback
from gear_sonic.research.scene_distillation.tasks import validate_task, verify_native_motion


def score_scene(task, tracked, reference, forces, body_names, *, terminated, valid_steps):
    """Score measured native traces; floor support by feet is allowed, other contacts are not."""
    tracked, reference, forces = map(np.asarray, (tracked, reference, forces))
    if tracked.shape != reference.shape or tracked.ndim != 3 or tracked.shape[-1] != 3:
        raise ValueError("Invalid measured trajectory")
    if forces.ndim != 4 or forces.shape[1:] != (len(body_names), len(task["obstacles"]) + 1, 3):
        raise ValueError("Missing pair-resolved obstacle/floor contacts")
    if not all(np.isfinite(x).all() for x in (tracked, reference, forces)):
        raise ValueError("Nonfinite physical evidence")
    n = min(valid_steps, len(tracked))
    if n < 2 or len(forces) < 4 * n:
        raise ValueError("Incomplete contact or trajectory evidence")
    force_norm = np.linalg.norm(forces[: 4 * n], axis=-1)
    nonfeet = [
        i
        for i, name in enumerate(body_names)
        if name not in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    undesired = max(
        float(force_norm[:, :, 1:].max(initial=0)), float(force_norm[:, nonfeet, 0].max(initial=0))
    )
    root = tracked[:n, 0]
    distance = np.linalg.norm(root - task["goal_xyz"], axis=-1)
    speed = np.linalg.norm(np.diff(root, axis=0), axis=-1) / 0.02
    hold = task["hold_ticks"]
    body_mean = float(np.linalg.norm(tracked[:n] - reference[:n], axis=-1).mean())
    root_max = float(np.linalg.norm((tracked[:n, 0] - reference[:n, 0])[:, :2], axis=-1).max())
    checks = {
        "whole_motion": bool(not terminated and body_mean <= 0.1 and root_max <= 0.25),
        "environment_contacts": bool(undesired <= 1.0),
        "goal_reached": bool(distance[-1] <= task["goal_tolerance_m"]),
        "terminal_hold": bool(
            n > hold
            and np.all(distance[-hold:] <= task["goal_tolerance_m"])
            and np.all(speed[-hold:] <= task["terminal_speed_mps"])
        ),
    }
    return {
        "checks": checks,
        "state": "complete" if all(checks.values()) else "failed",
        "max_undesired_environment_normal_force_n": undesired,
        "final_goal_distance_m": float(distance[-1]),
        "mean_body_error_m": body_mean,
        "max_root_xy_error_m": root_max,
        "valid_control_steps": n,
    }


class SceneTeacherQualificationCallback(FoundationCollectionCallback):
    def __init__(self, *args, task_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_path = Path(task_path)
        self.task = validate_task(json.loads(self.task_path.read_text()))
        self.contact_rows = []

    def _pre_evaluate_policy(self, reset_env=True):
        super()._pre_evaluate_policy(reset_env)
        if (
            self.env.num_envs != 1
            or sha(self.env.config.scene_usd_path) != self.task["scene_usd_sha256"]
        ):
            raise ValueError("Scene teacher audit must execute exactly one bound collision scene")
        keys = list(self.env._motion_lib.curr_motion_keys)
        verify_native_motion(self.task, self.env._motion_lib)
        if keys != ["hindsight_" + self.task["motion_id"]]:
            raise ValueError("Scene qualification loaded the wrong reference")
        self.contact_bodies = list(self.env.env.scene["robot"].body_names)
        self.sensors = [
            self.env.env.scene.sensors[f"navigation_contact_{name}"] for name in self.contact_bodies
        ]
        for sensor in self.sensors:
            if sensor.contact_physx_view.filter_count != len(self.task["obstacles"]) + 1:
                raise ValueError("Native contact filter count mismatch")
        self.original_step = self.env.env.sim.step

        def step(*args, **kwargs):
            result = self.original_step(*args, **kwargs)
            self.contact_rows.append(
                np.stack(
                    [
                        s.contact_physx_view.get_contact_force_matrix(dt=0.005)
                        .cpu()
                        .numpy()
                        .reshape(-1, 3)
                        .copy()
                        for s in self.sensors
                    ]
                )
            )
            return result

        self.env.env.sim.step = step

    def _post_evaluate_policy(self, eval_res):
        self.env.env.sim.step = self.original_step
        result = super()._post_evaluate_policy(eval_res)
        output = Path(self.output_dir)
        contacts = output / "environment-contacts.npz"
        forces = np.asarray(self.contact_rows)
        np.savez_compressed(
            contacts,
            force_w=forces,
            body_names=np.asarray(self.contact_bodies),
            dt=np.asarray(0.005),
        )
        mapping = output / "contact-mapping.json"
        write_new(
            mapping,
            [
                {
                    "body_paths": list(s.body_physx_view.prim_paths),
                    "filter_paths": list(s.cfg.filter_prim_paths_expr),
                    "filter_count": int(s.contact_physx_view.filter_count),
                }
                for s in self.sensors
            ],
        )
        metrics = eval_res["all_metrics_dict"]
        expected = int(self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)[0])
        score = score_scene(
            self.task,
            self.pred_pos_all[0],
            self.gt_pos_all[0],
            forces,
            self.contact_bodies,
            terminated=bool(metrics["terminated"][0]),
            valid_steps=int(round(metrics["progress"][0] * expected)),
        )
        evidence = [
            contacts,
            mapping,
            self.task_path,
            output / f"hindsight_{self.task['motion_id']}.npz",
            output / "collection.json",
        ]
        score.update(
            teacher_sha256=self.lock["teacher_sha256"],
            task_sha256=sha(self.task_path),
            control_dt=0.02,
            contact_dt=0.005,
            contact_measure="pair_resolved_normal_force_not_friction",
            evidence=[{"path": str(p), "sha256": sha(p)} for p in evidence],
        )
        write_new(output / "scene-qualification.json", score)
        return result
