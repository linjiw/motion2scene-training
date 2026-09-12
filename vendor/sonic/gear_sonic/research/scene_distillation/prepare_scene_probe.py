"""Prepare two bounded scene teacher probes, including an explicitly new terminal-hold reference."""

import argparse
import json
from pathlib import Path
import tarfile

import joblib
import numpy as np

from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)
from gear_sonic.research.hindsight_training.runtime import write_new
from gear_sonic.research.scene_distillation.tasks import binding, validate_task


def prepare(task_catalog, foundation_packet, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    for motion_id in ("00185", "00490"):
        dest = output / motion_id
        (dest / "motions").mkdir(parents=True)
        original_task = Path(task_catalog) / f"{motion_id}-n3.json"
        task = validate_task(json.loads(original_task.read_text()))
        with np.load(task["reference"]["path"], allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        fps = float(arrays["fps"])
        extra = int(round(1.2 * fps))
        qpos = np.concatenate([arrays["qpos"], np.repeat(arrays["qpos"][-1:], extra, axis=0)])
        reference = dest / "hold-reference.npz"
        # Only these fields are consumed by the task-label and sensor interfaces.
        np.savez_compressed(
            reference,
            qpos=qpos,
            time_s=np.arange(len(qpos)) / fps,
            fps=np.asarray(fps),
            body_names=arrays["body_names"],
            joint_names=arrays["joint_names"],
        )
        entry = qpos_to_sonic_motion_entry(
            qpos, source_fps=fps, canonicalize_horizontal_origin=False
        )
        key = "hindsight_" + motion_id
        motion_file = dest / "motions" / (key + ".pkl")
        save_sonic_motion_file(motion_file, motion_key=key, motion_entry=entry)
        joblib.dump(
            {key: {"length": len(qpos), "fps": fps}}, dest / "motions/metadata.pkl", compress=3
        )
        task.update(
            reference=binding(reference),
            native_motion=binding(motion_file),
            parent_task=binding(original_task),
            task_id=motion_id + "-n3-hold",
            terminal_reference_amendment="Append 1.2 seconds of final pose; new unqualified continuation",
            deadline_ticks=int(np.ceil((len(qpos) - 1) / fps / 0.02)) + 1,
        )
        validate_task(task)
        task_path = dest / "task.json"
        write_new(task_path, task)
        lock = json.loads((Path(foundation_packet) / "experiment-lock.json").read_text())
        lock.update(
            train_ids=[motion_id],
            max_control_steps=650,
            purpose="Exact scene and terminal hold teacher probe; no optimizer updates",
        )
        write_new(dest / "collection-lock.json", lock)
        base = json.loads((Path(foundation_packet) / "collection-0-command.json").read_text())
        replacements = {
            "++num_envs": "1",
            "++eval_output_dir": str(dest / "metrics"),
            "++eval_base_dir": str(dest / "hydra"),
            "++manager_env.config.terrain_type": "scene_usd",
            "++manager_env.commands.motion.motion_lib_cfg.motion_file": str(dest / "motions"),
            "++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load": "1",
            "++callbacks.im_eval._target_": (
                "gear_sonic.research.scene_distillation.scene_qualification."
                "SceneTeacherQualificationCallback"
            ),
            "++callbacks.im_eval.collection_lock": str(dest / "collection-lock.json"),
        }
        command = [
            (
                x.split("=", 1)[0] + "=" + replacements[x.split("=", 1)[0]]
                if x.split("=", 1)[0] in replacements
                else x
            )
            for x in base
        ]
        command += [
            f"++callbacks.im_eval.task_path={task_path}",
            "++manager_env._target_=gear_sonic.research.scene_distillation.scene_env.SceneQualificationEnvCfg",
            f"++manager_env.config.scene_usd_path={task['scene_usd_path']}",
            f"++manager_env.config.navigation_task_path={task_path}",
        ]
        commands.append({"motion_id": motion_id, "command": command, "timeout_seconds": 600})
    write_new(output / "commands.json", commands)
    with tarfile.open(output / "source-before-launch.tar.gz", "w:gz") as archive:
        for path in Path("gear_sonic/research/scene_distillation").glob("*.py"):
            archive.add(path)
    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-catalog", type=Path, required=True)
    parser.add_argument("--foundation-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.task_catalog, args.foundation_packet, args.output)
