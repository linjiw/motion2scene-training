"""Run a predeclared, serial full-command/navigation physics capture panel.

Both arms construct the same navigation checkpoint. Every attempt is retained;
task failures are results, whereas process failures abort the panel. The input
plan binds tasks, checkpoint and a previously validated native launch template.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.tasks import validate_task


def run_panel(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_text())
    for bound in [plan["checkpoint"], plan["template_command"], plan["template_config"]]:
        if sha(bound["path"]) != bound["sha256"]:
            raise ValueError("Panel input changed")
    tasks = []
    for entry in plan["tasks"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Panel task changed")
        task = validate_task(json.loads(Path(entry["path"]).read_text()))
        if Path(task["task_id"]).name != task["task_id"]:
            raise ValueError("Task ID must be a directory basename")
        tasks.append((entry, task))
    if len({t["task_id"] for _, t in tasks}) != len(tasks):
        raise ValueError("Duplicate panel task")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.json", plan)
    repo = Path(__file__).resolve().parents[3]
    sources = {str(p): sha(p) for p in Path(__file__).parent.glob("*.py")}
    write_new(output / "source-sha256.json", sources)
    template = json.loads(Path(plan["template_command"]["path"]).read_text())
    base_config = json.loads(Path(plan["template_config"]["path"]).read_text())
    ledger = []
    for entry, task in tasks:
        for arm in ("full", "nav"):
            name = arm + "-" + task["task_id"]
            folder = output / name
            folder.mkdir()
            config = dict(base_config)
            config.update(
                output=str(folder / "task"),
                task_path=entry["path"],
                max_steps=task["deadline_ticks"],
                student_checkpoint=plan["checkpoint"]["path"],
                student_sha256=plan["checkpoint"]["sha256"],
                actor_profile=(
                    "motion_full_current_v2" if arm == "full" else "nav_goal_map_localization_v2"
                ),
            )
            config_path = output / (name + "-config.json")
            write_new(config_path, config)
            callback = (
                "RecordNavigationFullCommandCallback"
                if arm == "full"
                else "RecordNavigationMotorCallback"
            )
            replacements = {
                "++seed": str(plan["seed"]),
                "++eval_output_dir": str(folder / "unused"),
                "++eval_base_dir": str(folder / "hydra"),
                "++callbacks.im_eval.stage_config": str(config_path),
                "++callbacks.im_eval._target_": (
                    "gear_sonic.research.scene_distillation.render_navigation_comparison."
                    + callback
                ),
                "++manager_env.config.navigation_task_path": entry["path"],
                "++manager_env.config.scene_usd_path": task["scene_usd_path"],
                "++manager_env.commands.motion.motion_lib_cfg.motion_file": str(
                    Path(task["native_motion"]["path"]).parent
                ),
            }
            command = [
                (
                    a.split("=", 1)[0] + "=" + replacements[a.split("=", 1)[0]]
                    if a.split("=", 1)[0] in replacements
                    else a
                )
                for a in template
            ]
            write_new(folder / "command.json", command)
            started = time.monotonic()
            with (folder / "process.log").open("x") as log:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    env=dict(
                        os.environ, PYTHONPATH=str(repo), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                )
            receipt = dict(
                stage=name, exit_code=result.returncode, wall_seconds=time.monotonic() - started
            )
            write_new(folder / "process-result.json", receipt)
            if result.returncode:
                raise RuntimeError(f"Native process failed: {name}; see retained log")
            receipt["score"] = json.loads((folder / "task/task-result.json").read_text())
            receipt["panel_group"] = entry["panel_group"]
            ledger.append(receipt)
            write_new(folder / "panel-result.json", receipt)
            print(json.dumps(receipt), flush=True)
    write_new(output / "results.json", ledger)
    write_new(
        output / "completion.json",
        dict(state="complete", attempts=len(ledger), plan_sha256=sha(plan_path)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_panel(args.plan, args.output)
