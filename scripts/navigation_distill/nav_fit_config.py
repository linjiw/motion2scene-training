"""Write a hash-bound navigation_motor fit config (teacher_positive or motor_recovery view)."""

import argparse
import json
import os
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motor", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--view", choices=["teacher_positive", "motor_recovery"], default="teacher_positive")
    p.add_argument("--task-manifest", type=Path, help="tasks/manifest.json (recovery view)")
    p.add_argument("--warm-start", type=Path, help="initial navigation checkpoint")
    p.add_argument("--recovery-fraction", type=float, default=0.0)
    p.add_argument("--fresh-behavior", help="sha256 of the navigation checkpoint that drove fresh recoveries")
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=91370)
    p.add_argument("--objective", default="structured_motor")
    p.add_argument("--no-localization", action="store_true")
    p.add_argument("--approach-weight", type=float, help="braking-aware sampling weight for rows within --approach-radius of the goal")
    p.add_argument("--approach-radius", type=float, default=1.0)
    p.add_argument("--purpose", default="")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    packet = Path(os.environ["NAV_PACKET"])
    ids = json.loads((packet / "ids.json").read_text())
    catalog = Path(os.environ.get("ANCESTRY_CATALOG", packet.parent / "m2s-hindsight-dataset-v1-20260911/catalog.json"))
    localization = not a.no_localization
    c = dict(
        motor_checkpoint=str(a.motor.resolve()), motor_sha256=sha(a.motor),
        teacher_sha256=ids["teacher_sha256"],
        dataset_manifest=str(a.manifest.resolve()), dataset_manifest_sha256=sha(a.manifest),
        ancestry_catalog=str(catalog.resolve()), ancestry_catalog_sha256=sha(catalog),
        updates=a.updates, batch_size=a.batch, width=512, wall_cap_seconds=1800, device="cuda",
        seed=a.seed, learning_rate=a.lr, command_weight=0.1,
        actor_profile="nav_goal_map_localization_v2" if localization else "nav_goal_map_v1",
        objective=a.objective, localization=localization, dataset_view=a.view, purpose=a.purpose,
    )
    if a.view == "motor_recovery":
        c.update(task_manifest=str(a.task_manifest.resolve()), task_manifest_sha256=sha(a.task_manifest),
                 recovery_fraction=a.recovery_fraction)
        if a.fresh_behavior:
            c["fresh_recovery_behavior_sha256"] = a.fresh_behavior
    if a.approach_weight is not None:
        c.update(approach_weight=a.approach_weight, approach_radius_m=a.approach_radius)
    if a.warm_start:
        c.update(initial_navigation_checkpoint=str(a.warm_start.resolve()),
                 initial_navigation_sha256=sha(a.warm_start))
    a.out.write_text(json.dumps(c, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
