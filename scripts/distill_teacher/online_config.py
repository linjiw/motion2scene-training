"""Write a hash-bound OnlineMotorCallback stage config."""
import argparse, json, os
from pathlib import Path
from gear_sonic.research.hindsight_training.runtime import sha
p = argparse.ArgumentParser()
p.add_argument("--student", type=Path, required=True)
p.add_argument("--replay", type=Path, required=True)
p.add_argument("--output", type=Path, required=True, help="stage training directory (must not exist)")
p.add_argument("--cycles", type=int, required=True)
p.add_argument("--rollout-steps", type=int, default=32)
p.add_argument("--updates-per-cycle", type=int, default=64)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--future-weight", type=float, default=0.1)
p.add_argument("--p-start", type=float, default=0.5)
p.add_argument("--p-end", type=float, default=0.05)
p.add_argument("--save-cycles", type=int, default=25)
p.add_argument("--seed", type=int, default=91370)
p.add_argument("--wall", type=int, default=7200)
p.add_argument("--takeover-probability", type=float, default=0.15)
p.add_argument("--takeover-trigger", type=float, default=0.1)
p.add_argument("--config", type=Path, required=True)
a = p.parse_args()
packet = Path(os.environ.get("DISTILL_PACKET", Path(__file__).resolve().parents[2] / "workspace/distill-8192"))
ids = json.loads((packet / "ids.json").read_text())
c = dict(
    cycles=a.cycles, rollout_steps=a.rollout_steps, updates_per_cycle=a.updates_per_cycle,
    wall_cap_seconds=a.wall, batch_size=a.batch, learning_rate=a.lr,
    token_weight=0.1, future_weight=a.future_weight,
    student_checkpoint=str(a.student.resolve()), student_sha256=sha(a.student),
    replay_manifest=str(a.replay.resolve()), replay_manifest_sha256=sha(a.replay),
    output=str(a.output.resolve()), teacher_sha256=ids["teacher_sha256"], train_ids=ids["train_ids"],
    seed=a.seed, nominal_start_fraction=0.2, burn_in_ticks=10,
    teacher_probability_start=a.p_start, teacher_probability_end=a.p_end,
    takeover_ticks=16, takeover_trigger_m=a.takeover_trigger, takeover_probability=a.takeover_probability, takeover_support_m=0.1,
    save_cycles=a.save_cycles,
)
a.config.write_text(json.dumps(c, indent=2))
print("wrote", a.config)
