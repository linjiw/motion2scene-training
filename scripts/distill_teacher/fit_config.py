"""Write a hash-bound anticipatory motor fit config."""
import argparse, json, os
from pathlib import Path
from gear_sonic.research.hindsight_training.runtime import sha
p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--init", type=Path, required=True)
p.add_argument("--updates", type=int, required=True)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--future-weight", type=float, default=0.1)
p.add_argument("--save-interval", type=int, default=2000)
p.add_argument("--seed", type=int, default=91360)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
packet = Path(os.environ.get("DISTILL_PACKET", Path(__file__).resolve().parents[2] / "workspace/distill-8192"))
ids = json.loads((packet / "ids.json").read_text())
config = dict(
    motor_architecture="anticipatory", current_frame_extension=True,
    reference_layout_version="native_q_then_v_v1", objective="public",
    teacher_checkpoint=ids["teacher_checkpoint"], teacher_sha256=ids["teacher_sha256"],
    dataset_manifest=str(a.manifest.resolve()), dataset_manifest_sha256=sha(a.manifest),
    initialize_checkpoint=str(a.init.resolve()), initialize_sha256=sha(a.init),
    train_ids=ids["train_ids"], updates=a.updates, batch_size=a.batch, learning_rate=a.lr,
    token_weight=0.1, future_weight=a.future_weight, save_interval=a.save_interval,
    seed=a.seed, wall_cap_seconds=7200, device="cuda",
)
a.out.write_text(json.dumps(config, indent=2))
print("wrote", a.out)
