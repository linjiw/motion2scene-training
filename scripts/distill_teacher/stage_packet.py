"""Stage a distillation packet: teacher checkpoint + config, repaired motion splits, ids.json, collection lock.

Usage: stage_packet.py --packet DIR --teacher-run workspace/teacher-8192-500/tracking-run-1 \
           --checkpoint model_step_000500.pt --motions workspace/teacher-8192-500/motions
"""
import argparse, hashlib, json, os, shutil
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--packet", type=Path, required=True)
p.add_argument("--teacher-run", type=Path, required=True, help="run dir holding the checkpoint and config.yaml")
p.add_argument("--checkpoint", default="model_step_000500.pt")
p.add_argument("--motions", type=Path, required=True, help="dir with train/ and development/ motion folders")
a = p.parse_args()
packet = a.packet.resolve()
(packet / "teacher").mkdir(parents=True, exist_ok=False)
for sub in ("motions", "collect", "offline", "online", "eval"):
    (packet / sub).mkdir(exist_ok=True)
source = (a.teacher_run / a.checkpoint).resolve()
teacher = packet / "teacher" / a.checkpoint  # eval_agent_trl.py reads config.yaml beside it
os.symlink(source, teacher)
shutil.copy2(a.teacher_run / "config.yaml", packet / "teacher/config.yaml")
for split in ("train", "development"):
    os.symlink((a.motions / split).resolve(), packet / "motions" / split)
ids = {split: sorted(q.stem.split("_")[-1] for q in (packet / "motions" / split).glob("hindsight_*.pkl"))
       for split in ("train", "development")}
if set(ids["train"]) & set(ids["development"]):
    raise ValueError("train/development overlap")
digest = sha256(teacher)
(packet / "ids.json").write_text(json.dumps(dict(
    train_ids=ids["train"], development_ids=ids["development"],
    teacher_checkpoint=str(teacher), teacher_sha256=digest), indent=2))
(packet / "collect/collection-lock.json").write_text(json.dumps(dict(
    teacher_checkpoint=str(teacher), teacher_sha256=digest, train_ids=ids["train"],
    max_control_steps=5000, max_root_xy_m=0.25, max_body_mean_m=0.10, minimum_prefix_rows=20,
    precision="fp32",
    purpose="same-state teacher collection for anticipatory motor distillation"), indent=2))
print(json.dumps(dict(packet=str(packet), train=len(ids["train"]), development=len(ids["development"]),
                      teacher_sha256=digest), indent=2))
