#!/usr/bin/env python3
"""Stage the teacher-8192-500 review packet and write its evaluation lock (no Isaac launch).

Creates <review>/eval/<arm>/{model_step_XXXXXX.pt -> source checkpoint (symlink), config.yaml}
so eval_agent_trl.py finds config.yaml beside the checkpoint (eval_agent_trl.py:94-102) and does
not copy the checkpoint (the name already equals model_step_<global_step>.pt, :613-619).
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_common import (  # noqa: E402
    ARMS, DEFAULT_REVIEW, LAUNCH_ORDER, LEDGER, PACKET, PRIOR_PREVIOUS8000_DEV, SEED, SPLITS,
    checkpoint_name, run_dir, sha256, split_keys,
)

EXPECTED_VIDEO_IDS = ["00802", "00047", "00921", "00333", "00796", "00865", "00421", "00044"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--selection-lock", type=Path, default=None,
                        help="default: <review>/video-selection-lock.json")
    parser.add_argument("--skip-hash", action="store_true", help="testing only")
    args = parser.parse_args()
    review = args.review
    if args.selection_lock is None:
        args.selection_lock = review / "video-selection-lock.json"
    if not args.selection_lock.exists():
        sys.exit(f"{args.selection_lock} missing: write the video selection before staging")
    if (review / "evaluation-lock.json").exists():
        sys.exit(f"{review}/evaluation-lock.json exists; refusing to re-stage")

    selection = json.loads(args.selection_lock.read_text())
    if selection["video_ids"] != EXPECTED_VIDEO_IDS:
        sys.exit(f"selection lock video_ids changed: {selection['video_ids']}")
    keys = {split: split_keys(split) for split in SPLITS}
    for split, count in SPLITS.items():
        if len(keys[split]) != count:
            sys.exit(f"{split}: expected {count} motions, found {len(keys[split])}")
    featured = []
    for pick in selection["picks"]:
        key = f"hindsight_{pick['id']}"
        if key not in keys[pick["split"]]:
            sys.exit(f"featured {key} not in {pick['split']}")
        featured.append({"motion_key": key, **{k: pick[k] for k in (
            "split", "stratum", "category", "seconds", "route", "reasons", "prompts")}})

    arms = {}
    for arm, spec in ARMS.items():
        digest = None if args.skip_hash else sha256(spec["source"])
        if digest is not None and digest != spec["sha256"]:
            sys.exit(f"{arm}: sha256 {digest} != expected {spec['sha256']}")
        arms[arm] = {"source": str(spec["source"]), "sha256": digest or spec["sha256"],
                     "global_step": spec["global_step"], "label": spec["label"]}

    config = PACKET / "tracking-run-1/config.yaml"
    review.mkdir(parents=True, exist_ok=True)
    for arm, spec in ARMS.items():
        folder = review / "eval" / arm
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / checkpoint_name(arm)
        if not link.exists():
            os.symlink(spec["source"], link)
        shutil.copy2(config, folder / "config.yaml")
        arms[arm].update(staged_checkpoint=str(link), config_sha256=sha256(folder / "config.yaml"))
        for split in SPLITS:
            run_dir(review, arm, split).mkdir(parents=True, exist_ok=True)

    lock_copy = review / "video-selection-lock.json"
    if args.selection_lock.resolve() != lock_copy.resolve():
        shutil.copy2(args.selection_lock, lock_copy)
    tools = review / "tools"
    tools.mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    if here != tools.resolve():
        for source in here.iterdir():
            if source.suffix in {".py", ".sh"}:
                shutil.copy2(source, tools / source.name)

    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Matched native Isaac Lab tracking evaluation + mesh replay of the "
        "8192-env x 500-iteration teacher against its release initialization and the previous "
        "8000-iteration teacher. Fixed before any 8192-500 evaluation output exists.",
        "arms": arms,
        "config": {"source": str(config), "sha256": sha256(config),
                   "note": "one config for all arms, as in m2s-sonic-qualification-20260912"},
        "seed": SEED,
        "splits": {split: {"motion_file": str(PACKET / "motions" / split),
                           "num_envs": SPLITS[split], "motion_keys": keys[split]}
                   for split in SPLITS},
        "launch_order": [f"{arm}/{split}" for arm, split in LAUNCH_ORDER],
        "callback": "gear_sonic.research.hindsight_training.pose_capture."
        "PoseCaptureQualificationCallback",
        "motion_ledger_sha256": sha256(LEDGER),
        "video_selection_lock_sha256": sha256(review / "video-selection-lock.json"),
        "featured_videos": featured,
        "reproduction_gate": {
            "run": "previous8000/development",
            "recorded": str(PRIOR_PREVIOUS8000_DEV),
            "recorded_sha256": sha256(PRIOR_PREVIOUS8000_DEV),
            "expect": "11/20 completed, mean progress 0.714801 (different host; informative)",
        },
        "metric_definitions": {
            "completed": "native terminated == false (anchor_pos, anchor_ori_full, ee_body_pos, "
            "foot_pos_xyz; time_out excluded; no root-XY guard)",
            "valid_frames": "T if completed else round(native_progress * motion_num_steps)",
            "prefix_mpjpe_g_mm": "mean over valid frames and 14 bodies of |tracked-reference|",
            "prefix_mpjpe_l_mm": "same, pelvis-relative",
            "root_xy": "pelvis XY distance tracked vs reference",
            "paired": "improved = A failed & B completed; regressed = reverse; exact sign test",
        },
    }
    (review / "evaluation-lock.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps({"review": str(review), "featured": [f["motion_key"] for f in featured],
                      "arms": {a: v["staged_checkpoint"] for a, v in arms.items()}}, indent=2))


if __name__ == "__main__":
    main()
