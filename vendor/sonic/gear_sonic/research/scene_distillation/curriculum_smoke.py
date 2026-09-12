"""CPU-only, matched-budget curriculum ablation on whole-qualified recorded clips."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.commands import PROFILES, MaskedMotionFoundation
from gear_sonic.research.scene_distillation.train import main as train, make_decoder


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_config.read_text())
    source = json.loads(args.collection.read_text())
    if source["teacher_sha256"] != base["teacher_sha256"] or source.get("precision") != "fp32":
        raise ValueError("Smoke requires matched FP32 teacher collection")
    manifest = {
        "schema": "bfm_executed_foundation_v1",
        "source": "executed_native_teacher",
        "teacher_sha256": source["teacher_sha256"],
        "episodes": [],
        "parents": [{"path": str(args.collection.resolve()), "sha256": sha(args.collection)}],
    }
    heldout = []
    split_receipt = []
    for episode in source["episodes"]:
        if not episode.get("whole_motion_eligible") or episode["terminated"]:
            continue
        if sha(episode["path"]) != episode["sha256"]:
            raise ValueError("Source episode changed")
        with np.load(episode["path"], allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files if k != "query_mask"}
        n = episode["rows"]
        arrays["reference_phase"] = np.arange(n, dtype=np.float32) / n
        # A contiguous held-out block in every quarter, with a two-frame gap.
        validation = np.zeros(n, dtype=bool)
        training = np.ones(n, dtype=bool)
        for part in np.array_split(np.arange(n), 4):
            start = int(part[len(part) * 3 // 5])
            stop = min(int(part[-1]) + 1, start + max(1, len(part) // 5))
            validation[start:stop] = True
            training[max(0, start - 2) : min(n, stop + 2)] = False
        path = args.output / f"train-{episode['motion_id']}.npz"
        np.savez_compressed(path, **{k: v[training] for k, v in arrays.items()})
        validation_path = args.output / f"validation-{episode['motion_id']}.npz"
        np.savez_compressed(validation_path, **{k: v[validation] for k, v in arrays.items()})
        heldout.append({k: torch.from_numpy(v[validation]) for k, v in arrays.items()})
        manifest["episodes"].append(
            {
                **episode,
                "path": str(path.resolve()),
                "sha256": sha(path),
                "rows": int(training.sum()),
                "source": "executed_native_teacher",
            }
        )
        split_receipt.append(
            {
                "motion_id": episode["motion_id"],
                "source_sha256": episode["sha256"],
                "training_indices": np.flatnonzero(training).tolist(),
                "validation_indices": np.flatnonzero(validation).tolist(),
                "validation_sha256": sha(validation_path),
            }
        )
    if len(heldout) < 2:
        raise ValueError("Need at least two whole-qualified clips")
    write_new(
        args.output / "split.json",
        {
            "episodes": split_receipt,
            "amendment": (
                "Retain all recorded rows of whole-qualified nonterminated episodes; "
                "replace old prefix mask."
            ),
            "limitation": "Within-clip held-out blocks; no unseen-motion or physical success claim.",
        },
    )
    manifest_path = args.output / "collection.json"
    write_new(manifest_path, manifest)
    config = {k: v for k, v in base.items() if not k.startswith(("initialize_", "resume_"))}
    config.update(
        device="cpu",
        updates=args.updates,
        batch_size=32,
        seed=91240,
        wall_cap_seconds=1800,
        save_interval=args.updates,
        prior_action_weight=1.0,
        foundation_sampling="reference_quarters",
        dataset_manifest=str(manifest_path.resolve()),
        dataset_manifest_sha256=sha(manifest_path),
        train_ids=[e["motion_id"] for e in manifest["episodes"]],
        purpose="CPU curriculum smoke; within-clip validation only",
    )
    torch.set_num_threads(2)
    decoder = make_decoder(config["teacher_checkpoint"], "cpu")

    def evaluate(model):
        model.eval()
        result = {}
        with torch.no_grad():
            for profile in ("full", "root", "navigation"):
                per_clip = []
                for e in heldout:
                    errors = []
                    for start in range(0, len(e["proprio"]), 32):
                        b = {k: v[start : start + 32] for k, v in e.items()}
                        mask = torch.zeros_like(b["control_mask"])
                        mask[:, list(PROFILES[profile])] = True
                        tokens = model.prior_step(
                            b["proprio"], b["controls"], mask & b["control_mask"]
                        )["tokens"]
                        errors.append(
                            (decoder(tokens, b["proprio"]) - b["teacher_actions"]).square().mean(-1)
                        )
                    per_clip.append(float(torch.cat(errors).mean()))
                result[profile] = {
                    "macro_action_mse": float(np.mean(per_clip)),
                    "per_clip_mse": per_clip,
                }
        return result

    torch.manual_seed(config["seed"])
    decoder = make_decoder(config["teacher_checkpoint"], "cpu")
    initial = MaskedMotionFoundation()
    results = {
        "initial": evaluate(initial),
        "physics_steps": 0,
        "claim": "Offline engineering ablation, not navigation qualification",
    }
    for arm in ("uniform", "curriculum"):
        arm_config = copy.deepcopy(config)
        if arm == "curriculum":
            arm_config["command_curriculum"] = [
                {"until_step": args.updates // 3, "weights": {"full": 1.0}},
                {"until_step": 2 * args.updates // 3, "weights": {"full": 0.25, "root": 0.75}},
                {
                    "until_step": args.updates,
                    "weights": {"full": 0.2, "root": 0.2, "navigation": 0.6},
                },
            ]
        path = args.output / f"{arm}.json"
        write_new(path, arm_config)
        train(SimpleNamespace(config=path, output=args.output / arm))
        saved = torch.load(
            args.output / arm / f"step-{args.updates:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = MaskedMotionFoundation()
        model.load_state_dict(saved["model"])
        results[arm] = evaluate(model)
        write_new(args.output / f"{arm}-evaluation.json", results[arm])
    write_new(args.output / "results.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=240)
    args = parser.parse_args()
    if not 6 <= args.updates <= 1000:
        parser.error("Smoke budget must be 6..1000 updates per arm")
    run(args)
