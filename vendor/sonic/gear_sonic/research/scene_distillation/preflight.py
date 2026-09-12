"""Read-only training readiness report; never starts simulation or optimization."""

import argparse
import json
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.navigation_runtime import (
    require_public_prior_qualification,
)
from gear_sonic.research.scene_distillation.train import load_episodes


def check(config):
    problems = []
    for field in ("teacher_checkpoint", "dataset_manifest"):
        value = config.get(field)
        if not value or not Path(value).is_file():
            problems.append(f"Missing {field}")
    if config.get("stage") == "navigation":
        for field in ("foundation_checkpoint",):
            value = config.get(field)
            if not value or not Path(value).is_file():
                problems.append(f"Missing {field}")
        binding = config.get("foundation_qualification")
        if not binding:
            problems.append("Missing public-prior command qualification")
        else:
            try:
                require_public_prior_qualification(binding, config["foundation_sha256"])
            except (ValueError, KeyError, OSError) as error:
                problems.append(str(error))
    if not problems:
        try:
            if sha(config["teacher_checkpoint"]) != config["teacher_sha256"]:
                raise ValueError("Teacher checkpoint changed")
            if sha(config["dataset_manifest"]) != config["dataset_manifest_sha256"]:
                raise ValueError("Dataset manifest changed")
            episodes = load_episodes(
                config["dataset_manifest"],
                config["teacher_sha256"],
                set(config["train_ids"]),
                config["stage"],
                allow_exploratory_queries=config.get("allow_exploratory_queries", False),
            )
            return {"ready": True, "stage": config["stage"], "eligible_episodes": len(episodes)}
        except (ValueError, KeyError, OSError) as error:
            problems.append(str(error))
    return {
        "ready": False,
        "stage": config.get("stage"),
        "reasons": problems,
        "optimizer_updates": 0,
        "physics_steps": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    result = check(json.loads(parser.parse_args().config.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 2)
