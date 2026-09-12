#!/usr/bin/env python3
"""Bind existing controlled-ladder CSV/SONIC pairs with conversion provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    converter_path = (
        Path(candidates["source_repo"])
        / "gear_sonic/dataset_generation/kimodo_motion_adapter.py"
    )
    candidate_hash = sha256(args.candidates)
    written = 0
    for ladder in candidates["ladders"]:
        for level in ladder["levels"]:
            csv_path = Path(level["csv"])
            motion_path = Path(level["sonic_motion"])
            if sha256(csv_path) != level["csv_sha256"]:
                raise ValueError(f"CSV hash mismatch: {csv_path}")
            if sha256(motion_path) != level["sonic_motion_sha256"]:
                raise ValueError(f"SONIC motion hash mismatch: {motion_path}")
            manifest = motion_path.with_suffix(motion_path.suffix + ".manifest.json")
            payload = {
                "schema_version": "motion2scene_sonic_conversion_provenance_v1",
                "converter": "gear_sonic.dataset_generation.kimodo_motion_adapter",
                "converter_source": {
                    "path": str(converter_path),
                    "sha256": sha256(converter_path),
                },
                "motion_key": level["motion_key"],
                "source_fps": 30.0,
                "canonicalize_horizontal_origin": True,
                "scene_start_xyz": [0.0, 0.0, 0.0],
                "scene_yaw": 0.0,
                "input": {"path": str(csv_path), "sha256": sha256(csv_path)},
                "output": {"path": str(motion_path), "sha256": sha256(motion_path)},
                "candidate_manifest": {
                    "path": str(args.candidates.resolve()),
                    "sha256": candidate_hash,
                },
            }
            temporary = manifest.with_suffix(manifest.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, manifest)
            written += 1
    print(f"bound {written} motion conversions to {args.candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
