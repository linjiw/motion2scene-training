#!/usr/bin/env python3
"""Read-only enforcement of a declared native runtime environment before launch."""

import json
import os
from pathlib import Path
import subprocess

from motion2scene_freeze_runtime_assets import INVENTORY_PROGRAM, bind_editable_sources
from motion2scene_timing_diagnostic import artifact, checked


def effective_cache(environ=None):
    environ = os.environ if environ is None else environ
    return (
        Path(environ.get("ISAACLAB_USD_CACHE_DIR", "~/.cache/isaaclab/usd"))
        .expanduser()
        .resolve(strict=True)
        / "g1_model_12_dex"
    )


def validate_environment(environment, assets, executable, *, environ=None):
    """Return actual checked identities; mismatches have launched no simulation."""
    environment_path = checked(Path(environment["path"]), environment["sha256"])
    expected = json.loads(environment_path.read_text())
    assets_path = checked(Path(assets["path"]), assets["sha256"])
    refs = json.loads(assets_path.read_text())
    if environment not in refs:
        raise ValueError("the explicit environment must be bound in runtime assets")
    cache = effective_cache(environ)
    if cache != Path(expected["native_usd_cache"]).resolve(strict=True):
        raise ValueError("effective runtime USD cache differs from the declared native body")
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    inventory = bind_editable_sources(
        json.loads(subprocess.check_output([str(executable), "-c", INVENTORY_PROGRAM], text=True))
    )
    if inventory["editable_sources"] != expected["editable_sources"]:
        raise ValueError("editable runtime source differs from its frozen identity")
    for key in ("python", "executable", "python_base_prefix", "packages"):
        actual_value, expected_value = inventory[key], expected[key]
        if key == "packages":
            # Distribution discovery order is not installation identity. Keep
            # duplicates, while canonicalizing all fields of each record.
            actual_value = sorted(json.dumps(row, sort_keys=True) for row in actual_value)
            expected_value = sorted(json.dumps(row, sort_keys=True) for row in expected_value)
        if actual_value != expected_value:
            raise ValueError(f"actual runtime inventory differs in {key}")
    return dict(
        schema="motion2scene_native_runtime_preflight_v1",
        environment=artifact(environment_path),
        runtime_assets=artifact(assets_path),
        effective_native_usd_cache=str(cache),
        checked_artifacts=len(refs),
        editable_sources_equal=True,
        runtime_inventory_equal=True,
        status="configuration_verified_before_launch",
        new_physics_steps=0,
    )
