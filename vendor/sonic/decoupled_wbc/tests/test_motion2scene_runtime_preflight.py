"""A different effective robot cache/source cannot be credited as the frozen runtime."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import motion2scene_preflight_runtime as preflight  # noqa: E402
from motion2scene_timing_diagnostic import artifact, write_new  # noqa: E402


def fixture(tmp_path, monkeypatch):
    cache = tmp_path / "cache/g1_model_12_dex"
    cache.mkdir(parents=True)
    asset = cache / "main.usd"
    asset.write_bytes(b"synthetic native body")
    inventory = dict(
        python="synthetic",
        executable="/synthetic/python",
        python_base_prefix="/synthetic",
        packages=[{"name": "same", "location": "one"}, {"name": "same", "location": "two"}],
        editable_sources={"/synthetic/source": {"commit": "a"}},
    )
    write_new(tmp_path / "environment.json", dict(**inventory, native_usd_cache=str(cache)))
    env_ref = artifact(tmp_path / "environment.json")
    write_new(tmp_path / "assets.json", [env_ref, artifact(asset)])
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda *a, **k: json.dumps(inventory))
    monkeypatch.setattr(preflight, "bind_editable_sources", lambda x: x)
    return env_ref, artifact(tmp_path / "assets.json"), cache, inventory


def test_canonical_equivalent_cache_is_accepted_without_physics(tmp_path, monkeypatch):
    env, assets, cache, _ = fixture(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    alias.symlink_to(cache.parent, target_is_directory=True)
    result = preflight.validate_environment(
        env, assets, "/synthetic/python", environ={"ISAACLAB_USD_CACHE_DIR": str(alias)}
    )
    assert result["status"] == "configuration_verified_before_launch"
    assert result["new_physics_steps"] == 0


def test_other_existing_cache_is_rejected(tmp_path, monkeypatch):
    env, assets, _, _ = fixture(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="effective runtime USD cache"):
        preflight.validate_environment(
            env, assets, "/synthetic/python", environ={"ISAACLAB_USD_CACHE_DIR": str(other)}
        )


def test_changed_native_body_is_rejected(tmp_path, monkeypatch):
    env, assets, cache, _ = fixture(tmp_path, monkeypatch)
    (cache / "main.usd").write_bytes(b"different geometry")
    with pytest.raises(ValueError):
        preflight.validate_environment(
            env, assets, "/synthetic/python", environ={"ISAACLAB_USD_CACHE_DIR": str(cache.parent)}
        )


def test_new_editable_change_is_rejected(tmp_path, monkeypatch):
    env, assets, cache, inventory = fixture(tmp_path, monkeypatch)
    inventory["editable_sources"]["/synthetic/source"]["commit"] = "changed"
    with pytest.raises(ValueError, match="editable runtime source"):
        preflight.validate_environment(
            env, assets, "/synthetic/python", environ={"ISAACLAB_USD_CACHE_DIR": str(cache.parent)}
        )


def test_equal_name_distribution_order_is_not_a_runtime_change(tmp_path, monkeypatch):
    env, assets, cache, inventory = fixture(tmp_path, monkeypatch)
    inventory["packages"].reverse()
    result = preflight.validate_environment(
        env, assets, "/synthetic/python", environ={"ISAACLAB_USD_CACHE_DIR": str(cache.parent)}
    )
    assert result["runtime_inventory_equal"] is True
    inventory["packages"].pop()
    with pytest.raises(ValueError, match="inventory differs in packages"):
        preflight.validate_environment(
            env, assets, "/synthetic/python", environ={"ISAACLAB_USD_CACHE_DIR": str(cache.parent)}
        )
