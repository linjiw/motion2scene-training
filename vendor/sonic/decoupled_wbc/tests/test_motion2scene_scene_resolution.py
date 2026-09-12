"""Exercise the real launcher's pre-output, nonphysical resolution path."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
from motion2scene_scene_resolution import validate_scene_handoff  # noqa: E402


@pytest.fixture
def inputs(tmp_path):
    scene = tmp_path / "original file.usda"
    scene.write_text("Not a simulator scene: resolution test only")
    unused = tmp_path / "never_invoked_python"
    marker = tmp_path / "interpreter_was_invoked"
    unused.write_text("#!/bin/sh\ntouch " + str(marker) + "\nexit 99\n")
    unused.chmod(0o755)
    motion = tmp_path / "motion.pkl"
    checkpoint = tmp_path / "checkpoint.pt"
    motion.write_text("Never loaded")
    checkpoint.write_text("Never loaded")
    definition = dict(
        scene_id="logical_alias",
        scene=dict(
            path=str(scene), sha256="sha256:" + hashlib.sha256(scene.read_bytes()).hexdigest()
        ),
    )
    command = [
        "bash",
        str(ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"),
        "--scene",
        definition["scene_id"],
        "--scene-package",
        str(tmp_path),
        "--motion",
        str(motion),
        "--checkpoint",
        str(checkpoint),
        "--python",
        str(unused),
        "--out",
        str(tmp_path / "uncreated"),
    ]
    return command, definition, marker


def test_bad_alias_rejected_before_output_or_interpreter(inputs):
    command, scene, marker = inputs
    with pytest.raises(ValueError, match="not found"):
        validate_scene_handoff(command, scene)
    assert not Path(command[command.index("--out") + 1]).exists()
    assert not marker.exists()


def test_explicit_path_keeps_logical_id_without_output_or_auto_inference(inputs):
    command, scene, marker = inputs
    command += ["--scene-usd", scene["scene"]["path"]]
    result = validate_scene_handoff(command, scene)
    assert result["scene_id"] == "logical_alias"
    assert result["resolved_scene_usd"] == scene["scene"]["path"]
    assert result["scene_sha256"] == scene["scene"]["sha256"]
    assert not Path(command[command.index("--out") + 1]).exists()
    assert not marker.exists()  # Default max-steps=auto must not invoke Python.


def test_legacy_command_requires_resolved_asset_equality(inputs):
    command, scene, marker = inputs
    path = Path(scene["scene"]["path"])
    path = path.rename(path.parent / "logical_alias.usda")
    scene["scene"]["path"] = str(path)
    assert validate_scene_handoff(command, scene)["scene_id"] == "logical_alias"
    assert not marker.exists()


def test_existing_wrong_basename_file_cannot_replace_bound_scene(inputs):
    command, scene, marker = inputs
    wrong = Path(scene["scene"]["path"]).parent / "logical_alias.usda"
    wrong.write_text("Different scene")
    with pytest.raises(ValueError, match="different scene identity or asset path"):
        validate_scene_handoff(command, scene)
    assert not marker.exists()


@pytest.mark.parametrize(
    "change",
    [
        "wrong_explicit",
        "duplicate",
        "hash",
        "scene_override",
        "logical_override",
        "terrain_override",
        "resolve_flag",
    ],
)
def test_conflicting_scene_handoffs_are_rejected(inputs, change):
    command, scene, marker = inputs
    command += ["--scene-usd", scene["scene"]["path"]]
    if change == "wrong_explicit":
        command[-1] += ".missing"
    elif change == "duplicate":
        command += ["--scene-usd", scene["scene"]["path"]]
    elif change == "hash":
        scene["scene"]["sha256"] = "sha256:" + "0" * 64
    elif change == "resolve_flag":
        command += ["--resolve-only"]
    else:
        key = {
            "scene_override": "manager_env.config.scene_usd_path",
            "logical_override": "dataset_scene_id",
            "terrain_override": "manager_env.config.terrain_type",
        }[change]
        command += ["--extra", "++" + key + "=different"]
    with pytest.raises(ValueError):
        validate_scene_handoff(command, scene)
    assert not marker.exists()
    assert not Path(command[command.index("--out") + 1]).exists()


def test_actual_shell_explicit_path_has_no_legacy_fallback(inputs):
    command, scene, marker = inputs
    command += ["--scene-usd", scene["scene"]["path"] + ".missing", "--resolve-only"]
    result = subprocess.run(command, capture_output=True, check=False)
    assert result.returncode == 2
    assert b"missing required path" in result.stderr
    assert not marker.exists()


def test_collector_emits_explicit_path_for_different_logical_scene_id(tmp_path):
    from decoupled_wbc.tests.test_motion2scene_timed_schedule_collector import (
        args_fixture,
        collector,
    )

    args = args_fixture(tmp_path)
    scene = json.loads(args.scene_definition.read_text())
    scene["scene_id"] = "primary_empty_bootstrap"
    args.scene_definition.write_text(json.dumps(scene))
    collector.prepare(args)
    manifest, bank, definition = collector.verify_manifest(args.out)
    for cell in manifest["cells"]:
        command = cell["command"]
        assert command[command.index("--scene") + 1] == "primary_empty_bootstrap"
        assert command[command.index("--scene-usd") + 1] == str(
            Path(scene["scene"]["path"]).resolve()
        )
        assert not Path(cell["output"]).exists()
        legacy = dict(
            cell,
            command=command[: command.index("--scene-usd")]
            + command[command.index("--scene-usd") + 2 :],
        )
        with pytest.raises(ValueError, match="not found"):
            collector.validate_collection_context(legacy, manifest, bank, definition)
