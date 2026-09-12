"""Check the actual shell scene handoff without output creation or simulation."""

import hashlib
from pathlib import Path
import subprocess


def validate_scene_handoff(command, scene):
    """Resolve the exact registered argv through the launcher's nonphysical path.

    Legacy ID/package commands remain valid only if they resolve the same bound
    asset. A separate logical scene ID never substitutes for a USD basename.
    """
    if (
        not isinstance(command, list)
        or len(command) < 2
        or command[0] != "bash"
        or Path(command[1]).name != "run_kimodo_sonic_rollout.sh"
        or "--resolve-only" in command
        or command.count("--scene-usd") > 1
    ):
        raise ValueError(
            "one physical rollout command with an optional explicit scene path required"
        )
    expected = Path(scene["scene"]["path"]).resolve()
    if "--scene-usd" in command:
        index = command.index("--scene-usd")
        if index + 1 >= len(command) or Path(command[index + 1]).resolve() != expected:
            raise ValueError("explicit scene path differs from registered asset")
    # The shell appends extras last. Do not allow an override to replace the
    # scene resolved by the shell or its logical dataset identity afterward.
    for index, value in enumerate(command):
        if value == "--extra":
            if index + 1 >= len(command):
                raise ValueError("missing extra override argument")
            for item in command[index + 1].split():
                if item.lstrip("+").split("=", 1)[0] in (
                    "manager_env.config.scene_usd_path",
                    "manager_env.config.terrain_type",
                    "dataset_scene_id",
                ):
                    raise ValueError("extra overrides cannot replace the bound scene handoff")
    result = subprocess.run(
        command + ["--resolve-only"], capture_output=True, timeout=15, check=False
    )
    if result.returncode != 0:
        raise ValueError(
            "nonphysical scene resolution failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    fields = result.stdout.split(b"\0")
    if len(fields) != 3 or fields[-1] != b"":
        raise ValueError("launcher did not return the exact scene resolution receipt")
    identifier, resolved = (value.decode("utf-8") for value in fields[:2])
    if identifier != scene["scene_id"] or not resolved or Path(resolved).resolve() != expected:
        raise ValueError("launcher resolves a different scene identity or asset path")
    digest = "sha256:" + hashlib.sha256(expected.read_bytes()).hexdigest()
    if digest != scene["scene"]["sha256"]:
        raise ValueError("resolved scene asset hash differs from registration")
    return dict(
        scene_id=identifier,
        resolved_scene_usd=str(expected),
        scene_sha256=digest,
        resolution_only=True,
        output_directories_created=False,
        python_or_simulator_invoked=False,
    )
