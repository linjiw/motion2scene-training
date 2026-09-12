#!/usr/bin/env python3
"""Freeze native G1 assets, dynamic config sources and the installed runtime inventory."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import candidate, closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402


def native_source_assets(root, cache):
    """Follow the actual G1 URDF and motion-loader MJCF mesh references."""
    description = root / "gear_sonic/data/assets/robot_description"
    urdf = description / "urdf/g1/main.urdf"
    mjcf = description / "mjcf/g1_29dof_rev_1_0.xml"
    paths = {urdf, mjcf}
    for mesh in ET.parse(urdf).getroot().iter("mesh"):
        name = mesh.attrib["filename"]
        if name.startswith("package://robot_description/"):
            path = description / name.removeprefix("package://robot_description/")
        elif "://" in name:
            raise ValueError("unresolved remote native mesh cannot be frozen")
        else:
            path = urdf.parent / name
        paths.add(path.resolve(strict=True))
    tree = ET.parse(mjcf).getroot()
    compiler = tree.find("compiler")
    mesh_dir = mjcf.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    for mesh in tree.iter("mesh"):
        if "file" in mesh.attrib:
            paths.add((mesh_dir / mesh.attrib["file"]).resolve(strict=True))
    for item in (urdf, mjcf):
        if list(ET.parse(item).getroot().iter("include")):
            raise ValueError("new native include requires explicit dependency traversal")
    if not (cache / "main.usd").is_file():
        raise ValueError("qualified native converted G1 cache is required")
    # USD layers, conversion settings and asset hash are all part of the imported body.
    paths.update(p.resolve() for p in cache.rglob("*") if p.is_file())
    return {p.resolve(strict=True) for p in paths}


def validate_usd_dependencies(cache, receipt):
    """Require all composed geometry layers/assets; declare renderer MDL outside scope."""
    if not isinstance(receipt, dict) or set(receipt) != {"layers", "assets", "unresolved"}:
        raise ValueError("explicit composed USD dependency receipt required")
    paths = {Path(value).resolve(strict=True) for value in receipt["layers"] + receipt["assets"]}
    if (cache / "main.usd").resolve(strict=True) not in paths:
        raise ValueError("actual converted root layer is missing from composed dependencies")
    unresolved = sorted(set(receipt["unresolved"]))
    if any(not value.lower().endswith(".mdl") for value in unresolved):
        raise ValueError("unresolved non-MDL USD dependency could change physical geometry")
    return paths, dict(
        resolved_layers=receipt["layers"],
        resolved_assets=receipt["assets"],
        renderer_mdl_dependencies_outside_scope=unresolved,
        renderer_reproduction_claim=False,
        scope="composed physical USD geometry; MDL shading is outside the ideal PhysX ray/contact protocol",
    )


USD_DEPENDENCY_PROGRAM = """
from pxr import Sdf, UsdUtils
import json, sys
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(sys.argv[1]))
print(json.dumps(dict(layers=[layer.realPath for layer in layers], assets=assets, unresolved=unresolved)))
"""


def composed_usd_dependencies(cache, executable, inventory):
    """Read installed USD libraries on CPU, without starting Kit or a simulator."""
    candidates = set()
    for row in inventory["packages"]:
        location = Path(row["location"]) / "isaacsim/extscache"
        candidates.update(
            path.resolve() for path in location.glob("omni.usd.libs-*") if path.is_dir()
        )
    if len(candidates) != 1:
        raise ValueError(
            "exactly one installed native USD library set is required for dependency audit"
        )
    library = candidates.pop()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(library) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(library / "bin"),
            str(Path(inventory["python_base_prefix"]) / "lib"),
            environment.get("LD_LIBRARY_PATH", ""),
        ]
    )
    completed = subprocess.run(
        [executable, "-c", USD_DEPENDENCY_PROGRAM, str(cache / "main.usd")],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    paths, receipt = validate_usd_dependencies(cache, json.loads(completed.stdout))
    receipt["usd_library_root"] = str(library)
    receipt["audit_warnings"] = completed.stderr
    return paths, receipt


def editable_source_identity(source):
    """Bind the source commit plus every nonignored working-tree modification."""
    source = Path(source).resolve(strict=True)

    def git(*args, check=True):
        return subprocess.run(["git", "-C", str(source), *args], capture_output=True, check=check)

    result = git("rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        # A local editable directory without Git still needs exact file identities.
        paths = sorted(p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        return dict(
            source_directory=str(source),
            git_repository=None,
            git_commit=None,
            non_git_source_files=[artifact(path) for path in paths],
            dirty_files=[],
            scope="all existing non-bytecode files in non-Git editable source",
        )
    repository = Path(result.stdout.decode().strip()).resolve()
    prefix = str(source.relative_to(repository))
    commit = git("rev-parse", "HEAD").stdout.decode().strip()

    # Query from the repository root: diff includes both index/worktree changes;
    # --no-renames preserves removed and added names explicitly.
    def files(*args):
        output = subprocess.check_output(["git", "-C", str(repository), *args, "--", prefix])
        return {value.decode() for value in output.split(b"\0") if value}

    changed = files("diff", "--name-only", "--no-renames", "-z", "HEAD")
    changed |= files("ls-files", "--others", "--exclude-standard", "-z")
    dirty = []
    for name in sorted(changed):
        path = repository / name
        if path.is_file():
            dirty.append(dict(relative_path=name, deleted=False, **artifact(path)))
        elif not path.exists():
            dirty.append(dict(relative_path=name, deleted=True))
        else:
            raise ValueError("changed editable source must resolve to a file or explicit deletion")
    return dict(
        source_directory=str(source),
        git_repository=str(repository),
        git_commit=commit,
        source_subdirectory=prefix,
        dirty_files=dirty,
        scope=(
            "Git HEAD plus staged/unstaged and nonignored untracked contents within editable source; "
            "ignored files and OS binaries are outside this identity"
        ),
    )


def bind_editable_sources(inventory):
    """Keep the actual editable target, not only site-packages distribution metadata."""
    sources = {}
    for row in inventory["packages"]:
        direct = row.get("direct_url")
        if not isinstance(direct, dict) or direct.get("dir_info", {}).get("editable") is not True:
            continue
        uri = urlparse(direct.get("url", ""))
        if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
            raise ValueError("editable package requires a resolvable local source URL")
        source = Path(unquote(uri.path)).resolve(strict=True)
        if str(source) not in sources:
            sources[str(source)] = editable_source_identity(source)
        row["editable_source_identity"] = str(source)
    inventory["editable_sources"] = sources
    return inventory


def dynamic_sources(root, release_config):
    configs = sorted((root / "gear_sonic/config").rglob("*.yaml")) + [release_config]
    seeds, unresolved = {root / "gear_sonic/eval_agent_trl.py"}, set()
    replacements = (
        ("groot.rl.trl.", "gear_sonic.trl."),
        ("groot.rl.envs.", "gear_sonic.envs."),
        ("groot.rl.utils.", "gear_sonic.utils."),
        ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
        ("groot.rl.agents.", "gear_sonic.trl."),
    )
    for path in configs:
        raw = path.read_text()
        for old, new in replacements:
            raw = raw.replace(old, new)
        for target in re.findall(r"_target_:\s*['\"]?([\w.]+)", raw):
            if not target.startswith("gear_sonic."):
                continue
            parts = target.split(".")
            found = None
            while len(parts) > 1 and found is None:
                found = candidate(".".join(parts), root)
                parts.pop()
            if found:
                seeds.add(found)
            else:
                unresolved.add(target)
    return closure(seeds), configs, sorted(unresolved)


INVENTORY_PROGRAM = """
import importlib.metadata as metadata
import json, platform, sys
from pathlib import Path
rows = []
for dist in metadata.distributions():
    direct = dist.read_text('direct_url.json')
    rows.append(dict(name=dist.metadata['Name'], version=dist.version,
                     location=str(Path(dist.locate_file('')).resolve()),
                     direct_url=json.loads(direct) if direct else None))
print(json.dumps(dict(python=sys.version, python_base_prefix=sys.base_prefix,
                      executable=str(Path(sys.executable).resolve()),
                      platform=platform.platform(), packages=sorted(rows, key=lambda r: r['name']))))
"""


def create(collection, out):
    manifest = json.loads(collection.read_text())
    for ref in manifest["dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    cache = Path(os.environ.get("ISAACLAB_USD_CACHE_DIR", "~/.cache/isaaclab/usd")).expanduser()
    cache = cache.resolve(strict=True) / "g1_model_12_dex"
    request = json.loads(
        checked(Path(manifest["request"]["path"]), manifest["request"]["sha256"]).read_text()
    )
    config = Path(request["controller"]["path"]).parent / "config.yaml"
    source, configs, unresolved = dynamic_sources(ROOT, config)
    assets = native_source_assets(ROOT, cache)
    executable = manifest["implementation"]["python"]
    inventory = json.loads(
        subprocess.check_output([executable, "-c", INVENTORY_PROGRAM], text=True)
    )
    inventory = bind_editable_sources(inventory)
    usd_paths, usd_receipt = composed_usd_dependencies(cache, executable, inventory)
    assets |= usd_paths
    driver = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "environment.json",
        dict(
            **inventory,
            gpu_driver=driver,
            native_usd_cache=str(cache),
            effective_environment={
                "ISAACLAB_USD_CACHE_DIR": str(cache.parent),
                "ISAACLAB_USD_CACHE_DIR_raw": os.environ.get("ISAACLAB_USD_CACHE_DIR"),
            },
            usd_dependencies=usd_receipt,
            preflight_enforced=False,
            preflight_requirement=(
                "Before primary execution, resolve the command process's effective USD cache and compare "
                "with this declaration; recheck exact editable-source identities and native artifact hashes. "
                "The current snapshot does not enforce the environment in the running collector."
            ),
            scope=(
                "package/direct-URL/editable-source identity and physical USD dependency snapshot; "
                "MDL rendering, ignored source files and binary-identical OS reproduction are not claimed"
            ),
        ),
    )
    existing = {Path(ref["path"]).resolve() for ref in manifest["runtime_artifacts"]}
    additional = (assets | source | set(configs)) - existing
    additional.add(out / "environment.json")
    refs = [artifact(path) for path in sorted(additional)]
    write_new(out / "runtime_assets.json", refs)
    write_new(
        out / "result.json",
        dict(
            schema="motion2scene_native_runtime_freeze_v1",
            source_collection=artifact(collection),
            implementation=artifact(Path(__file__)),
            runtime_assets=artifact(out / "runtime_assets.json"),
            native_asset_count=len(assets),
            dynamic_source_count=len(source),
            config_count=len(configs),
            additional_artifact_count=len(refs),
            unresolved_local_config_targets=unresolved,
            adopted=False,
            new_physics_steps=0,
            editable_source_count=len(inventory["editable_sources"]),
            composed_usd_layer_count=len(usd_receipt["resolved_layers"]),
            renderer_mdl_dependencies_outside_scope=usd_receipt[
                "renderer_mdl_dependencies_outside_scope"
            ],
            effective_environment_enforced=False,
            scope="review snapshot before final protocol adoption; sources must remain hash-identical",
        ),
    )
    print(
        json.dumps(
            dict(artifacts=len(refs), native_assets=len(assets), unresolved_targets=unresolved)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    create(args.collection, args.out)
