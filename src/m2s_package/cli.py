"""Install, verify and run the portable Motion2Scene research checkout."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def root_path():
    root = Path(os.environ.get("M2S_PACKAGE_ROOT", Path(__file__).resolve().parents[2]))
    if not (root / "manifests/catalog.json").exists():
        raise ValueError(
            "Use an editable checkout or set M2S_PACKAGE_ROOT to the data repository"
        )
    return root.resolve()


def safe_extract(archive, destination, verified_files=None):
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tar:
        pending = []
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not member.isfile():
                raise ValueError(f"Unsafe archive member: {member.name}")
            if target.exists():
                expected = (verified_files or {}).get(member.name)
                if expected is None or digest(target) != expected:
                    raise FileExistsError(
                        f"Refusing to overwrite changed/unverified {target}"
                    )
            else:
                pending.append(member)
        tar.extractall(destination, members=pending, filter="data")


def native(root, module, arguments):
    sonic = root / "vendor/sonic"
    env = dict(
        os.environ, PYTHONPATH=str(sonic), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"
    )
    subprocess.run(
        [sys.executable, "-m", module, *map(str, arguments)],
        cwd=sonic,
        env=env,
        check=True,
    )


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    sub.add_parser("verify")
    extract = sub.add_parser("unpack")
    extract.add_argument("names", nargs="*")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--cuda", action="store_true")
    doctor.add_argument("--profile", choices=["base", "view", "student", "teacher"])
    teacher = sub.add_parser("teacher-prepare")
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument(
        "--num-envs",
        type=int,
        choices=[128, 256, 512, 1024, 2048, 4096, 8192],
        default=128,
    )
    teacher.add_argument("--iterations", type=int, default=32000)
    launch = sub.add_parser("teacher-train")
    launch.add_argument("packet", type=Path)
    smoke = sub.add_parser("student-smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--updates", type=int, default=240)
    generator = sub.add_parser("generator-smoke")
    generator.add_argument("--output", type=Path, required=True)
    generator.add_argument("--updates", type=int, default=20)
    scenes = sub.add_parser("scene-tasks")
    scenes.add_argument("--output", type=Path, required=True)
    view = sub.add_parser("view")
    view.add_argument("motion", type=Path)
    view.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "teacher-prepare" and args.iterations <= 0:
        parser.error("iterations must be positive")
    if args.command == "student-smoke" and not 6 <= args.updates <= 1000:
        parser.error("student smoke updates must be 6..1000")
    if args.command == "generator-smoke" and not 1 <= args.updates <= 1000:
        parser.error("generator smoke updates must be 1..1000")
    root = root_path()
    workspace = (args.workspace or root / "workspace").resolve()
    catalog = json.loads((root / "manifests/catalog.json").read_text())
    if args.command == "catalog":
        print(json.dumps(catalog, indent=2))
    elif args.command in ("verify", "unpack"):
        names = getattr(args, "names", [])
        known = {b["name"] for b in catalog["bundles"]}
        if set(names) - known:
            raise ValueError("Unknown bundle name")
        for bundle in catalog["bundles"]:
            if names and bundle["name"] not in names:
                continue
            archive = root / bundle["archive"]
            if digest(archive) != bundle["sha256"]:
                raise ValueError(
                    f"Bundle missing or changed: {archive}; run git lfs pull"
                )
            if args.command == "unpack":
                manifest = json.loads((root / bundle["manifest"]).read_text())
                safe_extract(
                    archive,
                    workspace,
                    {f["path"]: f["sha256"] for f in manifest["files"]},
                )
                for item in manifest["files"]:
                    if digest(workspace / item["path"]) != item["sha256"]:
                        raise ValueError(f"Extracted file changed: {item['path']}")
            print(bundle["name"], "verified")
        if (
            args.command == "unpack"
            and (workspace / "vendor/sonic/gear_sonic/data").exists()
        ):
            link = root / "vendor/sonic/gear_sonic/data"
            expected = workspace / "vendor/sonic/gear_sonic/data"
            if link.is_symlink() and link.resolve() != expected.resolve():
                link.unlink()
            if not link.exists():
                link.symlink_to(
                    os.path.relpath(
                        workspace / "vendor/sonic/gear_sonic/data", link.parent
                    ),
                    target_is_directory=True,
                )
    elif args.command == "doctor":
        from m2s_package.readiness import inspect

        result = inspect(
            root,
            workspace,
            args.profile or ("teacher" if args.cuda else "base"),
            args.cuda,
        )
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(1)
    elif args.command == "teacher-prepare":
        parent = args.output.resolve().with_name(args.output.name + "-parent")
        parent.mkdir(parents=True, exist_ok=False)
        (parent / "inputs").mkdir()
        shutil.copy2(
            workspace / "models/sonic_release.pt", parent / "inputs/sonic_release.pt"
        )
        original = workspace / "m2s-sonic-teacher-student-v1-20260911"
        command = json.loads((original / "command.json").read_text())
        command["cwd"] = str(root / "vendor/sonic")
        command["argv"][0] = sys.executable
        save(parent / "command.json", command)
        shutil.copy2(original / "plan.json", parent / "plan.json")
        native(
            root,
            "gear_sonic.research.hindsight_training.prepare_repaired",
            [
                "--dataset",
                workspace / "m2s-sonic-repaired-v1_1-20260912",
                "--parent",
                parent,
                "--output",
                args.output.resolve(),
                "--num-envs",
                args.num_envs,
                "--iterations",
                args.iterations,
            ],
        )
        native(
            root,
            "gear_sonic.research.hindsight_training.validate",
            [args.output.resolve()],
        )
    elif args.command == "teacher-train":
        native(
            root,
            "gear_sonic.research.hindsight_training.launch",
            [args.packet.resolve()],
        )
    elif args.command == "student-smoke":
        prep = args.output.resolve().with_name(args.output.name + "-inputs")
        prep.mkdir(parents=True, exist_ok=False)
        source = workspace / "m2s-bfm-100motions-20260912"
        manifest = json.loads((source / "collection-0/collection.json").read_text())
        for episode in manifest["episodes"]:
            episode["path"] = str(source / "collection-0" / Path(episode["path"]).name)
        save(prep / "collection.json", manifest)
        config = json.loads((source / "fit-2/config.json").read_text())
        config["teacher_checkpoint"] = str(workspace / "models/previous_teacher.pt")
        save(prep / "base.json", config)
        save(
            prep / "relocation.json",
            {
                "source_manifest_sha256": digest(
                    source / "collection-0/collection.json"
                ),
                "note": "New local manifest; episode bytes and hashes unchanged",
            },
        )
        native(
            root,
            "gear_sonic.research.scene_distillation.curriculum_smoke",
            [
                "--base-config",
                prep / "base.json",
                "--collection",
                prep / "collection.json",
                "--output",
                args.output.resolve(),
                "--updates",
                args.updates,
            ],
        )
    elif args.command == "generator-smoke":
        sys.path.insert(0, str(root / "vendor/sonic"))
        from m2s_package.generator import fit

        fit(
            workspace / "m2s-hindsight-events-100-20260911/labels.pt",
            args.output.resolve(),
            args.updates,
        )
    elif args.command == "scene-tasks":
        native(
            root,
            "gear_sonic.research.scene_distillation.tasks",
            [
                "--dataset",
                workspace / "m2s-hindsight-dataset-v1-20260911",
                "--output",
                args.output.resolve(),
            ],
        )
    elif args.command == "view":
        from m2s_package.view import render

        render(args.motion.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
