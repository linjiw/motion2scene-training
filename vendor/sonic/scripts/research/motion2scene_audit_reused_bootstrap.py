#!/usr/bin/env python3
"""Recompute one unchanged bootstrap through its original archived collector.

This worker supplies evidence, not adoption or permission to reuse a corpus.
The caller must separately validate the same-corpus reuse declaration. All
project imports in the child come from the original SHA-bound source snapshot.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import subprocess
import sys

SCHEMA = "motion2scene_archived_bootstrap_audit_v1"
COLLECTOR = "scripts/research/motion2scene_collect_timed_schedules.py"


def artifact(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("canonical artifact reference required")
    if not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute():
        raise ValueError("absolute artifact path required")
    actual = artifact(ref["path"])
    if ref["sha256"] != actual["sha256"]:
        raise ValueError("bootstrap audit artifact hash mismatch: " + ref["path"])
    return Path(actual["path"])


def read(ref):
    return json.loads(checked(ref).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_exact(actual, expected, label):
    # JSON canonicalization also distinguishes booleans from integer labels.
    if canonical(actual) != canonical(expected):
        raise ValueError("archived bootstrap differs: " + label)


def verify_references(value):
    """Bind raw captures before trusted local pickle decoding."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            checked(value)
        else:
            for child in value.values():
                verify_references(child)
    elif isinstance(value, list):
        for child in value:
            verify_references(child)


def prepare_closure(manifest, result, collection):
    source = (collection / "source_snapshot").resolve(strict=True)
    entries = manifest["dependencies"]
    collectors = [r for r in entries if r["path"].endswith("/" + COLLECTOR)]
    if len(collectors) != 1:
        raise ValueError("one original collector required")
    repository = Path(collectors[0]["path"]).parents[2].resolve()
    bound, overlay = {}, {}
    for ref in entries:
        original = Path(ref["path"]).resolve()
        relative = original.relative_to(repository)
        snapshot = checked(ref["snapshot"])
        if snapshot != source / relative or ref["sha256"] != ref["snapshot"]["sha256"]:
            raise ValueError("source snapshot must preserve original path and bytes")
        if snapshot in bound or original in overlay:
            raise ValueError("duplicate or ambiguous original source")
        bound[snapshot] = ref["snapshot"]
        overlay[original] = snapshot
    for ref in result["analysis_implementation"]:
        previous = Path(ref["path"]).resolve()
        if previous not in overlay:
            raise ValueError("original analysis imported outside execution closure")
        if ref["sha256"] != bound[overlay[previous]]["sha256"]:
            raise ValueError("original analysis bytes differ from execution snapshot")
        checked(ref["snapshot"])
        if ref["snapshot"]["sha256"] != ref["sha256"]:
            raise ValueError("original analysis snapshot differs")
    # Native runtime declarations can contain identity-only Python assets that
    # were never imported by the collector. Do not promote them into code.
    identity = {}
    if manifest.get("runtime_assets_declaration"):
        for ref in read(manifest["runtime_assets_declaration"]):
            original = Path(ref["path"]).resolve()
            if original.suffix in {".py", ".pyc"}:
                target = overlay.get(original, original)
                if artifact(target)["sha256"] != ref["sha256"]:
                    raise ValueError("original runtime identity asset changed: " + str(original))
                identity[original] = target
    identity.update(overlay)
    return source, repository, bound, identity


def install_guards(source, repository, bound, identity, resolver_commands=()):
    """Reject external project code before execution and all filesystem writes.

    Only read-only pathlib identity reads are redirected to original bytes.
    importlib open_code and exec/compile events cannot use this exemption.
    Installed NumPy/stdlib code remains available; this is not an OS sandbox.
    """
    sys.dont_write_bytecode = True
    source, repository = Path(source).resolve(), Path(repository).resolve()
    sys.path[:] = [
        p
        for p in sys.path
        if p
        and (
            not Path(p).resolve().is_relative_to(repository)
            or Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())
        )
    ]
    sys.path[:0] = [str(source), str(source / "scripts/research")]
    blocked = {"torch", "isaaclab", "omni", "pxr", "mujoco", "onnxruntime", "joblib"}
    names = {"gear_sonic", "decoupled_wbc", "hallucination"}
    names.update(p.stem for p in bound if p.parent == source / "scripts/research")
    identity_reads, resolver_reads = [], []
    allowed_launch = [None]
    original_run = subprocess.run
    resolver_map = {}
    for command in resolver_commands:
        if command[0] != "bash" or "--resolve-only" in command:
            raise ValueError("registered physical argv required for resolver")
        original = Path(command[1]).resolve()
        archived = identity.get(original)
        if (
            archived != source / "scripts/research/run_kimodo_sonic_rollout.sh"
            or archived not in bound
        ):
            raise ValueError("exact archived launcher required for resolver")
        checked(bound[archived])
        resolver_map[tuple(command + ["--resolve-only"])] = archived
    resolver_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "PYTHONPATH", "PYTHONHOME"}
        and not k.startswith("BASH_FUNC_")
    }
    resolver_env.update(PATH="/usr/bin:/bin", CUDA_VISIBLE_DEVICES="")

    def archived_resolve_run(command, *args, **kwargs):
        key = tuple(command) if isinstance(command, list) else ()
        if (
            args
            or key not in resolver_map
            or kwargs != {"capture_output": True, "timeout": 15, "check": False}
        ):
            raise PermissionError("only exact registered nonphysical scene resolver may launch")
        archived = resolver_map[key]
        checked(bound[archived])
        actual = ["/bin/bash", "--noprofile", "--norc", str(archived), *command[2:]]
        previous = allowed_launch[0]
        allowed_launch[0] = (actual, resolver_env)
        try:
            result = original_run(
                actual, capture_output=True, timeout=15, check=False, env=resolver_env, cwd="/"
            )
        finally:
            allowed_launch[0] = previous
        resolver_reads.append(
            {
                "original_argv": command,
                "archived_argv": actual,
                "launcher": bound[archived],
                "exit_status": result.returncode,
                "stdout_hex": result.stdout.hex(),
                "stderr_hex": result.stderr.hex(),
                "scope": "SHA-identical shell --resolve-only path; exits before mkdir/Python/simulator",
            }
        )
        return result

    subprocess.run = archived_resolve_run
    allowed_open = [None]
    original_open = io.open

    def identity_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if path in identity:
                if mode != "rb":
                    raise PermissionError("original identity read must be binary and read-only")
                target = identity[path]
                identity_reads.append({"original": str(path), "read_from": str(target)})
                previous = allowed_open[0]
                allowed_open[0] = target
                try:
                    return original_open(target, mode, *args, **kwargs)
                finally:
                    allowed_open[0] = previous
        return original_open(file, mode, *args, **kwargs)

    io.open = identity_open

    trusted_roots = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
    worker_path = Path(__file__).resolve()

    def check_code(path):
        path = Path(path).resolve()
        if path.is_relative_to(source):
            if path not in bound:
                raise PermissionError("unbound archived code: " + str(path))
        elif path != worker_path and not any(path.is_relative_to(root) for root in trusted_roots):
            raise PermissionError(
                "code outside original closure or Python installation: " + str(path)
            )

    class ArchivedFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                raise ImportError("CPU archived audit forbids simulator/model import: " + fullname)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec and spec.origin not in (None, "built-in", "frozen"):
                origin = Path(spec.origin).resolve()
                if fullname.split(".")[0] in names and origin not in bound:
                    raise ImportError("project import outside original closure: " + fullname)
                check_code(origin)
            return spec

    sys.meta_path.insert(0, ArchivedFinder())

    def guard(event, args):
        if event == "subprocess.Popen":
            if (
                allowed_launch[0] is None
                or (args[1], args[3]) != allowed_launch[0]
                or args[2] != "/"
            ):
                raise PermissionError("archived audit forbids unregistered process launch")
        if event in {
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "socket.connect",
        }:
            raise PermissionError("archived audit cannot launch processes or use network")
        if event in {
            "os.remove",
            "os.rename",
            "os.mkdir",
            "os.rmdir",
            "os.link",
            "os.symlink",
            "os.truncate",
            "os.chmod",
            "os.utime",
        }:
            raise PermissionError("archived audit is read-only")
        if event == "import" and args[0].split(".")[0] in blocked:
            raise ImportError("CPU archived audit forbids simulator/model import: " + args[0])
        if event == "exec":
            filename = args[0].co_filename
            if not filename.startswith("<"):
                check_code(filename)
        if event == "compile" and isinstance(args[1], str) and not args[1].startswith("<"):
            check_code(args[1])
        if event != "open":
            return
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(x in mode for x in "wax+")) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        ):
            raise PermissionError("archived audit is read-only")
        if isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix in {".py", ".pyc"} and allowed_open[0] != path:
                check_code(path)

    sys.addaudithook(guard)
    return identity_reads, names, resolver_reads


def import_receipt(source, bound, names):
    origins = []
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if name.split(".")[0] in names or path.is_relative_to(source):
            if path not in bound:
                raise ValueError("unarchived project module executed: " + name)
            checked(bound[path])
            origins.append({"module": name, "source": bound[path]})
    return origins


def child_audit(reuse_spec_ref):
    if not sys.flags.isolated or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("isolated interpreter and hidden CUDA required")
    sys.dont_write_bytecode = True
    spec = read(reuse_spec_ref)
    if spec.get("schema") != "motion2scene_same_corpus_bootstrap_reuse_v1":
        raise ValueError("unsupported bootstrap declaration")
    result = read(spec["collection"])
    manifest = read(spec["manifest"])
    collection = checked(spec["collection"]).parent
    if checked(spec["manifest"]) != collection / "manifest.json":
        raise ValueError("original collection and manifest locations differ")
    require_exact(result["manifest"], spec["manifest"], "manifest reference")
    if manifest.get("policy") is not None or manifest.get("script_parameters") is not None:
        raise ValueError("reuse worker accepts forced bootstrap only; no learned model")
    cells, rows = manifest["cells"], result["rows"]
    if len(cells) != 7 or len(rows) != 7 or len({c["cell_id"] for c in cells}) != 7:
        raise ValueError("exactly seven distinct original bootstrap branches required")
    if any(c["timed_schedule_mode"] != "forced" for c in cells):
        raise ValueError("only forced complete bootstrap schedules can be reused")
    verify_references(rows)
    source, repository, bound, identity = prepare_closure(manifest, result, collection)
    identity_reads, names, resolver_reads = install_guards(
        source, repository, bound, identity, [cell["command"] for cell in cells]
    )
    collector = importlib.import_module("motion2scene_collect_timed_schedules")
    if Path(collector.__file__).resolve() != source / COLLECTOR:
        raise ValueError("original source_snapshot collector required")
    verified, bank, scene = collector.verify_manifest(collection, execution=False)
    require_exact(verified, manifest, "verified original manifest")
    recomputed, summary = [], []
    for cell, expected in zip(cells, rows, strict=True):
        if expected["cell_id"] != cell["cell_id"]:
            raise ValueError("original branch order differs")
        attempt_path = Path(cell["output"]) / "attempt.json"
        require_exact(artifact(attempt_path), expected["attempt"], "attempt identity")
        if json.loads(attempt_path.read_text())["exit_status"] != 0:
            raise ValueError("complete original attempt required; no recovery or resimulation")
        actual, _, _ = collector.analyze_cell(cell, manifest, bank, scene)
        actual["attempt"] = artifact(attempt_path)
        require_exact(actual, expected, "full physical row " + cell["cell_id"])
        if (
            actual["measurement_admitted"] is not True
            or actual["task_outcome_admitted"] is not True
        ):
            raise ValueError("bootstrap reuse requires complete known measurements")
        recomputed.append(actual)
        summary.append(
            {
                "cell_id": cell["cell_id"],
                "pass": actual["pass"],
                "physics_steps": actual["physics_steps"],
                "costs": actual["costs"],
                "full_row_sha256": "sha256:"
                + hashlib.sha256(canonical(actual).encode()).hexdigest(),
            }
        )
    steps = sum(r["physics_steps"] for r in recomputed)
    if steps != 8344 or steps != result["physics_steps"] or steps != spec["recorded_physics_steps"]:
        raise ValueError("original seven-branch accounting differs")
    require_exact(recomputed, rows, "all original physical rows")
    origins = import_receipt(source, bound, names)
    checked(spec["collection"])
    checked(spec["manifest"])
    return {
        "schema": SCHEMA,
        "status": "complete",
        "reuse_spec": reuse_spec_ref,
        "collection": spec["collection"],
        "manifest": spec["manifest"],
        "collector": artifact(source / COLLECTOR),
        "source_root": str(source),
        "original_dependency_count": len(bound),
        "imported_project_modules": origins,
        "identity_reads": [json.loads(r) for r in sorted({canonical(r) for r in identity_reads})],
        "nonphysical_resolver_invocations": resolver_reads,
        "all_rows_exact": True,
        "rows": summary,
        "complete_episode_count": 7,
        "pass_count": sum(r["pass"] for r in recomputed),
        "physical_failure_count": sum(not r["pass"] for r in recomputed),
        "recorded_physics_steps": steps,
        "new_physics_steps": 0,
        "original_writes": 0,
        "collector_analyze_called": False,
        "isolated_python": bool(sys.flags.isolated),
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "interpreter": artifact(sys.executable),
        "python_version": sys.version,
        "worker": artifact(__file__),
        "scope": (
            "Exact original archived-code physical audit only; adoption and same-corpus "
            "authorization remain caller requirements."
        ),
    }


def audit(reuse_spec_ref, interpreter):
    """Run archived validation in a clean CPU process; never mutate the caller."""
    checked(reuse_spec_ref)
    executable = Path(interpreter).absolute()
    if not executable.is_file():
        raise ValueError("audit interpreter does not exist")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    command = [str(executable), "-I", str(Path(__file__).resolve()), "--child"]
    completed = subprocess.run(
        command,
        input=canonical(reuse_spec_ref),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd="/",
        timeout=600,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("archived bootstrap child failed: " + completed.stderr[-6000:])
    receipt = json.loads(completed.stdout)
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "complete":
        raise ValueError("invalid archived bootstrap child receipt")
    require_exact(receipt["reuse_spec"], reuse_spec_ref, "child reuse declaration")
    require_exact(receipt["worker"], artifact(__file__), "child worker identity")
    receipt["child_command"] = command
    # Successful diagnostic text is not part of the immutable scientific receipt.
    # Nonzero child status already includes stderr in the raised exception.
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reuse-spec", type=Path)
    parser.add_argument("--reuse-sha256")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.child:
        ref = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(sys.stderr):
            receipt = child_audit(ref)
    else:
        if not args.reuse_spec or not args.reuse_sha256:
            parser.error("--reuse-spec and --reuse-sha256 are required")
        receipt = audit(
            {"path": str(args.reuse_spec.resolve()), "sha256": args.reuse_sha256}, args.python
        )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
