import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

WORKER = (
    Path(__file__).resolve().parents[2] / "scripts/research/motion2scene_audit_reused_bootstrap.py"
)
spec = importlib.util.spec_from_file_location("bootstrap_archive_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def ref(path):
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def closure(tmp_path):
    repository = tmp_path / "repository"
    original = repository / worker.COLLECTOR
    source = tmp_path / "collection/source_snapshot"
    archived = source / worker.COLLECTOR
    analysis = tmp_path / "collection/analysis_source_snapshot" / worker.COLLECTOR
    for path in (original, archived, analysis):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ORIGINAL = True\n")
    dependency = {**ref(original), "snapshot": ref(archived)}
    manifest = {"dependencies": [dependency]}
    result = {"analysis_implementation": [{**ref(original), "snapshot": ref(analysis)}]}
    return manifest, result, source.parent, repository, original, archived


def test_manifest_changes_cannot_select_unbound_archive(tmp_path):
    manifest, result, collection, _, _, archived = closure(tmp_path)
    archived.write_text("ORIGINAL = False\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        worker.prepare_closure(manifest, result, collection)


def test_current_project_change_does_not_replace_old_execution(tmp_path):
    manifest, result, collection, _, original, archived = closure(tmp_path)
    original.write_text("ORIGINAL = False\n")
    source, _, bound, identity = worker.prepare_closure(manifest, result, collection)
    assert identity[original] == archived
    assert bound[archived] == manifest["dependencies"][0]["snapshot"]
    assert source == collection / "source_snapshot"


def test_analysis_code_must_match_original_execution(tmp_path):
    manifest, result, collection, _, _, _ = closure(tmp_path)
    result["analysis_implementation"][0]["sha256"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="analysis bytes differ"):
        worker.prepare_closure(manifest, result, collection)


@pytest.mark.parametrize("changed", [{"pass": 0}, {"pass": False, "cost": None}])
def test_full_row_comparison_rejects_type_or_field_changes(changed):
    with pytest.raises(ValueError, match="differs"):
        worker.require_exact(changed, {"pass": False}, "row")


@pytest.mark.parametrize(
    "operation",
    [
        "Path(target).write_text('modified')",
        "importlib.util.spec_from_file_location('innocent_alias', target).loader.exec_module("
        "importlib.util.module_from_spec(importlib.util.spec_from_file_location('innocent_alias', target)))",
        "exec(compile('VALUE = 1', target, 'exec'), {})",
        "__import__('torch')",
        "subprocess.run([sys.executable, '-c', 'print(1)'])",
        "subprocess.Popen([sys.executable, '-c', 'print(1)'])",
    ],
)
def test_guard_rejects_write_foreign_import_compile_gpu_or_launch(tmp_path, operation):
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "foreign.py"
    target.write_text("raise RuntimeError('FOREIGN CODE EXECUTED')\n")
    code = f"""
import importlib.util, json, subprocess, sys
from pathlib import Path
s = importlib.util.spec_from_file_location("worker", {str(WORKER)!r})
w = importlib.util.module_from_spec(s); s.loader.exec_module(w)
target = {str(target)!r}
w.install_guards(Path({str(source)!r}), Path({str(tmp_path / 'repo')!r}), {{}}, {{}})
try:
    {operation}
except (PermissionError, ImportError) as error:
    print(type(error).__name__)
else:
    raise AssertionError("guard did not reject")
"""
    completed = subprocess.run([sys.executable, "-I", "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert target.read_text() == "raise RuntimeError('FOREIGN CODE EXECUTED')\n"


def test_identity_overlay_reads_old_bytes_without_import_exemption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    archived = source / "original.py"
    archived.write_text("VALUE = 'old'\n")
    current = tmp_path / "repo/current.py"
    current.parent.mkdir()
    current.write_text("raise RuntimeError('CURRENT CODE EXECUTED')\n")
    code = f"""
import importlib.util, json, sys
from pathlib import Path
s = importlib.util.spec_from_file_location("worker", {str(WORKER)!r})
w = importlib.util.module_from_spec(s); s.loader.exec_module(w)
source, original, current = map(Path, {[str(source), str(archived), str(current)]!r})
reads, _, _ = w.install_guards(source, current.parent, {{original: {ref(archived)!r}}}, {{current: original}})
assert current.read_bytes() == b"VALUE = 'old'\\n"
try:
    s = importlib.util.spec_from_file_location('alias', current)
    s.loader.exec_module(importlib.util.module_from_spec(s))
except PermissionError:
    print(json.dumps(reads))
else:
    raise AssertionError('identity read became code permission')
"""
    completed = subprocess.run([sys.executable, "-I", "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [{"original": str(current), "read_from": str(archived)}]


def test_parent_rejects_changed_reuse_spec_before_launch(tmp_path, monkeypatch):
    path = tmp_path / "spec.json"
    path.write_text("{}")
    reference = ref(path)
    path.write_text('{"changed": true}')
    monkeypatch.setattr(
        worker.subprocess, "run", lambda *a, **k: pytest.fail("launched changed spec")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        worker.audit(reference, sys.executable)


def test_only_exact_archived_resolver_runs_without_bash_environment(tmp_path):
    source = tmp_path / "source"
    launcher = source / "scripts/research/run_kimodo_sonic_rollout.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/bash\nprintf 'logical\\0/fixture.usda\\0'\n")
    original = tmp_path / "repo/scripts/research/run_kimodo_sonic_rollout.sh"
    original.parent.mkdir(parents=True)
    original.write_text("#!/bin/bash\nexit 99\n")
    sentinel = tmp_path / "forbidden_environment_execution"
    bash_env = tmp_path / "bash_env"
    bash_env.write_text(f"touch {sentinel}\n")
    command = ["bash", str(original), "--scene", "logical"]
    code = f"""
import importlib.util, json, os, subprocess, sys
from pathlib import Path
s = importlib.util.spec_from_file_location("worker", {str(WORKER)!r})
w = importlib.util.module_from_spec(s); s.loader.exec_module(w)
source, archived, original = map(Path, {[str(source), str(launcher), str(original)]!r})
os.environ['BASH_ENV'] = {str(bash_env)!r}
_, _, receipts = w.install_guards(source, original.parents[2],
    {{archived: {ref(launcher)!r}}}, {{original: archived}}, [{command!r}])
command = {command!r}
result = subprocess.run(command + ['--resolve-only'], capture_output=True, timeout=15, check=False)
assert result.returncode == 0 and result.stdout == b'logical\\0/fixture.usda\\0'
for changed in (command, command + ['--resolve-only', '--extra']):
    try:
        subprocess.run(changed, capture_output=True, timeout=15, check=False)
    except PermissionError:
        pass
    else:
        raise AssertionError('unregistered launch allowed')
assert len(receipts) == 1 and receipts[0]['archived_argv'][3] == str(archived)
print('exact archived resolver only')
"""
    completed = subprocess.run([sys.executable, "-I", "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()
