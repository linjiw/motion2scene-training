"""Storage maintenance must preserve scientific bytes and skip unfinished writes."""

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import compact_motion2scene_inventories as storage  # noqa: E402


def inventory(root, name, contents=b"same", complete=True):
    folder = root / name
    folder.mkdir(parents=True)
    path = folder / storage.INVENTORY
    path.write_bytes(contents)
    if complete:
        (folder / "success_manifest.json").write_text("{}")
    return path


def apply(root, plan):
    with (root / "journal.jsonl").open("w") as journal:
        return storage.compact(plan, journal, min_age_seconds=0)


def test_preserves_bytes_paths_and_existing_external_hardlinks(tmp_path):
    a = inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    c = inventory(tmp_path, "c", b"different")
    external = tmp_path / "external-copy.json"
    os.link(b, external)
    before = {path: path.read_bytes() for path in (a, b, c, external)}
    result = apply(tmp_path, storage.audit(tmp_path, [], 0))
    assert result["replacements"] == 1
    assert result["allocated_bytes_released"] == 0  # external link retains old blocks
    assert result["verified_paths"] == 3
    assert result["verified_unique_inodes"] == 2
    assert a.stat().st_ino == b.stat().st_ino != c.stat().st_ino
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert apply(tmp_path, storage.audit(tmp_path, [], 0))["replacements"] == 0


def test_reports_actually_freed_allocation(tmp_path):
    inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    allocated = b.stat().st_blocks * 512
    result = apply(tmp_path, storage.audit(tmp_path, [], 0))
    assert result["allocated_bytes_released"] == allocated


def test_excludes_active_recent_and_unfinished_files(tmp_path):
    a = inventory(tmp_path, "a")
    inventory(tmp_path, "active")
    inventory(tmp_path, "unfinished", complete=False)
    plan = storage.audit(tmp_path, ["active"], 0)
    assert [row["path"] for row in plan["files"]] == [str(a)]
    assert storage.audit(tmp_path, [], 600)["files"] == []


def test_rechecks_completion_before_apply(tmp_path):
    a = inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    plan = storage.audit(tmp_path, [], 0)
    (b.parent / "success_manifest.json").unlink()
    result = apply(tmp_path, plan)
    assert result["skipped"] == [str(b)]
    assert a.stat().st_ino != b.stat().st_ino


def test_open_writer_is_skipped_even_with_completion_marker(tmp_path):
    inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    plan = storage.audit(tmp_path, [], 0)
    with b.open("ab"):
        result = apply(tmp_path, plan)
    assert result["skipped"] == [str(b)]
    assert result["replacements"] == 0


def test_changed_source_aborts_before_any_replacement(tmp_path):
    a = inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    c = inventory(tmp_path, "c")
    plan = storage.audit(tmp_path, [], 0)
    c.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since audit"):
        apply(tmp_path, plan)
    assert a.stat().st_ino != b.stat().st_ino


def test_wrong_anchor_digest_is_rejected(tmp_path):
    a = inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    plan = storage.audit(tmp_path, [], 0)
    for row in plan["files"]:
        row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="anchor hash mismatch"):
        apply(tmp_path, plan)
    assert a.stat().st_ino != b.stat().st_ino


def test_symlinks_and_outside_root_are_rejected(tmp_path):
    a = inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    plan = storage.audit(tmp_path, [], 0)
    b.unlink()
    b.symlink_to(a)
    with pytest.raises(ValueError, match="symlink"):
        apply(tmp_path, plan)
    plan["files"][0]["path"] = "/outside/native_collision_inventory.json"
    with pytest.raises(ValueError, match="below the root"):
        apply(tmp_path, plan)


def test_different_permissions_are_not_combined(tmp_path):
    a = inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    b.chmod(0o444)
    assert apply(tmp_path, storage.audit(tmp_path, [], 0))["replacements"] == 0
    assert a.stat().st_ino != b.stat().st_ino


def test_atomic_replacement_of_one_alias_preserves_other_alias(tmp_path):
    a = inventory(tmp_path, "a")
    b = inventory(tmp_path, "b")
    apply(tmp_path, storage.audit(tmp_path, [], 0))
    temporary = b.with_suffix(".tmp")
    temporary.write_bytes(b"new capture")
    os.replace(temporary, b)
    assert a.read_bytes() == b"same"
    assert b.read_bytes() == b"new capture"
