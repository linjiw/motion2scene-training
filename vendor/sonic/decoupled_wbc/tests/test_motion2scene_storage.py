from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_storage import deduplicate_inventories


def test_identical_completed_inventories_share_bytes_and_preserve_paths(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    files = []
    for i, text in enumerate(("same", "same", "different")):
        path = tmp_path / f"rollouts/{i}/trajectories/native_collision_inventory.json"
        path.parent.mkdir(parents=True)
        path.write_text(text)
        files.append(path)
    report = deduplicate_inventories([tmp_path])
    assert len(report["links"]) == 1
    assert [p.read_text() for p in files] == ["same", "same", "different"]
    assert files[0].stat().st_ino == files[1].stat().st_ino
    assert files[0].stat().st_ino != files[2].stat().st_ino
    assert deduplicate_inventories([tmp_path])["links"] == []


def test_unfinished_collection_cannot_be_deduplicated(tmp_path):
    with pytest.raises(ValueError, match="completed"):
        deduplicate_inventories([tmp_path])
