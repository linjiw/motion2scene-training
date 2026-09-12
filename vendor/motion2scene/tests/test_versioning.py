from __future__ import annotations

from pathlib import Path

import pytest

from motion2scene.dataset.versioning import build_corpus_manifest, verify_corpus_manifest


def test_manifest_is_sorted_and_scoped_by_explicit_patterns(tmp_path: Path) -> None:
    (tmp_path / "motions").mkdir()
    (tmp_path / "motions/b.csv").write_text("b")
    (tmp_path / "motions/a.csv").write_text("a")
    (tmp_path / "later.bin").write_text("not in snapshot")

    manifest = build_corpus_manifest(
        tmp_path,
        dataset_id="pilot",
        version="v1-m",
        created_at="2026-09-04T00:00:00-04:00",
        includes=("motions/*.csv",),
    )

    assert [item["path"] for item in manifest["artifacts"]] == [
        "motions/a.csv",
        "motions/b.csv",
    ]
    assert manifest["artifact_count"] == 2
    assert manifest["total_size_bytes"] == 2


@pytest.mark.parametrize("pattern", ("/tmp/*.csv", "../*.csv"))
def test_manifest_rejects_patterns_outside_root(tmp_path: Path, pattern: str) -> None:
    with pytest.raises(ValueError, match="root-relative"):
        build_corpus_manifest(
            tmp_path,
            dataset_id="pilot",
            version="v1",
            created_at="2026-09-04T00:00:00-04:00",
            includes=(pattern,),
        )


def test_manifest_refuses_empty_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="matched no files"):
        build_corpus_manifest(
            tmp_path,
            dataset_id="pilot",
            version="v1",
            created_at="2026-09-04T00:00:00-04:00",
            includes=("motions/*.csv",),
        )


def test_verifier_detects_mutation_after_snapshot(tmp_path: Path) -> None:
    motion = tmp_path / "motion.csv"
    motion.write_text("before")
    manifest = build_corpus_manifest(
        tmp_path,
        dataset_id="pilot",
        version="v1",
        created_at="2026-09-04T00:00:00-04:00",
        includes=("motion.csv",),
    )
    assert verify_corpus_manifest(tmp_path, manifest) == ()

    motion.write_text("after")
    errors = verify_corpus_manifest(tmp_path, manifest)
    assert any("size mismatch" in error for error in errors)
    assert any("hash mismatch" in error for error in errors)
