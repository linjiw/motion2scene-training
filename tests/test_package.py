import io
import tarfile

import pytest

from m2s_package.cli import digest, safe_extract


def test_extract_rejects_traversal_and_links(tmp_path):
    for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
        archive = tmp_path / "bad.tar"
        with tarfile.open(archive, "w") as tar:
            item = tarfile.TarInfo(name)
            item.type = kind
            item.linkname = "/tmp/elsewhere"
            tar.addfile(item)
        with pytest.raises(ValueError):
            safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_extract_and_no_overwrite(tmp_path):
    archive = tmp_path / "ok.tar"
    with tarfile.open(archive, "w") as tar:
        item = tarfile.TarInfo("data/motion.txt")
        item.size = 3
        tar.addfile(item, io.BytesIO(b"abc"))
    safe_extract(archive, tmp_path / "out")
    assert (
        digest(tmp_path / "out/data/motion.txt")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    with pytest.raises(FileExistsError):
        safe_extract(archive, tmp_path / "out")


def test_resume_extraction_only_accepts_matching_bytes(tmp_path):
    archive = tmp_path / "ok.tar"
    with tarfile.open(archive, "w") as tar:
        item = tarfile.TarInfo("motion.txt")
        item.size = 3
        tar.addfile(item, io.BytesIO(b"abc"))
    out = tmp_path / "out"
    safe_extract(archive, out)
    hashes = {"motion.txt": digest(out / "motion.txt")}
    safe_extract(archive, out, hashes)
    (out / "motion.txt").write_text("changed")
    with pytest.raises(FileExistsError):
        safe_extract(archive, out, hashes)
    assert (out / "motion.txt").read_text() == "changed"


def test_readiness_reports_missing_dependencies(tmp_path, monkeypatch):
    from m2s_package.readiness import inspect

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert inspect(tmp_path, tmp_path, "base")["ready"]
    view = inspect(tmp_path, tmp_path, "view")
    assert not view["ready"]
    assert "Missing module: matplotlib" in view["issues"]
    assert not inspect(tmp_path, tmp_path, "teacher")["ready"]


def test_invalid_budget_does_not_create_outputs(tmp_path):
    from m2s_package.cli import main

    output = tmp_path / "bad"
    with pytest.raises(SystemExit) as error:
        main(["teacher-prepare", "--output", str(output), "--iterations", "0"])
    assert error.value.code == 2
    assert not output.exists()
    assert not output.with_name("bad-parent").exists()
