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
