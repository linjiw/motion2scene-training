"""Share byte-identical immutable inventories after collection completion."""

import hashlib
import os
from pathlib import Path


def deduplicate_inventories(collections):
    """Atomic hardlinks retain every path and byte; never touch unfinished captures."""
    anchors, links = {}, []
    for collection in collections:
        collection = Path(collection)
        if not (collection / "result.json").is_file():
            raise ValueError("only completed collections can share inventory storage")
        for path in sorted(
            collection.glob("rollouts/*/trajectories/native_collision_inventory.json")
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular immutable inventory files required")
            before = path.stat()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("inventory changed during storage comparison")
            key = (before.st_dev, digest, before.st_size)
            anchor = anchors.setdefault(key, path)
            if anchor.stat().st_ino == before.st_ino:
                continue
            temporary = path.with_name(path.name + f".dedup-{os.getpid()}")
            os.link(anchor, temporary, follow_symlinks=False)
            os.replace(temporary, path)
            links.append(
                dict(
                    path=str(path),
                    anchor=str(anchor),
                    sha256=digest,
                    bytes_preserved=before.st_size,
                    allocated_bytes_released=before.st_blocks * 512 if before.st_nlink == 1 else 0,
                )
            )
    return dict(
        links=links,
        allocated_bytes_released=sum(r["allocated_bytes_released"] for r in links),
        content_or_paths_removed=False,
    )
