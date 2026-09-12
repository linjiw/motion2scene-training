"""Deduplicate completed collision snapshots without changing their paths or bytes.

This maintenance command is deliberately outside the hash-bound simulation source
closure. Shared inventories must remain immutable: collect reruns into fresh output
directories, and use atomic replacement rather than in-place edits for shared files.
"""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid

INVENTORY = "native_collision_inventory.json"
MARKERS = ("success_manifest.json", "result.json", "release.json")


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completion_marker(path, root):
    for parent in path.parents:
        if parent == root:
            break
        for name in MARKERS:
            marker = parent / name
            if marker.is_file():
                return marker
    return None


def writable_inodes():
    """Linux /proc check, in addition to completion, age and explicit exclusions."""
    result = set()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / "fdinfo").iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                fields = dict(line.split(":", 1) for line in descriptor.read_text().splitlines())
                if int(fields["flags"].strip(), 8) & os.O_ACCMODE == os.O_RDONLY:
                    continue
                info = (process / "fd" / descriptor.name).stat()
            except (OSError, KeyError, ValueError):
                continue
            result.add((info.st_dev, info.st_ino))
    return result


def checked_path(row, root, excluded, cutoff, writers):
    path = Path(row["path"])
    if path.name != INVENTORY or not path.is_relative_to(root):
        raise ValueError(f"not an inventory below the root: {path}")
    if path.resolve() != path:
        raise ValueError(f"symlink inventory or ancestor: {path}")
    if any(path.is_relative_to(root / name) for name in excluded):
        return None
    info = path.lstat()
    expected = row["dev"], row["ino"], row["size"], row["mtime_ns"]
    if not stat.S_ISREG(info.st_mode) or signature(info) != expected:
        raise ValueError(f"inventory changed since audit: {path}")
    if info.st_mtime > cutoff or (info.st_dev, info.st_ino) in writers:
        return None
    if completion_marker(path, root) is None:
        return None
    return path


def audit(root, excluded, min_age_seconds):
    started = time.time()
    writers = writable_inodes()
    rows, hashes = [], {}
    last_progress = started
    for folder, directories, files in os.walk(root):
        directories[:] = sorted(
            name
            for name in directories
            if not name.startswith(".")
            and not any((Path(folder) / name).is_relative_to(root / item) for item in excluded)
        )
        if INVENTORY not in files:
            continue
        path = Path(folder) / INVENTORY
        info = path.lstat()
        row = dict(
            path=str(path),
            dev=info.st_dev,
            ino=info.st_ino,
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
            blocks=info.st_blocks,
            nlink=info.st_nlink,
        )
        if checked_path(row, root, excluded, started - min_age_seconds, writers) is None:
            continue
        key = signature(info)
        if key not in hashes:
            hashes[key] = digest_file(path)
            if signature(path.stat()) != key:
                raise ValueError(f"inventory changed while hashing: {path}")
        rows.append(dict(row, sha256=hashes[key]))
        if time.time() - last_progress >= 20:
            print(json.dumps(dict(audited_paths=len(rows), hashed_inodes=len(hashes))), flush=True)
            last_progress = time.time()
    return dict(root=str(root), excluded=excluded, started=started, files=rows)


def compact(plan, journal, min_age_seconds=600):
    """Apply a hash audit, journal each replacement, then hash-check all kept paths.

    The caller must exclude active experiment directories and hold the root lock.
    Recent, open-for-write and unfinished inventories are also skipped. Arbitrary
    noncooperating in-place writers cannot be made safe by a maintenance lock.
    """
    root = Path(plan["root"]).resolve()
    excluded = plan.get("excluded", [])
    cutoff = time.time() - min_age_seconds
    writers = writable_inodes()
    groups = defaultdict(list)
    skipped = []
    # Validate every source before the first mutation; never trust filenames as hashes.
    for row in plan["files"]:
        path = checked_path(row, root, excluded, cutoff, writers)
        if path is None:
            skipped.append(row["path"])
            continue
        info = path.stat()
        groups[(row["dev"], row["sha256"], info.st_mode, info.st_uid, info.st_gid)].append(row)
    released = replacements = verified = 0
    verification_hashes = {}
    for group in groups.values():
        anchor = Path(group[0]["path"])
        expected_hash = group[0]["sha256"]
        if digest_file(anchor) != expected_hash:
            raise ValueError(f"anchor hash mismatch: {anchor}")
        anchor_signature = signature(anchor.stat())
        for row in group:
            path = checked_path(row, root, excluded, cutoff, writers)
            if path is None:
                raise ValueError(f"inventory no longer eligible: {row['path']}")
            before = path.stat()
            if signature(anchor.stat()) != anchor_signature:
                raise ValueError(f"anchor changed: {anchor}")
            if (before.st_dev, before.st_ino) == anchor_signature[:2]:
                continue
            # Reuse the audit hash only while inode, size and nanosecond mtime
            # still match. The complete file was hashed in the read-only audit.
            temporary = path.with_name(f".{INVENTORY}.compact-{uuid.uuid4().hex}")
            record = dict(
                path=str(path),
                anchor=str(anchor),
                sha256=expected_hash,
                original=row,
                allocated_bytes_released=(before.st_blocks * 512 if before.st_nlink == 1 else 0),
            )
            journal.write(json.dumps(dict(event="intent", **record)) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            try:
                os.link(anchor, temporary, follow_symlinks=False)
                if signature(path.stat()) != signature(before):
                    raise ValueError(f"inventory changed before replacement: {path}")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            released += record["allocated_bytes_released"]
            replacements += 1
            journal.write(json.dumps(dict(event="replaced", **record)) + "\n")
            journal.flush()
        # Verify every retained pathname, hashing shared bytes once per inode.
        for row in group:
            path = Path(row["path"])
            info = path.stat()
            key = signature(info)
            if key not in verification_hashes:
                verification_hashes[key] = digest_file(path)
            if verification_hashes[key] != row["sha256"] or info.st_size != row["size"]:
                raise ValueError(f"post-compaction verification failed: {path}")
            if signature(path.stat()) != key:
                raise ValueError(f"inventory changed during verification: {path}")
            verified += 1
        print(
            json.dumps(
                dict(
                    verified_paths=verified,
                    replacements=replacements,
                    released_gib=round(released / 2**30, 3),
                )
            ),
            flush=True,
        )
    return dict(
        root=str(root),
        excluded=excluded,
        skipped=skipped,
        verified_paths=verified,
        replacements=replacements,
        allocated_bytes_released=released,
        verified_unique_inodes=len(verification_hashes),
        content_or_paths_removed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--exclude", action="append", default=[], help="active directory below root"
    )
    parser.add_argument("--min-age-seconds", type=int, default=600)
    parser.add_argument("--plan", type=Path, help="reuse an existing read-only hash audit")
    parser.add_argument("--report", type=Path, required=True, help="new output directory")
    parser.add_argument("--apply", action="store_true", help="default only audits")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.min_age_seconds < 0:
        parser.error("minimum age cannot be negative")
    for name in args.exclude:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            parser.error("exclusions must be relative directories below root")
    args.report.mkdir(parents=True, exist_ok=False)
    with (root / ".inventory-storage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.plan:
            plan = json.loads(args.plan.read_text())
            if Path(plan["root"]) != root:
                parser.error("audit belongs to a different root")
            plan["excluded"] = sorted(set(plan.get("excluded", []) + args.exclude))
        else:
            plan = audit(root, args.exclude, args.min_age_seconds)
        (args.report / "audit.json").write_text(json.dumps(plan, indent=2) + "\n")
        if args.apply:
            with (args.report / "journal.jsonl").open("x") as journal:
                result = compact(plan, journal, args.min_age_seconds)
            (args.report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
