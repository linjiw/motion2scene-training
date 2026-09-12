"""Build deterministic, content-addressed corpus snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def build_corpus_manifest(
    root: Path,
    *,
    dataset_id: str,
    version: str,
    created_at: str,
    includes: Iterable[str],
) -> dict:
    """Hash all files matching explicit root-relative globs.

    Explicit include patterns keep a later rollout from silently changing a raw-motion snapshot.
    The output path itself is not implicit, avoiding a self-referential hash.
    """
    root = root.resolve()
    paths: set[Path] = set()
    patterns = tuple(includes)
    if not patterns:
        raise ValueError("at least one include pattern is required")
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"include pattern must be root-relative: {pattern}")
        paths.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError("include patterns matched no files")
    for path in paths:
        if not path.is_relative_to(root):
            raise ValueError(f"resolved artifact escapes corpus root: {path}")

    artifacts = []
    for path in sorted(paths):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "motion2scene_corpus_manifest_v1",
        "dataset_id": dataset_id,
        "version": version,
        "created_at": created_at,
        "root": str(root),
        "include_patterns": list(patterns),
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(item["size_bytes"] for item in artifacts),
        "artifacts": artifacts,
    }


def verify_corpus_manifest(root: Path, manifest: dict) -> tuple[str, ...]:
    """Return every missing, escaped, resized, or rehashed artifact in a snapshot."""
    root = root.resolve()
    errors: list[str] = []
    for artifact in manifest.get("artifacts", []):
        relative = Path(str(artifact.get("path", "")))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            errors.append(f"artifact escapes corpus root: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != artifact.get("size_bytes"):
            errors.append(
                f"size mismatch for {relative}: {actual_size} != {artifact.get('size_bytes')}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != artifact.get("sha256"):
            errors.append(
                f"hash mismatch for {relative}: {actual_hash} != {artifact.get('sha256')}"
            )
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--include", action="append", required=True, dest="includes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = build_corpus_manifest(
        args.root,
        dataset_id=args.dataset_id,
        version=args.version,
        created_at=args.created_at,
        includes=args.includes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {payload['artifact_count']} artifacts / "
        f"{payload['total_size_bytes']} bytes to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
