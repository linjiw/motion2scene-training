# Collision inventory storage maintenance

`native_collision_inventory.json` captures composed collision geometry, including
large mesh arrays. Independent captures and portable exports frequently repeat
exactly the same bytes. Keep the scientific records and their existing SHA-256
identities; share identical completed inventories through filesystem hardlinks.
Files with different bytes remain separate, even if their scenes look equivalent.

## September 10, 2026 cleanup

The completed [cleanup receipt](/home/linjiw/research-data/groot-wbc/.storage-audits/collision-inventories-20260910/result.json)
records **358,761,410,560 bytes (334.123 GiB) reclaimed**, 2,594 pathname
replacements, and 3,804 retained inventory paths verified against their original
SHA-256 hashes. Those eligible paths now share 334 unique inodes. The hash audit,
replacement journal, tool snapshot, and before/after storage snapshots are saved
beside the receipt.

Across the data root, collision inventories occupied about **71 GiB** after
cleanup, and the entire research directory occupied **159 GiB**. These are live
snapshots: the excluded support-validation panel continued collecting throughout
maintenance. Three uncompleted/interrupted captures without completion markers
were also left untouched, including the archived M8 reboot attempt. No inventory
pathname, scientific file content, trajectory sample, or label was removed.

The remaining space includes distinct snapshots and the excluded active panel;
it is not all further reclaimable by exact-byte deduplication. The tests below
passed (12 tests), and Black/Ruff checks passed.

## Repeatable maintenance

The standalone maintenance command is
`scripts/research/compact_motion2scene_inventories.py`. It does not import or
modify the hash-bound simulator, scorer, or training code. It requires no Isaac
runtime and works with Python 3.10 or later on Linux.

Identify currently running collection directories and pass each as an
`--exclude` relative to the data root. The example excludes the support-validation
panel that was active during the September 10 cleanup. Use new report directories
for each invocation:

```bash
python3 scripts/research/compact_motion2scene_inventories.py \
  --root /home/linjiw/research-data/groot-wbc \
  --exclude m2s-support-validation-panel-20260910-v1 \
  --report /absolute/path/to/new-inventory-audit

python3 scripts/research/compact_motion2scene_inventories.py \
  --root /home/linjiw/research-data/groot-wbc \
  --exclude m2s-support-validation-panel-20260910-v1 \
  --plan /absolute/path/to/new-inventory-audit/audit.json \
  --report /absolute/path/to/new-inventory-compaction \
  --apply
```

The first command hashes files without changing inventory storage. The second
rechecks the audit and applies atomic replacements. Omit `--plan` to audit and
apply in one invocation. Rerun on newly completed collections to reclaim further
duplication; there is no background service or change to current collection jobs.

The tool skips files newer than ten minutes, files open for writing by visible
processes, and files without an enclosing success/result/release marker. It
rejects symlink paths and changed audit entries. Explicit active-directory
exclusions remain necessary: the maintenance lock coordinates this tool, not
simulator processes. Completion markers are a storage eligibility check, not a
new scientific outcome assessment.

The report retains the original file identities, hashes, sizes, timestamps and
link counts in `audit.json`, each replacement in `journal.jsonl`, and measured
reclaimed allocation plus verification counts in `result.json`. Reclaimed bytes
count only the last removed link to an old inode; external links can reduce the
savings. Every eligible retained path is verified against its original SHA-256
after compaction. Paths and bytes are preserved; inode identities and some file
timestamps change because aliases share an inode.

Treat shared inventories as immutable. Run new captures in fresh output
directories. An in-place write to one hardlink changes all its aliases; use an
atomic replacement with a separately written file when replacing an inventory.
Copy/export tools can expand hardlinks again, so retain hardlinks when packaging
archives and rerun maintenance after producing independent expanded copies.

This reduces disk allocation and permits shared filesystem caching. It does not
remove teacher/student episodes or change sample weighting, and it does not
eliminate repeated JSON parsing. A future recording format could store mesh
geometry once and reference it from small per-capture records, but that needs a
versioned reader/exporter change and fresh provenance registration. Existing
hash-bound captures should not be silently converted to that format.

Validation:

```bash
.venv_research/bin/python -m pytest -q \
  decoupled_wbc/tests/test_compact_motion2scene_inventories.py \
  decoupled_wbc/tests/test_motion2scene_storage.py
.venv_research/bin/python -m black --check \
  scripts/research/compact_motion2scene_inventories.py \
  decoupled_wbc/tests/test_compact_motion2scene_inventories.py
.venv_research/bin/python -m ruff check \
  scripts/research/compact_motion2scene_inventories.py \
  decoupled_wbc/tests/test_compact_motion2scene_inventories.py
```
