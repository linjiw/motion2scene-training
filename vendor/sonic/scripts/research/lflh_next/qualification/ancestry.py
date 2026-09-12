"""Bind existing selected converted motions to local retarget arrays without publishing data."""

import json
from pathlib import Path

import numpy as np

from scripts.research.lflh_next.qualification.audit import OUT, PREVIOUS, sha

root = Path("/home/linjiw/dataset/amass-licensed")
retarget = root / "retargeted-candidates/fleaven-275c4c4a028cadf5241c10a613c2f11913531ad2/g1"
lookup = {}
for p in retarget.rglob("*.npy"):
    key = "_".join(p.relative_to(retarget).with_suffix("").parts)
    lookup.setdefault(key, []).append(p)
rows = []
for ref in json.loads((PREVIOUS / "design.json").read_text())["source_bindings"]:
    name = Path(ref["path"]).stem
    matches = lookup.get(name, [])
    csvpath = root / "phase-g-csv-fmt8" / f"{name}.csv"
    row = dict(
        npz=ref["path"],
        npz_sha256=sha(ref["path"]),
        split=ref["split"],
        retarget_matches=len(matches),
        csv_exists=csvpath.exists(),
        raw_AMASS_ancestry="UNVERIFIED",
        other_project_split_membership="UNVERIFIED",
    )
    if len(matches) == 1 and csvpath.exists():
        a = np.load(matches[0], allow_pickle=False)
        b = np.loadtxt(csvpath, delimiter=",")
        row.update(
            retarget_path=str(matches[0]),
            retarget_sha256=sha(matches[0]),
            csv_path=str(csvpath),
            csv_sha256=sha(csvpath),
            shape_match=a.shape == b.shape,
        )
        if a.shape == b.shape:
            row["max_csv_rounding_difference"] = float(np.max(np.abs(a - b)))
            row["matches_8_decimal_export"] = row["max_csv_rounding_difference"] <= 5.1e-9
    rows.append(row)
(OUT / "ancestry-ledger.json").write_text(json.dumps(rows, indent=2) + "\n")
print(
    "Files:",
    len(rows),
    "unique retarget matches:",
    sum(r["retarget_matches"] == 1 for r in rows),
    "rounded CSV matches:",
    sum(r.get("matches_8_decimal_export", False) for r in rows),
)
