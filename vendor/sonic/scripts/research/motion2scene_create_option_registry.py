#!/usr/bin/env python3
"""Bind online reference choices to measured source-specific option qualifications."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402


def create(results, source, out):
    rows, motions, evidence = [], {}, []
    controller = None
    for path in results:
        result = json.loads(path.read_text())
        manifest = json.loads(
            checked(Path(result["manifest"]["path"]), result["manifest"]["sha256"]).read_text()
        )
        checkpoint = manifest["implementation"]["checkpoint"]
        if controller is None:
            controller = checkpoint
        elif checkpoint != controller:
            raise ValueError("qualification results use different controllers")
        evidence.append(artifact(path))
        for row in result["rows"]:
            if row["source"] != source or not row["qualified"]:
                continue
            name = row["option"]["reference"]
            cell = next(c for c in manifest["cells"] if c["cell_id"] == row["cell_id"])
            motion = cell["motion"] if name == "neutral" else cell["alternate_motion"]
            ref = {key: motion[key] for key in ("path", "sha256")}
            checked(Path(ref["path"]), ref["sha256"])
            if name in motions and motions[name] != ref:
                raise ValueError("a reference name resolves to multiple motion assets")
            motions[name] = ref
            rows.append(row)
    if "neutral" not in motions or "d040" not in motions:
        raise ValueError("installed adapter initializes neutral and d040 before extra references")
    out.mkdir(parents=True, exist_ok=False)
    qualification = out / "qualification.json"
    write_new(qualification, {"source_results": evidence, "rows": rows})
    references = []
    for name in ["neutral", "d040"] + sorted(set(motions) - {"neutral", "d040"}):
        references.append(
            {
                "name": name,
                "motion": motions[name],
                "qualified_entry_times_s": sorted(
                    {
                        row["option"]["entry_time_s"]
                        for row in rows
                        if row["option"]["reference"] == name
                    }
                ),
                "qualification": artifact(qualification),
            }
        )
    write_new(
        out / "registry.json",
        {
            "schema": "motion2scene_qualified_option_registry_v1",
            "source": source,
            "controller": controller,
            "references": references,
            "implementation": artifact(Path(__file__)),
            "scope": (
                "finite tested source/approach/seed/scene contexts; "
                "qualification does not certify arbitrary obstacles"
            ),
            "unqualified_transitions": [
                "adaptation_to_different_adaptation",
                "arbitrary duration",
                "steering",
                "stopping",
            ],
        },
    )
    print(json.dumps(artifact(out / "registry.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    create(args.results, args.source, args.out)
