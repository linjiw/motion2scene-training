#!/usr/bin/env python3
"""Export a completed acquisition checkpoint with separate immutable corpus datasets.

Each corpus retains every bootstrap/teacher/pre-update-student execution. This
driver uses the existing native portable exporter and never executes physics.
Recorded fit weights remain historical source data, not refreshed physical gaps.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_checkpoint_controls import checkpoint_slots  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_export_timed_schedule_dataset import export  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def export_checkpoint(audit_path, out):
    audit_ref = artifact(audit_path)
    audit = read_checked(audit_ref)
    plan = read_checked(audit["plan"])
    if audit["budget"] != 2 or len(audit["corpora"]) != 12:
        raise ValueError("this release requires the complete original twelve-corpus M2 audit")
    snapshot = []
    for corpus in audit["corpora"]:
        slots = checkpoint_slots(plan, corpus["run_id"], 2)
        source_models = [artifact(s["model"]) for s in slots]
        trained = read_checked(source_models[-1])
        if (
            source_models[-1] != corpus["checkpoint"]
            or trained["status"] != "complete"
            or trained["audit_errors"]
            or [g["collection"] for g in read_checked(trained["teachers"])]
            != [artifact(s["teacher"]) for s in slots]
        ):
            raise ValueError("exact complete audited teacher/model prefix required")
        collections = [artifact(s["teacher"]) for s in slots]
        collections += [artifact(s["student"]) for s in slots[1:]]
        if len(collections) != 5 or len({r["path"] for r in collections}) != 5:
            raise ValueError(
                "M2 requires three teachers and two distinct actual student executions"
            )
        snapshot.append(
            dict(
                run_id=corpus["run_id"],
                collections=collections,
                models=source_models,
                teachers=trained["teachers"],
                physical_cost=corpus["physical_cost"],
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(audit_path, out / "source_audit.json")
    shutil.copyfile(audit["plan"]["path"], out / "source_plan.json")
    write_new(
        out / "registration.json",
        dict(
            audit=audit_ref,
            plan=audit["plan"],
            corpora=snapshot,
            implementation=artifact(Path(__file__)),
            exporter=artifact(
                ROOT / "scripts/research/motion2scene_export_timed_schedule_dataset.py"
            ),
            new_physics_steps=0,
        ),
    )
    records = []
    for item in snapshot:
        print(json.dumps(dict(status="exporting_corpus", run_id=item["run_id"])), flush=True)
        started = time.monotonic()
        path = out / "corpora" / item["run_id"]
        export(path, [Path(r["path"]) for r in item["collections"]], Path(item["teachers"]["path"]))
        manifest = json.loads((path / "manifest.json").read_text())
        if (
            manifest["counts"]["episodes"] != 23
            or manifest["counts"]["physics_steps"] != item["physical_cost"]["total_recorded_steps"]
        ):
            raise ValueError(
                "portable corpus must preserve all assigned measured acquisition costs"
            )
        provenance = out / "fit_provenance" / item["run_id"]
        provenance.mkdir(parents=True)
        models = []
        for index, model_ref in enumerate(item["models"]):
            trained = read_checked(model_ref)
            registration = read_checked(trained["registration"])
            folder = provenance / f"model_{index:03d}"
            folder.mkdir()
            refs = dict(
                result=model_ref,
                registration=trained["registration"],
                teachers=trained["teachers"],
                policy=trained["policy"],
            )
            if registration["replay_weights"] is not None:
                refs["replay_weights"] = registration["replay_weights"]
            for key, source in refs.items():
                checked = artifact(Path(source["path"]))
                if checked != source:
                    raise ValueError("source fit changed during export")
                shutil.copyfile(
                    source["path"], folder / (key + (".npz" if key == "policy" else ".json"))
                )
            models.append(
                dict(
                    checkpoint=index,
                    original=model_ref,
                    files={
                        p.name: dict(path=str(p.relative_to(out)), sha256=artifact(p)["sha256"])
                        for p in sorted(folder.iterdir())
                    },
                )
            )
        record = dict(
            run_id=item["run_id"],
            dataset=str(path.relative_to(out)),
            dataset_manifest_sha256=artifact(path / "manifest.json")["sha256"],
            counts=manifest["counts"],
            physical_cost=item["physical_cost"],
            models=models,
            export_wall_seconds=time.monotonic() - started,
        )
        records.append(record)
        write_new(out / "receipts" / (item["run_id"] + ".json"), record)
        print(
            json.dumps(
                dict(
                    status="corpus_exported",
                    run_id=item["run_id"],
                    seconds=record["export_wall_seconds"],
                )
            ),
            flush=True,
        )
    toolkit = out / "tools/portable_baseline"
    sources = closure(
        [
            ROOT / "scripts/research/motion2scene_schedule_dataset_baseline.py",
            ROOT / "scripts/research/motion2scene_acquisition_dataset_baseline.py",
        ]
    )
    for source in sorted(sources):
        target = toolkit / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Namespace initializers avoid importing optional robot/runtime dependencies.
    for name in (
        "gear_sonic",
        "gear_sonic/dataset_generation",
        "gear_sonic/dataset_generation/hallucination",
    ):
        (toolkit / name / "__init__.py").write_text(
            '"""Portable pure Python subset; no simulator assets."""\n'
        )
    (toolkit / "requirements.txt").write_text("numpy>=1.24\n")
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")
    (out / "DATA_CARD.md").write_text(
        "# Completed M2 acquisition comparison\n\n"
        "Twelve separate development corpora: four construction/replay arms and three corpus seeds. "
        "All 276 recorded episodes and 328,992 physics steps are retained, including bootstrap and pre-update "
        "students. Episodes with the same bytes are distinct acquisitions when their source identities differ. "
        "The prior historical startup unknown reservation remains separately described by the source audit; "
        "it is not a measured step or silently imputed failure. No held-out policy episodes are included.\n\n"
        "Each corpus dataset independently exports physical records, measured contacts, causal sensor inputs, "
        "full teacher continuations and immutable collector source snapshots. Fit provenance contains M0/M1/M2 "
        "ridge policies and their exact teachers, initialization settings and recorded replay weights. "
        "Reconstruction uses these previously verified weights; "
        "it does not turn fitting residuals into physical gaps. "
        "The actual pre-update students retain their generating historical policies.\n\n"
        "Raw pretrained weights and full motion/reference banks are excluded. Recorded reference commands remain "
        "identified as execution data; repository/upstream licensing applies. This package neither requalifies "
        "absent motion assets nor supplies a new independent test distribution.\n"
    )
    return write_new(
        out / "release.json",
        dict(
            schema="motion2scene_acquisition_portable_release_v1",
            registration=artifact(out / "registration.json"),
            source_audit=audit_ref,
            budget=2,
            corpora=records,
            total_episodes=sum(r["counts"]["episodes"] for r in records),
            total_physics_steps=sum(r["counts"]["physics_steps"] for r in records),
            toolkit_files=[
                dict(path=str(p.relative_to(out)), sha256=artifact(p)["sha256"])
                for p in sorted(toolkit.rglob("*"))
                if p.is_file()
            ],
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.audit, args.out)))
