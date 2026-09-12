#!/usr/bin/env python3
"""Export traceable result tables and local package checks; never upload a submission."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import subprocess

from audit_motion2scene_submission import DATA, DOC, ROOT, Audit, dump

OUT = DOC / "submission"
EVIDENCE = OUT / "evidence"


def protocol_history(audit):
    """Inventory every historical document revision without changing its contents."""
    versions = []
    for path in sorted(DOC.glob("*.md")):
        history = subprocess.check_output(
            ["git", "log", "--format=%H|%cI", "--", str(path)], text=True
        ).splitlines()
        records = []
        for entry in history:
            commit, when = entry.split("|", 1)
            body = subprocess.check_output(
                ["git", "show", f"{commit}:docs/motion2scene/{path.name}"]
            )
            records.append(
                {
                    "commit": commit,
                    "commit_time": when,
                    "document_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                }
            )
        versions.append({"current": audit.pin(path), "git_versions": records})
    timeline = []
    for family, names in (
        ("m2s-icra-v1", ("registration.json", "proposals.json")),
        (
            "m2s-icra-learning-v1",
            ("registration.json", "comparison_366.json", "comparison_540.json"),
        ),
        ("m2s-icra-nominal-v1", ("registration.json", "proposals.json")),
        ("m2s-icra-nominal-learning-v1", ("registration.json",)),
        ("m2s-submission-command-audit-v1", ("registration.json",)),
    ):
        for name in names:
            path = DATA / family / name
            timeline.append(
                {
                    "artifact": audit.pin(path),
                    "filesystem_mtime_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                }
            )
    dump(
        EVIDENCE / "protocol-history.json",
        {
            "historical_documents": versions,
            "local_timeline": timeline,
            "scope": (
                "Complete git revision/hash inventory of Motion2Scene root Markdown documents. "
                "This inventory is not a raw reproduction of every historical experiment. "
                "Local file mtimes and git timestamps are provenance clues, not trusted external "
                "registration timestamps or an exhaustive log of human inspection."
            ),
        },
    )


def csv_file(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (EVIDENCE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                    for k, v in row.items()
                }
            )


def export_tables(a):
    csv_file("seed-level-outcomes.csv", a["seed_level_rows"])
    csv_file("matched-commands.csv", a["command_lookup"]["conditions"])
    csv_file(
        "acquisition.csv",
        [
            {"study": name, **r}
            for name, study in a["studies"].items()
            for r in study["acquisition"]
        ],
    )
    csv_file(
        "proposal-dispositions.csv",
        [
            {"study": name, **r}
            for name, study in a["studies"].items()
            for r in study["proposals"]["rows"]
        ],
    )
    csv_file(
        "paired-comparisons.csv",
        [
            {"study": name, **r}
            for name, study in a["studies"].items()
            for r in study["primary"]["registered_pairs"]
        ],
    )
    csv_file(
        "background-endpoints.csv",
        [
            {"study": name, "background": kind, "arm": arm, **r}
            for name, study in a["studies"].items()
            for kind, arms in study["primary"]["backgrounds"].items()
            for arm, r in arms.items()
        ],
    )
    csv_file(
        "group-observability.csv",
        [
            {"study": name, "arm": arm, **r}
            for name, study in a["studies"].items()
            for arm, diagnostic in study.get("fit_diagnostics", {}).items()
            for r in diagnostic["groups_detail"]
        ],
    )
    csv_file("rollout-costs.csv", a["costs"])
    summaries = []
    for study, prefix in (("robust", "icra_eval_"), ("nominal", "nom_eval_")):
        rows = [
            r
            for r in a["seed_level_rows"]
            if r["cell_id"].startswith(prefix) and r["suite"] == "traversal"
        ]
        for axes in (("source",), ("layout",), ("source", "layout")):
            groups = defaultdict(list)
            for row in rows:
                groups[(row["arm"], *(row[k] for k in axes))].append(row)
            for key, rr in sorted(groups.items()):
                summaries.append(
                    {
                        "study": study,
                        "grouping": "+".join(axes),
                        "arm": key[0],
                        **dict(zip(axes, key[1:])),
                        "n": len(rr),
                        "pass": sum(r["pass"] for r in rr),
                        "requests": sum(r["action"] for r in rr),
                        "physics_seeds": sorted({r["physics_seed"] for r in rr}),
                        "cell_ids": [r["cell_id"] for r in rr],
                        "scope": "Descriptive; repeats retained together, not independent motion draws",
                    }
                )
    csv_file("carrier-layout-results.csv", summaries)


def claims(audit, a, d, manifest):
    transition_record = DATA / "m2s-development-transition-bank-v2/run_record.json"
    transition = audit.read(transition_record)
    assert len(transition["cells"]) == 12
    dump(
        EVIDENCE / "excluded-costs.json",
        {
            "transition_bank": {
                "record": audit.pin(transition_record),
                "rollout_process_hours": transition["budget"]["actual_contended_gpu_hours"],
                "physical_executions": len(transition["cells"]),
                "inside_historical_6_6_boundary": False,
            },
            "inherited_training": "Excluded; total method training cost not reconstructed or claimed",
            "original_selector_fit_cpu_wall_time": None,
        },
    )
    specs = [
        (
            "protocol",
            "Table I; Section IV",
            ["robust", "nominal"],
            ["/studies/robust/registrations", "/studies/nominal/registrations"],
            ["/provenance"],
            "Outcome-informed revision; 31/36 nominal conditions already observed.",
        ),
        (
            "acquisition",
            "Table II; Section V-A; abstract",
            ["nominal"],
            ["/studies/nominal/acquisition", "/studies/nominal/composition"],
            ["/registered_predictions/nominal"],
            "Equal groups, unequal acquisition compute.",
        ),
        (
            "nominal_passage",
            "Table III; Figure 2; Section V-B; abstract",
            ["nominal", "robust", "audit"],
            ["/comparators", "/command_lookup", "/seed_level_rows"],
            [],
            "Finite-bank result; matched lookup is post hoc and conditional on checked runtime.",
        ),
        (
            "paired_nominal",
            "Table IV; Section V-B",
            ["nominal"],
            [
                "/studies/nominal/primary/registered_pairs",
                "/studies/nominal/primary/clustered_descriptions",
            ],
            [],
            "Condition-level independence unestablished; no equivalence or population CI.",
        ),
        (
            "robust_failure",
            "Table V; Section V-C",
            ["robust"],
            ["/studies/robust/composition", "/studies/robust/primary"],
            ["/registered_predictions/robust"],
            "Unequal acquired corpora; distinct from nominal study.",
        ),
        (
            "backgrounds",
            "Section V-B",
            ["robust"],
            ["/studies/robust/primary/backgrounds"],
            [],
            "Parent policies only; refusal continues walking.",
        ),
        (
            "execution",
            "Figure 1; Sections III-A/B",
            ["nominal"],
            ["/studies/nominal/pairs", "/raw_checks"],
            [],
            "One named illustrative training pair; root height is not a clearance certificate.",
        ),
        (
            "envelope",
            "Figure 3; Sections III-B and V-C",
            [],
            ["/envelope"],
            ["/provenance/envelope_protocol_discrepancies"],
            "Post-hoc finite grid; conditional 1-D derivation, not a full pose-space theorem.",
        ),
        (
            "observability",
            "Figure 4; Section V-D; abstract",
            ["nominal"],
            ["/studies/nominal/fit_diagnostics", "/studies/nominal/acquisition"],
            [],
            "Post-hoc exact-input floor for fitting; does not make passage failure unavoidable.",
        ),
        (
            "sensitivity",
            "Section V-E",
            ["robust", "nominal", "audit"],
            ["/command_lookup"],
            ["/registered_refits", "/nominal_sensitivity"],
            "Post-hoc recorded-input/lookup analysis; no generator-level source holdout.",
        ),
        (
            "costs",
            "Table VI; Section V-F",
            ["robust", "nominal", "audit"],
            ["/costs", "/studies/nominal/proposals/costs", "/studies/nominal/proposals/seconds"],
            [],
            "6.577196 process-hours exclude inherited training, earlier work and CPU search.",
        ),
        (
            "method",
            "Section III; equations 1-3",
            ["robust", "nominal"],
            [
                "/studies/robust/registrations",
                "/studies/nominal/registrations",
                "/studies/nominal/fit_diagnostics",
            ],
            [],
            "Frozen command/scorer/model parameters; explanation adds no new guarantee.",
        ),
    ]
    records = []
    protocols = {
        "robust": ["M2S_ICRA_V1.md", "M2S_ICRA_EXECUTION_V1.md"],
        "nominal": ["M2S_ICRA_NOMINAL_V1.md"],
        "audit": ["SUBMISSION_COMMAND_AUDIT_V1.md"],
    }
    script_refs = [
        audit.pin(ROOT / "scripts/research" / name)
        for name in (
            "audit_motion2scene_submission.py",
            "motion2scene_submission_diagnostics.py",
            "render_motion2scene_submission.py",
            "package_motion2scene_submission.py",
        )
    ]
    for ident, location, studies, ap, dp, limit in specs:
        refs = []
        for filename, content, pointers in (("analysis.json", a, ap), ("diagnostics.json", d, dp)):
            for pointer in pointers:
                value = content
                for part in pointer.lstrip("/").split("/"):
                    value = value[int(part)] if isinstance(value, list) else value[part]
                refs.append({"artifact": audit.pin(EVIDENCE / filename), "json_pointer": pointer})
        ids = []
        for r in a["seed_level_rows"]:
            family = (
                "nominal"
                if r["cell_id"].startswith("nom_")
                else "robust" if r["cell_id"].startswith("icra_") else "audit"
            )
            if family in studies:
                ids.append(r["cell_id"])
        if ident == "execution":
            ids = next(
                p["row_ids"]
                for p in a["studies"]["nominal"]["pairs"]
                if p["group_id"] == "nom_41001_analytic_00"
            )
        if ident == "costs":
            refs.append(
                {
                    "artifact": audit.pin(EVIDENCE / "excluded-costs.json"),
                    "json_pointer": "/transition_bank",
                }
            )
        if ident == "protocol":
            refs.append(
                {
                    "artifact": audit.pin(EVIDENCE / "protocol-history.json"),
                    "json_pointer": "/local_timeline",
                }
            )
        records.append(
            {
                "claim_id": ident,
                "manuscript_location": location,
                "analysis_evidence": refs,
                "analysis_scripts": script_refs,
                "protocols": [audit.pin(DOC / p) for s in studies for p in protocols[s]],
                "dataset_ids": [
                    a["studies"][s][k]
                    for s in studies
                    if s != "audit"
                    for k in ("construction", "learning")
                ]
                + (["m2s-submission-command-audit-v1"] if "audit" in studies else []),
                "supporting_cell_ids": ids,
                "configuration_and_execution_refs": (
                    "Resolve every cell ID in manifest.rows: result, manifest, trajectory, physics, "
                    "controller, configuration_sha256, policy, and inherited_source_commit. "
                    "Exact dependency hashes are in manifest.inputs and expected_pin_checks."
                ),
                "limitation": limit,
            }
        )
        if ident == "envelope":
            records[-1]["dataset_ids"] = [
                "m2s-envelope-tradeoff-v1",
                "m2s-development-transition-bank-v2",
            ]
            records[-1]["protocols"] = [audit.ref(a["envelope"]["protocol"])]
            records[-1]["analysis_scripts"] = script_refs + [
                audit.pin(ROOT / "scripts/research/motion2scene_envelope_tradeoff.py")
            ]
            records[-1]["geometric_record_inputs"] = [
                audit.ref(a["envelope"][key]) for key in ("bank", "refinement")
            ]
        if ident == "costs":
            records[-1]["dataset_ids"].append("m2s-development-transition-bank-v2")
    dump(
        EVIDENCE / "manuscript-claims.json",
        {
            "schema_version": 1,
            "manuscript": audit.pin(OUT / "paper.tex"),
            "pdf": audit.pin(OUT / "paper.pdf"),
            "record_manifest": audit.pin(EVIDENCE / "manifest.json"),
            "audit_checkout_commit": manifest["audit_checkout_commit"],
            "execution_commit_limit": (
                "Original source_commit is inherited lineage; exact execution checkout is not "
                "independently established. Use the recorded per-file and configuration hashes."
            ),
            "claims": records,
            "other_evidence": [
                audit.pin(OUT / "LITERATURE_AND_REQUIREMENTS.md"),
                audit.pin(EVIDENCE / "video-provenance.json"),
            ],
        },
    )


def package_checks(audit, a, manifest):
    pdf = OUT / "paper.pdf"
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    metadata = dict(line.split(":", 1) for line in info.splitlines() if ":" in line)
    metadata = {k: v.strip() for k, v in metadata.items()}
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    urls = subprocess.check_output(["pdfinfo", "-url", str(pdf)], text=True)
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    markers = [
        x
        for x in ("linjiw", "/home/", "groot-wbc", "github.com", "nvlabs.github.io")
        if x in (text + json.dumps(metadata)).lower()
    ]
    video = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(OUT / "motion2scene-anonymous.mp4"),
            ],
            text=True,
        )
    )
    frames = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=interlaced_frame",
                "-of",
                "json",
                str(OUT / "motion2scene-anonymous.mp4"),
            ],
            text=True,
        )
    )["frames"]
    stream = video["streams"][0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    font_rows = fonts.splitlines()[2:]
    template = audit.read(OUT / "template-provenance.json")
    checks = {
        "official_template_class_unchanged": (
            audit.pin(OUT / "ieeeconf.cls")["sha256"] == "sha256:" + template["class_sha256"]
        ),
        "eight_pages_or_fewer": int(metadata["Pages"]) <= 8,
        "letter_paper": "612 x 792" in metadata["Page size"],
        "fonts_embedded": bool(font_rows) and all(r.split()[-5] == "yes" for r in font_rows),
        "no_pdf_url_annotations": len(urls.splitlines()) <= 1,
        "no_known_identifying_markers": not markers,
        "no_author_metadata": not metadata.get("Author"),
        "no_overfull_tex_boxes": "Overfull" not in (OUT / "paper.log").read_text(),
        "video_duration": float(video["format"]["duration"]) <= 180,
        "video_size": int(video["format"]["size"]) <= 20_000_000,
        "video_height": stream["height"] >= 480,
        "video_fps": numerator / denominator >= 20,
        "video_progressive_all_frames": bool(frames)
        and all(not f["interlaced_frame"] for f in frames),
        "all_outcomes_rescored": len(a["raw_checks"]) == 854
        and all(r["scorer_equal"] for r in a["raw_checks"]),
        "no_original_pin_mismatch": not manifest["unresolved_pin_checks"],
        "matched_commands_complete": a["command_lookup"]["missing_actions"] == 0,
        "matching_conditions": not a["command_lookup"]["reuse_mismatches"],
    }
    dump(
        EVIDENCE / "package-checks.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "pdf": audit.pin(pdf),
            "pdf_metadata": metadata,
            "pdf_fonts": fonts,
            "pdf_urls": urls,
            "identifying_markers": markers,
            "video": audit.pin(OUT / "motion2scene-anonymous.mp4"),
            "video_probe": video,
            "video_frames_checked": len(frames),
            "scope": "Local mechanical checks only; not portal acceptance or independent scientific review.",
            "submission_ready": False,
            "remaining": [
                "Author order, affiliations and PaperPlaza PINs not supplied",
                "Earlier AI-use history needs author reconciliation",
                "Independent reader review not performed by this drafting agent",
                "Portal field entry, validation and upload receipt not completed",
            ],
        },
    )
    assert all(checks.values()), checks
    return checks


def main():
    audit = Audit()
    a = audit.read(EVIDENCE / "analysis.json")
    d = audit.read(EVIDENCE / "diagnostics.json")
    m = audit.read(EVIDENCE / "manifest.json")
    assert m["analysis_output"]["sha256"] == audit.pin(EVIDENCE / "analysis.json")["sha256"]
    for doc in (
        d,
        audit.read(EVIDENCE / "render-inputs.json"),
        audit.read(EVIDENCE / "video-provenance.json"),
    ):
        for ref in doc["inputs"]:
            audit.ref(ref)
    export_tables(a)
    protocol_history(audit)
    claims(audit, a, d, m)
    checks = package_checks(audit, a, m)
    files = [
        audit.pin(p)
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name != "package-manifest.json" and p.suffix != ".log"
    ]
    dump(
        EVIDENCE / "package-manifest.json",
        {
            "schema_version": 1,
            "artifacts": files,
            "submission_ready": False,
            "review_upload_candidates": ["paper.pdf", "motion2scene-anonymous.mp4"],
            "local_only": "All other artifacts contain provenance and may contain identifying paths.",
        },
    )
    print(json.dumps({"local_checks": checks, "submission_ready": False}), flush=True)


if __name__ == "__main__":
    main()
