#!/usr/bin/env python3
"""Publish policy recordings with separate, immutable scorer-version children.

This wrapper is deliberately outside the execution package. It never combines
different passage receipt schemas into a common scorer. Each child uses its own
unchanged scoring closure; the top-level inventory counts every assigned slot,
including the distinct physical and sensor lengths of partial attempts.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
SCHEMA = "motion2scene_policy_panel_portable_v1"
EXPORTER = Path("scripts/research/motion2scene_export_timed_schedule_dataset.py")
WORKER = "motion2scene_policy_release_worker.py"
SCORERS = tuple(
    "gear_sonic/dataset_generation/hallucination/" + name + ".py"
    for name in ("motion2scene_passage", "motion2scene_course", "motion2scene_timed_outcome")
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), size_bytes=path.stat().st_size)


def checked(item):
    path = Path(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError("source artifact changed: " + str(path))
    return path


def safe_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError("portable path escapes release")
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_closure(root):
    """Resolve only local Python imports; importing production code is unnecessary."""
    root = Path(root).resolve()
    pending, done = [root / EXPORTER], set()
    while pending:
        path = pending.pop().resolve()
        if path in done:
            continue
        if not path.is_relative_to(root):
            raise ValueError("source closure escapes supplied version")
        done.add(path)
        for parent in path.parents:
            if not parent.is_relative_to(root):
                break
            if (parent / "__init__.py").is_file():
                pending.append(parent / "__init__.py")
        for node in ast.walk(ast.parse(path.read_text())):
            modules, bases = [], [root, root / "scripts/research", path.parent]
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.level:
                    bases = [path.parents[node.level - 1]]
                modules += [
                    ((node.module + ".") if node.module else "") + a.name for a in node.names
                ]
            for module in modules:
                for base in bases:
                    stem = base.joinpath(*module.split(".")) if module else base
                    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                        if candidate.is_file() and candidate.resolve().is_relative_to(root):
                            pending.append(candidate)
                            break
    return sorted(done)


def relocate_exporter(text, repository):
    """Change provenance path mapping only; scoring and ROOT imports are unchanged."""
    old = 'Path(dependency["path"]).relative_to(ROOT)'
    mapped = 'Path(dependency["path"]).relative_to(SOURCE_REPOSITORY_ROOT)'
    declaration = "SOURCE_REPOSITORY_ROOT = Path(" + repr(str(Path(repository).resolve())) + ")"
    if old in text:
        if text.count(old) != 1 or "SOURCE_REPOSITORY_ROOT" in text:
            raise ValueError("unrecognized exporter relocation pattern")
        text = text.replace(
            "ROOT = Path(__file__).resolve().parents[2]",
            "ROOT = Path(__file__).resolve().parents[2]\n" + declaration,
        )
        return text.replace(old, mapped)
    if text.count(mapped) != 1:
        raise ValueError("exporter lacks the audited provenance path mapping")
    tree = ast.parse(text)
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCE_REPOSITORY_ROOT" for t in node.targets)
    ]
    if len(declarations) != 1 or ast.literal_eval(declarations[0].value.args[0]) != str(
        Path(repository).resolve()
    ):
        raise ValueError("existing provenance root differs from declared original workspace")
    return text


def count_slots(slots):
    identities = [(s["collection_sha256"], s["original_cell_id"]) for s in slots]
    if not slots or len(identities) != len(set(identities)):
        raise ValueError("every actual assigned slot must appear once")
    captures = [s["source_capture"]["path"] for s in slots]
    if len(captures) != len(set(captures)):
        raise ValueError("same physical capture supplied more than once")
    counts = Counter(assigned_slots=len(slots), teacher_targets=0)
    for slot in slots:
        status = slot["capture_status"]
        if status not in ("complete_episode", "partial_attempt"):
            raise ValueError("explicit completion status required")
        counts[status + "_count"] += 1
        for key in ("physical_rows", "sensor_packets", "physics_steps"):
            if type(slot[key]) is not int or slot[key] < 0:
                raise ValueError("actual nonnegative array counts required")
            counts[key] += slot[key]
            counts[("complete_" if status == "complete_episode" else "partial_") + key] += slot[key]
        known = slot["task_outcome_admitted"]
        if type(known) is not bool or type(slot["pass"]) is not bool:
            raise ValueError("explicit physical outcome status required")
        if slot["pass"] and (not known or status != "complete_episode"):
            raise ValueError("a pass requires a complete admitted physical outcome")
        cost = slot["costs"]["passage_time_s"]
        if not slot["pass"] and cost is not None:
            raise ValueError("failure/unknown passage cost cannot be fabricated")
        counts[
            (
                "pass_count"
                if slot["pass"]
                else "known_failure_count" if known else "unknown_outcome_count"
            )
        ] += 1
    for key in (
        "complete_episode_count",
        "partial_attempt_count",
        "pass_count",
        "known_failure_count",
        "unknown_outcome_count",
        "complete_physical_rows",
        "complete_sensor_packets",
        "complete_physics_steps",
        "partial_physical_rows",
        "partial_sensor_packets",
        "partial_physics_steps",
    ):
        counts.setdefault(key, 0)
    return dict(sorted(counts.items()))


def compare_combined_audit(slots, combined):
    """Bind every portable assigned slot to the separately completed physical audit."""
    if combined.get("status") != "complete" or combined["assigned_episodes"] != len(slots):
        raise ValueError("completed combined physical audit required")
    if len(combined["rows"]) != len(slots):
        raise ValueError("combined audit omits assigned attempts")
    used = set()
    for slot in slots:
        candidates = [
            (i, row)
            for i, row in enumerate(combined["rows"])
            if row["result"]["sha256"] == slot["collection_sha256"]
            and slot["source_capture"]["path"]
            in [ref["path"] for ref in row["raw_artifacts"].values()]
        ]
        if len(candidates) != 1 or candidates[0][0] in used:
            raise ValueError("assigned slot lacks distinct combined-audit capture identity")
        index, row = candidates[0]
        used.add(index)
        expected = dict(
            scene_id=row["scene_id"],
            configured_policy_id=row["policy_id"],
            measurement_admitted=row["measurement_admitted"],
            physics_steps=row["physics_steps"],
            costs=row["costs"],
            physical_rows=row["outcome"]["physical_rows_recorded"],
            sensor_packets=row["outcome"]["sensor_packets_recorded"],
            task_outcome_admitted=row["outcome"]["task_outcome"] != "unknown",
        )
        expected["pass"] = row["passed"]
        if any(slot[key] != value for key, value in expected.items()):
            raise ValueError("portable slot differs from independent combined physical audit")


def freeze(spec_path, work):
    """Bind completed audits/collections and source versions before any export."""
    spec = read(spec_path)
    if spec["schema"] != SCHEMA or not spec["children"]:
        raise ValueError("explicit policy release specification required")
    checked(spec["combined_audit"])
    for item in spec.get("provenance_items", []):
        checked(item["artifact"])
    if len({c["id"] for c in spec["children"]}) != len(spec["children"]):
        raise ValueError("distinct scorer child IDs required")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    children = []
    for child in spec["children"]:
        name = child["id"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError("simple portable child ID required")
        source = Path(child["source_root"]).resolve()
        dest = work / "sources" / name
        sources = []
        for path in source_closure(source):
            relative = path.relative_to(source)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            if relative == EXPORTER:
                data = relocate_exporter(data.decode(), spec["repository_root"]).encode()
            target.write_bytes(data)
            sources.append(
                dict(source=artifact(path), snapshot=artifact(target), relative=str(relative))
            )
        scorer_ids = {name: digest(dest / name) for name in SCORERS}
        if scorer_ids != child["scoring_sha256"]:
            raise ValueError("declared scorer version differs from frozen closure")
        identity_copies = {}
        for item in child["collections"]:
            result = read(checked(item))
            manifest = read(checked(result["manifest"]))
            declaration = manifest.get("runtime_assets_declaration")
            if declaration:
                for asset in read(checked(declaration)):
                    original = Path(asset["path"])
                    if original.suffix == ".py" and original.is_relative_to(
                        spec["repository_root"]
                    ):
                        key = str(original)
                        if key not in identity_copies:
                            target = (
                                work
                                / "asset_identity"
                                / name
                                / original.relative_to(spec["repository_root"])
                            )
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(checked(asset).read_bytes())
                            identity_copies[key] = dict(original=asset, snapshot=artifact(target))
            paths = {
                str(Path(x["path"]).relative_to(spec["repository_root"])): x
                for x in manifest["dependencies"]
            }
            for scorer, sha in scorer_ids.items():
                if scorer not in paths or paths[scorer]["sha256"] != sha:
                    raise ValueError("export scorer does not match actual executed source version")
            cells = {c["cell_id"] for c in manifest["cells"]}
            if (
                len(cells) != len(manifest["cells"])
                or len(result["rows"]) != len(cells)
                or {r["cell_id"] for r in result["rows"]} != cells
            ):
                raise ValueError("all assigned collection slots must have a retained result")
        if child.get("preexported_child"):
            checked(child["preexported_child"])
        children.append(
            dict(
                **child,
                isolated_source_root=str(dest.resolve()),
                frozen_sources=sources,
                asset_identity_copies=list(identity_copies.values()),
            )
        )
    for name in (Path(__file__).name, WORKER):
        shutil.copy2(Path(__file__).with_name(name), work / name)
    registration = dict(
        schema=SCHEMA,
        specification=artifact(spec_path),
        combined_audit=spec["combined_audit"],
        children=children,
        repository_root=spec["repository_root"],
        data_root=spec["data_root"],
        expected_counts=spec["expected_counts"],
        wrapper_sources=[artifact(work / name) for name in (Path(__file__).name, WORKER)],
        role="new CPU packaging only; original scorer versions and assigned outcomes retained",
        new_physics_steps=0,
    )
    write_new(work / "registration.json", registration)
    return artifact(work / "registration.json")


def run_worker(worker, action, config):
    with tempfile.TemporaryDirectory(prefix="m2s-policy-native-") as directory:
        directory = Path(directory)
        config_path, receipt = directory / "config.json", directory / "receipt.json"
        write_new(config_path, config)
        command = [
            sys.executable,
            "-I",
            "-B",
            str(worker),
            action,
            "--config",
            str(config_path),
            "--receipt",
            str(receipt),
        ]
        result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(
                "source-isolated policy release worker failed:\n" + result.stderr[-10000:]
            )
        return read(receipt)


def inventory(root):
    return [
        dict(path=str(path.relative_to(root)), sha256=digest(path), size_bytes=path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in (root / "manifest.json", root / "audit.json")
    ]


def export(work, out):
    work, out = Path(work).resolve(), Path(out).resolve()
    registration = read(work / "registration.json")
    spec = read(checked(registration["specification"]))
    checked(registration["combined_audit"])
    for source in registration["wrapper_sources"]:
        checked(source)
    out.mkdir(parents=True, exist_ok=False)
    provenance = out / "provenance"
    provenance.mkdir()
    write_new(provenance / "registration.json", registration)
    write_new(provenance / "specification.json", spec)
    write_new(provenance / "combined_audit.json", read(checked(spec["combined_audit"])))
    for index, item in enumerate(spec.get("provenance_items", [])):
        path = checked(item["artifact"])
        if path.suffix not in (".json", ".md", ".patch", ".txt"):
            raise ValueError("release provenance must exclude model/reference binary assets")
        shutil.copy2(path, provenance / f"evidence_{index:03d}_{path.name}")
    for name in (Path(__file__).name, WORKER):
        shutil.copy2(work / name, out / name)
    children, slots = [], []
    for child in registration["children"]:
        for source in child["frozen_sources"]:
            checked(source["snapshot"])
        for item in child["collections"]:
            checked(item)
        folder = out / "children" / child["id"]
        config = dict(
            source_root=child["isolated_source_root"],
            child_root=str(folder),
            repository_root=registration["repository_root"],
            data_root=registration["data_root"],
            collections=child["collections"],
            asset_identity_copies=child["asset_identity_copies"],
        )
        if child.get("preexported_child"):
            previous = checked(child["preexported_child"])
            old_manifest = read(previous)
            if [(v["path"], v["sha256"]) for v in old_manifest["source_results"]] != [
                (v["path"], v["sha256"]) for v in child["collections"]
            ]:
                raise ValueError("pre-exported child collections differ from frozen assignments")
            shutil.copytree(previous.parent, folder)
            receipt = dict(
                reused_unchanged_child=child["preexported_child"],
                audit_scope="complete native audit is repeated after top-level assembly",
            )
        else:
            receipt = run_worker(work / WORKER, "export", config)
        manifest = read(folder / "manifest.json")
        if manifest["counts"]["teacher_decisions"] or read(folder / "teachers.json"):
            raise ValueError("actual policy panel must not invent matched teacher targets")
        contexts = []
        episode_map = {
            (e["collection_sha256"], e["original_cell_id"]): e for e in manifest["episodes"]
        }
        for study_index, item in enumerate(child["collections"]):
            result = read(checked(item))
            source_manifest = read(checked(result["manifest"]))
            scene = read(checked(source_manifest["scene_definition"]))
            cells = {c["cell_id"]: c for c in source_manifest["cells"]}
            for row in result["rows"]:
                cell = cells[row["cell_id"]]
                key = (item["sha256"], row["cell_id"])
                complete = key in episode_map
                if complete:
                    episode = episode_map[key]
                    assessment = episode["assessment"]
                    counts = dict(
                        physical_rows=episode["physical_control_frames"],
                        sensor_packets=episode["physical_control_frames"],
                        physics_steps=episode["physics_steps"],
                    )
                    capture = episode["source_trajectory"]
                    directory = episode["directory"]
                else:
                    directory = f"provenance/study_{study_index:03d}/failed_{row['cell_id']}"
                    failures = [
                        f for f in manifest["failed_attempts"] if f["directory"] == directory
                    ]
                    if len(failures) != 1:
                        raise ValueError("assigned slot missing from child package")
                    assessment = failures[0]["assessment"]
                    attempt = read(checked(row["attempt"]))
                    registry = read(checked(source_manifest["registry"]))
                    # Frame count is the verified request field, not a dataset default.
                    request = (
                        registry["request"]
                        if isinstance(registry["request"], dict)
                        and "options" in registry["request"]
                        else read(checked(registry["request"]))
                    )
                    context = dict(
                        scene_definition=scene,
                        attempt=attempt,
                        declared_neutral_self_pairs=source_manifest["declared_neutral_self_pairs"],
                        reference_frames=request["expected_loaded_frames"],
                        phase_ticks=source_manifest["phase_ticks"],
                    )
                    context_path = f"provenance/partial_{len(contexts):03d}_{child['id']}.json"
                    write_new(out / context_path, context)
                    contexts.append(dict(directory=directory, context_path=context_path))
                    counts = dict(
                        physical_rows=row["outcome"]["physical_rows_recorded"],
                        sensor_packets=row["outcome"]["sensor_packets_recorded"],
                        physics_steps=row["outcome"]["physics_steps_recorded"],
                    )
                    capture = row["raw_artifacts"].get(
                        "aborted_raw_recording", row["raw_artifacts"].get("trajectory")
                    )
                    if capture is None:
                        raise ValueError("partial physical source binding required")
                slot = dict(
                    assigned_slot_id=f"{child['id']}:{study_index:03d}:{row['cell_id']}",
                    study_index=study_index,
                    child_id=child["id"],
                    collection_sha256=item["sha256"],
                    original_cell_id=row["cell_id"],
                    capture_status="complete_episode" if complete else "partial_attempt",
                    directory=directory,
                    source_capture=capture,
                    scene_id=scene["scene_id"],
                    beam_count=len(scene["beams"]),
                    physics_seed=cell["runtime_seed"],
                    mode=cell["timed_schedule_mode"],
                    forced_option_id=cell["forced_option_id"],
                    configured_policy_id=cell.get("policy_id"),
                    model=source_manifest.get("policy"),
                    script_parameters=source_manifest.get("script_parameters"),
                    preferred_option_id=cell.get("preferred_option_id"),
                    measurement_admitted=assessment["measurement_admitted"],
                    task_outcome_admitted=assessment.get(
                        "task_outcome_admitted", assessment["outcome"]["task_outcome"] != "unknown"
                    ),
                    costs=assessment["costs"],
                    **counts,
                )
                slot["pass"] = assessment["pass"]
                slots.append(slot)
        child_info = dict(
            id=child["id"],
            directory=str(folder.relative_to(out)),
            manifest_sha256=digest(folder / "manifest.json"),
            criterion=child["criterion"],
            scoring_sha256=child["scoring_sha256"],
            partial_contexts=contexts,
            source_root=str((folder / "provenance/exporter_source").relative_to(out)),
        )
        write_new(provenance / (child["id"] + "_export.json"), receipt)
        children.append(child_info)
    counts = count_slots(slots)
    compare_combined_audit(slots, read(checked(registration["combined_audit"])))
    if counts != registration["expected_counts"]:
        raise ValueError("actual assigned-slot counts differ from combined audited registration")
    write_new(out / "assigned_slots.json", slots)
    write_new(
        out / "SCORING_VERSIONS.json",
        dict(
            schema=SCHEMA,
            children=children,
            common_passage_dictionary=False,
            teacher_targets=0,
            note=(
                "Each child keeps its execution-time scorer. Later corrections are separate provenance; "
                "partial physical failures do not become completed episodes."
            ),
        ),
    )
    (out / "DATA_CARD.md").write_text(
        "# Actual policy panel\n\n"
        "All assigned physical attempts are retained. Each source-version child has its own "
        "native scorer and exact passage receipts. Complete episodes and partial attempts, "
        "physical rows and causal sensor packets, are counted separately. The top-level audit "
        "reconstructs partial environmental contacts and task outcomes from numeric arrays. "
        "No teacher targets, missing observations, failure costs, pretrained models or raw "
        "motion banks are invented or redistributed. This is development execution evidence.\n"
    )
    write_new(
        out / "manifest.json",
        dict(
            schema=SCHEMA,
            children=children,
            counts=counts,
            files=inventory(out),
            repository_root=registration["repository_root"],
            data_root=registration["data_root"],
            combined_audit=registration["combined_audit"],
            new_physics_steps=0,
        ),
    )
    report = audit(out)
    write_new(out / "audit.json", report)
    return report


def audit(out):
    out = Path(out).resolve()
    manifest = read(out / "manifest.json")
    if manifest["schema"] != SCHEMA or inventory(out) != manifest["files"]:
        raise ValueError("mixed-source portable inventory changed")
    slots = read(out / "assigned_slots.json")
    counts = count_slots(slots)
    compare_combined_audit(slots, read(out / "provenance/combined_audit.json"))
    if counts != manifest["counts"]:
        raise ValueError("assigned-slot accounting changed")
    receipts, seen = [], set()
    for child in manifest["children"]:
        folder = safe_path(out, child["directory"])
        if digest(folder / "manifest.json") != child["manifest_sha256"]:
            raise ValueError("immutable child manifest changed")
        source = safe_path(out, child["source_root"])
        if {name: digest(source / name) for name in SCORERS} != child["scoring_sha256"]:
            raise ValueError("portable child scorer identity changed")
        receipt = run_worker(
            out / WORKER,
            "audit",
            dict(
                source_root=str(source),
                child_root=str(folder),
                portable_root=str(out),
                repository_root=manifest["repository_root"],
                data_root=manifest["data_root"],
                partial_contexts=child["partial_contexts"],
            ),
        )
        child_manifest = read(folder / "manifest.json")
        full = {e["directory"]: e for e in child_manifest["episodes"]}
        partial = {e["directory"]: e["audit"] for e in receipt["partial_attempt_audits"]}
        assigned = [s for s in slots if s["child_id"] == child["id"]]
        if {s["directory"] for s in assigned} != set(full) | set(partial):
            raise ValueError("top-level assigned inventory omits or adds physical attempts")
        for slot in assigned:
            seen.add(slot["assigned_slot_id"])
            study = folder / "provenance" / f"study_{slot['study_index']:03d}"
            invocation = read(study / "manifest.json")
            cells = [
                cell for cell in invocation["cells"] if cell["cell_id"] == slot["original_cell_id"]
            ]
            if len(cells) != 1:
                raise ValueError("portable assigned cell differs from original invocation")
            cell = cells[0]
            configured = dict(
                physics_seed=cell["runtime_seed"],
                mode=cell["timed_schedule_mode"],
                forced_option_id=cell["forced_option_id"],
                configured_policy_id=cell.get("policy_id"),
                model=invocation.get("policy"),
                script_parameters=invocation.get("script_parameters"),
                preferred_option_id=cell.get("preferred_option_id"),
            )
            if any(slot[key] != value for key, value in configured.items()):
                raise ValueError("portable policy identity differs from its actual invocation")
            if slot["directory"] in full:
                episode = full[slot["directory"]]
                expected = dict(
                    physical_rows=episode["physical_control_frames"],
                    sensor_packets=episode["physical_control_frames"],
                    physics_steps=episode["physics_steps"],
                )
                assessment = episode["assessment"]
            else:
                expected = partial[slot["directory"]]
                assessment = dict(
                    measurement_admitted=False,
                    task_outcome_admitted=expected["task_outcome_admitted"],
                    costs=expected["costs"],
                )
                assessment["pass"] = expected["pass_label"]
            for key in ("physical_rows", "sensor_packets", "physics_steps"):
                if slot[key] != expected[key]:
                    raise ValueError("top-level count differs from native physical audit")
            if "task_outcome_admitted" not in assessment:
                assessment = dict(
                    assessment,
                    task_outcome_admitted=assessment["outcome"]["task_outcome"] != "unknown",
                )
            for key in ("pass", "measurement_admitted", "costs", "task_outcome_admitted"):
                if slot[key] != assessment[key]:
                    raise ValueError("top-level outcome differs from native physical audit")
        receipts.append(dict(child_id=child["id"], **receipt))
    if len(seen) != len(slots):
        raise ValueError("unrecognized or duplicate assigned slot")
    return dict(
        schema=SCHEMA,
        counts=counts,
        child_audits=receipts,
        separate_scorer_versions=True,
        every_assigned_slot_retained=True,
        partial_contact_and_outcome_recomputed=True,
        teacher_targets=0,
        new_physics_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "export", "audit", "archive"))
    parser.add_argument("--specification", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.specification, args.work)
    elif args.action == "export":
        result = export(args.work, args.out)
    elif args.action == "audit":
        result = audit(args.out)
    else:
        audit(args.out)
        archive = args.out.with_suffix(".tar.gz")
        with archive.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as tar:
            tar.add(args.out, arcname=args.out.name)
        result = artifact(archive)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
