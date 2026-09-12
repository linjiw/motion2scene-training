#!/usr/bin/env python3
"""Isolated worker for one scorer version of a portable policy release.

The caller supplies a frozen source closure. No runtime module is imported until
the source and file-access guards have been installed. Portable audits require
only NumPy and the bundled Python sources, including for aborted captures.
"""

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def install_identity_overlay(records):
    """Redirect pathlib hash reads of declared assets to exact frozen bytes.

    Python imports use importlib's open_code, not this pathlib/io.open overlay.
    They remain subject to the production-source guard and origin audit.
    """
    mapping, reads = {}, []
    for record in records:
        source = Path(record["original"]["path"]).resolve()
        target = Path(record["snapshot"]["path"]).resolve()
        expected = record["original"]["sha256"]
        if source.suffix != ".py" or expected != record["snapshot"]["sha256"]:
            raise ValueError("identity overlay must preserve declared Python asset bytes")
        if "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen identity asset changed")
        if source in mapping and mapping[source] != target:
            raise ValueError("ambiguous identity-read overlay")
        mapping[source] = target
    original_open = io.open

    def frozen_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if path in mapping:
                if any(flag in mode for flag in "wax+"):
                    raise PermissionError("frozen identity overlay is read-only")
                reads.append(str(path))
                file = mapping[path]
        return original_open(file, mode, *args, **kwargs)

    io.open = frozen_open
    return reads


def activate(source, repository, data_root, portable_root=None):
    source, repository, data_root = map(
        lambda p: Path(p).resolve(), (source, repository, data_root)
    )
    portable = Path(portable_root).resolve() if portable_root else None
    sys.path[:0] = [str(source), str(source / "scripts/research")]

    def guard(event, args):
        if event == "import" and portable is not None:
            if args[0].split(".")[0] in {
                "torch",
                "isaaclab",
                "omni",
                "pxr",
                "onnxruntime",
                "joblib",
            }:
                raise ImportError("portable audit forbids simulator/model/pickle dependencies")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        in_venv = any(p.name.startswith(".venv") for p in path.parents)
        if portable is not None:
            if (path.is_relative_to(repository) and not in_venv) or (
                path.is_relative_to(data_root) and not path.is_relative_to(portable)
            ):
                raise PermissionError(
                    "portable audit requires external project/experiment read: " + str(path)
                )
        elif (
            path.is_relative_to(repository)
            and path.suffix in (".py", ".pyc")
            and not in_venv
            and not path.is_relative_to(source)
        ):
            raise PermissionError("unfrozen production Python read: " + str(path))

    sys.addaudithook(guard)
    return importlib.import_module("motion2scene_export_timed_schedule_dataset")


def assert_import_origins(source):
    source = Path(source).resolve()
    modules = []
    for name, module in list(sys.modules.items()):
        value = getattr(module, "__file__", None)
        if value and ("motion2scene" in name or name.startswith(("gear_sonic.", "decoupled_wbc."))):
            path = Path(value).resolve()
            if path.suffix == ".py":
                if not path.is_relative_to(source):
                    raise ValueError("mixed scorer import: " + name)
                modules.append(dict(module=name, source=str(path.relative_to(source))))
    return sorted(modules, key=lambda row: row["module"])


def decode_payload(arrays, metadata):
    """Decode the existing exporter's numeric-only representation, recursively."""

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"npz_array"}:
                return arrays[value["npz_array"]]
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value

    return decode(metadata)


def partial_audit(exporter, child, failure, context, *, compare=True):
    """Recompute an aborted attempt without reopening its original pickle/assets.

    The partial interface and physical payload intentionally have independent
    lengths. Contact evidence can establish task failure while completion and
    the unrecorded sensor/teacher phases remain unavailable.
    """
    import numpy as np

    child = Path(child)
    folder = exporter.safe_path(child, failure["directory"])
    raw = folder / "aborted_raw_recording.npz"
    if raw.exists():
        decoded = decode_payload(
            exporter.array_file(raw), read(folder / "aborted_raw_recording_metadata.json")
        )
        if set(decoded) != {"0"} or not isinstance(decoded["0"], dict):
            raise ValueError("one preserved aborted environment required")
        payload = dict(decoded["0"], fps=50)
    elif (folder / "partial_trajectory.npz").exists():
        payload = decode_payload(
            exporter.array_file(folder / "partial_trajectory.npz"),
            read(folder / "partial_trajectory_metadata.json"),
        )
    else:
        raise ValueError("independent partial audit requires actual physical arrays")
    interface = read(folder / "partial_interface.json")

    def select(normal, aborted):
        paths = [folder / name for name in (normal, aborted) if (folder / name).exists()]
        if len(paths) != 1:
            raise ValueError("one unambiguous complete counterpart stream required")
        return paths[0]

    pairs = exporter.array_file(
        select("environment_pair_contacts.npz", "aborted_environment_pair_contacts.npz")
    )
    net = exporter.array_file(select("all_body_contacts.npz", "aborted_all_body_contacts.npz"))
    mapping = read(
        select("environment_contact_mapping.json", "aborted_environment_contact_mapping.json")
    )
    steps = pairs["physics_steps"]
    scene = context["scene_definition"]
    contact = exporter.audit_environment_contacts(
        pairs,
        net,
        mapping,
        len(steps),
        neutral_self_pairs=context["declared_neutral_self_pairs"],
        beam_paths=exporter.scene_beam_paths(scene),
    )
    observations = interface.get("observations", [])
    assessment = failure["assessment"]
    outcome = exporter.classify_timed_attempt(
        payload,
        observations,
        reference_frames=context["reference_frames"],
        phase_ticks=context["phase_ticks"],
        exit_status=context["attempt"]["exit_status"],
        contact_audit=contact,
        physics_steps=steps,
    )
    physical_rows = len(np.asarray(payload["root_pos_w"]))
    if (
        outcome["physical_rows_recorded"] != physical_rows
        or outcome["sensor_packets_recorded"] != len(observations)
        or outcome["physics_steps_recorded"] != len(steps)
    ):
        raise ValueError("partial classifier counts differ from actual arrays")
    # This release permits fully observed partial physics, not guessed counts.
    if not contact["complete_synchronized_streams"]:
        raise ValueError("partial contact streams cannot independently establish accounting")
    if compare and (
        contact != assessment["contact_audit"]
        or outcome != assessment["outcome"]
        or assessment["physics_steps"] != len(steps)
        or assessment["measurement_admitted"]
        or assessment["pass"]
        or assessment["task_outcome_admitted"] != (outcome["task_outcome"] != "unknown")
        or any(
            assessment["costs"][key] is not None
            for key in ("passage_time_s", "whole_episode_time_s", "positive_mechanical_work_j")
        )
    ):
        raise ValueError("partial recorded outcome/contact/null-cost assessment differs")
    return dict(
        physical_rows=physical_rows,
        sensor_packets=len(observations),
        physics_steps=len(steps),
        contact_audit=contact,
        outcome=outcome,
        phase_availability=outcome["phase_availability"],
        task_outcome_admitted=outcome["task_outcome"] != "unknown",
        measurement_admitted=False,
        pass_label=False,
        costs=assessment["costs"],
        physical_arrays_and_counterpart_order_recomputed=True,
        absent_rows_or_costs_invented=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    source = Path(config["source_root"])
    child = Path(config["child_root"])
    identity_reads = install_identity_overlay(config.get("asset_identity_copies", []))
    exporter = activate(
        source,
        config["repository_root"],
        config["data_root"],
        config.get("portable_root") if args.action == "audit" else None,
    )
    if args.action == "export":
        exporter.export(child, [Path(item["path"]) for item in config["collections"]], None, False)
        report = read(child / "audit.json")
        partials = []
    else:
        report = exporter.audit(child)
        partials = []
        manifest = read(child / "manifest.json")
        for failure in manifest["failed_attempts"]:
            contexts = [
                c for c in config["partial_contexts"] if c["directory"] == failure["directory"]
            ]
            if len(contexts) != 1:
                raise ValueError("every partial attempt requires exactly one immutable context")
            context = read(Path(config["portable_root"]) / contexts[0]["context_path"])
            partials.append(
                dict(
                    directory=failure["directory"],
                    audit=partial_audit(exporter, child, failure, context),
                )
            )
    write_new(
        args.receipt,
        dict(
            complete_episode_audit=report,
            partial_attempt_audits=partials,
            source_imports=assert_import_origins(source),
            source_version_isolated=True,
            external_project_and_experiment_reads_forbidden=args.action == "audit",
            frozen_declared_asset_identity_reads=sorted(set(identity_reads)),
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    main()
