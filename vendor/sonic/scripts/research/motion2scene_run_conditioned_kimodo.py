#!/usr/bin/env python3
"""Register/run one constrained CPU Kimodo candidate, preserving uncorrected output."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path("/home/linjiw/groot-wbc-sonic-sim-trackb")
KIMODO = Path("/home/linjiw/kimodo")
DATA = Path("/home/linjiw/research-data/groot-wbc")


def artifact(path):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": "sha256:" + sha, "size_bytes": path.stat().st_size}


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check(ref):
    if artifact(Path(ref["path"]))["sha256"] != ref["sha256"]:
        raise ValueError(f"registered artifact changed: {ref['path']}")


def register(out):
    proposal_path = DATA / "m2s-kimodo-constraint-api-audit-v1/proposal.json"
    proposal = json.loads(proposal_path.read_text())
    parent_path = DATA / "m2s-longer-reference-development-v1/registration.json"
    parent = json.loads(parent_path.read_text())
    for ref in proposal["assets"].values():
        check(ref)
    check(proposal["constraints"])
    check(proposal["source_csv"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "generated").mkdir()
    sources = {Path(ref["path"]) for ref in parent["source_files"]}
    sources.add(Path(__file__).resolve())
    sources.update(
        (ROOT / "gear_sonic/__init__.py", ROOT / "gear_sonic/dataset_generation/__init__.py")
    )
    sources.update((KIMODO / "kimodo").rglob("*.py"))
    sources.update(
        p
        for p in (KIMODO / "kimodo/assets/skeletons/g1skel34").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    snapshots = []
    for path in sorted(sources):
        relative = (
            Path("repository") / path.relative_to(ROOT)
            if path.is_relative_to(ROOT)
            else Path("kimodo") / path.relative_to(KIMODO)
        )
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        snapshots.append({**artifact(path), "snapshot": artifact(dest)})
    (out / "constraints.json").write_bytes(Path(proposal["constraints"]["path"]).read_bytes())
    checkpoint_dir = Path(proposal["assets"]["checkpoint"]["path"]).parent
    model_assets = [artifact(p) for p in sorted(checkpoint_dir.rglob("*")) if p.is_file()]
    # Reuse the known native-FK audit, adapting only this candidate's artifact/key names.
    audit_source = DATA / "m2s-longer-reference-development-v1/audit_reference.py"
    audit_driver = (
        audit_source.read_text()
        .replace("neutral.pkl", "conditioned.pkl")
        .replace("longer_development_41002_neutral", "longer_development_41002_conditioned")
    )
    (out / "audit_reference.py").write_text(audit_driver)
    environment = {
        **parent["environment"],
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PYTHONPATH": str(out / "source_snapshot/repository")
        + ":"
        + str(out / "source_snapshot/kimodo"),
    }
    driver = out / "source_snapshot/repository/scripts/research" / Path(__file__).name
    registration = {
        "schema": "motion2scene_single_conditioned_prior_registration_v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "proposal": artifact(proposal_path),
        "proposal_contents": proposal,
        "source_generation_registration": artifact(parent_path),
        "assets": proposal["assets"],
        "model_snapshot_assets": model_assets,
        "planned_neutral_source_csv": proposal["source_csv"],
        "qualified_neutral_native": artifact(
            DATA / "m2s-longer-reference-development-v1/native_reference_50hz.npz"
        ),
        "constraints": artifact(out / "constraints.json"),
        "source_files": snapshots,
        "driver": artifact(driver),
        "audit_driver": artifact(out / "audit_reference.py"),
        "audit_driver_parent": artifact(audit_source),
        "environment": environment,
        "command": [
            str(ROOT / ".venv_kimodo/bin/python"),
            str(driver),
            "generate",
            "--out",
            str(out),
        ],
        "model_call": proposal["model_call"],
        "seed": proposal["seed"],
        "versions": {
            k: importlib.metadata.version(k) for k in ("numpy", "torch", "scipy", "safetensors")
        },
        "maximum_attempts": 1,
        "maximum_inference_seconds": 1800,
        "outputs": [
            "raw_model_motion.npz",
            "generated/conditioned.csv",
            "generation_execution.json",
            "conditioned.pkl",
            "conditioned.pkl.manifest.json",
            "native_reference_50hz.npz",
            "kinematic_audit.json",
            "condition_residuals.json",
            "native_guard_comparison.json",
        ],
        "ancestry": (
            "One new neural candidate conditioned on planned six-second development-neutral "
            "reference; not achieved history or held-out ancestry."
        ),
        "admission": proposal["preregister_before_inference"],
        "failure_policy": (
            "Retain raw candidate and failed checks; no repeat, clipping, postprocess, prefix copy, "
            "retiming, padding, option insertion or physical execution."
        ),
        "evaluation_layouts_queried": False,
        "physics_authorized_by_this_registration": False,
    }
    write(out / "registration.json", registration)
    (out / "registration.sha256").write_text(artifact(out / "registration.json")["sha256"] + "\n")
    print(
        json.dumps(
            {
                "registration": artifact(out / "registration.json"),
                "command": registration["command"],
            }
        )
    )


def generate(out):
    registration = json.loads((out / "registration.json").read_text())
    if (out / "generation_started.json").exists():
        raise ValueError("the registered single attempt already started")
    for key, value in registration["environment"].items():
        if os.environ.get(key) != value:
            raise ValueError(f"registered environment differs: {key}")
    for ref in list(registration["assets"].values()) + registration["model_snapshot_assets"]:
        check(ref)
    for ref in registration["source_files"]:
        check(ref["snapshot"])
    check(registration["constraints"])
    check(registration["driver"])
    started = time.monotonic()
    write(
        out / "generation_started.json",
        {
            "registration": artifact(out / "registration.json"),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "maximum_attempts": 1,
        },
    )
    status, error = "failed", None
    try:
        from generate_kimodo_motions import register_cached_encoder
        from kimodo import load_model
        from kimodo.constraints import load_constraints_lst
        from kimodo.exports.mujoco import MujocoQposConverter
        import numpy as np
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        if torch.cuda.is_available():
            raise ValueError("CUDA must be unavailable")
        register_cached_encoder(Path(registration["assets"]["prompt_cache"]["path"]), "cpu")
        torch.manual_seed(registration["seed"])
        model, resolved = load_model("kimodo-g1-rp", device="cpu", return_resolved_name=True)
        constraints = load_constraints_lst(
            str(out / "constraints.json"), model.skeleton, device="cpu"
        )
        output = model(constraint_lst=constraints, **registration["model_call"])
        np.savez_compressed(out / "raw_model_motion.npz", **output)
        restored = dict(output)
        restored["root_positions"] = np.array(output["root_positions"], copy=True)
        xz = registration["proposal_contents"]["canonicalization"][
            "subtract_from_all_condition_XZ_m"
        ]
        restored["root_positions"][..., 0] += xz[0]
        restored["root_positions"][..., 2] += xz[1]
        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(restored, device="cpu", mujoco_rest_zero=False)
        if qpos.shape != (1, 180, 36):
            raise ValueError(f"unexpected generated shape: {qpos.shape}")
        converter.save_csv(qpos, str(out / "generated/conditioned.csv"))
        write(
            out / "generated/conditioned.json",
            {
                "registration": artifact(out / "registration.json"),
                "resolved_model": resolved,
                "source_fps": 30,
                "raw_motion": artifact(out / "raw_model_motion.npz"),
                "converted_csv": artifact(out / "generated/conditioned.csv"),
                "coordinate_restore": registration["proposal_contents"]["canonicalization"],
                "joint_clipping": False,
                "postprocessing": False,
                "copied_prefix": False,
            },
        )
        status = "generated_retained_pending_audit"
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    write(
        out / "generation_execution.json",
        {
            "registration": artifact(out / "registration.json"),
            "status": status,
            "error": error,
            "wall_seconds": time.monotonic() - started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "physics_steps": 0,
            "cuda_used": False,
            "maximum_attempts": 1,
        },
    )
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "generate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (register if args.mode == "register" else generate)(args.out)
