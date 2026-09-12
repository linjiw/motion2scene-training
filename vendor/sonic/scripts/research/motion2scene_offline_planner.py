"""Register and execute one file-only CPU SONIC planner proposal, never physics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.dataset_generation.hallucination.motion2scene_planner_adapter import (  # noqa: E402
    COMMON_MODES,
    INPUT_SPEC,
    OUTPUT_SPEC,
    deployment_context,
    kinematic_diagnostics,
    make_inputs,
    named_qpos_to_mujoco,
    valid_output,
    validate_inputs,
)
from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    KIMODO_G1_JOINT_NAMES,
    qpos_to_sonic_motion_entry,
)
from gear_sonic.dataset_generation.motion_prefilter import load_joint_limits  # noqa: E402


def artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": "sha256:" + digest.hexdigest()}


def checked(ref):
    if artifact(ref["path"]) != ref:
        raise ValueError(f"artifact changed: {ref['path']}")
    return Path(ref["path"])


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).absolute()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "onnx", "numpy", "scipy", "joblib")
        },
    }


def prepare(args):
    registry = json.loads(Path(args.registry).read_text())
    reference = registry["references"][0]
    motion = checked(reference["motion"])
    qualification = checked(reference["qualification"])
    rows = json.loads(qualification.read_text())["rows"]
    matches = [row for row in rows if row["option"]["option_id"] == "walk"]
    if len(matches) != 1 or not matches[0]["qualified"] or matches[0]["reset_count"]:
        raise ValueError("expected one qualified, reset-free neutral walk")
    row = matches[0]
    trajectory = checked(row["trajectory"])
    bank_path = checked(row["bank"])
    payload = joblib.load(trajectory)
    reference_qpos = np.asarray(payload["reference_g1_qpos"])
    with np.load(bank_path, allow_pickle=False) as bank:
        if float(bank["fps"]) != 50.0 or payload["fps"] != 50.0:
            raise ValueError("deployment-parity source must be loaded at 50 Hz")
        if not np.array_equal(reference_qpos[:, :3], bank["root_xyz"][0]):
            raise ValueError("recorded neutral root differs from qualified loaded bank")
        if not np.array_equal(reference_qpos[:, 7:], bank["joint_pos"][0]):
            raise ValueError("recorded neutral joints differ from qualified loaded bank")
    if (
        payload["quat_format"] != "wxyz"
        or payload["reference_g1_qpos_dof_order"] != payload["dof_order"]
    ):
        raise ValueError("unbound reference quaternion/joint order")
    if not np.all(np.asarray(payload["motion_id"]) == 0):
        raise ValueError("context source contains reference switches")
    if not np.allclose(payload["motion_time_s"], np.arange(len(reference_qpos)) / 50, atol=1e-6):
        raise ValueError("reference clock is not contiguous from zero")
    qpos = named_qpos_to_mujoco(reference_qpos, payload["dof_joint_names"])
    context, timing = deployment_context(qpos, args.current_frame)
    w, x, y, z = context[0, 0, 3:7]
    heading = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    inputs = make_inputs(
        context,
        mode=args.mode,
        target_vel=args.speed,
        height=args.height,
        seed=args.seed,
        heading_rad=heading,
    )
    # Validate the limit file and environment before creating an immutable registration.
    limit_names, limits = load_joint_limits(args.joint_limits)
    kinematic_diagnostics(np.repeat(context[0, :1], 24, axis=0), context, limit_names, limits)
    env = environment()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "inputs.npz", **inputs)
    np.savez(out / "source_reference.npz", qpos_mujoco=qpos, fps=np.array(50.0))
    source_paths = [
        Path(__file__),
        REPO / "gear_sonic/dataset_generation/hallucination/motion2scene_planner_adapter.py",
        REPO / "gear_sonic/dataset_generation/kimodo_motion_adapter.py",
        REPO / "gear_sonic/dataset_generation/deployable_retiming.py",
        REPO / "gear_sonic/dataset_generation/motion_prefilter.py",
        REPO / "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py",
        REPO / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp",
        REPO
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_tensorrt.hpp",
    ]
    manifest = {
        "schema": "m2s_offline_planner_registration_v1",
        "registered_at_utc": utc(),
        "scope": "fresh development CPU proposal; no physical qualification or achieved-history conditioning",
        "model": artifact(args.model),
        "registry": artifact(args.registry),
        "qualified_neutral_motion": artifact(motion),
        "qualification": artifact(qualification),
        "trajectory": artifact(trajectory),
        "bank": artifact(bank_path),
        "joint_limits": artifact(args.joint_limits),
        "source_files": [artifact(p) for p in source_paths],
        "inputs": artifact(out / "inputs.npz"),
        "source_reference": artifact(out / "source_reference.npz"),
        "input_spec": INPUT_SPEC,
        "output_spec": OUTPUT_SPEC,
        "context": timing,
        "joint_names": list(KIMODO_G1_JOINT_NAMES),
        "controls": {
            "mode": args.mode,
            "mode_name": COMMON_MODES[args.mode],
            "speed_m_s": args.speed,
            "height_m": args.height,
            "random_seed": args.seed,
            "heading_rad": heading,
            "specific_targets_enabled": False,
            "allowed_frame_counts": [36, 40, 44],
            "height_scope": "mode default when -1; no generic walk height guarantee",
        },
        "environment": env,
        "session": {
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": 2,
            "inter_op_threads": 1,
        },
        "maximum_session_run_calls": 1,
        "expected_outputs": [
            "raw_outputs.npz",
            "candidate_30hz.csv",
            "candidate_30hz.pkl",
            "diagnostics.json",
            "result.json",
        ],
        "output_hash_policy": "unknown before inference; actual output hashes recorded in result.json afterward",
        "output_convention": (
            "native valid 30 Hz frames only; world origin retained; "
            "no stitching, padding, or 50 Hz resampling"
        ),
        "physical_rollout_steps": 0,
    }
    write_json(out / "manifest.json", manifest)
    return artifact(out / "manifest.json")


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    out = manifest_path.parent
    for name in (
        "model",
        "registry",
        "qualified_neutral_motion",
        "qualification",
        "trajectory",
        "bank",
        "joint_limits",
        "inputs",
        "source_reference",
    ):
        checked(manifest[name])
    for ref in manifest["source_files"]:
        checked(ref)
    if environment() != manifest["environment"]:
        raise ValueError("CPU environment differs from the registration")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("run with CUDA_VISIBLE_DEVICES='' as an explicit CPU-only guard")
    with np.load(checked(manifest["inputs"]), allow_pickle=False) as packet:
        inputs = {name: packet[name] for name in packet.files}
    validate_inputs(inputs)
    # Exclusive marker prevents retries or second inference under one registration.
    write_json(out / "attempt.json", {"started_at_utc": utc(), "manifest": artifact(manifest_path)})
    started = time.perf_counter()
    result = {
        "schema": "m2s_offline_planner_result_v1",
        "manifest": artifact(manifest_path),
        "session_run_calls": 0,
        "physical_rollout_steps": 0,
        "physical_qualification": "not_run",
        "outputs": [],
        "started_at_utc": utc(),
    }
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = manifest["session"]["intra_op_threads"]
        options.inter_op_num_threads = manifest["session"]["inter_op_threads"]
        session = ort.InferenceSession(
            manifest["model"]["path"], sess_options=options, providers=["CPUExecutionProvider"]
        )
        result["session_initialization_seconds"] = time.perf_counter() - started
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("unexpected execution provider")
        ort_names = {"float32": "tensor(float)", "int64": "tensor(int64)", "int32": "tensor(int32)"}
        for actual, spec in (
            (session.get_inputs(), INPUT_SPEC),
            (session.get_outputs(), OUTPUT_SPEC),
        ):
            if {value.name for value in actual} != set(spec):
                raise ValueError("loaded graph tensor names differ from audited contract")
            for value in actual:
                dtype, shape = spec[value.name]
                if value.type != ort_names[dtype] or tuple(value.shape) != shape:
                    raise ValueError(f"loaded graph tensor differs: {value.name}")
        inference_start = time.perf_counter()
        result["session_run_calls"] = 1
        raw = session.run(list(OUTPUT_SPEC), inputs)
        result["inference_seconds"] = time.perf_counter() - inference_start
        outputs = dict(zip(OUTPUT_SPEC, raw, strict=True))
        np.savez(out / "raw_outputs.npz", **outputs)
        result["outputs"].append(artifact(out / "raw_outputs.npz"))
        candidate = valid_output(outputs, inputs)
        names, limits = load_joint_limits(manifest["joint_limits"]["path"])
        diagnostics = kinematic_diagnostics(candidate, inputs["context_mujoco_qpos"], names, limits)
        diagnostics["output_anchor_reference_time_s"] = manifest["context"][
            "output_anchor_reference_time_s"
        ]
        diagnostics["scope"] = (
            "reference proposal only; no achieved-state splice, controller execution, or traversal outcome"
        )
        np.savetxt(out / "candidate_30hz.csv", candidate, delimiter=",", fmt="%.9g")
        entry = qpos_to_sonic_motion_entry(
            candidate, source_fps=30, canonicalize_horizontal_origin=False
        )
        joblib.dump({"sonic_planner_development_candidate": entry}, out / "candidate_30hz.pkl")
        write_json(out / "diagnostics.json", diagnostics)
        result["outputs"].extend(
            artifact(out / name)
            for name in ("candidate_30hz.csv", "candidate_30hz.pkl", "diagnostics.json")
        )
        result["status"] = "unqualified_reference_candidate_created"
    except Exception as exc:
        result["status"] = "infrastructure_or_contract_failure"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["finished_at_utc"] = utc()
        write_json(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("out", "model", "registry", "joint-limits"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--current-frame", type=int, default=15)
    prep.add_argument("--mode", type=int, choices=sorted(COMMON_MODES), default=1)
    prep.add_argument("--speed", type=float, default=0.6)
    prep.add_argument("--height", type=float, default=-1)
    prep.add_argument("--seed", type=int, default=1234)
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args) if args.command == "prepare" else run(args.manifest), indent=2))


if __name__ == "__main__":
    main()
