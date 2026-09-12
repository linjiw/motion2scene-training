#!/usr/bin/env python3
"""Prepare and inspect native Kimodo constraints without loading/generating a model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kimodo.constraints import (
    EndEffectorConstraintSet,
    FullBodyConstraintSet,
    Root2DConstraintSet,
    load_constraints_lst,
    save_constraints_lst,
)
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.motion_rep import KimodoMotionRep
from kimodo.sanitize import sanitize_text
from kimodo.skeleton import G1Skeleton34
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/linjiw/research-data/groot-wbc")
KIMODO = Path("/home/linjiw/kimodo")


def artifact(path):
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(out):
    torch.set_num_threads(4)
    source_registration = DATA / "m2s-longer-reference-development-v1/registration.json"
    original = json.loads(source_registration.read_text())
    for ref in original["assets"].values():
        if artifact(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("pinned generation asset changed")
    csv = DATA / (
        "m2s-longer-reference-development-v1/generated/"
        "000_a_person_walks_at_a_steady_pace_in_a_straight_li_s0.csv"
    )
    qpos = np.loadtxt(csv, delimiter=",").astype(np.float32)
    if qpos.shape != (180, 36) or not np.isfinite(qpos).all():
        raise ValueError("requires the existing six-second neutral source")
    skeleton = G1Skeleton34()
    converter = MujocoQposConverter(skeleton)
    motion = converter.qpos_to_motion_dict(qpos, source_fps=30)
    roundtrip = converter.dict_to_qpos(motion, device="cpu", mujoco_rest_zero=False)
    offset = motion["smooth_root_pos"][0, [0, 2]].clone()
    for key in ("root_positions", "smooth_root_pos", "posed_joints"):
        motion[key] = motion[key].clone()
        motion[key][..., 0] -= offset[0]
        motion[key][..., 2] -= offset[1]
    full = torch.tensor(list(range(16)) + [150, 165, 179])
    low = torch.tensor([60, 75, 90, 105])
    low_positions = motion["posed_joints"][low].clone()
    low_positions[:, skeleton.root_idx, 1] -= 0.085
    constraints = [
        Root2DConstraintSet(
            skeleton,
            torch.arange(180),
            motion["smooth_root_pos"][:, [0, 2]],
            global_root_heading=motion["global_root_heading"],
        ),
        FullBodyConstraintSet(
            skeleton,
            full,
            motion["posed_joints"][full],
            motion["global_rot_mats"][full],
            motion["smooth_root_pos"][full][:, [0, 2]],
        ),
        EndEffectorConstraintSet(
            skeleton,
            low,
            low_positions,
            motion["global_rot_mats"][low],
            motion["smooth_root_pos"][low][:, [0, 2]],
            joint_names=["Hips"],
        ),
    ]
    prompt = (
        "A person walks at a steady pace in a straight line, then near the middle "
        "lowers into a crouch to pass under an obstacle, stands upright again, "
        "and continues walking."
    )
    cache = Path(original["assets"]["prompt_cache"]["path"])
    with np.load(cache, allow_pickle=True) as arrays:
        if sanitize_text(prompt) not in arrays["prompts"].tolist():
            raise ValueError("exact sanitized prompt unavailable in pinned cache")
    out.mkdir(parents=True, exist_ok=False)
    save_constraints_lst(str(out / "constraints.json"), constraints)
    # Round-trip the actual saved input file through the installed public loader.
    loaded = load_constraints_lst(str(out / "constraints.json"), skeleton, device="cpu")
    rep = KimodoMotionRep(skeleton, fps=30)
    observed, mask = rep.create_conditions_from_constraints_batched(
        loaded, torch.tensor([180]), to_normalize=False, device="cpu"
    )
    if not torch.isfinite(observed).all() or mask.shape != (1, 180, 417):
        raise ValueError("native condition construction failed")
    np.savez_compressed(
        out / "condition_channels.npz", observed=observed.numpy(), mask=mask.numpy()
    )
    implementation = [
        Path(__file__),
        ROOT / "gear_sonic/dataset_generation/kimodo_motion_adapter.py",
        ROOT / "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py",
        ROOT / "gear_sonic/utils/motion_lib/torch_humanoid_batch.py",
    ]
    implementation += [
        KIMODO / p
        for p in (
            "kimodo/constraints.py",
            "kimodo/model/kimodo_model.py",
            "kimodo/model/twostage_denoiser.py",
            "kimodo/model/cfg.py",
            "kimodo/model/load_model.py",
            "kimodo/motion_rep/reps/base.py",
            "kimodo/motion_rep/reps/kimodo_motionrep.py",
            "kimodo/motion_rep/conditioning.py",
            "kimodo/exports/mujoco.py",
            "kimodo/exports/motion_io.py",
            "kimodo/postprocess.py",
            "kimodo/scripts/generate.py",
            "kimodo/skeleton/base.py",
            "kimodo/skeleton/definitions.py",
            "docs/source/user_guide/constraints.md",
            "docs/source/key_concepts/limitations.md",
        )
    ]
    sources = []
    for index, path in enumerate(implementation):
        snapshot = out / "source_snapshot" / f"{index:02d}_{path.name}"
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    heading = motion["global_root_heading"][0]
    proposal = {
        "schema": "motion2scene_conditioned_prior_experiment_proposal_v1",
        "status": "PROPOSED_NOT_INFERRED_NOT_PHYSICALLY_QUALIFIED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_registration": artifact(source_registration),
        "source_csv": artifact(csv),
        "assets": original["assets"],
        "constraints": artifact(out / "constraints.json"),
        "model_call": {
            "prompts": prompt,
            "num_frames": 180,
            "num_denoising_steps": 100,
            "num_samples": 1,
            "multi_prompt": False,
            "cfg_type": "separated",
            "cfg_weight": [2.0, 2.0],
            "post_processing": False,
            "return_numpy": True,
            "first_heading_angle": float(torch.atan2(heading[1], heading[0])),
        },
        "seed": 41002,
        "device": "cpu",
        "maximum_inference_attempts": 1,
        "environment": {**original["environment"], "PYTHONPATH": str(ROOT)},
        "prefix_kind": "planned qualified-neutral reference, not achieved state/history",
        "fullbody_condition_frames": full.tolist(),
        "low_pelvis_condition_frames": low.tolist(),
        "low_pelvis_drop_m": 0.085,
        "soft_low_interval_s": [2.0, 3.5],
        "canonicalization": {
            "subtract_from_all_condition_XZ_m": offset.tolist(),
            "restore_before_csv_mujoco_XY_m": [float(offset[1]), float(offset[0])],
            "root_height_unchanged_except_explicit_Hips_targets": True,
        },
        "fixed_candidate_entry_reference_phase_s": 0.4,
        "fixed_candidate_return_reference_phase_s": 5.5,
        "preregister_before_inference": [
            "complete runner/import-closure hashes and exact cache/checkpoint pins",
            "raw generated motion before MuJoCo hinge projection and CSV after conversion",
            "fixed single attempt; no clipping, blending, padding, retiming or automatic resampling",
            "native 29-joint limits at1e-6rad, full 180 source and299 loaded frames",
            "report full conditioning errors, all approach state/velocity and root-route errors",
            "existing entry/return joins: max joint delta<=.05rad and root delta<=.01m",
            "low-root target and native full-body envelope measured through2.0–3.5s, no beam or clearance claim",
            "failed gates retain candidate; no registry insertion or physics from this proposal",
            "later matched-history empty execution must verify actual transition and recovery separately",
        ],
        "evaluation_exclusion": "No V2/V3 reserved layout, geometry query or outcome used.",
        "source_snapshot": sources,
    }
    write(out / "proposal.json", proposal)
    report = {
        "schema": "motion2scene_native_constraint_api_audit_v1",
        "proposal": artifact(out / "proposal.json"),
        "constraints": artifact(out / "constraints.json"),
        "conditioning": artifact(out / "condition_channels.npz"),
        "source_qpos_shape": list(qpos.shape),
        "conditioning_shape": list(mask.shape),
        "conditioned_channels": int(mask.sum()),
        "fullbody_rotation_channels": int(mask[0, full, rep.slice_dict["global_rot_data"]].sum()),
        "hip_rotation_channels": int(mask[0, low, rep.slice_dict["global_rot_data"]].sum()),
        "converter_roundtrip_max_hinge_rad": float(abs(roundtrip[:, 7:] - qpos[:, 7:]).max()),
        "converter_roundtrip_max_root_m": float(abs(roundtrip[:, :3] - qpos[:, :3]).max()),
        "canonical_smooth_root_xz_at_zero_m": motion["smooth_root_pos"][0, [0, 2]].tolist(),
        "joint_name_expansion_Hips": skeleton.expand_joint_names(["Hips"]),
        "prompt_cache_exact_match": True,
        "model_loaded": False,
        "inference_calls": 0,
        "physics_steps": 0,
        "cuda_used": False,
        "scope": "Native input serialization, FK conversion and mask construction only; no generated output.",
    }
    write(out / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
