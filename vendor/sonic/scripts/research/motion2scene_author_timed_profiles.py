"""Preregister and author two finite local-crouch candidates on a qualified carrier."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import joblib  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import init_humanoid_fk  # noqa: E402
from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    G1_ISAACLAB_TO_MUJOCO_DOF,
    qpos_to_sonic_motion_entry,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    DEFAULT_RAMP,
    DEFAULT_RANGE_KEEP,
    MAX_CROUCH_EXCURSION_RAD,
    adaptation_profile,
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.motion_prefilter import load_joint_limits  # noqa: E402
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF  # noqa: E402


def register(out, neutral, qualification):
    expected = "sha256:ef33c2139f7a6278d670c012edc9db70671d08b6842e8757d15dc6a6b61b3990"
    checked(neutral, expected)
    result = json.loads(qualification.read_text())
    if result.get("neutral_horizon_qualified") != 1 or not all(
        result["rows"][0]["full_horizon_checks"].values()
    ):
        raise ValueError("requires the complete qualified six-second neutral horizon")
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        out / "registration.json",
        {
            "schema": "motion2scene_authored_duration_profiles_v1",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "neutral": artifact(neutral),
            "neutral_qualification": artifact(qualification),
            "actual_neutral_bank": result["rows"][0]["bank"],
            "mjcf": artifact(DEFAULT_G1_MJCF),
            "operator": "local_crouch",
            "station_fraction": 0.55,
            "target_drop_m": 0.085,
            "profiles": [
                {"label": "short", "window_half_width": 0.18},
                {"label": "sustained", "window_half_width": 0.48},
            ],
            "ramp": DEFAULT_RAMP,
            "range_keep": DEFAULT_RANGE_KEEP,
            "max_excursion_rad": MAX_CROUCH_EXCURSION_RAD,
            "waist_use_fraction": 0.0,
            "source_frames": 180,
            "source_fps": 30,
            "expected_loaded_frames": 299,
            "proposed_entry_tick": 15,
            "proposed_return_tick": 265,
            "reference_guard": {"joint_jump_rad": 0.05, "root_jump_m": 0.01},
            "reference_resampling": (
                "native CPU Humanoid_Batch quaternion/axis-angle interpolation, matching actual loader"
            ),
            "later_return_audit": (
                "report every legal tick265..297 and its remaining tail; "
                "do not silently alter proposed schedule"
            ),
            "implementation": sources,
            "scope": (
                "CPU authored candidates only; no new prior sampling, physical qualification, padding or retiming"
            ),
        },
    )
    print(json.dumps({"registration": artifact(out / "registration.json")}))


def author(out):
    import torch

    if torch.cuda.is_available():
        raise ValueError("authoring must run with CUDA_VISIBLE_DEVICES='' on CPU")
    reg = json.loads((out / "registration.json").read_text())
    for name in ("neutral", "neutral_qualification", "actual_neutral_bank", "mjcf"):
        checked(Path(reg[name]["path"]), reg[name]["sha256"])
    for ref in reg["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "author_attempt.json",
        {
            "registration": artifact(out / "registration.json"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    started = time.monotonic()
    entry = next(iter(joblib.load(reg["neutral"]["path"]).values()))
    if entry["fps"] != 30 or len(entry["dof"]) != 180:
        raise ValueError("qualified neutral source shape/rate mismatch")
    qpos = np.c_[
        entry["root_trans_offset"], entry["root_rot"][:, [3, 0, 1, 2]], entry["dof"]
    ].astype(float)
    fk = init_humanoid_fk()

    def native_qpos(motion):
        with torch.no_grad():
            native = fk.fk_batch(
                torch.as_tensor(motion["pose_aa"], dtype=torch.float32)[None],
                torch.as_tensor(motion["root_trans_offset"], dtype=torch.float32)[None],
                return_full=True,
                fps=30,
                target_fps=50,
                interpolate_data=True,
            )
        arrays = {
            k: v.detach().cpu().numpy()[0] for k, v in native.items() if isinstance(v, torch.Tensor)
        }
        return np.c_[
            arrays["global_translation"][:, 0], arrays["global_rotation"][:, 0], arrays["dof_pos"]
        ]

    neutral50 = native_qpos(entry)
    bank = np.load(reg["actual_neutral_bank"]["path"])
    bank_joint_error = float(
        abs(bank["joint_pos"][0][:, G1_ISAACLAB_TO_MUJOCO_DOF] - neutral50[:, 7:]).max()
    )
    bank_root_error = float(abs(bank["root_xyz"][0] - neutral50[:, :3]).max())
    if bank_joint_error > 1e-4 or bank_root_error > 1e-4:
        raise ValueError("source-clock interpolation differs from the physically loaded neutral")
    _, limits = load_joint_limits(reg["mjcf"]["path"])
    rows = []
    for profile in reg["profiles"]:
        label = profile["label"]
        adapted, report = local_crouch(
            qpos,
            reg["station_fraction"],
            target_drop_m=reg["target_drop_m"],
            window=profile["window_half_width"],
            ramp=reg["ramp"],
            range_keep=reg["range_keep"],
            max_excursion=reg["max_excursion_rad"],
            waist_use_fraction=reg["waist_use_fraction"],
            mjcf_path=reg["mjcf"]["path"],
        )
        np.savetxt(out / f"{label}.csv", adapted, delimiter=",", fmt="%.12g")
        converted = qpos_to_sonic_motion_entry(
            adapted, source_fps=30, canonicalize_horizontal_origin=False
        )
        joblib.dump({f"authored_six_second_{label}": converted}, out / f"{label}.pkl")
        candidate50 = native_qpos(converted)
        joint = abs(candidate50[:, 7:] - neutral50[:, 7:]).max(axis=1)
        root = np.linalg.norm(candidate50[:, :3] - neutral50[:, :3], axis=1)
        eligible = (joint <= reg["reference_guard"]["joint_jump_rad"]) & (
            root <= reg["reference_guard"]["root_jump_m"]
        )
        alpha = adaptation_profile(
            route_progress(qpos[:, :2]),
            reg["station_fraction"],
            window=profile["window_half_width"],
            ramp=reg["ramp"],
        )
        active = np.flatnonzero(alpha > 1e-9)
        core = np.flatnonzero(alpha > 0.9)
        violations = np.maximum(
            np.maximum(limits[:, 0] - adapted[:, 7:], adapted[:, 7:] - limits[:, 1]), 0
        )
        requested = [
            {
                "tick": tick,
                "time_s": tick / 50,
                "joint_jump_rad": float(joint[tick]),
                "root_jump_m": float(root[tick]),
                "passes_reference_guard": bool(eligible[tick]),
            }
            for tick in (reg["proposed_entry_tick"], reg["proposed_return_tick"])
        ]
        later = [
            {
                "tick": int(tick),
                "time_s": float(tick / 50),
                "remaining_command_ticks": 298 - int(tick),
                "joint_jump_rad": float(joint[tick]),
                "root_jump_m": float(root[tick]),
            }
            for tick in np.flatnonzero(eligible)
            if 265 <= tick <= 297
        ]
        np.savez_compressed(
            out / f"{label}_reference_guard.npz",
            qpos_mujoco_50hz=candidate50,
            neutral_qpos_mujoco_50hz=neutral50,
            joint_jump_rad=joint,
            root_jump_m=root,
            eligible=eligible,
            source_alpha=alpha,
            source_fps=np.array(30.0),
        )
        row = {
            "label": label,
            "motion": artifact(out / f"{label}.pkl"),
            "csv": artifact(out / f"{label}.csv"),
            "reference_guard": artifact(out / f"{label}_reference_guard.npz"),
            "operator_report": asdict(report),
            "source_frames": len(adapted),
            "source_fps": 30,
            "anticipated_loaded_frames": len(candidate50),
            "root_xy_exact": bool(np.array_equal(adapted[:, :2], qpos[:, :2])),
            "root_quaternion_exact": bool(np.array_equal(adapted[:, 3:7], qpos[:, 3:7])),
            "maximum_native_joint_limit_violation_rad": float(violations.max()),
            "active_source_interval_s": [float(active[0] / 30), float(active[-1] / 30)],
            "core_source_interval_s": [float(core[0] / 30), float(core[-1] / 30)],
            "core_source_sample_span_s": float((core[-1] - core[0]) / 30),
            "proposed_schedule_reference_checks": requested,
            "later_return_guard_candidates": later,
            "physical_qualification": None,
            "guard_scope": (
                "source interpolation matched to loaded neutral; "
                "actual alternate bank/jump must be checked in simulator"
            ),
        }
        rows.append(row)
        write_new(
            out / f"{label}.pkl.manifest.json",
            {"neutral": reg["neutral"], "registration": artifact(out / "registration.json"), **row},
        )
    result = {
        "schema": "motion2scene_authored_duration_profiles_result_v1",
        "registration": artifact(out / "registration.json"),
        "rows": rows,
        "neutral_loaded_joint_interpolation_max_error_rad": bank_joint_error,
        "neutral_loaded_root_interpolation_max_error_m": bank_root_error,
        "physics_steps": 0,
        "wall_seconds": time.monotonic() - started,
        "scope": "authored reference candidates and kinematic guard estimates only; all proposals retained",
    }
    write_new(out / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "author"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--neutral", type=Path)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.stage == "register":
        register(args.out, args.neutral, args.qualification)
    else:
        author(args.out)
