#!/usr/bin/env python3
"""Register one explicit hinge-box projection and neutral/prior transition splice."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/linjiw/groot-wbc-sonic-sim-trackb")
DATA = Path("/home/linjiw/research-data/groot-wbc")
NEUTRAL = DATA / "m2s-longer-reference-development-v1"
RAW = DATA / "m2s-kimodo-conditioned-lowheight-development-v1"


def artifact(path):
    path = Path(path)
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def write(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check(ref):
    if artifact(ref["path"]) != ref:
        raise ValueError(f"Registered artifact changed: {ref['path']}")


def quintic(t):
    t = np.asarray(t, dtype=np.float64)
    if np.any((t < 0) | (t > 1)):
        raise ValueError("quintic requires t in [0,1]")
    return t**3 * (10 + t * (-15 + 6 * t))


def slerp_wxyz(a, b, weight):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    weight = np.asarray(weight, dtype=float)[..., None]
    sine = np.sin(angle)
    safe = np.where(sine > 1e-8, sine, 1)
    spherical = np.sin((1 - weight) * angle) / safe * a + np.sin(weight * angle) / safe * b
    linear = (1 - weight) * a + weight * b
    result = np.where(sine > 1e-8, spherical, linear)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def canonical(qpos):
    result = np.array(qpos, dtype=float, copy=True)
    result[:, :2] -= result[0, :2].copy()
    return result


def project_and_splice(neutral, raw, limits):
    neutral, raw, limits = np.asarray(neutral), np.asarray(raw), np.asarray(limits)
    if neutral.shape != (180, 36) or raw.shape != neutral.shape or limits.shape != (29, 2):
        raise ValueError("Expected two 180 x 36 sources and 29 native joint intervals")
    if not all(np.isfinite(a).all() for a in (neutral, raw, limits)):
        raise ValueError("Nonfinite source/limits")
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Reversed joint limits")
    if np.any(np.linalg.norm(neutral[:, 3:7], axis=1) < 1e-8) or np.any(
        np.linalg.norm(raw[:, 3:7], axis=1) < 1e-8
    ):
        raise ValueError("Zero root quaternion")
    projected = raw.copy()
    projected[:, 7:] = np.clip(raw[:, 7:], limits[:, 0], limits[:, 1])
    phase = np.arange(180, dtype=float) / 30
    weight = np.ones(180, dtype=float)
    weight[phase <= 1] = 0
    weight[phase >= 5] = 0
    entering = (phase > 1) & (phase < 1.4)
    returning = (phase > 4.5) & (phase < 5)
    weight[entering] = quintic((phase[entering] - 1) / 0.4)
    weight[returning] = 1 - quintic((phase[returning] - 4.5) / 0.5)
    result = (1 - weight[:, None]) * neutral + weight[:, None] * projected
    blending = entering | returning
    result[blending, 3:7] = slerp_wxyz(
        neutral[blending, 3:7], projected[blending, 3:7], weight[blending]
    )
    # Explicit authored prefix/tail, and untouched projected core, including quaternion bytes.
    result[weight == 0] = neutral[weight == 0]
    result[weight == 1] = projected[weight == 1]
    return result, projected, weight


def native_model(mjcf):
    xml = ET.parse(mjcf).getroot().find("worldbody")
    joints = [j for j in xml.findall(".//joint") if j.get("type") != "free"]
    names = [j.get("name") for j in joints]
    limits = np.array([[float(v) for v in j.get("range").split()] for j in joints])
    spheres = []
    for body in xml.findall(".//body"):
        if body.get("name") not in ("left_ankle_roll_link", "right_ankle_roll_link"):
            continue
        for geom in body.findall("geom"):
            if (
                geom.get("mesh")
                or geom.get("contype") == "0"
                or geom.get("type", "sphere") != "sphere"
            ):
                continue
            spheres.append(
                {
                    "body": body.get("name"),
                    "local_center_m": [float(x) for x in geom.get("pos").split()],
                    "radius_m": float(geom.get("size")),
                }
            )
    if len(spheres) != 8:
        raise ValueError("Native contact-sphere contract changed")
    return names, limits, spheres


def register(out):
    from bundle_motion2scene_sources import closure

    out.mkdir(parents=True, exist_ok=False)
    sources = closure(
        [
            Path(__file__).resolve(),
            ROOT / "gear_sonic/dataset_generation/kimodo_motion_adapter.py",
            ROOT / "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py",
        ]
    )
    snapshots = []
    for path in sorted(sources):
        dest = out / "source_snapshot/repository" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        snapshots.append({"original": artifact(path), "snapshot": artifact(dest)})
    mjcf = ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
    names, limits, spheres = native_model(mjcf)
    inputs = {
        "neutral_csv": NEUTRAL
        / "generated/000_a_person_walks_at_a_steady_pace_in_a_straight_li_s0.csv",
        "neutral_native": NEUTRAL / "native_reference_50hz.npz",
        "neutral_pkl": NEUTRAL / "neutral.pkl",
        "neutral_qualification": DATA / "m2s-longer-neutral-qualification-v2/result.json",
        "raw_prior_csv": RAW / "generated/conditioned.csv",
        "raw_full_skeleton": RAW / "raw_model_motion.npz",
        "raw_completed": RAW / "completed.json",
        "raw_registration": RAW / "registration.json",
        "raw_native": RAW / "native_reference_50hz.npz",
        "native_mjcf": mjcf,
    }
    driver = out / "source_snapshot/repository/scripts/research" / Path(__file__).name
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONPATH": str(out / "source_snapshot/repository"),
    }
    registration = {
        "schema": "motion2scene_single_prior_projection_splice_registration_v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "One authored derivative of rejected neural prior, planned reference only; "
            "no inference or physics"
        ),
        "inputs": {k: artifact(v) for k, v in inputs.items()},
        "source_files": snapshots,
        "driver": artifact(driver),
        "environment": env,
        "command": [
            str(ROOT / ".venv_isaaclab/bin/python"),
            str(driver),
            "author",
            "--out",
            str(out),
        ],
        "working_directory": str(ROOT),
        "versions": {
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "joblib")
        },
        "operator": {
            "source_frames": 180,
            "source_fps": 30,
            "target_fps": 50,
            "canonicalization": (
                "For each parent independently subtract its initial XY; "
                "preserve height, orientation, clock"
            ),
            "projection": "Clamp only violating 29 hinge values to native closed joint boxes",
            "joint_names": names,
            "native_joint_intervals_rad": limits.tolist(),
            "prefix_neutral_inclusive_s": [0, 1.0],
            "tail_neutral_inclusive_s": [5.0, 179 / 30],
            "enter_blend_s": [1.0, 1.4],
            "return_blend_s": [4.5, 5.0],
            "weight": "s(u)=6u^5-15u^4+10u^3; candidate weight s on entry, 1-s on return",
            "position_and_hinges": "Pointwise affine blend, no resampling or root floor correction",
            "orientation": (
                "Shortest-arc normalized WXYZ quaternion SLERP in blend interior only; "
                "parent values exact outside"
            ),
        },
        "audit": {
            "numerical_joint_tolerance_rad": 1e-6,
            "entry_reference_phase_s": 0.3,
            "return_reference_phase_s": 5.3,
            "position_guard_max_joint_rad": 0.05,
            "position_guard_pelvis_distance_m": 0.01,
            "ground_z_m": 0,
            "sole_spheres": spheres,
            "ground_numeric_tolerance_m": 1e-6,
            "floor_gate": (
                "Flag any sole sphere surface below fixed ground minus tolerance; "
                "report neutral/raw comparators unchanged"
            ),
            "foot_scope": (
                "Eight native MJCF sole spheres only; not mesh or cooked PhysX clearance "
                "and not dynamic contact"
            ),
            "low_interval_s": [2.0, 3.5],
            "descriptive_low_drop_threshold_m": 0.04,
            "velocity_acceleration": (
                "Source30Hz finite differences whole sequence and blend windows; "
                "compare neutral/raw/derived, no hardware bounds assumed"
            ),
        },
        "maximum_authored_candidates": 1,
        "failure_policy": (
            "Retain candidate and failed checks; no further correction, root height adjustment, "
            "reinference or automatic physical admission"
        ),
        "inference_calls": 0,
        "physics_steps": 0,
        "evaluation_queries": 0,
        "source_ancestry": "Source41002 development neutral and its one softly conditioned Kimodo sample",
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


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def limit_report(dof, limits):
    excess = np.maximum(limits[:, 0] - dof, 0) + np.maximum(dof - limits[:, 1], 0)
    return {
        "max_excess_rad": float(excess.max()),
        "violating_frames": int(np.any(excess > 1e-6, axis=1).sum()),
        "violating_joint_samples": int(np.sum(excess > 1e-6)),
    }


def derivatives(qpos):
    phases = np.arange(len(qpos), dtype=float) / 30
    intervals = {"whole": [0, phases[-1]], "entry_blend": [1.0, 1.4], "return_blend": [4.5, 5.0]}
    angular = (
        Rotation.from_quat(qpos[:-1, [4, 5, 6, 3]]).inv()
        * Rotation.from_quat(qpos[1:, [4, 5, 6, 3]])
    ).as_rotvec() * 30
    signals = {"joint": qpos[:, 7:], "root_xyz": qpos[:, :3]}
    rows = {}
    for interval, (start, stop) in intervals.items():
        # Include finite-difference stencils touching each blend boundary.
        vmask = (phases[1:] >= start) & (phases[:-1] <= stop)
        amask = (phases[2:] >= start) & (phases[:-2] <= stop)
        row = {}
        for key, signal in signals.items():
            row[key + "_max_abs_velocity"] = float(
                np.abs(np.diff(signal, axis=0)[vmask] * 30).max()
            )
            row[key + "_max_abs_acceleration"] = float(
                np.abs(np.diff(signal, n=2, axis=0)[amask] * 900).max()
            )
        row["root_rotation_max_speed_rad_s"] = float(np.linalg.norm(angular[vmask], axis=1).max())
        rows[interval] = row
    return rows


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def sole_heights(native, spheres):
    positions, names = native["global_translation"], list(native["body_names"])
    heights = []
    for sphere in spheres:
        index = names.index(sphere["body"])
        local = np.asarray(sphere["local_center_m"])
        center = (
            np.einsum("tij,j->ti", native["global_rotation_mat"][:, index], local)
            + positions[:, index]
        )
        heights.append(center[:, 2] - sphere["radius_m"])
    return np.stack(heights, axis=1)


def author(out):
    import torch

    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import init_humanoid_fk
    from gear_sonic.dataset_generation.kimodo_motion_adapter import (
        KIMODO_G1_JOINT_NAMES,
        load_kimodo_qpos_csv,
        qpos_to_sonic_motion_entry,
        save_sonic_motion_file,
    )

    if (out / "author_started.json").exists():
        raise ValueError("The registered single authored attempt already started")
    registration = json.loads((out / "registration.json").read_text())
    for key, value in registration["environment"].items():
        if os.environ.get(key) != value:
            raise ValueError(f"Registered environment differs: {key}")
    for ref in registration["inputs"].values():
        check(ref)
    for ref in registration["source_files"]:
        check(ref["snapshot"])
    check(registration["driver"])
    if torch.cuda.is_available():
        raise ValueError("CPU only")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    started = time.monotonic()
    write(
        out / "author_started.json",
        {
            "registration": artifact(out / "registration.json"),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        inputs, specification = registration["inputs"], registration["operator"]
        assert tuple(specification["joint_names"]) == KIMODO_G1_JOINT_NAMES
        neutral = canonical(load_kimodo_qpos_csv(inputs["neutral_csv"]["path"]))
        raw = canonical(load_kimodo_qpos_csv(inputs["raw_prior_csv"]["path"]))
        limits = np.asarray(specification["native_joint_intervals_rad"])
        derived, projected, weight = project_and_splice(neutral, raw, limits)
        (out / "generated").mkdir()
        np.savetxt(out / "generated/derived.csv", derived, delimiter=",", fmt="%.18e")
        np.savez_compressed(
            out / "authored_stages_30hz.npz",
            neutral_canonical=neutral,
            raw_prior_canonical=raw,
            projected_prior=projected,
            derived=derived,
            candidate_weight=weight,
            phase_s=np.arange(180) / 30,
            joint_names=np.array(specification["joint_names"]),
        )
        entry = qpos_to_sonic_motion_entry(derived, source_fps=30)
        save_sonic_motion_file(
            out / "derived.pkl",
            motion_key="longer_development_41002_projected_splice",
            motion_entry=entry,
        )
        write(
            out / "derived.pkl.manifest.json",
            {
                "schema": "motion2scene_authored_prior_motion_v1",
                "source_csv": artifact(out / "generated/derived.csv"),
                "registration": artifact(out / "registration.json"),
                "motion_key": "longer_development_41002_projected_splice",
                "source_fps": 30,
                "motion_file": artifact(out / "derived.pkl"),
                "origin": "Explicit hinge projection and neutral/prior splice; not raw model output",
            },
        )
        fk = init_humanoid_fk()
        with torch.no_grad():
            loaded = fk.fk_batch(
                torch.as_tensor(entry["pose_aa"])[None],
                torch.as_tensor(entry["root_trans_offset"])[None],
                return_full=True,
                fps=30,
                target_fps=50,
                interpolate_data=True,
            )
        native = {
            name: value.detach().cpu().numpy()[0]
            for name, value in loaded.items()
            if isinstance(value, torch.Tensor)
        }
        n = len(native["global_translation"])
        native.update(
            phase_s=np.arange(n) / 50,
            body_names=np.array(fk.body_names),
            joint_names=np.array(specification["joint_names"]),
        )
        np.savez_compressed(out / "native_reference_50hz.npz", **native)
        comparators = {
            "neutral": load_npz(inputs["neutral_native"]["path"]),
            "raw_prior": load_npz(inputs["raw_native"]["path"]),
            "derived": native,
        }
        for value in comparators.values():
            assert np.array_equal(value["phase_s"], native["phase_s"])
            assert np.array_equal(value["body_names"], native["body_names"])
            assert np.array_equal(value["joint_names"], native["joint_names"])
        phase = native["phase_s"]
        neutral_native = comparators["neutral"]
        joint_delta = abs(native["dof_pos"] - neutral_native["dof_pos"])
        root_delta = np.linalg.norm(
            native["global_translation"][:, 0] - neutral_native["global_translation"][:, 0], axis=1
        )
        guards = []
        for t in [0.3, 5.3]:
            i = np.flatnonzero(abs(phase - t) < 1e-8).item()
            guards.append(
                {
                    "reference_phase_s": t,
                    "max_joint_difference_rad": float(joint_delta[i].max()),
                    "pelvis_distance_m": float(root_delta[i]),
                    "max_joint_velocity_difference_rad_s": float(
                        abs(native["dof_vels"][i] - neutral_native["dof_vels"][i]).max()
                    ),
                    "passes_position_guard": bool(
                        joint_delta[i].max() <= 0.05 and root_delta[i] <= 0.01
                    ),
                }
            )
        sphere_heights = {
            key: sole_heights(value, registration["audit"]["sole_spheres"])
            for key, value in comparators.items()
        }
        np.savez_compressed(
            out / "native_sole_surface_heights.npz", **sphere_heights, phase_s=phase
        )
        floor_report = {
            key: {
                "signed_surface_height_m": stats(value),
                "frames_with_any_sphere_below_minus_1e_6_m": int(
                    np.any(value < -1e-6, axis=1).sum()
                ),
                "sphere_samples_below_minus_1e_6_m": int((value < -1e-6).sum()),
                "per_sphere_minimum_m": value.min(axis=0).tolist(),
            }
            for key, value in sphere_heights.items()
        }
        correction = projected[:, 7:] - raw[:, 7:]
        prefix, tail = np.arange(180) <= 30, np.arange(180) >= 150
        low = (phase >= 2) & (phase <= 3.5)
        root_drop = (
            neutral_native["global_translation"][:, 0, 2] - native["global_translation"][:, 0, 2]
        )
        top_drop = neutral_native["global_translation"][:, :, 2].max(axis=1) - native[
            "global_translation"
        ][:, :, 2].max(axis=1)
        intervals, active_start = [], None
        for i, active in enumerate(root_drop >= 0.04):
            if active and active_start is None:
                active_start = i
            if active_start is not None and (not active or i == len(phase) - 1):
                end = i if active else i - 1
                intervals.append(
                    {
                        "first_phase_s": float(phase[active_start]),
                        "last_phase_s": float(phase[end]),
                        "samples": end - active_start + 1,
                    }
                )
                active_start = None
        checks = {
            "source_shape": derived.shape == (180, 36),
            "native_frames299": n == 299,
            "source_finite": bool(np.isfinite(derived).all()),
            "native_finite": all(
                bool(np.isfinite(v).all())
                for v in native.values()
                if np.issubdtype(v.dtype, np.number)
            ),
            "source_box_limits": limit_report(derived[:, 7:], limits)["max_excess_rad"] <= 1e-6,
            "native_box_limits": limit_report(native["dof_pos"], limits)["max_excess_rad"] <= 1e-6,
            "source_prefix_exact": bool(np.array_equal(derived[prefix], neutral[prefix])),
            "source_tail_exact": bool(np.array_equal(derived[tail], neutral[tail])),
            "native_prefix_joint_root_exact": bool(
                np.max(joint_delta[phase <= 1]) == 0 and np.max(root_delta[phase <= 1]) == 0
            ),
            "native_tail_joint_root_exact": bool(
                np.max(joint_delta[phase >= 5]) == 0 and np.max(root_delta[phase >= 5]) == 0
            ),
            "registered_entry_return_guards": all(row["passes_position_guard"] for row in guards),
            "sole_spheres_not_below_ground": bool(sphere_heights["derived"].min() >= -1e-6),
        }
        result = {
            "schema": "motion2scene_prior_projection_splice_result_v1",
            "registration": artifact(out / "registration.json"),
            "status": (
                "kinematic_checks_pass_physics_unqualified"
                if all(checks.values())
                else "retained_with_failed_kinematic_checks"
            ),
            "checks": checks,
            "all_registered_kinematic_checks_pass": all(checks.values()),
            "physics_admitted": False,
            "inference_calls": 0,
            "physics_steps": 0,
            "evaluation_queries": 0,
            "authored_candidates": 1,
            "source_frames": 180,
            "source_fps": 30,
            "native_frames": n,
            "native_last_phase_s": float(phase[-1]),
            "source_joint_names": specification["joint_names"],
            "hinge_projection": {
                "changed_source_frames": int(np.any(correction != 0, axis=1).sum()),
                "changed_joint_samples": int((correction != 0).sum()),
                "max_correction_rad": float(abs(correction).max()),
                "per_joint_max_correction_rad": abs(correction).max(axis=0).tolist(),
            },
            "splice": {
                "changed_frames_relative_raw_canonical": int(np.any(derived != raw, axis=1).sum()),
                "changed_frames_relative_projected_canonical": int(
                    np.any(derived != projected, axis=1).sum()
                ),
                "source_prefix_max_abs_difference": float(
                    abs(derived[prefix] - neutral[prefix]).max()
                ),
                "source_tail_max_abs_difference": float(abs(derived[tail] - neutral[tail]).max()),
                "derivation": specification,
            },
            "joint_limits": {
                key: {
                    "source30hz": limit_report(qpos[:, 7:], limits),
                    "native50hz": limit_report(comparators[key]["dof_pos"], limits),
                }
                for key, qpos in [("neutral", neutral), ("raw_prior", raw), ("derived", derived)]
            },
            "entry_return_reference_guards": guards,
            "source_finite_difference_diagnostics": {
                key: derivatives(qpos)
                for key, qpos in [("neutral", neutral), ("raw_prior", raw), ("derived", derived)]
            },
            "native_sole_floor_geometry": floor_report,
            "low_interval_s": [2, 3.5],
            "low_interval_pelvis_drop_m": stats(root_drop[low]),
            "low_interval_top_body_origin_drop_m": stats(top_drop[low]),
            "pelvis_drop_at_least_0_04m_intervals": intervals,
            "root_xyz_height_range_m": [float(derived[:, 2].min()), float(derived[:, 2].max())],
            "native_horizontal_root_deviation_from_neutral_m": stats(
                np.linalg.norm(
                    native["global_translation"][:, 0, :2]
                    - neutral_native["global_translation"][:, 0, :2],
                    axis=1,
                )
            ),
            "limits": [
                "Authored derivative; does not change rejection of uncorrected neural prior.",
                "Reference guards do not qualify controller transitions, duration, stability or floor contact.",
                (
                    "Sole clearance uses eight native MJCF spheres; "
                    "no mesh/world beam clearance or cooked PhysX test."
                ),
                "Body-origin height drop is not an executed or collider clearance envelope.",
                (
                    "Finite differences are descriptive; no actuator speed/acceleration limits "
                    "or dynamics admission assumed."
                ),
            ],
            "artifacts": [
                artifact(out / p)
                for p in [
                    "generated/derived.csv",
                    "authored_stages_30hz.npz",
                    "derived.pkl",
                    "derived.pkl.manifest.json",
                    "native_reference_50hz.npz",
                    "native_sole_surface_heights.npz",
                ]
            ],
        }
        write(out / "result.json", result)
        write(
            out / "execution.json",
            {
                "status": "completed",
                "wall_seconds": time.monotonic() - started,
                "result": artifact(out / "result.json"),
            },
        )
        print(
            json.dumps(
                {
                    "result": artifact(out / "result.json"),
                    "checks": checks,
                    "projection": result["hinge_projection"],
                    "floor": floor_report,
                    "guards": guards,
                    "low": result["low_interval_pelvis_drop_m"],
                }
            )
        )
    except Exception as error:
        write(
            out / "execution.json",
            {
                "status": "failed",
                "wall_seconds": time.monotonic() - started,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "author"])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    {"register": register, "author": author}[args.mode](args.out.resolve())


if __name__ == "__main__":
    main()
