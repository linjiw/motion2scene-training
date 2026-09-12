#!/usr/bin/env python3
"""Reference-versus-executed construction with identical native collision shapes.

Reference poses use the commanded, loader-resampled full schedule, including
entry and return. Native URDF forward kinematics is checked against measured
poses from the same qualified executions before screening any candidate.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_prepare_acquisition_pools import read_bound  # noqa: E402
from motion2scene_timed_scene_screen import beam_clearance  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (  # noqa: E402
    acquisition_queues,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)


def urdf_poses(urdf, root_position, root_quaternion, joint_angles, joint_names, body_names):
    """Batched URDF tree FK, with named joints and wxyz input/output quaternions."""
    xml = ET.parse(urdf).getroot()
    joints = xml.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    roots = {link.get("name") for link in xml.findall("link")} - children
    q = np.asarray(joint_angles, float)
    xyz, quat = np.asarray(root_position, float), np.asarray(root_quaternion, float)
    names = tuple(joint_names)
    if len(roots) != 1 or len(set(names)) != len(names) or q.shape != (len(xyz), len(names)):
        raise ValueError("one rooted URDF and uniquely named joint columns required")
    if (
        xyz.shape != (len(q), 3)
        or quat.shape != (len(q), 4)
        or not all(np.isfinite(v).all() for v in (q, xyz, quat))
    ):
        raise ValueError("finite aligned root and joint poses required")
    rotations = {roots.pop(): Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()}
    positions = {next(iter(rotations)): xyz}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in positions:
                continue
            origin = joint.find("origin")
            offset = np.fromstring(
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            rpy = np.fromstring(
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            if offset.shape != (3,) or rpy.shape != (3,):
                raise ValueError("three-coordinate joint origins required")
            rotation = rotations[parent] @ Rotation.from_euler("xyz", rpy).as_matrix()
            position = positions[parent] + np.einsum("tij,j->ti", rotations[parent], offset)
            kind = joint.get("type")
            if kind != "fixed":
                if joint.get("name") not in names or kind not in (
                    "revolute",
                    "continuous",
                    "prismatic",
                ):
                    raise ValueError("every moving joint must be explicitly measured")
                axis_node = joint.find("axis")
                axis = np.fromstring(
                    axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" "
                )
                if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1):
                    raise ValueError("unit joint axis required")
                angle = q[:, names.index(joint.get("name"))]
                if kind == "prismatic":
                    position = position + np.einsum("tij,tj->ti", rotation, angle[:, None] * axis)
                else:
                    rotation = rotation @ Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
            positions[child], rotations[child] = position, rotation
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("disconnected or cyclic URDF")
    xyz = np.stack([positions[name] for name in body_names], axis=1)
    matrices = np.stack([rotations[name] for name in body_names], axis=1)
    quat = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(*xyz.shape[:2], 4)
    return xyz, quat[..., [3, 0, 1, 2]]


def build_reference_envelopes(registration, urdf):
    source = read_bound(registration["option_qualification_inputs"][0])
    manifest = read_bound(source["manifest"])
    geometry = read_bound(manifest["geometry"])
    shapes = {"outer": {}, "inner": {}}
    for shape in geometry["shapes"]:
        capsule = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        shapes["outer"].setdefault(shape["owner"], []).append(capsule)
        if shape["role"] == "native_primitive_subset":
            shapes["inner"].setdefault(shape["owner"], []).append(capsule)
    envelopes, validation = {}, []
    for row in source["rows"]:
        ref = read_bound(row["evidence"])["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
        if (
            payload["quat_format"] != "wxyz"
            or payload["dof_order"] != payload["reference_g1_qpos_dof_order"]
        ):
            raise ValueError(
                "declared common measured/reference joint and quaternion order required"
            )
        names, bodies = payload["dof_joint_names"], payload["body_names"]
        actual_xyz, actual_quat = urdf_poses(
            urdf, payload["root_pos_w"], payload["root_quat_w"], payload["dof_pos"], names, bodies
        )
        position_error = np.linalg.norm(actual_xyz - payload["body_pos_w"], axis=-1)
        measured_quat = np.asarray(payload["body_quat_w"], dtype=float)
        measured_quat = measured_quat / np.linalg.norm(measured_quat, axis=-1, keepdims=True)
        dots = np.abs(np.sum(actual_quat * measured_quat, axis=-1))
        angle_error = 2 * np.arccos(np.clip(dots, 0, 1))
        validation.append(
            dict(
                option_id=row["cell_id"],
                trajectory=ref,
                max_position_error_m=float(position_error.max()),
                max_orientation_error_rad=float(angle_error.max()),
            )
        )
        if position_error.max() > 0.005 or angle_error.max() > 0.01:
            raise ValueError(f"native FK pose convention check failed: {validation[-1]}")
        reference = np.asarray(payload["reference_g1_qpos"])
        reference_xyz, reference_quat = urdf_poses(
            urdf, reference[:, :3], reference[:, 3:7], reference[:, 7:], names, bodies
        )
        envelopes[row["cell_id"]] = {
            kind: body_capsules_world(reference_xyz, reference_quat, bodies, capsules=capsules)
            for kind, capsules in shapes.items()
        }
    return envelopes, validation


def run(pools, out):
    registration = read_bound(artifact(pools / "registration.json"))
    source = read_bound(artifact(pools / "result.json"))
    urdf = ROOT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "experiment.json",
        dict(
            pools=artifact(pools / "result.json"),
            registration=artifact(pools / "registration.json"),
            urdf=artifact(urdf),
            implementation=artifact(Path(__file__)),
            intervention="commanded reference FK versus measured executed body poses; same native shapes",
            maximum_fk_position_error_m=0.005,
            maximum_fk_orientation_error_rad=0.01,
            physical_steps=0,
        ),
    )
    envelopes, validation = build_reference_envelopes(registration, urdf)
    write_new(out / "kinematics.json", dict(rows=validation))
    summaries = []
    ids = list(envelopes)
    offsets = registration["offsets_world_xyz_yaw"]
    for pool in source["pools"]:
        candidates = read_bound(pool["candidates"])["rows"]
        outer = np.full((len(candidates), len(ids)), 0.2)
        inner = np.full_like(outer, 0.2)
        counts = np.ones_like(outer, dtype=int)
        queries = 0
        start = time.monotonic()
        for i, candidate in enumerate(candidates):
            # Native scene geometry is shared byte-for-byte with executed construction.
            scene = read_bound(candidate["definition"])
            for a, option in enumerate(ids):
                inner[i, a] = min(
                    beam_clearance(envelopes[option]["inner"], b, offsets[0])
                    for b in scene["beams"]
                )
                nominal = min(
                    beam_clearance(envelopes[option]["outer"], b, offsets[0])
                    for b in scene["beams"]
                )
                outer[i, a] = nominal
                queries += 2 * len(scene["beams"])
                if nominal >= registration["margin_m"]:
                    for offset in offsets[1:]:
                        outer[i, a] = min(
                            outer[i, a],
                            min(
                                beam_clearance(envelopes[option]["outer"], b, offset)
                                for b in scene["beams"]
                            ),
                        )
                        queries += len(scene["beams"])
                    counts[i, a] = len(offsets)
            if (i + 1) % 100 == 0:
                print(
                    json.dumps(dict(seed=pool["seed"], screened=i + 1, total=len(candidates))),
                    flush=True,
                )
        queues = acquisition_queues(candidates, ids, outer, counts, inner)
        folder = out / f"seed_{pool['seed']}"
        folder.mkdir()
        np.savez_compressed(
            folder / "geometry.npz",
            candidate_ids=[c["candidate_id"] for c in candidates],
            option_ids=ids,
            outer_minimum_m=outer,
            inner_minimum_m=inner,
            counts=counts,
        )
        rows = queues["analytic_contrast"]
        definitions = {
            candidate["candidate_id"]: candidate["definition"] for candidate in candidates
        }
        for row in rows:
            row["definition"] = definitions[row["candidate_id"]]
        queue = write_new(folder / "reference_contrast.json", dict(rows=rows, seed=pool["seed"]))
        summaries.append(
            dict(
                seed=pool["seed"],
                queue=queue,
                eligible=len(rows),
                queries=queries,
                seconds=time.monotonic() - start,
                candidates=pool["candidates"],
            )
        )
    write_new(
        out / "result.json",
        dict(
            experiment=artifact(out / "experiment.json"),
            pools=summaries,
            kinematics=artifact(out / "kinematics.json"),
            physical_steps=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.pools, args.out)
