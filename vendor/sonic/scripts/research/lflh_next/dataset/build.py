"""Package frozen hindsight proposals, motion references and every sampled scene."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
import torch

from scripts.research.lflh_next.navigation.hindsight_study import LocalGenerator
from scripts.research.lflh_next.navigation.route_study import GEOMETRY, MJCF

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT.parent / "research-data/m2s-hindsight-events-100-20260911"
ROUTE = ROOT.parent / "research-data/m2s-kimodo-route-100-20260911"
SAMPLING = ROOT.parent / "research-data/m2s-sampling-debug-20260911"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def shape_record(spec):
    x, y, z, kind, yaw = map(float, spec)
    kind = int(kind)
    return {
        "position_m": [x, y, z],
        "quaternion_wxyz": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
        "yaw_rad": yaw,
        "shape": ["beam", "box", "sphere"][kind],
        "dimensions_m": [[0.3, 1.6, 0.2], [0.6, 0.6, 0.8], [0.6, 0.6, 0.6]][kind],
        "radius_m": 0.3 if kind == 2 else None,
        "semantic_asset": None,
        "mounting_or_support_qualified": False,
    }


def main(out):
    start = time.perf_counter()
    torch.set_num_threads(2)
    (out / "motions").mkdir(exist_ok=False)
    for name in ("scenes", "assets", "code", "dataset"):
        (out / name).mkdir(exist_ok=False)
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "state": "running",
        "physics_steps": 0,
        "optimizer_updates": 0,
        "new_clearance_queries": 0,
        "kinematic_calls": 0,
        "physical_passes": None,
        "physical_failures": None,
    }
    try:
        inputs = json.loads((SOURCE / "manifest.json").read_text())
        for row in inputs:
            assert sha(row["path"]) == row["sha256"], row["path"]
        design = json.loads((SOURCE / "design.json").read_text())
        ids = design["fit_ids"] + design["development_ids"]
        inventory = {r["id"]: r for r in json.loads((ROUTE / "inventory.json").read_text())}
        assert len(ids) == len(set(ids)) == 120
        assert not (
            {inventory[i]["group"] for i in ids[:100]} & {inventory[i]["group"] for i in ids[100:]}
        )
        labels = torch.load(SOURCE / "labels.pt", weights_only=True, map_location="cpu")
        model = LocalGenerator()
        checkpoint = SOURCE / "hindsight-801.pt"
        model.load_state_dict(
            torch.load(checkpoint, weights_only=True, map_location="cpu")["state"]
        )
        model.eval()
        with torch.inference_mode():
            probability = model(labels["features"], labels["recipes"]).softmax(-1)
        probability = probability.clone()
        previous = torch.load(SOURCE / "probabilities.pt", weights_only=True)["hindsight-801"]
        receipt["replayed_probability_max_absolute_error"] = float(
            (probability[100:] - previous).abs().max()
        )
        assert torch.allclose(probability[100:], previous, atol=1e-7, rtol=0)
        probability[100:] = previous
        shutil.copy2(checkpoint, out / "assets/hindsight-801.pt")
        shutil.copy2(SOURCE / "labels.pt", out / "assets/frozen-proposal-inputs.pt")
        shutil.copy2(GEOMETRY, out / "assets/native-clearance-geometry.json")
        shutil.copytree(MJCF.parent, out / "assets/native_model/mjcf")
        shutil.copytree(
            (MJCF.parent / "../meshes/g1").resolve(), out / "assets/native_model/meshes/g1"
        )
        sampler = ROOT / "scripts/research/lflh_next/navigation/sampling.js"
        assert sha(sampler) == sha(SAMPLING / "viewer/sampling.js")
        shutil.copy2(sampler, out / "code/sampling.js")
        requests = []
        for i, id in enumerate(ids):
            seed = int(hashlib.sha256(f"dataset-v1:{id}".encode()).hexdigest()[:8], 16)
            for count in (3, 5):
                requests.append(
                    dict(
                        weights=probability[i].tolist(),
                        mask=(labels["lower"][i] >= 0.02).tolist(),
                        count=count,
                        seed=seed,
                        avoidOverlap=True,
                        specs=labels["specs"][i].tolist(),
                    )
                )
        ts = time.perf_counter()
        sampler_js = "const fs=require('fs'),s=require(process.argv[1]);const a=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>s.sampleSet(x))));"
        process = subprocess.run(
            ["node", "-e", sampler_js, str(out / "code/sampling.js")],
            input=json.dumps(requests),
            capture_output=True,
            text=True,
            check=True,
        )
        sets = json.loads(process.stdout)
        receipt["measured_sampling_seconds"] = time.perf_counter() - ts
        data = json.loads(
            (SAMPLING / "viewer/data.js").read_text().removeprefix("const DATA=").removesuffix(";")
        )
        meshes = data["mesh"]
        write(out / "assets/visual-meshes.json", meshes)
        native = mujoco.MjModel.from_xml_path(str(MJCF))
        state = mujoco.MjData(native)
        joint_names = [
            mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, native.njnt)
        ]
        body_names = [
            mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_BODY, j) for j in range(1, native.nbody)
        ]
        geom_ids = [g["id"] for g in meshes]
        old_display = {m["id"]: m for m in json.loads((SOURCE / "display.json").read_text())}
        catalog, assignments, obstacle_rows = [], [], []
        maximum_replay_error = 0.0
        for i, id in enumerate(ids):
            row = inventory[id]
            src = Path(row["path"])
            assert sha(src) == row["sha256"]
            inputs.append({"path": str(src), "sha256": row["sha256"]})
            destination = out / "motions" / id
            destination.mkdir()
            shutil.copy2(src, destination / "source.csv")
            q = np.loadtxt(src, delimiter=",")
            assert q.shape == (row["frames"], native.nq) and np.isfinite(q).all()
            q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=-1, keepdims=True)
            body_p, body_q, geom_p, geom_q = [], [], [], []
            for frame in q:
                state.qpos[:] = frame
                mujoco.mj_kinematics(native, state)
                receipt["kinematic_calls"] += 1
                body_p.append(state.xpos[1:].copy())
                body_q.append(state.xquat[1:].copy())
                geom_p.append(state.geom_xpos[geom_ids].copy())
                quat = np.empty((len(geom_ids), 4))
                for k, g in enumerate(geom_ids):
                    mujoco.mju_mat2Quat(quat[k], state.geom_xmat[g])
                geom_q.append(quat)
            body_p, body_q = np.array(body_p), np.array(body_q)
            geom_p, geom_q = np.array(geom_p), np.array(geom_q)
            shift = np.r_[body_p[0, 0, :2], body_p[:, :, 2].min() - 0.1]
            q[:, :3] -= shift
            body_p -= shift
            geom_p -= shift
            fps = float(row["fps"])
            if id in old_display:
                old = old_display[id]
                indices = np.rint(np.array(old["times"]) * fps).astype(int)
                error = float(np.abs(q[indices] - np.asarray(old["qpos"])).max())
                maximum_replay_error = max(maximum_replay_error, error)
                assert error < 1e-10, (id, error)
            np.savez_compressed(
                destination / "reference.npz",
                qpos=q,
                time_s=np.arange(len(q)) / fps,
                body_position_m=body_p,
                body_quaternion_wxyz=body_q,
                geom_position_m=geom_p,
                geom_quaternion_wxyz=geom_q,
                fps=fps,
                source_to_scene_translation_m=-shift,
                joint_names=np.array(joint_names),
                body_names=np.array(body_names),
                geom_ids=np.array(geom_ids),
            )
            candidates = []
            for j, spec in enumerate(labels["specs"][i]):
                lo, up = float(labels["lower"][i, j]), float(labels["inner_upper"][i, j])
                cf = labels["cf_upper"][i, j].tolist()
                candidates.append(
                    {
                        "cell": j,
                        **shape_record(spec),
                        "spec_xyzkindyaw": spec.tolist(),
                        "anchor_time_s": float(labels["recipes"][i, j, 0]) * 3 * (len(q) - 1) / fps,
                        "raw_q": float(probability[i, j]),
                        "reference_lower_bound_m": lo,
                        "reference_inner_upper_bound_m": up,
                        "alternative_inner_upper_bounds_m": cf,
                        "reference_clear": lo >= 0.02,
                        "penetration_witness": up < 0,
                        "kinematic_contrast": lo >= 0.02 and min(cf) < 0,
                        "mask_reason": (
                            None
                            if lo >= 0.02
                            else (
                                "penetration_witness"
                                if up < 0
                                else "uncertain_or_insufficient_clearance"
                            )
                        ),
                    }
                )
            write(destination / "candidates.json", candidates)
            sampled = np.unique(np.linspace(0, len(q) - 1, 60).round().astype(int))
            write(
                destination / "viewer.json",
                {
                    "id": id,
                    "times": (sampled / fps).tolist(),
                    "root": q[:, :3].tolist(),
                    "geom_position_m": geom_p[sampled].tolist(),
                    "geom_quaternion_wxyz": geom_q[sampled].tolist(),
                },
            )
            metadata = {
                **row,
                "split": "train" if i < 100 else "development",
                "source_sha256": row["sha256"],
                "reference_npz": f"motions/{id}/reference.npz",
                "source_to_scene_translation_m": (-shift).tolist(),
                "coordinate_frame": "z up, metres, source heading retained; inherited body-origin height convention",
                "quaternion_order": "wxyz",
                "joint_position_units": "radians",
                "teacher_eligible": False,
                "teacher_actions": None,
                "physical_passage": None,
                "executed_contact_log": None,
                "student_sensor_stream": None,
                "isaac_rendered_frames": 0,
                "native_model_sha256": sha(MJCF),
                "generator_sha256": sha(checkpoint),
            }
            write(destination / "metadata.json", metadata)
            catalog.append(
                {
                    k: metadata[k]
                    for k in (
                        "id",
                        "split",
                        "category",
                        "prompts",
                        "frames",
                        "fps",
                        "group",
                        "joint_limit_excess_rad",
                        "teacher_eligible",
                    )
                }
            )
            for offset, count in enumerate((3, 5)):
                result = sets[2 * i + offset]
                if count == 5:
                    assert [d["cell"] for d in result["draws"][:3]] == [
                        d["cell"] for d in sets[2 * i]["draws"]
                    ]
                scene_id = f"{id}-n{count}"
                draws = []
                for draw in result["draws"]:
                    c = candidates[draw["cell"]] if draw["cell"] is not None else None
                    draws.append({**draw, "obstacle": c})
                    obstacle_rows.append(
                        {
                            "scene_id": scene_id,
                            "motion_id": id,
                            "draw": draw["draw"],
                            "status": draw["status"],
                            "cell": draw["cell"],
                            "conditional_q": draw["conditionalQ"],
                            "raw_q": c["raw_q"] if c else None,
                            "shape": c["shape"] if c else None,
                            "x_m": c["position_m"][0] if c else None,
                            "y_m": c["position_m"][1] if c else None,
                            "z_m": c["position_m"][2] if c else None,
                            "yaw_rad": c["yaw_rad"] if c else None,
                            "clearance_lower_m": c["reference_lower_bound_m"] if c else None,
                        }
                    )
                physical_labels = {
                    "passage": None,
                    "failure": None,
                    "censored": None,
                    "contact_log": None,
                    "teacher_actions": None,
                }
                write(
                    out / "scenes" / f"{scene_id}.json",
                    {
                        "scene_id": scene_id,
                        "motion_id": id,
                        "split": metadata["split"],
                        "generator": "hindsight-801",
                        "generator_sha256": sha(checkpoint),
                        "reference": f"../motions/{id}/reference.npz",
                        "sampling": {**result, "draws": draws},
                        "physical_labels": physical_labels,
                        "teacher_eligible": False,
                        "status": "reference_geometry_only",
                        "isaac_render_status": "BLOCKED_GPU_PREFLIGHT",
                    },
                )
                assignments.append(
                    {
                        "scene_id": scene_id,
                        "motion_id": id,
                        "split": metadata["split"],
                        "requested_obstacles": count,
                        "sampled_obstacles": count - result["abstentions"],
                        "abstentions": result["abstentions"],
                        "seed": result["seed"],
                        "all_sampled_reference_clear": all(
                            d["obstacle"]["reference_clear"] for d in draws if d["obstacle"]
                        ),
                        "kinematic_contrast_obstacles": sum(
                            d["obstacle"]["kinematic_contrast"] for d in draws if d["obstacle"]
                        ),
                        "physics_runs": 0,
                        "passes": None,
                        "physical_failures": None,
                        "censoring": None,
                        "technical_failure": None,
                        "unrun": True,
                        "isaac_render_status": "BLOCKED_GPU_PREFLIGHT",
                        "teacher_eligible": False,
                    }
                )
        for name, rows in (("assignments", assignments), ("obstacles", obstacle_rows)):
            write(out / f"{name}.json", rows)
            with (out / f"{name}.csv").open("x") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write(out / "catalog.json", catalog)
        for path in (MJCF, GEOMETRY, ROUTE / "inventory.json", sampler, Path(__file__)):
            inputs.append({"path": str(path), "sha256": sha(path)})
        write(out / "inputs.json", inputs)
        shutil.copytree(SAMPLING / "viewer", out / "viewer")
        shutil.copy2(out / "viewer/plotly.js", out / "dataset/plotly.js")
        for src, dst in (("viewer.html", "index.html"), ("viewer.js", "app.js")):
            shutil.copy2(Path(__file__).with_name(src), out / "dataset" / dst)
        text = (out / "viewer/index.html").read_text()
        text = text.replace(
            "<body>",
            '<body><p style="padding:12px"><a href="../dataset/index.html" style="color:#61dfc2">Open frozen dataset v1: 120 references and 240 scene assignments →</a></p>',
            1,
        )
        (out / "viewer/index.html").write_text(text)
        receipt.update(
            state="complete",
            motion_references=len(ids),
            scene_assignments=len(assignments),
            frame_count=sum(r["frames"] for r in catalog),
            sampled_obstacles=sum(r["sampled_obstacles"] for r in assignments),
            abstentions=sum(r["abstentions"] for r in assignments),
            scenes_with_abstention=sum(r["abstentions"] > 0 for r in assignments),
            scenes_with_kinematic_contrast=sum(
                r["kinematic_contrast_obstacles"] > 0 for r in assignments
            ),
            inference_candidate_scores=probability.numel(),
            prior_display_qpos_max_absolute_error=maximum_replay_error,
            GPU_metrics=None,
            physical_teacher_eligible_scenes=0,
        )
    except Exception as error:
        receipt.update(state="failed", error=repr(error), retry_reason="No automatic retry")
        raise
    finally:
        receipt["measured_packaging_wall_seconds"] = time.perf_counter() - start
        write(out / "build-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
