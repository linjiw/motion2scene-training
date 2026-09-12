"""Build private standalone reference viewer with native G1 visual mesh assets."""

import json
from pathlib import Path

import mujoco
import numpy as np
from plotly.offline import get_plotlyjs
import torch

from scripts.research.lflh_next.navigation.study import MJCF, OUT
from scripts.research.lflh_next.qualification.split_safe_learning import sha


def cluster_mesh(vertices, faces, pitch=0.008):
    """Display-only vertex clustering; never used for geometric labels."""
    _, inverse = np.unique(np.floor(vertices / pitch).astype(np.int64), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    v = np.stack([np.bincount(inverse, weights=vertices[:, k]) / count for k in range(3)], axis=-1)
    f = inverse[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v, np.unique(f, axis=0)


def main():
    destination = OUT / "viewer"
    destination.mkdir(exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    geoms = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 1:
            continue
        mid = model.geom_dataid[g]
        va, fa = model.mesh_vertadr[mid], model.mesh_faceadr[mid]
        v = model.mesh_vert[va : va + model.mesh_vertnum[mid]]
        f = model.mesh_face[fa : fa + model.mesh_facenum[mid]]
        v, f = cluster_mesh(v, f)
        geoms.append(
            dict(id=g, v=v.round(6).tolist(), f=f.tolist(), color=model.geom_rgba[g, :3].tolist())
        )
    display = json.loads((OUT / "display.json").read_text())
    for motion in display:
        poses = []
        for q in motion.pop("qpos"):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            poses.append(
                [
                    dict(
                        p=data.geom_xpos[g["id"]].round(6).tolist(),
                        r=data.geom_xmat[g["id"]].round(7).tolist(),
                    )
                    for g in geoms
                ]
            )
        motion["mesh_poses"] = poses
    labels = torch.load(OUT / "labels.pt", weights_only=True)
    probs = torch.load(OUT / "probabilities.pt", weights_only=True)
    payload = dict(
        motions=display,
        mesh=geoms,
        spec=labels["specs"].tolist(),
        lower=labels["lower"][100:].tolist(),
        upper=labels["inner_upper"][100:].tolist(),
        prob={k: v.tolist() for k, v in probs.items()},
    )
    (destination / "data.js").write_text(
        "const DATA=" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";"
    )
    (destination / "plotly.js").write_text(get_plotlyjs())
    for name in ["index.html", "app.js"]:
        (destination / name).write_text(Path(__file__).with_name(name).read_text())
    assets = []
    for p in [MJCF, *sorted((MJCF.parent / "../meshes/g1").resolve().glob("*.STL"))]:
        assets.append(dict(path=str(p), sha256=sha(p)))
    (destination / "asset-receipt.json").write_text(
        json.dumps(
            dict(
                robot_assets=assets,
                visual_geoms=len(geoms),
                display_faces=sum(len(g["f"]) for g in geoms),
                display_vertices=sum(len(g["v"]) for g in geoms),
                simplification="8 mm vertex clusters; display only; labels use unchanged native bound geometry",
                obstacles="Exact previously labelled box, sphere, beam primitives. No downloaded indoor semantic asset; no transfer of labels to furniture",
                physics_steps=0,
                kinematic_calls=sum(len(m["times"]) for m in display),
            ),
            indent=2,
        )
    )
    print(destination / "index.html")


if __name__ == "__main__":
    main()
