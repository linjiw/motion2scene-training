"""Draw G1 reference poses from existing mesh transforms; no physics is run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/linjiw/research-data/m2s-hindsight-dataset-v1-20260911")


def rotation(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    meshes = json.loads((DATA / "assets/visual-meshes.json").read_text())
    camera = np.array([3.5, -6, 2.2])
    camera /= np.linalg.norm(camera)
    right = np.cross([0, 0, 1], camera)
    right /= np.linalg.norm(right)
    up = np.cross(camera, right)
    proj = np.stack([right, up, camera], -1)
    light = np.array([-0.4, -0.7, 1.0])
    light /= np.linalg.norm(light)
    manifest = []
    for name, clip, fraction in [
        ("walk", "00413", 0.25),
        ("turn", "00413", 0.60),
        ("crouch", "00916", 0.45),
        ("stand", "00413", 0.95),
    ]:
        data = np.load(DATA / f"motions/{clip}/reference.npz")
        index = round(fraction * (len(data["qpos"]) - 1))
        triangles, colors = [], []
        root_rotation = rotation(data["qpos"][index, 3:7])
        yaw = np.arctan2(root_rotation[1, 0], root_rotation[0, 0])
        canonical = rotation([np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2)])
        origin = data["qpos"][index, :3]
        for j, mesh in enumerate(meshes):
            rot = rotation(data["geom_quaternion_wxyz"][index, j])
            vertices = np.asarray(mesh["v"]) @ rot.T + data["geom_position_m"][index, j]
            vertices = (vertices - origin) @ canonical.T
            tri = vertices[np.asarray(mesh["f"], dtype=int)]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
            shade = 0.40 + 0.60 * np.clip(normal @ light, 0, 1)
            base = np.asarray(mesh["color"])
            base = np.clip(0.25 + 0.7 * base, 0, 1)
            colors.extend(np.clip(shade[:, None] * base, 0, 1))
            triangles.extend(tri @ proj)
        triangles, colors = np.array(triangles), np.array(colors)
        order = np.argsort(triangles[:, :, 2].mean(1))
        xy = triangles[:, :, :2]
        low, high = xy.min((0, 1)), xy.max((0, 1))
        span = high - low
        fig = plt.figure(figsize=(4.4, 6.5), facecolor="none")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_collection(PolyCollection(xy[order], facecolors=colors[order], edgecolors="none"))
        ax.set_xlim(low[0] - 0.08 * span[0], high[0] + 0.08 * span[0])
        ax.set_ylim(low[1] - 0.035 * span[1], high[1] + 0.035 * span[1])
        fig.savefig(ROOT / f"assets/g1_{name}.png", dpi=180, transparent=True)
        plt.close(fig)
        manifest.append(
            dict(
                asset=f"g1_{name}.png",
                clip=clip,
                reference_frame=index,
                role="reference-pose visualization, not an executed state",
                source=str(DATA / f"motions/{clip}/reference.npz"),
            )
        )
    (ROOT / "assets/pose_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
