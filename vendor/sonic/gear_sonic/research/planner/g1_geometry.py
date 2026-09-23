"""G1 forward kinematics from the Isaac Lab URDF, and per-link floor/top extents (numpy only).

The URDF (``robot_description/urdf/g1/main.urdf``) is what the Isaac tracker simulates. Its 29
revolute joints have the same origins, axes and limits as the motion-lib MJCF
(``robot_description/mjcf/g1_29dof_rev_1_0.xml``); a test checks this against MuJoCo when
MuJoCo is installed. Two surface proxies are exposed:

- ``collision``: the URDF collision shapes Isaac uses for contacts (spheres, cylinders and
  meshes; the head link has none),
- ``visual``: the URDF visual meshes (the physical surface; the head top comes from here).

Mesh extents use each mesh's convex hull, which has the same extreme points.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from gear_sonic.dataset_generation.kimodo_motion_adapter import KIMODO_G1_JOINT_NAMES

ROBOT_DESCRIPTION_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "data/assets/robot_description",
    Path(
        "/home/robotixx/motion2scene-training/vendor/sonic/gear_sonic/data/assets/robot_description"
    ),
)
URDF_RELATIVE = "urdf/g1/main.urdf"
HEAD_LINK = "head_link"
SENSOR_LINK = "mid360_link"
FLOOR_CONTACT_THRESHOLD_M = 0.05

# Contact groups reported in the posture table (link names are URDF links).
BODY_GROUPS = {
    "pelvis": ("pelvis", "pelvis_contour_link"),
    "torso": ("waist_yaw_link", "waist_roll_link", "torso_link", "logo_link"),
    "head": ("head_link",),
    "left_upper_arm": (
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
    ),
    "right_upper_arm": (
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
    ),
    "left_elbow_forearm": ("left_elbow_link",),
    "right_elbow_forearm": ("right_elbow_link",),
    "left_hand": (
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "left_hand_palm_link",
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
        "left_hand_index_0_link",
        "left_hand_index_1_link",
    ),
    "right_hand": (
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "right_hand_palm_link",
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link",
    ),
    "left_thigh": ("left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link"),
    "right_thigh": ("right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link"),
    "left_knee_shin": ("left_knee_link",),
    "right_knee_shin": ("right_knee_link",),
    "left_foot": ("left_ankle_pitch_link", "left_ankle_roll_link"),
    "right_foot": ("right_ankle_pitch_link", "right_ankle_roll_link"),
}
FOOT_GROUPS = ("left_foot", "right_foot")


def find_robot_description(explicit: str | Path | None = None) -> Path:
    candidates = [Path(explicit)] if explicit else list(ROBOT_DESCRIPTION_CANDIDATES)
    for root in candidates:
        if (root / URDF_RELATIVE).is_file():
            return root
    raise FileNotFoundError(f"no {URDF_RELATIVE} under {candidates}")


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quat_wxyz_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation for a fixed unit axis and a batch of angles -> (T, 3, 3)."""
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    angle = np.asarray(angle, dtype=np.float64)[:, None, None]
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def load_stl_vertices(path: str | Path) -> np.ndarray:
    """Unique vertices of a binary or ASCII STL file."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * count:
            records = np.frombuffer(
                data[84:],
                dtype=np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
            )
            return np.unique(records["v"].reshape(-1, 3).astype(np.float64), axis=0)
    vertices = [
        [float(v) for v in line.split()[1:4]]
        for line in data.decode("ascii", errors="ignore").splitlines()
        if line.strip().startswith("vertex")
    ]
    if not vertices:
        raise ValueError(f"no vertices in {path}")
    return np.unique(np.asarray(vertices), axis=0)


def hull_vertices(vertices: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import ConvexHull

        return vertices[ConvexHull(vertices).vertices]
    except Exception:  # degenerate or scipy missing: all vertices give the same extents
        return vertices


@dataclass(frozen=True)
class Shape:
    link: str
    kind: str
    rotation: np.ndarray
    offset: np.ndarray
    radius: float = 0.0
    half_length: float = 0.0
    half_extents: np.ndarray | None = None
    vertices: np.ndarray | None = None
    mesh: str = ""


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    rotation: np.ndarray
    offset: np.ndarray
    axis: np.ndarray
    limits: tuple[float, float] | None


class G1Geometry:
    """Batch FK over MuJoCo-order qpos ``(T, 36)`` and z extents per link and body group."""

    def __init__(self, robot_description: str | Path | None = None):
        self.root_dir = find_robot_description(robot_description)
        self.urdf_path = self.root_dir / URDF_RELATIVE
        tree = ET.parse(self.urdf_path).getroot()
        self.joints: list[Joint] = []
        children = set()
        for element in tree.findall("joint"):
            origin = element.find("origin")
            xyz = [
                float(v)
                for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
            ]
            rpy = [
                float(v)
                for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
            ]
            axis_el = element.find("axis")
            axis = (
                np.array([float(v) for v in axis_el.get("xyz").split()])
                if axis_el is not None
                else np.zeros(3)
            )
            limit = element.find("limit")
            limits = (
                (float(limit.get("lower")), float(limit.get("upper")))
                if (limit is not None and element.get("type") == "revolute")
                else None
            )
            joint = Joint(
                element.get("name"),
                element.get("type"),
                element.find("parent").get("link"),
                element.find("child").get("link"),
                rpy_matrix(rpy),
                np.asarray(xyz),
                axis,
                limits,
            )
            self.joints.append(joint)
            children.add(joint.child)
        links = [link.get("name") for link in tree.findall("link")]
        roots = [name for name in links if name not in children]
        if roots != ["pelvis"]:
            raise ValueError(f"expected the single root link 'pelvis', got {roots}")
        self.joints = self._topological(self.joints)
        self.revolute = {j.name: j for j in self.joints if j.kind == "revolute"}
        if set(self.revolute) != set(KIMODO_G1_JOINT_NAMES):
            raise ValueError("URDF revolute joints differ from the 29 named G1 joints")
        self.joint_limits = np.array([self.revolute[name].limits for name in KIMODO_G1_JOINT_NAMES])
        self.shapes = {"collision": [], "visual": []}
        self._mesh_cache: dict[str, np.ndarray] = {}
        for link in tree.findall("link"):
            for kind in ("collision", "visual"):
                for element in link.findall(kind):
                    self.shapes[kind].append(self._shape(link.get("name"), element))

    @staticmethod
    def _topological(joints: list[Joint]) -> list[Joint]:
        ordered, known = [], {"pelvis"}
        remaining = list(joints)
        while remaining:
            ready = [j for j in remaining if j.parent in known]
            if not ready:
                raise ValueError("URDF joint graph is not a tree rooted at pelvis")
            for joint in ready:
                ordered.append(joint)
                known.add(joint.child)
                remaining.remove(joint)
        return ordered

    def _shape(self, link: str, element) -> Shape:
        origin = element.find("origin")
        xyz = np.array(
            [
                float(v)
                for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
            ]
        )
        rpy = [
            float(v)
            for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
        ]
        geometry = element.find("geometry")[0]
        rotation = rpy_matrix(rpy)
        if geometry.tag == "sphere":
            return Shape(link, "sphere", rotation, xyz, radius=float(geometry.get("radius")))
        if geometry.tag == "cylinder":
            return Shape(
                link,
                "cylinder",
                rotation,
                xyz,
                radius=float(geometry.get("radius")),
                half_length=float(geometry.get("length")) / 2,
            )
        if geometry.tag == "box":
            size = np.array([float(v) for v in geometry.get("size").split()])
            return Shape(link, "box", rotation, xyz, half_extents=size / 2)
        if geometry.tag == "mesh":
            name = geometry.get("filename")
            path = self.root_dir / name.split("robot_description/", 1)[-1]
            scale = np.array([float(v) for v in geometry.get("scale", "1 1 1").split()])
            if name not in self._mesh_cache:
                self._mesh_cache[name] = hull_vertices(load_stl_vertices(path))
            return Shape(
                link,
                "mesh",
                rotation,
                xyz,
                vertices=self._mesh_cache[name] * scale,
                mesh=Path(name).name,
            )
        raise ValueError(f"unsupported URDF geometry {geometry.tag}")

    def link_poses(self, qpos: np.ndarray, joint_names: Sequence[str] = KIMODO_G1_JOINT_NAMES):
        """World rotation (T, 3, 3) and position (T, 3) of every URDF link."""
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError("qpos must be (T, 36) MuJoCo order")
        index = {name: i for i, name in enumerate(joint_names)}
        poses = {"pelvis": (quat_wxyz_matrix(qpos[:, 3:7]), qpos[:, :3].copy())}
        for joint in self.joints:
            parent_r, parent_p = poses[joint.parent]
            rotation = parent_r @ joint.rotation
            position = parent_p + parent_r @ joint.offset
            if joint.kind == "revolute":
                rotation = rotation @ axis_angle_matrix(joint.axis, qpos[:, 7 + index[joint.name]])
            elif joint.kind != "fixed":
                raise ValueError(f"unsupported joint type {joint.kind}")
            poses[joint.child] = (rotation, position)
        return poses

    def link_z_extents(
        self, poses, proxy: str = "collision"
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per link: lowest and highest world z of its ``proxy`` shapes, each (T,)."""
        extents: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for shape in self.shapes[proxy]:
            link_r, link_p = poses[shape.link]
            rotation = link_r @ shape.rotation
            center_z = link_p[:, 2] + (link_r @ shape.offset)[:, 2]
            if shape.kind == "sphere":
                low, high = center_z - shape.radius, center_z + shape.radius
            elif shape.kind == "cylinder":
                a_z = np.abs(rotation[:, 2, 2])
                reach = shape.half_length * a_z + shape.radius * np.sqrt(np.clip(1 - a_z**2, 0, 1))
                low, high = center_z - reach, center_z + reach
            elif shape.kind == "box":
                reach = np.abs(rotation[:, 2, :]) @ shape.half_extents
                low, high = center_z - reach, center_z + reach
            else:
                z = center_z[:, None] + rotation[:, 2, :] @ shape.vertices.T
                low, high = z.min(axis=1), z.max(axis=1)
            if shape.link in extents:
                old_low, old_high = extents[shape.link]
                low, high = np.minimum(old_low, low), np.maximum(old_high, high)
            extents[shape.link] = (low, high)
        return extents

    def group_z_extents(self, poses, proxy: str = "collision"):
        links = self.link_z_extents(poses, proxy)
        groups = {}
        for group, members in BODY_GROUPS.items():
            present = [links[m] for m in members if m in links]
            if present:
                groups[group] = (
                    np.min([p[0] for p in present], axis=0),
                    np.max([p[1] for p in present], axis=0),
                )
        return groups

    def joint_limit_excess(self, qpos: np.ndarray) -> np.ndarray:
        """(T, 29) radians beyond the URDF limits (0 inside), MuJoCo joint order."""
        joints = np.asarray(qpos, dtype=np.float64)[:, 7:]
        lower, upper = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return np.maximum(np.maximum(lower - joints, joints - upper), 0.0)
