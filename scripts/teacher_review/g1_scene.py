"""MuJoCo scene with two kinematic copies of the exact Isaac training asset (urdf/g1/main.urdf).

``robot/`` is drawn with its URDF colors; ``ghost/`` is a translucent blue reference.
Collision geoms (group 0) are hidden at render time; the floor is in group 2.
"""

import re

import mujoco
import numpy as np

from review_common import MESHDIR, MJ_JOINTS, TRACKED_BODIES, URDF

GHOST_RGBA = (0.20, 0.55, 1.0, 0.28)
FAIL_RGBA = np.array([0.85, 0.15, 0.15, 1.0])

SCENE_XML = """
<mujoco>
  <visual>
    <headlight ambient="0.40 0.40 0.40" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.96 0.97 0.99" rgb2="0.80 0.84 0.90"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1="0.90 0.91 0.93"
             rgb2="0.85 0.86 0.89" markrgb="0.55 0.57 0.62" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="1 -1 6" dir="-0.2 0.2 -1" directional="true" castshadow="true"
           diffuse="0.55 0.55 0.55"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" group="2"
          contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def g1_spec():
    text = URDF.read_text().replace("package://robot_description/meshes/g1/", "")
    compiler = (
        f'<mujoco><compiler meshdir="{MESHDIR}" discardvisual="false" fusestatic="false"/>'
        "</mujoco>"
    )
    text = re.sub(r"<robot([^>]*)>", lambda m: f"<robot{m.group(1)}>{compiler}", text, count=1)
    spec = mujoco.MjSpec.from_string(text)
    spec.worldbody.first_body().add_freejoint(name="root")
    return spec


class G1Scene:
    def __init__(self):
        spec = mujoco.MjSpec.from_string(SCENE_XML)
        for prefix, rgba in (("robot/", None), ("ghost/", GHOST_RGBA)):
            child = g1_spec()
            for geom in child.geoms:
                geom.contype = 0
                geom.conaffinity = 0
                if rgba is not None:
                    geom.rgba = rgba
            spec.worldbody.add_frame().attach_body(child.worldbody.first_body(), prefix, "")
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.addr = {}
        for prefix in ("robot/", "ghost/"):
            root = m.jnt_qposadr[m.joint(prefix + "root").id]
            joints = [m.jnt_qposadr[m.joint(prefix + name).id] for name in MJ_JOINTS]
            self.addr[prefix] = (int(root), np.asarray(joints, dtype=int))
        self.bodies = {
            prefix: np.asarray([m.body(prefix + name).id for name in TRACKED_BODIES])
            for prefix in ("robot/", "ghost/")
        }
        body_names = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        self.robot_geoms = np.asarray(
            [g for g, name in enumerate(body_names) if name.startswith("robot/")]
        )
        self.robot_rgba = m.geom_rgba[self.robot_geoms].copy()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0] = 0

    def set_pose(self, prefix, qpos36):
        root, joints = self.addr[prefix]
        self.data.qpos[root : root + 7] = qpos36[:7]
        self.data.qpos[joints] = qpos36[7:]

    def pose(self, robot_q, ghost_q):
        self.set_pose("robot/", robot_q)
        self.set_pose("ghost/", ghost_q)
        mujoco.mj_kinematics(self.model, self.data)

    def tint_robot(self, failed):
        if failed:
            blend = 0.45 * self.robot_rgba + 0.55 * FAIL_RGBA
            blend[:, 3] = 1.0
            self.model.geom_rgba[self.robot_geoms] = blend
        else:
            self.model.geom_rgba[self.robot_geoms] = self.robot_rgba

    def tracked_positions(self, prefix):
        return self.data.xpos[self.bodies[prefix]].copy()


def fk_check(scene, run, stride=1):
    """Max 14-body distance (mm) between MuJoCo FK of captured poses and native Isaac positions."""
    robot_err, ghost_err = [], []
    names = run.body_names
    order = [names.index(b) for b in TRACKED_BODIES]
    for k in range(0, run.valid, stride):
        scene.pose(run.robot_q[k], run.track_q[run.ghost_index(k)])
        robot = scene.tracked_positions("robot/")
        ghost = scene.tracked_positions("ghost/")
        robot_err.append(np.linalg.norm(robot - (run.tracked[k, order] - run.origin), axis=-1).max())
        ghost_err.append(np.linalg.norm(ghost - (run.reference[k, order] - run.origin), axis=-1).max())
    to_mm = lambda values: float(np.max(values) * 1000.0) if values else None  # noqa: E731
    med = lambda values: float(np.median(values) * 1000.0) if values else None  # noqa: E731
    return {
        "robot_fk_max_mm": to_mm(robot_err),
        "robot_fk_median_mm": med(robot_err),
        "ghost_fk_max_mm": to_mm(ghost_err),
        "ghost_fk_median_mm": med(ghost_err),
        "frames_checked": len(robot_err),
    }
