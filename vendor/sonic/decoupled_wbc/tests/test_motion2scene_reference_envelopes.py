from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_reference_envelopes import urdf_poses


def test_named_fk_applies_joint_motion_before_descendant_offset(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<robot name="test"><link name="root"/><link name="arm"/><link name="tip"/>
    <joint name="hinge" type="revolute"><parent link="root"/><child link="arm"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
    <joint name="end" type="fixed"><parent link="arm"/><child link="tip"/>
    <origin xyz="1 0 0"/></joint></robot>""")
    positions, rotations = urdf_poses(
        urdf, [[0, 0, 0]], [[1, 0, 0, 0]], [[np.pi / 2]], ["hinge"], ["tip", "root"]
    )
    np.testing.assert_allclose(positions[0], [[1, 1, 0], [0, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(np.abs(rotations[0, 0]), [2**-0.5, 0, 0, 2**-0.5], atol=1e-12)
    with pytest.raises(ValueError, match="explicitly measured"):
        urdf_poses(urdf, [[0, 0, 0]], [[1, 0, 0, 0]], np.empty((1, 0)), [], ["root"])
