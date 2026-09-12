import mujoco
import numpy as np
import torch

from scripts.research.lflh_next.multimotion.run import gaps
from scripts.research.lflh_next.qualification.audit import orientation_error, surface_witness


def test_orientation_comparison_is_sign_and_scale_invariant():
    q = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    np.testing.assert_allclose(orientation_error(q, -2 * q), 0, atol=1e-12)


def test_primitive_noncontainment_witness_is_not_only_a_root_test():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="link"><geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    witness = surface_witness(data, model, np.array([[0.0, 0.0, 0.0]]))
    assert witness > 0.4
    # Dense overlapping sphere centers can contain these finite samples.
    centers = np.stack([np.linspace(0, 0.5, 20), np.zeros(20), np.zeros(20)], axis=-1)
    assert surface_witness(data, model, centers) < 0


def test_temporal_subsampling_can_miss_penetration():
    full = torch.tensor([[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    obstacle = torch.tensor([[0.0, 0, 0, 2.0]])
    assert gaps(full[[0, 2]], obstacle).item() > 0.02
    assert gaps(full, obstacle).item() < 0
