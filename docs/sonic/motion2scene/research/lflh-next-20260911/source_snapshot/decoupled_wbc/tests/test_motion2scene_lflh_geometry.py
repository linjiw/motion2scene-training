import math

import pytest
import torch

from scripts.research.lflh_next.geometry import box_sdf, capsule_samples, soft_lower_min


def test_weighted_average_can_hide_penetration():
    d = torch.tensor([-0.001, 0.009])
    old = (torch.softmax(-d / 0.01, dim=0) * d).sum()
    assert old > 0 and d.min() < 0
    from gear_sonic.dataset_generation.hallucination.sdf_decoder import SdfChoiceDecoder

    box = {k: torch.tensor([v]) for k, v in {
        "centre_x": 0., "centre_y": 0., "centre_z": 0.,
        "half_along_m": 1., "half_lateral_m": 1., "half_vertical_m": 1., "yaw": 0.
    }.items()}
    measured_old = SdfChoiceDecoder().clearance(
        torch.tensor([[0.999, 0., 0.], [1.009, 0., 0.]]), torch.zeros(2), box
    )
    assert measured_old.item() > 0
    assert soft_lower_min(d, 0.01) < 0


def test_lower_min_bounds_and_gradients():
    d = torch.tensor([-0.13, 0.05, 0.8], dtype=torch.float64, requires_grad=True)
    low = soft_lower_min(d, 0.02)
    assert d.min() - 0.02 * math.log(len(d)) <= low <= d.min()
    low.backward()
    assert torch.isfinite(d.grad).all() and torch.all(d.grad >= 0)
    assert torch.allclose(d.grad.sum(), torch.tensor(1., dtype=d.dtype))


def test_axis_subsampling_is_optimistic_and_correction_bounds_it():
    a, b = torch.tensor([[-1., 0, 0]]), torch.tensor([[1., 0, 0]])
    p, r, cover = capsule_samples(a, b, torch.tensor([0.05]), 2)
    gap = box_sdf(p, torch.zeros(3), torch.tensor([0.1, 0.1, 0.1]), torch.tensor(0.)) - r
    dense = torch.linspace(-1, 1, 2001)
    dense_points = torch.stack((dense, dense*0, dense*0), -1)
    true_dense = box_sdf(dense_points, torch.zeros(3), torch.full((3,), 0.1), torch.tensor(0.)).min()-0.05
    assert gap.min() > 0 and true_dense < 0
    assert (gap-cover).min() <= true_dense


def test_rigid_transform_invariance():
    p = torch.tensor([[2., 0.3, 0.2], [0., 0., 0.]])
    c = torch.tensor([0.4, 0.1, 0.0]); h = torch.tensor([0.3, 0.5, 0.6])
    a = torch.tensor(0.7)
    rot = torch.tensor([[a.cos(), -a.sin(), 0], [a.sin(), a.cos(), 0], [0., 0., 1.]])
    t = torch.tensor([8., -2., 4.])
    assert torch.allclose(box_sdf(p, c, h, torch.tensor(0.2)), box_sdf(p@rot.T+t, c@rot.T+t, h, torch.tensor(0.9)), atol=1e-6)


@pytest.mark.parametrize('tau', [0, -1])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError):
        soft_lower_min(torch.ones(2), tau)


def test_sampling_validates_inputs():
    with pytest.raises(ValueError):
        capsule_samples(torch.zeros(1, 3), torch.ones(1, 3), torch.ones(1), 1)
    with pytest.raises(ValueError):
        capsule_samples(torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([-1.]), 3)
