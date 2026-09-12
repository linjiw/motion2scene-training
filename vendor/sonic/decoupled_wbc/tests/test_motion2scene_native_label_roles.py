import torch

from scripts.research.lflh_next.qualification.native_labels import bound_roles


def test_outer_mesh_overlap_is_not_a_penetration_witness():
    lower, upper = bound_roles(
        torch.tensor([-0.1, 0.4]),
        torch.tensor([0.1, 0.1]),
        torch.zeros(2),
        torch.tensor([False, True]),
    )
    assert lower < 0 and upper > 0


def test_clearance_uses_outer_geometry_as_well_as_primitives():
    lower, upper = bound_roles(
        torch.tensor([0.105, 1.0]),
        torch.tensor([0.1, 0.1]),
        torch.zeros(2),
        torch.tensor([False, True]),
    )
    assert lower < 0.02 and upper > 0.02


def test_spatial_cover_correction_does_not_create_negative_witness():
    lower, upper = bound_roles(
        torch.tensor([0.12]),
        torch.tensor([0.1]),
        torch.tensor([0.05]),
        torch.tensor([True]),
    )
    assert lower < 0 and upper > 0
