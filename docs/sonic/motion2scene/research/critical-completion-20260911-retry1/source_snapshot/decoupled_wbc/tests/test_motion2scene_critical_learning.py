import pytest
import torch

from scripts.research.lflh_next.critical import (
    cover_loss,
    critical_mask,
    observations,
    pair_loss,
    partition,
)


def test_partition_complete_disjoint():
    train, test = partition(32, 48, 4)
    assert not (train & test).any()
    assert (train | test).all()
    assert train.sum() == 1152 and test.sum() == 384


def test_unknown_is_not_negative_and_clear_alternatives_are_not_negatives():
    energy = torch.tensor([[[1.0]], [[2.0]], [[3.0]]], requires_grad=True)
    lower = torch.tensor([[[0.02]], [[0.03]], [[-0.1]]])
    upper = torch.tensor([[[0.04]], [[0.05]], [[0.1]]])
    train = torch.ones(1, 1, dtype=torch.bool)
    assert pair_loss(energy, lower, upper, train, 0.01) == 0
    assert not critical_mask(lower, upper, 0.01).any()
    upper[2] = -0.02
    crit = critical_mask(lower, upper, 0.01)
    assert crit[:, 0, 0].tolist() == [True, True, False]
    loss = pair_loss(energy, lower, upper, train, 0.01)
    loss.backward()
    assert energy.grad[0] < 0 and energy.grad[1] < 0 and energy.grad[2] > 0


def test_inconsistent_bounds_rejected():
    with pytest.raises(ValueError):
        critical_mask(torch.ones(1, 1, 1), -torch.ones(1, 1, 1), 0.01)


def test_cover_spreads_mass_and_retains_unsupported_target():
    e = torch.tensor([[[3.0, 0.0, 0.0]], [[1.0, 2.0, 3.0]]], requires_grad=True)
    positives = torch.tensor([[[True, True, False]], [[False, False, False]]])
    loss = cover_loss(e, positives, torch.ones(1, 3, dtype=torch.bool))
    loss.backward()
    assert e.grad[0, 0, 0] > 0 and e.grad[0, 0, 1] < 0
    assert torch.equal(e.grad[1], torch.zeros_like(e.grad[1]))


def test_withheld_values_never_enter_observations_or_losses():
    train = torch.tensor([[True, False], [True, False]])
    f = torch.ones(2, 2, 2, 2, requires_grad=True)
    requested = torch.ones(2, 2, 2, dtype=torch.bool)
    visible, mask = observations(f, requested, train)
    assert not visible[:, :, ~train].any() and not mask[:, :, ~train].any()
    visible.sum().backward()
    assert not f.grad[:, :, ~train].any()
    e = torch.randn(2, 2, 2, requires_grad=True)
    p = torch.ones(2, 2, 2, dtype=torch.bool)
    cover_loss(e, p, train).backward()
    assert not e.grad[:, ~train].any()


def test_mixed_geometry_dtypes_support_a_real_optimizer_step():
    from scripts.research.lflh_next.critical import CriticalNet, completion_fields

    field = completion_fields(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4, dtype=torch.float64))
    model = CriticalNet()
    optimizer = torch.optim.Adam(model.parameters())
    logits, reconstruction = model(
        torch.randn(2, 16, 6),
        torch.zeros(1, 2, 4, 4),
        torch.zeros_like(field),
        torch.zeros_like(field[:, :1]),
    )
    loss = logits.square().mean() + (reconstruction - field).square().mean()
    loss.backward()
    optimizer.step()
    assert field.dtype == torch.float32 and torch.isfinite(loss)


def test_probability_accounting_is_exhaustive_and_normalized():
    from scripts.research.lflh_next.critical_evaluate import summarize

    lower = torch.tensor([0.02, -0.1, -0.1, 0.03])
    upper = torch.tensor([0.04, -0.02, 0.1, 0.05])
    q, scores = summarize(torch.zeros(4), lower, upper, torch.tensor([True, False, False, False]))
    assert torch.allclose(q.sum(), torch.tensor(1.0))
    assert scores["critical_mass"] == 0.25
    assert scores["clear_mass"] == 0.5
    assert scores["penetration_witness_mass"] == 0.25
    assert scores["uncertain_mass"] == 0.25
