"""Checks for the finite-distribution support projection and abstention contract."""

import pytest
import torch

from scripts.research.lflh_next.constrained import project_clearance


def test_projection_retains_ratios_and_reports_rejection():
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    lower = torch.tensor([[0.02, 0.005, -0.2, 0.03]])
    result, retained = project_clearance(q, lower)
    torch.testing.assert_close(result, torch.tensor([[0.2, 0.0, 0.0, 0.8]]))
    torch.testing.assert_close(retained, torch.tensor([0.5]))


def test_empty_support_abstains_without_uniform_fallback():
    result, retained = project_clearance(torch.tensor([0.2, 0.8]), torch.tensor([-1.0, 0.0]))
    assert result.sum() == 0 and retained == 0
    assert torch.isfinite(result).all()


def test_positive_geometry_without_probability_support_abstains():
    result, retained = project_clearance(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0]))
    assert result.sum() == 0 and retained == 0


def test_projection_is_kl_closest_feasible_distribution():
    q = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    p, _ = project_clearance(q, torch.tensor([0.02, 0.02, -1.0]))
    kl = (p[:2] * (p[:2] / q[:2]).log()).sum()
    for weight in torch.linspace(0.01, 0.99, 99, dtype=torch.float64):
        alternative = torch.stack([weight, 1 - weight])
        assert (alternative * (alternative / q[:2]).log()).sum() >= kl - 1e-12


@pytest.mark.parametrize(
    "q,lower",
    [
        ([0.1, 0.1], [1.0, 1.0]),
        ([-0.1, 1.1], [1.0, 1.0]),
        ([float("nan"), 1.0], [1.0, 1.0]),
        ([0.5, 0.5], [1.0, float("nan")]),
    ],
)
def test_invalid_input_rejected(q, lower):
    with pytest.raises(ValueError):
        project_clearance(torch.tensor(q), torch.tensor(lower))


def test_subnormal_retained_probability_is_still_normalized():
    q = torch.tensor([1.0, 1e-310], dtype=torch.float64)
    result, retained = project_clearance(q, torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert retained > 0


def test_nonfinite_margin_rejected():
    with pytest.raises(ValueError):
        project_clearance(torch.tensor([1.0]), torch.tensor([1.0]), float("nan"))


def test_acceptance_weight_one_recovers_original_coverage():
    from scripts.research.lflh_next.constrained import coverage_acceptance_loss

    logits = torch.tensor([[0.2, -0.1, 0.7]], requires_grad=True)
    critical = torch.tensor([[True, True, False]])
    clear = torch.tensor([[True, True, False]])
    loss = coverage_acceptance_loss(logits, critical, clear, acceptance_weight=1.0)
    expected = -logits.log_softmax(-1)[critical].mean()
    torch.testing.assert_close(loss, expected)
    weighted = coverage_acceptance_loss(logits, critical, clear, acceptance_weight=2.0)
    assert weighted > loss


def test_no_critical_target_still_learns_acceptance():
    from scripts.research.lflh_next.constrained import coverage_acceptance_loss

    logits = torch.zeros(1, 3, requires_grad=True)
    loss = coverage_acceptance_loss(
        logits, torch.zeros(1, 3, dtype=torch.bool), torch.tensor([[True, False, False]])
    )
    loss.backward()
    assert logits.grad[0, 0] < 0 and (logits.grad[0, 1:] > 0).all()


def test_no_clear_target_does_not_invent_feasible_labels():
    from scripts.research.lflh_next.constrained import coverage_acceptance_loss

    logits = torch.zeros(1, 3, requires_grad=True)
    labels = torch.zeros(1, 3, dtype=torch.bool)
    loss = coverage_acceptance_loss(logits, labels, labels)
    loss.backward()
    assert loss == 0 and (logits.grad == 0).all()
