"""Transformer public-input boundaries and checkpoint architecture compatibility."""

import pytest
import torch

from gear_sonic.research.scene_distillation.commands import MaskedMotionFoundation, build_foundation


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = {
        "foundation_architecture": "transformer",
        "transformer": {"width": 32, "layers": 2, "heads": 4, "latent_dim": 8},
    }
    model = build_foundation(config).eval()
    p = torch.randn(2, 930)
    c = torch.randn(2, 79)
    c[:, :2] = torch.tensor([0.0, 1.0])
    mask = torch.ones(2, 79, dtype=torch.bool)
    return config, model, p, c, mask


def test_public_mask_hides_values_and_keeps_command_gradients():
    _, model, p, c, mask = fixture()
    mask[:, 8:] = False
    expected = model.prior_step(p, c, mask)["tokens"]
    changed = c.clone()
    changed[:, 8:] = float("nan")
    torch.testing.assert_close(
        model.prior_step(p, changed, mask)["tokens"], expected, rtol=0, atol=0
    )
    c.requires_grad_()
    model.prior_step(p, c, mask)["tokens"].square().sum().backward()
    assert c.grad[:, 2:8].abs().sum() > 0
    assert c.grad[:, 8:].abs().sum() == 0


def test_checkpoint_rebuild_and_prior_determinism():
    config, model, p, c, mask = fixture()
    other = build_foundation(config).eval()
    other.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        other.prior_step(p, c, mask)["tokens"], model.prior_step(p, c, mask)["tokens"]
    )
    with pytest.raises(TypeError):
        model.prior_step(p, c, mask, future_reference=torch.zeros(2, 640))
    posterior = model.posterior_step(p, torch.zeros(2, 1645), torch.zeros(2, 640), c, mask)
    assert torch.isfinite(posterior["kl"]).all()
    assert isinstance(build_foundation({}), MaskedMotionFoundation)
    with pytest.raises(ValueError):
        build_foundation({"foundation_architecture": "unknown"})
