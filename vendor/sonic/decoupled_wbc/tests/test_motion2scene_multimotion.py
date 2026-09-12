import torch

from scripts.research.lflh_next.multimotion.run import Model, candidates, gaps


def test_shapes_have_distinct_exact_proxy_clearance():
    points = torch.tensor([[0.0, 0.0, 0.0]])
    spec = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0]])
    torch.testing.assert_close(gaps(points, spec), torch.tensor([0.75, 0.6, 0.6]))
    assert (gaps(points, torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])) < 0).all()


def test_every_shape_and_candidate_retained():
    spec = candidates()
    assert spec.shape == (225, 4)
    for kind in range(3):
        assert (spec[:, 3] == kind).sum() == 75
    assert len(torch.unique(spec, dim=0)) == 225


def test_unconditional_ignores_motion():
    torch.manual_seed(0)
    model = Model("unconditional").eval()
    spec = torch.randn(225, 7)
    result = model(torch.randn(2, 32, 90), spec)
    torch.testing.assert_close(result[0], result[1])


def test_transformer_probability_and_gradient():
    model = Model("transformer")
    result = model(torch.randn(2, 32, 90), torch.randn(225, 7))
    q = result.softmax(-1)
    torch.testing.assert_close(q.sum(-1), torch.ones(2))
    loss = -q[:, 3].log().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
