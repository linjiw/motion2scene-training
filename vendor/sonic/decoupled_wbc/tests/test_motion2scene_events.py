import numpy as np
import torch

from scripts.research.lflh_next.navigation.hindsight_study import loss_fn
from scripts.research.lflh_next.navigation.motion_events import (
    chord_root,
    event_anchors,
    observations,
    velocity_acceleration,
)


def test_derivatives_use_seconds_and_do_not_make_constant_speed_critical():
    for fps in [30, 60]:
        t = np.arange(90) / fps
        p = np.stack([2 * t, np.zeros_like(t), 0.5 * t * t], -1)
        _, v, a = velocity_acceleration(p, fps)
        np.testing.assert_allclose(v[:, 0], 2, atol=1e-10)
        np.testing.assert_allclose(a[:, 2], 1, atol=1e-9)


def test_z_path_events_include_the_turns():
    vertices = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0], [3.0, 0.0]])
    root = np.concatenate([np.linspace(vertices[i], vertices[i + 1], 61)[:-1] for i in range(3)])
    pts = np.zeros((len(root), 7, 3))
    pts[:, :, :2] = root[:, None, :]
    pts[:, :, 2] = [0.8, 0.8, 1.5, 1.0, 1.0, 0.1, 0.1]
    rot = np.broadcast_to(np.eye(3), (len(root), 6, 3, 3)).copy()
    x, ev, tr = observations(pts, rot, 30)
    assert x.shape == (180, 64)
    anchors = event_anchors(ev[:, 0])
    assert min(abs(anchors - 60)) < 5 and min(abs(anchors - 120)) < 5
    assert np.isfinite(tr["turn_rate"]).all()


def test_shortcut_preserves_window_endpoints_and_vertical_motion():
    root = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.2], [2.0, 0.0, 0.5]])
    out, a, b = chord_root(root, 1, 1)
    np.testing.assert_allclose(out[:, 2], root[:, 2])
    np.testing.assert_allclose(out[[a, b]], root[[a, b]])
    np.testing.assert_allclose(out[1, :2], [1, 0])


def test_empty_near_still_penalizes_rejected_probability():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = loss_fn(logits, torch.zeros_like(logits), torch.tensor([[True, False]]))
    torch.testing.assert_close(loss, torch.tensor(2 * np.log(2), dtype=torch.float32))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
