"""Differentiable geometry bounds for proposed LfLH research.

Bounds concern the supplied capsule model at supplied frames, not robot dynamics,
native-mesh containment, continuous-time tracking, or physical passage.
"""

import torch


def box_sdf(points, centre, half, yaw):
    """Signed distance to a yaw-rotated box, with broadcast-compatible tensors."""
    delta = points - centre
    c, s = yaw.cos(), yaw.sin()
    local = torch.stack(
        (c * delta[..., 0] + s * delta[..., 1],
         -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), dim=-1
    )
    q = local.abs() - half
    return q.clamp_min(0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0)


def soft_lower_min(values, temperature, dim=-1):
    """Unnormalized log-sum-exp: min(x)-tau*log(N) <= result <= min(x)."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return -temperature * torch.logsumexp(-values / temperature, dim=dim)


def capsule_samples(starts, ends, radii, count):
    """Samples plus radius and spatial covering correction for every axis.

    For count equally spaced points, any axis point is within L/(2*(count-1))
    of a sample. Subtracting this from sampled SDF-radius gives a lower bound.
    There is no correction for time intervals not supplied by the caller.
    """
    if count < 2:
        raise ValueError('at least two axial samples required')
    if starts.shape != ends.shape or starts.shape[-1] != 3:
        raise ValueError('matching (...,3) endpoint arrays required')
    if not torch.isfinite(starts).all() or not torch.isfinite(ends).all():
        raise ValueError('finite capsule endpoints required')
    if not torch.isfinite(radii).all() or (radii < 0).any():
        raise ValueError('finite nonnegative radii required')
    fraction = torch.linspace(0, 1, count, dtype=starts.dtype, device=starts.device)
    points = starts[..., None, :] + fraction[:, None] * (ends-starts)[..., None, :]
    radius = torch.broadcast_to(radii, starts.shape[:-1])[..., None].expand(points.shape[:-1])
    cover = (ends-starts).norm(dim=-1) / (2 * (count-1))
    correction = cover[..., None].expand(points.shape[:-1])
    return points, radius, correction
