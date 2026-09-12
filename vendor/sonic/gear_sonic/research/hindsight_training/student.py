"""Scene-conditioned SONIC token student and frozen decoder distillation loss.

Inputs are measured history and commands available at the decision time.
Full future reference motion belongs exclusively to the tracking teacher.
"""

import torch
from torch import nn
from torch.nn import functional as F


def mlp(dimensions):
    layers = []
    for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(dimensions) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def masked_mean(features, mask):
    if mask.dtype != torch.bool or features.shape[:-1] != mask.shape:
        raise ValueError("Mask must be boolean and match the set dimensions")
    clean = torch.where(mask[..., None], features, torch.zeros_like(features))
    return clean.sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)


class SceneTokenStudent(nn.Module):
    """Prototype: causal proprioception + body-frame goal/start + obstacle set.

    Obstacle rows: center xyz, full bounding dimensions xyz, rotation columns
    1 and 2 (six numbers), shape one-hot (box/cylinder/other). These bounding
    features do not replace mesh collision checks. Optional route is a provided
    command, never the unobserved future of the executed motion.
    """

    def __init__(self, camera=False):
        super().__init__()
        self.camera = camera
        self.proprio = mlp([930, 256, 128])
        self.command = mlp([6, 64, 64])
        self.obstacle = mlp([15, 64, 64])
        # Flattening retains waypoint order; a set pool would erase route order.
        self.route = mlp([16 * 4, 64, 64])
        if camera:
            self.depth = nn.Sequential(
                nn.Conv2d(2, 16, 5, stride=2),
                nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.SiLU(),
            )
        self.head = mlp([320 + (64 if camera else 0), 256, 128, 64])

    def forward(
        self,
        *,
        proprio,
        start_goal_body,
        obstacles_body,
        obstacle_mask,
        route_body,
        route_mask,
        depth=None
    ):
        batch = proprio.shape[0]
        if proprio.shape != (batch, 930) or start_goal_body.shape != (batch, 6):
            raise ValueError("Expected 930D causal SONIC history and 6D body-frame start/goal")
        if (
            obstacles_body.ndim != 3
            or obstacles_body.shape[0] != batch
            or obstacles_body.shape[-1] != 15
        ):
            raise ValueError("Obstacle features must have shape (batch, obstacles, 15)")
        if route_body.shape != (batch, 16, 3) or route_mask.shape != (batch, 16):
            raise ValueError("Route must contain 16 ordered, masked command waypoints")
        if obstacle_mask.dtype != torch.bool or route_mask.dtype != torch.bool:
            raise ValueError("Validity masks must be boolean")
        obstacle_clean = torch.where(obstacle_mask[..., None], obstacles_body, 0.0)
        route_clean = torch.where(route_mask[..., None], route_body, 0.0)
        route_features = torch.cat([route_clean, route_mask[..., None].float()], dim=-1).flatten(1)
        values = [proprio, start_goal_body, obstacle_clean, route_features]
        if not all(torch.isfinite(x).all() for x in values):
            raise ValueError("Non-finite valid student observation")
        features = [
            self.proprio(proprio),
            self.command(start_goal_body),
            masked_mean(self.obstacle(obstacle_clean), obstacle_mask),
            self.route(route_features),
        ]
        if self.camera:
            if (
                depth is None
                or depth.shape != (batch, 2, 120, 160)
                or not torch.isfinite(depth).all()
            ):
                raise ValueError(
                    "Camera profile needs normalized depth and validity, (B,2,120,160)"
                )
            features.append(self.depth(depth))
        elif depth is not None:
            raise ValueError("Camera data supplied to a camera-disabled profile")
        continuous = self.head(torch.cat(features, dim=-1)).tanh() * (31.0 / 32) - 1.0 / 32
        quantized = (continuous * 16).round() / 16
        # Straight-through estimator; forward values exactly match the release FSQ lattice.
        return continuous + (quantized - continuous).detach()


class FrozenSonicDecoder(nn.Module):
    def __init__(self, policy_state):
        super().__init__()
        self.module = mlp([994, 2048, 2048, 1024, 1024, 512, 512, 29])
        prefix = "actor_module.decoders.g1_dyn.module."
        weights = {
            key[len(prefix) :]: value
            for key, value in policy_state.items()
            if key.startswith(prefix)
        }
        self.module.load_state_dict(weights, strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, token, proprio):
        if token.shape[-1] != 64 or proprio.shape[-1] != 930:
            raise ValueError("SONIC decoder requires token64 followed by proprioception930")
        return self.module(torch.cat([token, proprio], dim=-1))


def distillation_loss(student_tokens, teacher_tokens, teacher_action_mean, proprio, decoder):
    """Compare tokens and decoded action means on the same measured state.

    This supervised loss alone does not establish on-policy recovery or avoidance.
    """
    if teacher_tokens.shape != student_tokens.shape or teacher_action_mean.shape != (
        *student_tokens.shape[:-1],
        29,
    ):
        raise ValueError("Teacher/student dimensions do not match")
    with torch.no_grad():
        expected_action = decoder(teacher_tokens, proprio)
        if not torch.allclose(expected_action, teacher_action_mean, atol=2e-4, rtol=2e-4):
            raise ValueError("Teacher actions disagree with the bound decoder and observed state")
    token_loss = F.mse_loss(student_tokens, teacher_tokens.detach())
    action_loss = F.mse_loss(decoder(student_tokens, proprio), teacher_action_mean.detach())
    return {"loss": token_loss + action_loss, "token_mse": token_loss, "action_mse": action_loss}
