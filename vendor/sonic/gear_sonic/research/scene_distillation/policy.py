"""Route-free recurrent navigation actor with an explicit inference-time interface."""

import math

import torch
from torch import nn

from gear_sonic.research.hindsight_training.student import mlp

PUBLIC_SHAPES = {
    "proprio": (930,),
    "start_goal_body": (6,),
    "obstacles_body": (5, 15),
    "obstacle_mask": (5,),
}


def validate_public_observation(observation):
    if set(observation) != set(PUBLIC_SHAPES):
        raise ValueError(f"Public actor fields must be exactly {sorted(PUBLIC_SHAPES)}")
    batch = observation["proprio"].shape[0]
    device = observation["proprio"].device
    for key, shape in PUBLIC_SHAPES.items():
        value = observation[key]
        dtype = torch.bool if key == "obstacle_mask" else torch.float32
        if value.shape != (batch, *shape) or value.dtype != dtype or value.device != device:
            raise ValueError(f"Invalid public field {key}: shape, dtype or device")
        if key not in ("obstacles_body", "obstacle_mask") and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite public field {key}")
    mask = observation["obstacle_mask"]
    if not torch.isfinite(observation["obstacles_body"][mask]).all():
        raise ValueError("Non-finite valid obstacle")
    return batch


class PublicNavigationEncoder(nn.Module):
    """Global known-map, five-obstacle prototype; no image, route, phase or motion ID.

    A recurrent state stores only prior public observations. Its reset belongs to
    the episode boundary. Tokens drive the separately frozen SONIC decoder at 50 Hz.
    This untrained module has no physical avoidance or terminal-hold qualification.
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proprio = mlp([930, 256, 128])
        self.command = mlp([6, 64, 64])
        self.obstacle = mlp([15, 64, 64])
        self.query = nn.Linear(192, 64)
        self.null_obstacle = nn.Parameter(torch.zeros(1, 1, 64))
        self.memory = nn.GRUCell(257, hidden_dim)

    def encode_step(self, observation, hidden=None, *, episode_start=None):
        batch = validate_public_observation(observation)
        proprio = observation["proprio"]
        if hidden is None:
            hidden = proprio.new_zeros(batch, self.hidden_dim)
        if (
            hidden.shape != (batch, self.hidden_dim)
            or hidden.device != proprio.device
            or hidden.dtype != proprio.dtype
        ):
            raise ValueError("Recurrent state must match the current environment batch")
        if episode_start is not None:
            if (
                episode_start.shape != (batch,)
                or episode_start.dtype != torch.bool
                or episode_start.device != proprio.device
            ):
                raise ValueError("episode_start must be a boolean vector")
            hidden = torch.where(episode_start[:, None], torch.zeros_like(hidden), hidden)
        if not torch.isfinite(hidden).all():
            raise ValueError("Non-finite recurrent state")
        body = self.proprio(proprio)
        command = self.command(observation["start_goal_body"])
        mask = observation["obstacle_mask"]
        obstacles = torch.where(mask[..., None], observation["obstacles_body"], 0.0)
        encoded = self.obstacle(obstacles)
        encoded = torch.cat([encoded, self.null_obstacle.expand(batch, 1, 64)], dim=1)
        valid = torch.cat([mask, torch.ones(batch, 1, device=mask.device, dtype=torch.bool)], dim=1)
        query = self.query(torch.cat([body, command], dim=-1))
        scores = (encoded * query[:, None]).sum(-1) / math.sqrt(64)
        weights = scores.masked_fill(~valid, -torch.inf).softmax(dim=-1)
        scene = (encoded * weights[..., None]).sum(1)
        hidden = self.memory(
            torch.cat([body, command, scene, mask.float().mean(1, keepdim=True)], dim=-1), hidden
        )
        return hidden, body, weights


class NavigationTokenStudent(PublicNavigationEncoder):
    """Deterministic recurrent baseline, with diagnostic endpoint heads."""

    def __init__(self, hidden_dim=128):
        super().__init__(hidden_dim)
        self.token_head = mlp([hidden_dim, 128, 64])
        self.displacement_head = nn.Linear(hidden_dim, 3)
        self.arrival_head = nn.Linear(hidden_dim, 1)

    def forward_step(self, observation, hidden=None, *, episode_start=None):
        hidden, _, weights = self.encode_step(observation, hidden, episode_start=episode_start)
        continuous = self.token_head(hidden).tanh() * (31.0 / 32) - 1.0 / 32
        quantized = (continuous * 16).round() / 16
        return {
            "tokens": continuous + (quantized - continuous).detach(),
            "hidden": hidden,
            "displacement_body": self.displacement_head(hidden),
            "arrival_logit": self.arrival_head(hidden).squeeze(-1),
            "obstacle_attention": weights,
        }

    def forward_sequence(self, observations, episode_start, hidden=None):
        if set(observations) != set(PUBLIC_SHAPES):
            raise ValueError("Unknown sequence fields; privileged inputs are forbidden")
        batch, length = observations["proprio"].shape[:2]
        if length < 1 or episode_start.shape != (batch, length):
            raise ValueError("Invalid sequence length or reset mask")
        outputs = []
        for step in range(length):
            result = self.forward_step(
                {key: value[:, step] for key, value in observations.items()},
                hidden,
                episode_start=episode_start[:, step],
            )
            hidden = result["hidden"]
            outputs.append(result)
        return {key: torch.stack([row[key] for row in outputs], dim=1) for key in outputs[0]}
