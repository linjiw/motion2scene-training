"""Goal/map command completion through an unchanged anticipatory motor specialist."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.motor_runtime import load_motor
from gear_sonic.research.scene_distillation.navigation_data import load_successful_tasks


@dataclass(frozen=True)
class NavigationInput:
    """Actor-only API: no reference, phase, motion identifier, or teacher target."""

    proprio: torch.Tensor
    navigation_context: torch.Tensor
    obstacles_body: torch.Tensor
    obstacle_mask: torch.Tensor

    localization: torch.Tensor | None = None

    @classmethod
    def from_batch(cls, batch):
        return cls(
            **{
                k: batch.get(k) if k == "localization" else batch[k]
                for k in cls.__dataclass_fields__
            }
        )


class NavigationMotorStudent(nn.Module):
    """Predict detailed commands internally; all inherited motor weights remain frozen.

    The deployment API is goal/map-only. This bounded deterministic comparator
    tests command completion before direct reference/token or flow alternatives.
    """

    actor_profile = "nav_goal_map_v1"

    def __init__(self, motor, decoder, width=512, localization=False):
        super().__init__()
        if getattr(motor, "command_dim", None) != 114 or not hasattr(motor, "forecaster"):
            raise ValueError("Navigation bridge requires the verified anticipatory motor")
        self.use_localization = localization
        self.actor_profile = "nav_goal_map_localization_v2" if localization else "nav_goal_map_v1"
        self.motor, self.decoder = motor, decoder
        self.motor.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.obstacle_encoder = mlp([15, 64, 64])
        self.command_predictor = mlp(
            [930 + 10 + 64 + (4 if localization else 0), width, width, 114]
        )
        nn.init.zeros_(self.command_predictor[-1].weight)
        nn.init.zeros_(self.command_predictor[-1].bias)
        self.register_buffer("context_mean", torch.zeros(10))
        self.register_buffer("context_std", torch.ones(10))
        self.register_buffer("command_mean", torch.zeros(114))
        self.register_buffer("command_std", torch.ones(114))
        self.register_buffer("normalized", torch.tensor(False))

    def initialize_normalization(self, data):
        if self.normalized:
            return
        with torch.no_grad():
            for name, key in [("context", "navigation_context"), ("command", "controls")]:
                values = data[key]
                getattr(self, name + "_mean").copy_(values.mean(0))
                getattr(self, name + "_std").copy_(values.std(0, unbiased=False).clamp_min(0.05))
            self.normalized.fill_(True)

    def navigation_step(self, actor):
        if not isinstance(actor, NavigationInput):
            raise TypeError("Navigation requires the explicit public actor view")
        p, nav, obstacles, mask = (
            actor.proprio,
            actor.navigation_context,
            actor.obstacles_body,
            actor.obstacle_mask,
        )
        n = len(p)
        if (
            p.shape != (n, 930)
            or nav.shape != (n, 10)
            or obstacles.shape != (n, 5, 15)
            or mask.shape != (n, 5)
            or mask.dtype != torch.bool
        ):
            raise ValueError("Invalid navigation actor shapes")
        clean = torch.where(mask[..., None], obstacles, 0)
        if not all(torch.isfinite(v).all() for v in (p, nav, clean)):
            raise ValueError("Nonfinite public observation")
        if not (nav[:, 9] == 1).all():
            raise ValueError("This profile requires a complete known map")
        features = self.obstacle_encoder(clean.clamp(-20, 20))
        pooled = torch.where(mask[..., None], features, 0).sum(1)
        pooled = pooled / mask.sum(1, keepdim=True).clamp_min(1)
        history = (p - self.motor.input_mean[:930]) / self.motor.input_std[:930]
        context = (nav - self.context_mean) / self.context_std
        inputs = [history.clamp(-20, 20), context.clamp(-20, 20), pooled]
        if self.use_localization:
            velocity = actor.localization
            if velocity is None or velocity.shape != (n, 4) or not torch.isfinite(velocity).all():
                raise ValueError("Localized actor requires causal velocity and validity")
            if not ((velocity[:, 3] == 0) | (velocity[:, 3] == 1)).all():
                raise ValueError("Invalid localization validity bit")
            inputs.append(velocity.clamp(-20, 20))
        prediction = self.command_predictor(torch.cat(inputs, -1))
        commands = prediction * self.command_std + self.command_mean
        available = torch.ones_like(commands, dtype=torch.bool)
        available[:, 7] = False
        result = self.motor.prior_step(p, commands, available)
        return dict(**result, commands=commands, actions=self.decoder(result["tokens"], p))

    def full_step(self, proprio, controls, mask):
        """Explicit retention path, independent of every navigation parameter."""
        result = self.motor.prior_step(proprio, controls, mask)
        return dict(**result, actions=self.decoder(result["tokens"], proprio))


def make_navigation(config, device):
    if sha(config["motor_checkpoint"]) != config["motor_sha256"]:
        raise ValueError("Selected motor changed")
    motor, decoder, _ = load_motor(config["motor_checkpoint"], config["teacher_sha256"], device)
    return NavigationMotorStudent(
        motor, decoder, config.get("width", 512), config.get("localization", False)
    ).to(device)


def load_navigation(path, teacher_sha256, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["stage"] != "navigation_motor" or saved["config"]["teacher_sha256"] != teacher_sha256:
        raise ValueError("Navigation checkpoint contract mismatch")
    student = make_navigation(saved["config"], device)
    student.load_state_dict(saved["model"], strict=True)
    return student.eval()


# Current-command schema: arrival (7) is deliberately absent from the motor input.
COMMAND_GROUPS = {
    "heading": slice(0, 2),
    "velocity": slice(2, 5),
    "height": slice(5, 6),
    "yaw_rate": slice(6, 7),
    "keypoints": slice(8, 50),
    "joint_position": slice(50, 79),
    "joint_velocity": slice(79, 108),
    "orientation": slice(108, 114),
}


def navigation_loss(student, batch, command_weight=0.1, token_weight=0.1, objective="legacy"):
    result = student.navigation_step(NavigationInput.from_batch(batch))
    teacher_action = (result["actions"] - batch["teacher_actions"]).square().mean()
    error = ((result["commands"] - batch["controls"]) / student.command_std).square()
    if objective == "structured_motor":
        with torch.no_grad():
            available = torch.ones_like(batch["controls"], dtype=torch.bool)
            available[:, 7] = False
            target = student.full_step(batch["proprio"], batch["controls"], available)
        action = (result["actions"] - target["actions"]).square().mean()
        groups = {"command_" + k: error[:, v].mean() for k, v in COMMAND_GROUPS.items()}
        commands = torch.stack(list(groups.values())).mean()
        # Motor targets are representable at this exact history; teacher gap is diagnostic.
        return dict(
            loss=action + command_weight * commands,
            action_mse=action,
            teacher_action_mse=teacher_action,
            motor_teacher_action_mse=(target["actions"] - batch["teacher_actions"]).square().mean(),
            normalized_command_mse=commands,
            **groups,
        )
    if objective != "legacy":
        raise ValueError("Unknown navigation objective")
    tokens = (result["tokens"] - batch["teacher_tokens"]).square().mean()
    commands = error.mean()
    return dict(
        loss=teacher_action + token_weight * tokens + command_weight * commands,
        action_mse=teacher_action,
        token_mse=tokens,
        normalized_command_mse=commands,
    )


def fit(config, output):
    """Bounded task-positive fit; immutable motor retention replaces joint fine-tuning."""
    if not 1 <= config["updates"] <= 20000 or not 0 < config["wall_cap_seconds"] <= 1800:
        raise ValueError("Invalid navigation pilot budget")
    if config.get("dataset_view", "teacher_positive") == "motor_recovery":
        from gear_sonic.research.scene_distillation.navigation_recovery_data import (
            load_motor_recoveries,
        )

        episodes = load_motor_recoveries(config)
    elif config.get("dataset_view", "teacher_positive") == "teacher_positive":
        episodes = load_successful_tasks(config)
    else:
        raise ValueError("Unknown navigation dataset view")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "config.json", config)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = config.get("device", "cuda")
    student = make_navigation(config, device)
    if config.get("initial_navigation_checkpoint"):
        if sha(config["initial_navigation_checkpoint"]) != config["initial_navigation_sha256"]:
            raise ValueError("Initial navigation checkpoint changed")
        initial_checkpoint = torch.load(
            config["initial_navigation_checkpoint"], map_location=device, weights_only=False
        )
        if initial_checkpoint["stage"] != "navigation_motor" or any(
            initial_checkpoint["config"].get(key) != config.get(key)
            for key in ("motor_sha256", "teacher_sha256")
        ):
            raise ValueError("Navigation warm start changes the frozen motor or actor contract")
        if any(
            initial_checkpoint["config"].get(key, default) != config.get(key, default)
            for key, default in (("localization", False), ("width", 512))
        ):
            raise ValueError("Navigation warm start changes the actor shape")
        student.load_state_dict(initial_checkpoint["model"], strict=True)
    fields = [k for k in NavigationInput.__dataclass_fields__ if k != "localization"] + [
        "controls",
        "teacher_actions",
        "teacher_tokens",
    ]
    if config.get("localization", False):
        fields.append("localization")
    if config.get("dataset_view") == "motor_recovery":
        fields.append("motor_actions")
    rows = [{k: e["arrays"][k][e["arrays"]["query_mask"]] for k in fields} for e in episodes]
    if any(not len(e["proprio"]) for e in rows):
        raise ValueError("Empty admitted episode")
    data = {k: torch.cat([e[k] for e in rows]).to(device) for k in fields}
    student.initialize_normalization(data)
    optimizer = torch.optim.AdamW(
        [v for v in student.parameters() if v.requires_grad], lr=config.get("learning_rate", 1e-4)
    )
    rng = np.random.default_rng(config["seed"])
    offsets = np.cumsum([0] + [len(e["proprio"]) for e in rows])
    by_motion = {}
    for i, e in enumerate(episodes):
        by_motion.setdefault(e["motion_id"], []).append(i)
    groups = list(by_motion.values())
    recovery_fraction = config.get("recovery_fraction", 0.0)
    if not 0 <= recovery_fraction <= 1:
        raise ValueError("Invalid recovery replay mixture")
    by_role = {}
    for i, episode in enumerate(episodes):
        by_role.setdefault(episode.get("role", "replay"), {}).setdefault(
            episode["motion_id"], []
        ).append(i)
    role_groups = {role: list(motions.values()) for role, motions in by_role.items()}
    if recovery_fraction and "recovery" not in role_groups:
        raise ValueError("Recovery mixture has no qualified recovery episodes")
    frozen = {
        k: v.clone()
        for k, v in student.state_dict().items()
        if k.startswith(("motor.", "decoder."))
    }
    initial = student.full_step(
        data["proprio"][:32],
        data["controls"][:32],
        torch.ones_like(data["controls"][:32], dtype=torch.bool),
    )["actions"].detach()
    started, step = time.monotonic(), 0
    with (output / "metrics.jsonl").open("x") as log:
        for step in range(1, config["updates"] + 1):
            if time.monotonic() - started > config["wall_cap_seconds"]:
                raise TimeoutError("Navigation fit wall ceiling")
            # Motion first, then scene replica, then a valid decision; no fake sequence adjacency.
            chosen = []
            for _ in range(config["batch_size"]):
                role = (
                    "recovery"
                    if recovery_fraction and rng.random() < recovery_fraction
                    else "replay"
                )
                pool = role_groups.get(role)
                if not pool:
                    raise ValueError("Empty requested replay role")
                chosen.append(int(rng.choice(pool[int(rng.integers(len(pool)))])))
            ix = offsets[chosen] + (rng.random(len(chosen)) * np.diff(offsets)[chosen]).astype(int)
            batch = {k: v[torch.as_tensor(ix, device=device)] for k, v in data.items()}
            if step == 1:
                if "motor_actions" in batch:
                    availability = torch.ones_like(batch["controls"], dtype=torch.bool)
                    availability[:, 7] = False
                    torch.testing.assert_close(
                        student.full_step(batch["proprio"], batch["controls"], availability)[
                            "actions"
                        ],
                        batch["motor_actions"],
                        atol=2e-4,
                        rtol=2e-4,
                    )
                torch.testing.assert_close(
                    student.decoder(batch["teacher_tokens"], batch["proprio"]),
                    batch["teacher_actions"],
                    atol=2e-4,
                    rtol=2e-4,
                )
            losses = navigation_loss(
                student,
                batch,
                config.get("command_weight", 0.1),
                objective=config.get("objective", "legacy"),
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(
                [v for v in student.parameters() if v.requires_grad], 1, error_if_nonfinite=True
            )
            optimizer.step()
            if step % 20 == 0 or step == 1:
                log.write(
                    json.dumps(dict(step=step, **{k: float(v.detach()) for k, v in losses.items()}))
                    + "\n"
                )
                log.flush()
    assert all(torch.equal(v, student.state_dict()[k]) for k, v in frozen.items())
    with torch.no_grad():
        final = student.full_step(
            data["proprio"][:32],
            data["controls"][:32],
            torch.ones_like(data["controls"][:32], dtype=torch.bool),
        )["actions"]
        torch.testing.assert_close(initial, final, atol=0, rtol=0)
    torch.save(
        dict(
            stage="navigation_motor",
            model=student.state_dict(),
            config=config,
            optimizer=optimizer.state_dict(),
            step=step,
            rng_numpy=rng.bit_generator.state,
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all(),
        ),
        output / f"step-{step:06d}.pt",
    )
    write_new(
        output / "receipt.json",
        dict(
            state="complete",
            updates=step,
            wall_seconds=time.monotonic() - started,
            rows=len(data["proprio"]),
            episodes=len(episodes),
            motions=len(groups),
            ancestry_groups=len({e["ancestry"] for e in episodes}),
            frozen_motor_tensor_identity=True,
            full_command_action_identity=True,
            physical_success=None,
            evaluation_scope="in-sample positive control only",
            recovery_fraction=recovery_fraction,
            qualified_recovery_episodes=sum(e.get("role") == "recovery" for e in episodes),
            warm_start=bool(config.get("initial_navigation_checkpoint")),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
