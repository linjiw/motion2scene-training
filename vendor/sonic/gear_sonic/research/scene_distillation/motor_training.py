"""Full-command motor recovery with a frozen SONIC decoder and explicit support tiers."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.commands import build_foundation
from gear_sonic.research.scene_distillation.reference_layout import unpack_reference
from gear_sonic.research.scene_distillation.train import (
    load_episodes,
    make_decoder,
    train_foundation_step,
)
from gear_sonic.research.scene_distillation.transformer import TransformerMotionFoundation


class FullCommandFoundation(TransformerMotionFoundation):
    """Legacy 79 commands plus current target joint velocity29 and orientation6.

    No future trajectory, reference phase or motion identifier enters this actor.
    A zero-initialized extension preserves the original motor foundation exactly.
    """

    command_dim = 114

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_extension = nn.Sequential(nn.Linear(70, 256), nn.SiLU(), nn.Linear(256, 192))
        nn.init.zeros_(self.current_extension[-1].weight)
        nn.init.zeros_(self.current_extension[-1].bias)

    def _condition(self, proprio, controls, control_mask):
        if controls.shape != (len(proprio), 114) or control_mask.shape != controls.shape:
            raise ValueError("Extended full command requires B,114 values and masks")
        if control_mask.dtype != torch.bool:
            raise ValueError("Command mask must be boolean")
        base = super()._condition(proprio, controls[:, :79], control_mask[:, :79])
        clean = torch.where(control_mask[:, 79:], controls[:, 79:], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("Nonfinite available current command")
        return base + self.current_extension(torch.cat([clean, control_mask[:, 79:].float()], -1))


def current_frame_extension(future_reference, controls):
    """Reconstruct only frame zero of a verified native B,10,64 recording.

    Decoded physical frame: joint position29, velocity29, relative rotation6.
    Reference spacing starts at zero. Later frames are deliberately never read.
    """
    if future_reference.shape != (len(controls), 640) or controls.shape[1] != 79:
        raise ValueError("Unexpected native recording schema")
    current = unpack_reference(future_reference)[:, 0]
    if not torch.allclose(current[:, :29], controls[:, 50:], atol=1e-6, rtol=0):
        raise ValueError("Current target mismatch: cannot reconstruct public command")
    if not torch.isfinite(current).all():
        raise ValueError("Nonfinite current target")
    return current[:, 29:]


def build_motor(config):
    if (
        config.get("current_frame_extension")
        and config.get("reference_layout_version") != "native_q_then_v_v1"
    ):
        raise ValueError("Extended motor needs the verified native reference layout version")
    if config.get("motor_architecture") == "anticipatory":
        from gear_sonic.research.scene_distillation.anticipatory_motor import AnticipatoryMotor

        return AnticipatoryMotor(config["teacher_checkpoint"], config["teacher_sha256"])
    if config.get("current_frame_extension", False):
        return FullCommandFoundation(**config["transformer"])
    return build_foundation(config)


def initialize_motor(model, saved):
    own, source = model.state_dict(), saved["model"]
    if set(source) - set(own) or any(
        not k.startswith("current_extension.") for k in set(own) - set(source)
    ):
        raise ValueError("Unapproved motor architecture migration")
    model.load_state_dict({**own, **source}, strict=True)


def public_motor_loss(model, decoder, batch, token_weight=0.1, future_weight=0.0):
    result = model.prior_step(batch["proprio"], batch["controls"], batch["control_mask"])
    action = decoder(result["tokens"], batch["proprio"])
    action_mse = (action - batch["teacher_actions"]).square().mean()
    token_mse = (result["tokens"] - batch["teacher_tokens"]).square().mean()
    future = action_mse * 0
    if future_weight:
        if "forecast" not in result:
            raise ValueError("Future auxiliary loss requires an anticipatory motor")
        future = torch.nn.functional.smooth_l1_loss(
            result["forecast"] / model.frame_std,
            unpack_reference(batch["future_reference"]) / model.frame_std,
        )
    return dict(
        loss=action_mse + token_weight * token_mse + future_weight * future,
        prior_action_mse=action_mse,
        token_mse=token_mse,
        future_loss=future,
    )


def fit(config, output):
    if not 1 <= config["updates"] <= 100000 or not 0 < config["wall_cap_seconds"] <= 7200:
        raise ValueError("Invalid fit budget")
    for path_key, hash_key in [
        ("teacher_checkpoint", "teacher_sha256"),
        ("dataset_manifest", "dataset_manifest_sha256"),
        ("initialize_checkpoint", "initialize_sha256"),
    ]:
        if sha(config[path_key]) != config[hash_key]:
            raise ValueError(f"Changed {path_key}")
    episodes = load_episodes(
        config["dataset_manifest"],
        config["teacher_sha256"],
        set(config["train_ids"]),
        "foundation",
        allow_teacher_prefixes=True,
        allow_exploratory_queries=True,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "config.json", config)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = config.get("device", "cuda")
    model = build_motor(config).to(device)
    saved = torch.load(config["initialize_checkpoint"], map_location=device, weights_only=False)
    if saved["teacher_sha256"] != config["teacher_sha256"]:
        raise ValueError("Initialization teacher mismatch")
    initialize_motor(model, saved)
    decoder = make_decoder(config["teacher_checkpoint"], device)
    fields = [
        "proprio",
        "controls",
        "control_mask",
        "teacher_tokens",
        "teacher_actions",
        "privileged_state",
        "future_reference",
    ]
    if config.get("current_frame_extension", False):
        for e in episodes:
            extra = current_frame_extension(e["future_reference"], e["controls"])
            e["controls"] = torch.cat([e["controls"], extra], -1)
            e["control_mask"] = torch.cat(
                [e["control_mask"], torch.ones_like(extra, dtype=torch.bool)], -1
            )
    # Dense GPU buffers remove per-example Python work while retaining uniform episode sampling.
    offsets = np.cumsum([0] + [len(e["proprio"]) for e in episodes])
    data = {k: torch.cat([e[k] for e in episodes]).to(device) for k in fields}
    if hasattr(model, "initialize_normalization"):
        model.initialize_normalization(data)
    groups = [
        [i for i, e in enumerate(episodes) if bool(e["_exploratory_source"][0]) == flag]
        for flag in [False, True]
    ]
    groups = [np.array(g) for g in groups if g]
    rng = np.random.default_rng(config["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    started = time.monotonic()
    state = "failed"
    step = 0
    try:
        with (output / "metrics.jsonl").open("x") as log:
            for step in range(1, config["updates"] + 1):
                if time.monotonic() - started > config["wall_cap_seconds"]:
                    raise TimeoutError("Motor fit wall cap")
                g = groups[int(rng.integers(len(groups)))]
                selected = rng.choice(g, size=config["batch_size"])
                ix = offsets[selected] + (
                    rng.random(len(selected)) * (offsets[selected + 1] - offsets[selected])
                ).astype(int)
                indices = torch.as_tensor(ix, device=device)
                batch = {k: v[indices] for k, v in data.items()}
                if step == 1:
                    with torch.no_grad():
                        torch.testing.assert_close(
                            decoder(batch["teacher_tokens"], batch["proprio"]),
                            batch["teacher_actions"],
                            atol=2e-4,
                            rtol=2e-4,
                        )
                if config.get("objective", "public") == "cvae":
                    if config.get("current_frame_extension", False):
                        raise ValueError("CVAE control uses legacy commands")
                    losses = train_foundation_step(
                        model,
                        decoder,
                        batch,
                        0.001,
                        1.0,
                        command_weights={"full": 1.0},
                        full_token_weight=0.1,
                    )
                else:
                    losses = public_motor_loss(
                        model,
                        decoder,
                        batch,
                        config.get("token_weight", 0.1),
                        config.get("future_weight", 0.0),
                    )
                if not torch.isfinite(losses["loss"]):
                    raise ValueError("Nonfinite motor loss")
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                if step % 20 == 0 or step == 1:
                    log.write(
                        json.dumps(
                            dict(step=step, **{k: float(v.detach()) for k, v in losses.items()})
                        )
                        + "\n"
                    )
                    log.flush()
                if step % config.get("save_interval", 2000) == 0 or step == config["updates"]:
                    torch.save(
                        dict(
                            stage="motor_foundation",
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            config=config,
                            teacher_sha256=config["teacher_sha256"],
                            step=step,
                            qualification="unqualified_motor_research",
                            rng_torch=torch.get_rng_state(),
                            rng_numpy=rng.bit_generator.state,
                            rng_cuda=torch.cuda.get_rng_state_all(),
                        ),
                        output / f"step-{step:06d}.pt",
                    )
        state = "complete"
    finally:
        write_new(
            output / "receipt.json",
            dict(
                state=state,
                updates=step,
                wall_seconds=time.monotonic() - started,
                rows=int(offsets[-1]),
                episodes=len(episodes),
                decoder_frozen=True,
                physical_success=None,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    fit(json.loads(a.config.read_text()), a.output)


if __name__ == "__main__":
    main()
