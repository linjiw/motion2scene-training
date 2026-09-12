"""Bounded, hash-bound BFM foundation and recurrent residual-navigation fitting."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha, write_new
from gear_sonic.research.hindsight_training.student import FrozenSonicDecoder
from gear_sonic.research.scene_distillation.commands import (
    MaskedMotionFoundation,
    sample_command_mask,
)
from gear_sonic.research.scene_distillation.curriculum import (
    PhaseBalancedSampler,
    curriculum_mask,
    curriculum_weights,
)
from gear_sonic.research.scene_distillation.losses import variational_imitation_loss
from gear_sonic.research.scene_distillation.navigation import (
    FrozenFoundationNavigator,
    navigation_distillation_loss,
)
from gear_sonic.research.scene_distillation.policy import PUBLIC_SHAPES


def load_episodes(
    manifest_path,
    teacher_sha256,
    allowed_ids,
    stage,
    *,
    allow_exploratory_queries=False,
    allow_teacher_prefixes=False,
):
    manifest = json.loads(Path(manifest_path).read_text())
    expected = (
        "bfm_executed_foundation_v1" if stage == "foundation" else "bfm_executed_navigation_v1"
    )
    if manifest["schema"] != expected or manifest["teacher_sha256"] != teacher_sha256:
        raise ValueError("Dataset stage or teacher mismatch")
    for parent in manifest.get("parents", []):
        if sha(parent["path"]) != parent["sha256"]:
            raise ValueError("Aggregated collection manifest changed")
    exploratory = manifest.get("source") == "queried_native_teacher_on_student_state"
    if exploratory and not allow_exploratory_queries:
        raise ValueError("Exploratory student-state queries require an explicit experiment flag")
    episodes = []
    for episode in manifest["episodes"]:
        prefix = episode.get("source", manifest.get("source")) == "executed_native_teacher_prefix"
        if prefix and not allow_teacher_prefixes:
            raise ValueError(
                "Locally supported teacher prefixes require an explicit experiment flag"
            )
        exploratory = (
            episode.get("source", manifest.get("source"))
            == "queried_native_teacher_on_student_state"
        )
        if exploratory and not allow_exploratory_queries:
            raise ValueError(
                "Exploratory student-state queries require an explicit experiment flag"
            )
        if episode["split"] != "train" or episode["motion_id"] not in allowed_ids:
            raise ValueError("Unknown/development motion in student training manifest")
        if sha(episode["path"]) != episode["sha256"]:
            raise ValueError("Episode content changed")
        if not episode["eligible"]:
            continue
        if stage == "navigation":
            from gear_sonic.research.scene_distillation.scene_teacher import verify_scene_receipt

            verify_scene_receipt(episode["qualification"], teacher_sha256, episode["task_sha256"])
            if episode.get("task_path"):
                from gear_sonic.research.scene_distillation.tasks import validate_task

                if sha(episode["task_path"]) != episode["task_sha256"]:
                    raise ValueError("Navigation task changed")
                validate_task(json.loads(Path(episode["task_path"]).read_text()))
        with np.load(episode["path"], allow_pickle=False) as data:
            arrays = {k: torch.from_numpy(data[k].copy()) for k in data.files}
        n = episode["rows"]
        shapes = {"proprio": (930,), "teacher_actions": (29,)}
        if stage == "foundation":
            shapes.update(
                teacher_tokens=(64,),
                privileged_state=(1645,),
                future_reference=(640,),
                controls=(79,),
                control_mask=(79,),
            )
        else:
            shapes.update(PUBLIC_SHAPES)
            shapes.update(qualified_mask=())
            if "label_controls" in arrays:
                shapes.update(label_controls=(79,), label_control_mask=(79,))
        for key, shape in shapes.items():
            value = arrays[key]
            dtype = torch.bool if key.endswith("mask") else torch.float32
            if value.shape != (n, *shape) or value.dtype != dtype:
                raise ValueError(f"Invalid episode field {key}")
            if key != "teacher_actions" and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite episode field {key}")
        if stage == "navigation":
            from gear_sonic.research.scene_distillation.contracts import public_observation_sha256

            binding = episode["queries"]
            if sha(binding["path"]) != binding["sha256"]:
                raise ValueError("Same-state query ledger changed")
            queries = json.loads(Path(binding["path"]).read_text())
            if len(queries) != n:
                raise ValueError("Same-state query ledger length mismatch")
            for i, query in enumerate(queries):
                digest = public_observation_sha256({k: arrays[k][i].numpy() for k in PUBLIC_SHAPES})
                if (
                    query["tick"] != i
                    or query["observation_sha256"] != digest
                    or query["teacher_sha256"] != teacher_sha256
                    or query["task_sha256"] != episode["task_sha256"]
                    or query["qualification_sha256"] != episode["qualification"]["sha256"]
                    or query["qualified"] != bool(arrays["qualified_mask"][i])
                ):
                    raise ValueError(
                        "Teacher query does not match recorded state/task/availability"
                    )
        if exploratory or prefix or (stage == "foundation" and "query_mask" in arrays):
            mask = arrays["query_mask"]
            if mask.shape != (n,) or mask.dtype != torch.bool:
                raise ValueError("Invalid student-state query mask")
            arrays = {k: v[mask] for k, v in arrays.items()}
            n = int(mask.sum())
        if n < 2:
            raise ValueError("Episode too short")
        episodes.append(arrays)
    if not episodes:
        raise ValueError(
            "No eligible executed teacher episodes; missing labels cannot train a student"
        )
    return episodes


def make_decoder(checkpoint, device):
    return FrozenSonicDecoder(load_release_checkpoint(checkpoint)["policy_state_dict"]).to(device)


def train_foundation_step(
    model, decoder, batch, beta, prior_weight, decoder_atol=2e-4, *, command_weights=None
):
    mask = (
        sample_command_mask(batch["control_mask"])
        if command_weights is None
        else curriculum_mask(batch["control_mask"], command_weights)
    )
    output = model.posterior_step(
        batch["proprio"],
        batch["privileged_state"],
        batch["future_reference"],
        batch["controls"],
        mask,
        epsilon=torch.randn(len(mask), model.latent_dim, device=mask.device),
    )
    result = variational_imitation_loss(
        output,
        batch["proprio"],
        batch["teacher_tokens"],
        batch["teacher_actions"],
        decoder,
        beta=beta,
        decoder_atol=decoder_atol,
    )
    public = model.prior_step(batch["proprio"], batch["controls"], mask)
    prior_mse = (
        (decoder(public["tokens"], batch["proprio"]) - batch["teacher_actions"]).square().mean()
    )
    result["loss"] = result["loss"] + prior_weight * prior_mse
    return {
        "loss": result["loss"],
        "reconstruction": result["reconstruction"],
        "kl": result["kl"],
        "prior_action_mse": prior_mse,
    }


def train_navigation_step(model, episode, start, length, config):
    """Reconstruct recurrent state from the complete public prefix, then truncated BPTT."""
    hidden = None
    with torch.no_grad():
        for t in range(start):
            hidden, _, _ = model.encoder.encode_step(
                {k: episode[k][t : t + 1] for k in PUBLIC_SHAPES}, hidden
            )
    previous = None
    terms = []
    for t in range(start, min(start + length, len(episode["proprio"]))):
        output = model.forward_step({k: episode[k][t : t + 1] for k in PUBLIC_SHAPES}, hidden)
        hidden = output["hidden"]
        valid = episode["qualified_mask"][t : t + 1]
        if valid.any():
            loss = navigation_distillation_loss(
                output,
                episode["teacher_actions"][t : t + 1],
                valid,
                prior_weight=config["latent_kl_weight"],
                residual_weight=config["action_residual_weight"],
                smoothness_weight=config["action_smoothness_weight"] if previous is not None else 0,
                previous_residual=previous,
                continuation_mask=torch.ones_like(valid),
            )
            command_weight = config.get("navigation_command_weight", 0.0)
            if not np.isfinite(command_weight) or command_weight < 0:
                raise ValueError("Navigation command weight must be finite and nonnegative")
            if command_weight:
                from gear_sonic.research.scene_distillation.navigation import (
                    FOUNDATION_CONTROL_INDICES,
                )

                indices = list(FOUNDATION_CONTROL_INDICES)
                target = episode["label_controls"][t : t + 1, indices]
                available = episode["label_control_mask"][t : t + 1, indices] & valid[:, None]
                if not available.any():
                    raise ValueError("Missing qualified navigation-command labels")
                radius = (model.command_upper - model.command_lower) / 2
                command_mse = (
                    (((output["controls"][:, indices] - target) / radius)[available])
                    .square()
                    .mean()
                )
                loss["loss"] = loss["loss"] + command_weight * command_mse
            terms.append(loss["loss"])
        previous = output["action_residual"]
    if not terms:
        raise ValueError("Sequence has no qualified queries")
    return {"loss": torch.stack(terms).mean()}


def main(args):
    config = json.loads(args.config.read_text())
    stage = config["stage"]
    if stage not in ("foundation", "navigation") or not 1 <= config["updates"] <= 100000:
        raise ValueError("Invalid bounded training stage")
    if stage == "navigation":
        from gear_sonic.research.scene_distillation.navigation_runtime import (
            require_public_prior_qualification,
        )

        if not config.get("foundation_qualification"):
            raise ValueError("Missing public-prior command qualification for navigation fitting")
        require_public_prior_qualification(
            config["foundation_qualification"], config["foundation_sha256"]
        )
    if not 0 < config["wall_cap_seconds"] <= 86400 or not 0 < config["learning_rate"] <= 0.01:
        raise ValueError("Invalid wall cap or learning rate")
    teacher_sha = sha(config["teacher_checkpoint"])
    if teacher_sha != config["teacher_sha256"]:
        raise ValueError("Teacher checkpoint changed")
    if sha(config["dataset_manifest"]) != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    episodes = load_episodes(
        config["dataset_manifest"],
        teacher_sha,
        set(config["train_ids"]),
        stage,
        allow_exploratory_queries=config.get("allow_exploratory_queries", False),
        allow_teacher_prefixes=config.get("allow_teacher_prefixes", False),
    )
    curriculum = config.get("command_curriculum")
    if curriculum is not None:
        curriculum_weights(curriculum, 0)
    sampling = config.get("foundation_sampling", "episode_uniform")
    if sampling not in ("episode_uniform", "reference_quarters"):
        raise ValueError("Unknown foundation sampling strategy")
    sampler = PhaseBalancedSampler(episodes) if sampling == "reference_quarters" else None
    args.output.mkdir(parents=True, exist_ok=False)
    if sampler is not None:
        write_new(args.output / "phase-coverage.json", {"episode_quarter_rows": sampler.coverage()})
    write_new(args.output / "config.json", config)
    torch.set_num_threads(2)
    if config.get("precision") == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = torch.device(config.get("device", "cpu"))
    decoder = make_decoder(config["teacher_checkpoint"], device)
    if stage == "foundation":
        model = MaskedMotionFoundation().to(device)
    else:
        foundation_checkpoint = Path(config["foundation_checkpoint"])
        if sha(foundation_checkpoint) != config["foundation_sha256"]:
            raise ValueError("Foundation checkpoint changed")
        saved = torch.load(foundation_checkpoint, map_location=device, weights_only=False)
        if saved["teacher_sha256"] != teacher_sha or saved["stage"] != "foundation":
            raise ValueError("Foundation/decoder provenance mismatch")
        foundation = MaskedMotionFoundation().to(device)
        foundation.load_state_dict(saved["model"], strict=True)
        model = FrozenFoundationNavigator(
            foundation,
            decoder,
            command_lower=config["command_lower"],
            command_upper=config["command_upper"],
            residual_scale=config.get("latent_residual_scale", 0),
            action_residual_limit=config["action_residual_limit"],
        ).to(device)
    model.train()
    if config.get("initialize_checkpoint"):
        if config.get("resume_checkpoint"):
            raise ValueError("Choose weights-only initialization or optimizer continuation")
        if sha(config["initialize_checkpoint"]) != config["initialize_sha256"]:
            raise ValueError("Initialization checkpoint changed")
        saved = torch.load(config["initialize_checkpoint"], map_location=device, weights_only=False)
        if saved["stage"] != stage or saved["teacher_sha256"] != teacher_sha:
            raise ValueError("Initialization stage/teacher mismatch")
        model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]
    )
    start_step = 0
    if config.get("resume_checkpoint"):
        if sha(config["resume_checkpoint"]) != config["resume_sha256"]:
            raise ValueError("Resume checkpoint changed")
        saved = torch.load(config["resume_checkpoint"], map_location=device, weights_only=False)
        if saved["stage"] != stage or saved["teacher_sha256"] != teacher_sha:
            raise ValueError("Resume stage/teacher mismatch")
        if saved["config"]["learning_rate"] != config["learning_rate"]:
            raise ValueError("Optimizer continuation requires the recorded learning rate")
        for key in ("command_curriculum", "foundation_sampling", "dataset_manifest_sha256"):
            if saved["config"].get(key) != config.get(key):
                raise ValueError(f"Optimizer continuation changed {key}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_torch"].cpu())
        rng.bit_generator.state = saved["rng_numpy"]
        if device.type == "cuda":
            if not saved.get("rng_cuda"):
                raise ValueError("CUDA RNG state missing: exact offline continuation unavailable")
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["rng_cuda"]])
        start_step = saved["step"]
    started = time.monotonic()
    step = 0
    status = "training"
    try:
        with (args.output / "metrics.jsonl").open("x") as log:
            for next_step in range(1, config["updates"] + 1):
                if time.monotonic() - started > config["wall_cap_seconds"]:
                    status = "wall_cap"
                    break
                if stage == "foundation":
                    # Balance episodes, not their unequal numbers of frames.
                    samples = []
                    for _ in range(config["batch_size"]):
                        if sampler is None:
                            e = episodes[int(rng.integers(len(episodes)))]
                            i = int(rng.integers(len(e["proprio"])))
                        else:
                            episode_index, i = sampler.sample(rng)
                            e = episodes[episode_index]
                        fields = (
                            "proprio",
                            "teacher_tokens",
                            "teacher_actions",
                            "privileged_state",
                            "future_reference",
                            "controls",
                            "control_mask",
                        )
                        samples.append({k: e[k][i] for k in fields})
                    batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in samples[0]}
                    losses = train_foundation_step(
                        model,
                        decoder,
                        batch,
                        config["beta"],
                        config["prior_action_weight"],
                        config.get("decoder_atol", 2e-4),
                        command_weights=(
                            curriculum_weights(curriculum, start_step + next_step - 1)
                            if curriculum is not None
                            else None
                        ),
                    )
                else:
                    e = episodes[int(rng.integers(len(episodes)))]
                    e = {k: v.to(device) for k, v in e.items()}
                    start = int(rng.integers(len(e["proprio"])))
                    losses = train_navigation_step(
                        model, e, start, config["sequence_length"], config
                    )
                if not torch.isfinite(losses["loss"]):
                    raise ValueError("Non-finite optimization loss")
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                step = next_step
                log.write(
                    json.dumps({"step": step, **{k: float(v.detach()) for k, v in losses.items()}})
                    + "\n"
                )
                log.flush()
                if step % config.get("save_interval", 100) == 0 or step == config["updates"]:
                    torch.save(
                        {
                            "stage": stage,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": start_step + step,
                            "teacher_sha256": teacher_sha,
                            "config": config,
                            "rng_torch": torch.get_rng_state(),
                            "rng_numpy": rng.bit_generator.state,
                            "rng_cuda": (
                                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                            ),
                            "qualification": "unqualified_offline_student",
                        },
                        args.output / f"step-{start_step + step:06d}.pt",
                    )
            if status == "training":
                status = "complete"
    except Exception:
        status = "failed"
        raise
    finally:
        write_new(
            args.output / "receipt.json",
            {
                "state": status,
                "updates": step,
                "starting_step": start_step,
                "final_step": start_step + step,
                "stage": stage,
                "wall_seconds": time.monotonic() - started,
                "physics_steps": 0,
                "eligible_episodes": len(episodes),
                "teacher_sha256": teacher_sha,
                "physical_student_performance": None,
            },
        )
    print(json.dumps({"state": status, "updates": step, "output": str(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
