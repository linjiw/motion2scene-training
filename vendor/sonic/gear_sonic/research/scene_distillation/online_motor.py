"""Bounded native online motor DAgger with RSI, burn-in and measured teacher takeovers."""

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.motor_runtime import (
    current_orientation_observation,
    load_motor,
    motor_commands,
)
from gear_sonic.research.scene_distillation.motor_training import (
    current_frame_extension,
    public_motor_loss,
)
from gear_sonic.research.scene_distillation.train import load_episodes


class RecoveryTracker:
    """A bounded tracking takeover test; never scene/task recovery qualification."""

    def __init__(self, count, device, horizon=16, stable_ticks=4):
        self.remaining = torch.zeros(count, dtype=torch.long, device=device)
        self.good = torch.zeros_like(self.remaining)
        self.horizon, self.stable_ticks = horizon, stable_ticks
        self.started = self.recovered = self.failed = 0

    def begin(self, selected):
        selected = selected & (self.remaining == 0)
        self.remaining[selected] = self.horizon
        self.good[selected] = 0
        self.started += int(selected.sum())
        return self.remaining > 0

    def observe(self, boundary, supported):
        active = self.remaining > 0
        self.good = torch.where(
            active & supported & ~boundary, self.good + 1, torch.zeros_like(self.good)
        )
        self.remaining[active] -= 1
        ended = active & ((self.remaining == 0) | boundary)
        passed = ended & ~boundary & (self.good >= self.stable_ticks)
        self.recovered += int(passed.sum())
        self.failed += int((ended & ~passed).sum())
        self.remaining[ended] = 0
        return passed, ended

    def report(self):
        return dict(
            started=self.started,
            recovered=self.recovered,
            failed=self.failed,
            censored=int((self.remaining > 0).sum()),
            horizon_ticks=self.horizon,
            stable_ticks=self.stable_ticks,
            qualification="local_tracking_takeover_only",
        )


class OnlineMotorCallback:
    """Runs a finite training loop inside the existing initialized Isaac Lab driver."""

    def __init__(self, stage_config, **kwargs):
        self.config = json.loads(Path(stage_config).read_text())

    def on_step_end(self, args, state, control, **kwargs):
        c = self.config
        if (
            not 1 <= c["cycles"] <= 1000
            or not 1 <= c["rollout_steps"] <= 128
            or not 1 <= c["updates_per_cycle"] <= 64
        ):
            raise ValueError("Invalid bounded online schedule")
        if not 0 < c["wall_cap_seconds"] <= 7200 or not 1 <= c["batch_size"] <= 4096:
            raise ValueError("Invalid online resources")
        for path_key, hash_key in [
            ("student_checkpoint", "student_sha256"),
            ("replay_manifest", "replay_manifest_sha256"),
        ]:
            if sha(c[path_key]) != c[hash_key]:
                raise ValueError("Online input changed")
        output = Path(c["output"])
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "config.json", c)
        env, teacher = kwargs["env"], kwargs["model"].policy
        teacher.eval()
        teacher.eval_mode()
        teacher.requires_grad_(False)
        torch.set_num_threads(2)
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, decoder, mc = load_motor(c["student_checkpoint"], c["teacher_sha256"], env.device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=c["learning_rate"]
        )
        episodes = load_episodes(
            c["replay_manifest"],
            c["teacher_sha256"],
            set(c["train_ids"]),
            "foundation",
            allow_teacher_prefixes=True,
            allow_exploratory_queries=True,
        )
        fields = [
            "proprio",
            "controls",
            "control_mask",
            "teacher_tokens",
            "teacher_actions",
            "future_reference",
        ]
        for e in episodes:
            if mc.get("current_frame_extension", False):
                extra = current_frame_extension(e["future_reference"], e["controls"])
                e["controls"] = torch.cat([e["controls"], extra], -1)
                e["control_mask"] = torch.cat(
                    [e["control_mask"], torch.ones_like(extra, dtype=torch.bool)], -1
                )
        replay = {k: torch.cat([e[k] for e in episodes]).to(env.device) for k in fields}
        if hasattr(model, "initialize_normalization"):
            model.initialize_normalization(replay)
        del episodes
        # Capture the exact same-state inputs and targets used by the frozen teacher.
        captured = {}
        handles = []
        for name, module in [
            ("reference", teacher.actor_module.encoders["g1"].module[0]),
            ("decoder", teacher.actor_module.decoders["g1_dyn"].module[0]),
        ]:

            def hook(module, inputs, name=name):
                captured[name] = inputs[0].detach().reshape(env.num_envs, -1).clone()

            handles.append(module.register_forward_pre_hook(hook))
        env.set_is_evaluating(False)
        command = env.motion_command
        if command.cfg.start_from_first_frame or command.cfg.use_paired_motions:
            raise ValueError("Online RSI requires randomized reference starts")
        keys = list(env._motion_lib.curr_motion_keys)
        if {k[-5:] for k in keys} != set(c["train_ids"]):
            raise ValueError("Online resident train split mismatch")
        original_sampler = env._motion_lib.sample_time_steps

        def sample_times(*a, **kw):
            times = original_sampler(*a, **kw)
            return torch.where(
                torch.rand(times.shape, device=times.device) < c.get("nominal_start_fraction", 0.2),
                0,
                times,
            )

        env._motion_lib.sample_time_steps = sample_times
        observation = env.reset_all()
        teacher.init_rollout()
        age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        tracker = RecoveryTracker(env.num_envs, env.device, horizon=c.get("takeover_ticks", 16))
        motion_rows = torch.zeros(len(keys), dtype=torch.long, device=env.device)
        phase_rows = torch.zeros(4, dtype=torch.long, device=env.device)
        step = 0
        interventions = 0
        boundaries = 0
        burn_in_rows = 0
        began = time.monotonic()
        status = "failed"
        cycle = 0
        try:
            with (output / "metrics.jsonl").open("x") as log:
                for cycle in range(1, c["cycles"] + 1):
                    if time.monotonic() - began > c["wall_cap_seconds"]:
                        raise TimeoutError("Online motor wall cap")
                    chunk = {k: [] for k in fields}
                    with torch.no_grad():
                        for tick in range(c["rollout_steps"]):
                            captured.clear()
                            target = teacher.act_inference(
                                obs_dict=observation, skip_episode_attnmask=True
                            )
                            history = captured["decoder"][:, 64:]
                            controls, mask = motor_commands(
                                command,
                                env.env.scene.env_origins,
                                mc.get("current_frame_extension", False),
                                (
                                    current_orientation_observation(
                                        teacher.actor_module, observation
                                    )
                                    if mc.get("current_frame_extension", False)
                                    else None
                                ),
                            )
                            if mc.get("current_frame_extension", False):
                                extra = current_frame_extension(
                                    captured["reference"], controls[:, :79]
                                )
                                torch.testing.assert_close(
                                    extra, controls[:, 79:], atol=2e-5, rtol=2e-5
                                )
                            batch = dict(
                                proprio=history,
                                controls=controls,
                                control_mask=mask,
                                teacher_tokens=captured["decoder"][:, :64],
                                teacher_actions=target,
                                future_reference=captured["reference"],
                            )
                            for k, v in batch.items():
                                if not torch.isfinite(v).all():
                                    raise ValueError("Nonfinite online query")
                                chunk[k].append(v.detach().clone())
                            result = model.prior_step(history, controls, mask)
                            student = decoder(result["tokens"], history)
                            error = torch.linalg.vector_norm(
                                command.body_pos_w - command.robot_body_pos_w, dim=-1
                            ).mean(-1)
                            remaining = (
                                env._motion_lib.get_time_step_total(command.motion_ids)
                                - command.time_steps
                                - command.motion_start_time_steps
                            )
                            begin = (
                                (error > c.get("takeover_trigger_m", 0.1))
                                & (error < 0.25)
                                & (age >= c["burn_in_ticks"])
                                & (remaining > tracker.horizon + 2)
                            )
                            begin &= torch.rand(env.num_envs, device=env.device) < c.get(
                                "takeover_probability", 0.15
                            )
                            takeover = tracker.begin(begin)
                            probability = c["teacher_probability_start"] + (
                                c["teacher_probability_end"] - c["teacher_probability_start"]
                            ) * (cycle - 1) / max(1, c["cycles"] - 1)
                            burn = age < c["burn_in_ticks"]
                            intervene = (
                                burn
                                | takeover
                                | (torch.rand(env.num_envs, device=env.device) < probability)
                            )
                            action = torch.where(intervene[:, None], target, student)
                            mid = command.motion_ids.clone()
                            start = command.motion_start_time_steps.clone()
                            elapsed = command.time_steps.clone()
                            motion_rows += torch.bincount(mid, minlength=len(keys))
                            total = env._motion_lib.get_time_step_total(mid)
                            quarter = (((start + elapsed).float() / total) * 4).long().clamp(0, 3)
                            phase_rows += torch.bincount(quarter, minlength=4)
                            observation, _, done, _ = env.step({"actions": action})
                            boundary = (
                                done.reshape(-1).bool()
                                | (command.motion_ids != mid)
                                | (command.motion_start_time_steps != start)
                                | (command.time_steps <= elapsed)
                            )
                            after = torch.linalg.vector_norm(
                                command.body_pos_w - command.robot_body_pos_w, dim=-1
                            ).mean(-1)
                            tracker.observe(boundary, after <= c.get("takeover_support_m", 0.1))
                            age = torch.where(boundary, 0, age + 1)
                            interventions += int(intervene.sum())
                            burn_in_rows += int(burn.sum())
                            boundaries += int(boundary.sum())
                    fresh = {k: torch.cat(v) for k, v in chunk.items()}
                    del chunk
                    with torch.no_grad():
                        torch.testing.assert_close(
                            decoder(fresh["teacher_tokens"][:16], fresh["proprio"][:16]),
                            fresh["teacher_actions"][:16],
                            atol=2e-4,
                            rtol=2e-4,
                        )
                    with torch.enable_grad():
                        for update in range(c["updates_per_cycle"]):
                            n = c["batch_size"] // 2
                            i = torch.randint(len(fresh["proprio"]), (n,), device=env.device)
                            j = torch.randint(
                                len(replay["proprio"]), (c["batch_size"] - n,), device=env.device
                            )
                            batch = {k: torch.cat([fresh[k][i], replay[k][j]]) for k in fields}
                            loss = public_motor_loss(
                                model,
                                decoder,
                                batch,
                                c.get("token_weight", 0.1),
                                c.get("future_weight", 0.0),
                            )
                            if not torch.isfinite(loss["loss"]):
                                raise ValueError("Nonfinite online loss")
                            optimizer.zero_grad()
                            loss["loss"].backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1, error_if_nonfinite=True
                            )
                            optimizer.step()
                            step += 1
                    item = dict(
                        cycle=cycle,
                        updates=step,
                        transitions=cycle * c["rollout_steps"] * env.num_envs,
                        teacher_action_fraction=interventions
                        / (cycle * c["rollout_steps"] * env.num_envs),
                        burn_in_rows=burn_in_rows,
                        boundaries=boundaries,
                        **{k: float(v.detach()) for k, v in loss.items()},
                        recovery=tracker.report(),
                    )
                    log.write(json.dumps(item) + "\n")
                    log.flush()
                    if cycle % c.get("save_cycles", 50) == 0 or cycle == c["cycles"]:
                        torch.save(
                            dict(
                                stage="motor_foundation",
                                model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                config=mc,
                                teacher_sha256=c["teacher_sha256"],
                                step=step,
                                online_config=c,
                                qualification="unqualified_online_motor_research",
                            ),
                            output / f"step-{step:06d}.pt",
                        )
                        np.savez_compressed(
                            output / f"query-sample-{cycle:04d}.npz",
                            **{k: v[: min(512, len(v))].cpu().numpy() for k, v in fresh.items()},
                        )
                        write_new(output / f"progress-{cycle:04d}.json", item)
                    del fresh
            status = "complete"
        finally:
            env._motion_lib.sample_time_steps = original_sampler
            for h in handles:
                h.remove()
            write_new(
                output / "receipt.json",
                dict(
                    state=status,
                    cycles=cycle,
                    updates=step,
                    transitions=int(motion_rows.sum()),
                    teacher_actions=interventions,
                    burn_in_rows=burn_in_rows,
                    boundaries=boundaries,
                    motion_rows=dict(zip(keys, motion_rows.cpu().tolist())),
                    phase_rows=phase_rows.cpu().tolist(),
                    recovery=tracker.report(),
                    wall_seconds=time.monotonic() - began,
                    decoder_frozen=True,
                    optimizer_initialization="fresh_for_online_stage",
                    query_support="finite_pre_step_states_under_native_termination_no_0.10m_prefix_censor",
                    scene_qualified=False,
                ),
            )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
