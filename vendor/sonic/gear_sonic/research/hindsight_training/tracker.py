"""Strict SONIC warm start and measured accounting for the bounded tracker fit."""

import json
from pathlib import Path
import time

import torch
from transformers import TrainerCallback

from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha, write_new
from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer


class HindsightTrackerTrainer(TRLAuxLossPPOTrainer):
    def load_checkpoint(self, checkpoint_path, resume=False):
        if resume:
            raise ValueError("This packet initializes model weights with a fresh optimizer")
        checkpoint = load_release_checkpoint(checkpoint_path)
        model = self.accelerator.unwrap_model(self.model)
        model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        model.value_model.load_state_dict(checkpoint["value_state_dict"], strict=True)
        raw = getattr(self.optimizer, "optimizer", self.optimizer)
        if len(raw.state):
            raise ValueError("Optimizer state must be empty before the new fit")
        write_new(
            Path(self.config.hindsight_run_dir) / "initialization.json",
            {
                "checkpoint": str(checkpoint_path),
                "sha256": sha(checkpoint_path),
                "policy_strict": True,
                "critic_strict": True,
                "optimizer_state_entries": len(raw.state),
                "resume": False,
            },
        )
        return checkpoint


class TrackingReceiptCallback(TrainerCallback):
    def __init__(self, packet, output):
        self.packet = json.loads(Path(packet).read_text())
        self.output = Path(output)
        self.started = None
        self.steps = 0
        self.physics = 0
        self.exposure = {}
        self.subscription = None

    def on_train_begin(self, args, state, control, **kwargs):
        import omni.physx

        self.started = time.perf_counter()
        env = kwargs["env"]
        self.envs = int(env.num_envs)
        library = env._motion_lib
        expected = {"hindsight_" + x for x in self.packet["train_ids"]}
        actual = set(library.curr_motion_keys)
        if actual != expected:
            raise ValueError(f"Training motion assignment mismatch: {actual ^ expected}")
        self.old_step = env.step

        def observed_step(actions):
            keys = getattr(library, "curr_motion_keys", [])
            selected = getattr(env.motion_command, "motion_ids", None)
            if selected is not None:
                for i in selected.detach().cpu().tolist():
                    key = keys[int(i)]
                    self.exposure[key] = self.exposure.get(key, 0) + 1
            result = self.old_step(actions)
            self.steps += 1
            return result

        env.step = observed_step

        def physics_event(dt):
            self.physics += 1

        self.subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            physics_event
        )
        torch.cuda.reset_peak_memory_stats()
        write_new(
            self.output / "started.json",
            {
                "state": "training",
                "resident_motion_count": len(actual),
                "num_envs": self.envs,
                "starting_iteration": int(state.global_step),
            },
        )
        return control

    def report(self, iteration, status):
        return {
            "state": status,
            "iteration": int(iteration),
            "observed_env_step_calls": self.steps,
            "observed_env_transitions": self.steps * self.envs,
            "observed_physics_step_events": self.physics,
            "observed_env_physics_steps": self.physics * self.envs,
            "initialization_physics_steps": None,
            "measurement_scope": "Training callback lifetime; initialization not observed",
            "wall_seconds": time.perf_counter() - self.started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cuda_free_total_bytes": list(torch.cuda.mem_get_info()),
            "motion_exposure_transitions": self.exposure,
            "physical_navigation_passage": None,
        }

    def on_step_end(self, args, state, control, **kwargs):
        with (self.output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(self.report(state.global_step, "training")) + "\n")
        # Native PPO returns before on_train_end when the stop flag is set.
        # The model-save callback has already persisted this boundary checkpoint.
        if int(state.global_step) == self.packet["tracking"]["iterations"]:
            return self.on_train_end(args, state, control, **kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if (self.output / "training-receipt.json").exists():
            return control
        kwargs["env"].step = self.old_step
        self.subscription = None
        expected = self.packet["tracking"]["iterations"]
        path = self.output / f"model_step_{expected:06d}.pt"
        report = self.report(state.global_step, "complete")
        if int(state.global_step) != expected or not path.is_file():
            report["state"] = "incomplete"
        else:
            report["checkpoint"] = {"path": str(path), "sha256": sha(path)}
        write_new(self.output / "training-receipt.json", report)
        return control
