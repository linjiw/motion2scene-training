"""Continuation with optimizer history, online telemetry, and bounded termination."""

import json
from pathlib import Path

from transformers import TrainerCallback
import wandb

from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha, write_new
from gear_sonic.research.hindsight_training.tracker import TrackingReceiptCallback
from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer


def restore_optimizer_history(optimizer, scheduler, checkpoint):
    raw = getattr(optimizer, "optimizer", optimizer)
    if raw.state:
        raise ValueError("Expected an uninitialized optimizer before continuation")
    saved = checkpoint["optimizer_state_dict"]
    if [len(g["params"]) for g in raw.param_groups] != [
        len(g["params"]) for g in saved["param_groups"]
    ]:
        raise ValueError("Checkpoint optimizer groups do not match this architecture")
    optimizer.load_state_dict(saved)
    scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    return len(raw.state)


class LongTrackingTrainer(TRLAuxLossPPOTrainer):
    def load_checkpoint(self, checkpoint_path, resume=False):
        if resume:
            raise ValueError(
                "Use a new run directory; this continuation restores optimizer history explicitly"
            )
        checkpoint = load_release_checkpoint(checkpoint_path)
        model = self.accelerator.unwrap_model(self.model)
        model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        model.value_model.load_state_dict(checkpoint["value_state_dict"], strict=True)
        entries = restore_optimizer_history(self.optimizer, self.lr_scheduler, checkpoint)
        self.args.learning_rate = float(checkpoint["args"].learning_rate)
        write_new(
            Path(self.config.hindsight_run_dir) / "initialization.json",
            {
                "checkpoint": str(checkpoint_path),
                "sha256": sha(checkpoint_path),
                "source_iteration": int(checkpoint["state"].global_step),
                "policy_strict": True,
                "critic_strict": True,
                "optimizer_state_entries": entries,
                "optimizer_history_restored": True,
                "scheduler_history_restored": True,
                "environment_state_restored": False,
                "new_run_start_iteration": 0,
                "learning_rate": self.args.learning_rate,
                "reason": "Larger environment batch; fresh simulation state, additional bounded training budget",
            },
        )
        return checkpoint

    def train(self):
        try:
            return super().train()
        except BaseException as error:
            callback = getattr(self.env, "_hindsight_receipt_callback", None)
            if callback is not None and callback.started is not None:
                write_new(
                    Path(self.config.hindsight_run_dir) / "failure-receipt.json",
                    {
                        **callback.report(self.state.global_step, "technical_failure"),
                        "error": repr(error),
                    },
                )
            raise
        finally:
            # Native entry point uses os._exit after train(), so flush before returning.
            if wandb.run is not None:
                receipt = Path(self.config.hindsight_run_dir) / "training-receipt.json"
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                    wandb.run.summary["tracking_run_state"] = result["state"]
                    wandb.run.summary["final_checkpoint_sha256"] = result.get("checkpoint", {}).get(
                        "sha256"
                    )
                wandb.finish(exit_code=0 if receipt.exists() else 1)


class OnlineTrackingReceipt(TrackingReceiptCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if wandb.run is None or wandb.run.settings.mode != "online":
            raise ValueError("This run requires active online W&B logging")
        planned = self.packet["wandb"]
        if wandb.run.id != planned["id"] or wandb.run.entity != planned["entity"]:
            raise ValueError("W&B identity does not match the locked run")
        wandb.run.name = planned["name"]
        result = super().on_train_begin(args, state, control, **kwargs)
        kwargs["env"]._hindsight_receipt_callback = self
        write_new(
            self.output / "wandb-run.json",
            {
                "mode": wandb.run.settings.mode,
                "id": wandb.run.id,
                "url": wandb.run.url,
                "entity": wandb.run.entity,
                "project": wandb.run.project,
            },
        )
        return result


class TrackingWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        callback = getattr(kwargs["env"], "_hindsight_receipt_callback", None)
        values = dict(logs, global_step=int(state.global_step))
        if callback is not None:
            report = callback.report(state.global_step, "training")
            values.update(
                {
                    "hindsight/motions_seen": len(callback.exposure),
                    "hindsight/env_transitions": report["observed_env_transitions"],
                    "hindsight/env_physics_steps": report["observed_env_physics_steps"],
                    "hindsight/cuda_peak_allocated_MiB": report["cuda_peak_allocated_bytes"]
                    / 1024**2,
                    "hindsight/cuda_free_MiB": report["cuda_free_total_bytes"][0] / 1024**2,
                    "hindsight/elapsed_seconds": report["wall_seconds"],
                }
            )
            with (callback.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(values, default=lambda x: x.item()) + "\n")
        if state.is_world_process_zero and wandb.run is not None:
            wandb.log(values, step=state.global_step)
