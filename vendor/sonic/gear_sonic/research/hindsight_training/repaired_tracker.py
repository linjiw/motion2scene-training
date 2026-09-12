"""Fresh-optimizer repaired teacher with local metrics and preserved failure accounting."""

import json
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import write_new
from gear_sonic.research.hindsight_training.tracker import (
    HindsightTrackerTrainer,
    TrackingReceiptCallback,
)


class RepairedTeacherTrainer(HindsightTrackerTrainer):
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


class RepairedTrackingReceipt(TrackingReceiptCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        result = super().on_train_begin(args, state, control, **kwargs)
        kwargs["env"]._hindsight_receipt_callback = self
        return result

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with (self.output / "metrics.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        dict(logs, global_step=int(state.global_step)), default=lambda x: x.item()
                    )
                    + "\n"
                )
        return control
