"""Retain native matched-evaluation trajectories for auditable tracking videos."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback


class TrackingQualificationCallback(ImEvalCallback):
    """Use native evaluation unchanged and save its measured body trajectories."""

    def _post_evaluate_policy(self, eval_res):
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        keys = eval_res["all_metrics_dict"]["motion_keys"]
        names = list(self.env.motion_command.cmd_body_names)
        for index, key in enumerate(keys):
            np.savez_compressed(
                output / f"{key}.npz",
                reference=self.gt_pos_all[index],
                tracked=self.pred_pos_all[index],
                body_names=np.asarray(names),
            )
        (output / "trajectory-contract.json").write_text(
            json.dumps(
                {
                    "source": "Native ImEvalCallback measured simulated body positions",
                    "dt": float(self.env.env.step_dt),
                    "body_names": names,
                    "failure_handling": "Use native progress to censor post-failure/reset frames",
                    "cameras": False,
                },
                indent=2,
            )
        )
        return super()._post_evaluate_policy(eval_res)
