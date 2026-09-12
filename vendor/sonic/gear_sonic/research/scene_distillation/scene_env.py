"""Isaac Lab sensor configuration for exact-map teacher qualification."""

import json

from isaaclab.sensors import ContactSensorCfg
import numpy as np

from gear_sonic.envs.manager_env.modular_tracking_env_cfg import ModularTrackingEnvCfg
from gear_sonic.research.scene_distillation.tasks import validate_task


class SceneQualificationEnvCfg(ModularTrackingEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        task = validate_task(json.load(open(config["navigation_task_path"])))
        with np.load(task["reference"]["path"], allow_pickle=False) as data:
            names = [str(n) for n in data["body_names"] if str(n) != "world"]
        paths = ["/World/ground/terrain/Structure/Floor"] + [
            f"/World/ground/terrain/Obstacles/obstacle_{i}"
            for i in range(1, len(task["obstacles"]) + 1)
        ]
        for name in names:
            setattr(
                self.scene,
                f"navigation_contact_{name}",
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_0/Robot/{name}",
                    filter_prim_paths_expr=paths,
                    update_period=0.0,
                    history_length=0,
                    track_air_time=False,
                    force_threshold=0.0,
                ),
            )
