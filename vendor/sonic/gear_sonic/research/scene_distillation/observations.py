"""Route-free features for a known map of at most five exact collision primitives."""

import numpy as np

from gear_sonic.research.hindsight_training.observations import scene_features
from gear_sonic.research.scene_distillation.contracts import public_observation_sha256


def navigation_observation(*, proprio, root_xyz, root_wxyz, start_xyz, goal_xyz, obstacles):
    """Use measured SONIC history and root pose, plus an explicitly supplied scene map.

    Input obstacles follow scene_features' schema. The third shape channel means
    sphere in this profile. Unsupported meshes must not silently become spheres.
    Scene-known availability is distinct from a camera's visibility or field of view.
    """
    normalized = []
    for obstacle in obstacles:
        obstacle = dict(obstacle)
        if obstacle["shape"] == "beam":
            obstacle["shape"] = "box"
        if obstacle["shape"] not in ("box", "sphere", "cylinder"):
            raise ValueError("This profile supports box/beam, sphere and cylinder primitives only")
        dimensions = np.asarray(obstacle["full_dimensions_xyz"], dtype=float)
        if dimensions.shape != (3,):
            raise ValueError("Full dimensions must have length three")
        if obstacle["shape"] == "sphere" and not np.allclose(dimensions, dimensions[0]):
            raise ValueError("Sphere dimensions must be its equal full diameters")
        if obstacle["shape"] == "cylinder" and not np.isclose(dimensions[0], dimensions[1]):
            raise ValueError("Cylinder must have equal xy diameters and a local z axis")
        normalized.append(obstacle)
    features = scene_features(
        root_xyz=root_xyz,
        root_wxyz=root_wxyz,
        start_xyz=start_xyz,
        goal_xyz=goal_xyz,
        obstacles=normalized,
    )
    observation = {"proprio": np.asarray(proprio, dtype=np.float32)}
    observation.update(
        {key: features[key] for key in ("start_goal_body", "obstacles_body", "obstacle_mask")}
    )
    public_observation_sha256(observation)
    return observation
