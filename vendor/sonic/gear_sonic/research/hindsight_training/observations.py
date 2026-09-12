"""World-to-body feature conversion for measured state and supplied scene commands."""

import numpy as np


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-4:
        raise ValueError("Expected a finite unit quaternion in wxyz order")
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def scene_features(*, root_xyz, root_wxyz, start_xyz, goal_xyz, obstacles, route_xyz=None):
    """Create one padded observation. Obstacles use full dimensions, not half-extents.

    Obstacle mappings need center_xyz, quaternion_wxyz, full_dimensions_xyz, shape.
    The same world-frame convention must describe the robot, route, and obstacles.
    Up to five obstacles and sixteen ordered route command points are supported.
    """
    if len(obstacles) > 5:
        raise ValueError("More than five obstacles requires a separately configured profile")
    root = np.asarray(root_xyz, dtype=np.float64)
    rotation = rotation_wxyz(root_wxyz)
    command = (np.asarray([start_xyz, goal_xyz]) - root) @ rotation
    rows = np.zeros((5, 15), dtype=np.float32)
    mask = np.zeros(5, dtype=bool)
    for index, obstacle in enumerate(obstacles):
        center = (np.asarray(obstacle["center_xyz"]) - root) @ rotation
        dimensions = np.asarray(obstacle["full_dimensions_xyz"])
        if dimensions.shape != (3,) or np.any(dimensions <= 0):
            raise ValueError("Obstacle full dimensions must be positive xyz lengths")
        local_rotation = rotation.T @ rotation_wxyz(obstacle["quaternion_wxyz"])
        shape = [
            obstacle["shape"] == "box",
            obstacle["shape"] == "cylinder",
            obstacle["shape"] not in ("box", "cylinder"),
        ]
        rows[index] = np.concatenate([center, dimensions, local_rotation[:, :2].T.flatten(), shape])
        mask[index] = True
    route = np.zeros((16, 3), dtype=np.float32)
    route_mask = np.zeros(16, dtype=bool)
    if route_xyz is not None:
        points = np.asarray(route_xyz)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) > 16:
            raise ValueError("Route must contain at most sixteen ordered xyz command waypoints")
        route[: len(points)] = (points - root) @ rotation
        route_mask[: len(points)] = True
    if command.shape != (2, 3) or not all(np.isfinite(x).all() for x in (command, rows, route)):
        raise ValueError("Invalid or non-finite world-frame scene")
    return {
        "start_goal_body": command.astype(np.float32).flatten(),
        "obstacles_body": rows,
        "obstacle_mask": mask,
        "route_body": route,
        "route_mask": route_mask,
    }
