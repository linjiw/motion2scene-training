"""Evaluation-only goal/map-use ablations for the known-map navigation adapter (Phase 0.4).

Both flags change only what the actor is told about the task. The physical scene, the
contact sensors, the reference clock and the scorer keep the original task:

- ``goal_rotation_deg`` rotates the goal about the vertical axis through the start
  (world-frame yaw, counter-clockwise seen from +z) before it enters the actor
  observation. Success is still scored against the original goal; the task result
  additionally records final pelvis XY distance to the original goal, the rotated goal
  and the reference endpoint.
- ``zero_obstacles`` presents the map as empty: every primitive slot takes the padding
  used for absent primitives (zero geometry, ``obstacle_mask`` false), not raw zeros
  behind a true mask. The collision walls stay in the scene.

With neither flag the actor sees the task object itself, so default runs are unchanged.
"""

from dataclasses import dataclass
import math

import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha

# Phase 0.4 decision rule (docs/ROADMAP_20260923.md §8): the adapter ignores the goal if
# its final XY stays within 0.5 m of the reference endpoint in >=70% of tasks.
IGNORE_RADIUS_M = 0.5
IGNORE_FRACTION = 0.70
# Displacements shorter than this have no meaningful heading.
MIN_DISPLACEMENT_M = 0.3
# A displacement heading within this of the rotated goal's direction counts as following it.
HEADING_TOLERANCE_DEG = 45.0


@dataclass(frozen=True)
class ObservationAblation:
    goal_rotation_deg: float | None = None
    zero_obstacles: bool = False

    @property
    def active(self):
        return self.goal_rotation_deg is not None or self.zero_obstacles

    @classmethod
    def from_config(cls, config):
        rotation = config.get("goal_rotation_deg")
        zero = config.get("zero_obstacles", False)
        if rotation is not None and (
            isinstance(rotation, bool)
            or not isinstance(rotation, (int, float))
            or not math.isfinite(rotation)
        ):
            raise ValueError("goal_rotation_deg must be a finite number of degrees")
        if not isinstance(zero, bool):
            raise ValueError("zero_obstacles must be a boolean")
        return cls(None if rotation is None else float(rotation), zero)

    def as_record(self):
        return dict(goal_rotation_deg=self.goal_rotation_deg, zero_obstacles=self.zero_obstacles)


def rotate_goal_about_start(start_xyz, goal_xyz, degrees):
    """Yaw-rotate the goal about the vertical axis through the start; the goal height is kept."""
    start, goal = np.asarray(start_xyz, dtype=np.float64), np.asarray(goal_xyz, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([*start, *goal]).all():
        raise ValueError("Start and goal must be finite xyz positions")
    if not math.isfinite(degrees):
        raise ValueError("Goal rotation must be finite")
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = goal[:2] - start[:2]
    return [float(start[0] + c * dx - s * dy), float(start[1] + s * dx + c * dy), float(goal[2])]


def observation_task(task, ablation):
    """The task as the actor observes it: ``task`` itself unless an ablation is active."""
    if not ablation.active:
        return task
    view = dict(task)
    if ablation.goal_rotation_deg is not None:
        view["goal_xyz"] = rotate_goal_about_start(
            task["start_xyz"], task["goal_xyz"], ablation.goal_rotation_deg
        )
    if ablation.zero_obstacles:
        view["obstacles"] = []
    return view


def reference_endpoint(task):
    """Final reference pelvis position from the task's sha-bound reference-with-hold."""
    binding = task["reference"]
    if sha(binding["path"]) != binding["sha256"]:
        raise ValueError("Task reference changed")
    with np.load(binding["path"], allow_pickle=False) as reference:
        endpoint = np.asarray(reference["qpos"][-1, :3], dtype=np.float64)
    if not np.isfinite(endpoint).all():
        raise ValueError("Nonfinite reference endpoint")
    return endpoint.tolist()


def xy_distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a[:2] - b[:2]))


def ablation_record(task, ablation, final_root_xyz, reference_endpoint_xyz):
    """task-result.json block for an ablated run. Scoring stays on the original goal."""
    final = np.asarray(final_root_xyz, dtype=np.float64)
    if final.shape != (3,) or not np.isfinite(final).all():
        raise ValueError("Invalid final pelvis position")
    rotated = (
        None
        if ablation.goal_rotation_deg is None
        else rotate_goal_about_start(task["start_xyz"], task["goal_xyz"], ablation.goal_rotation_deg)
    )
    return dict(
        **ablation.as_record(),
        scoring_goal="original",
        start_xyz=[float(v) for v in task["start_xyz"]],
        original_goal_xyz=[float(v) for v in task["goal_xyz"]],
        rotated_goal_xyz=rotated,
        reference_endpoint_xyz=[float(v) for v in reference_endpoint_xyz],
        observed_obstacles=0 if ablation.zero_obstacles else len(task["obstacles"]),
        final_root_xyz=final.tolist(),
        final_xy_distance_to_original_goal_m=xy_distance(final, task["goal_xyz"]),
        final_xy_distance_to_rotated_goal_m=None if rotated is None else xy_distance(final, rotated),
        final_xy_distance_to_reference_endpoint_m=xy_distance(final, reference_endpoint_xyz),
    )


def _heading_deg(vector_xy):
    return math.degrees(math.atan2(vector_xy[1], vector_xy[0]))


def _wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def final_position_row(task, final_root_xyz, reference_endpoint_xyz, goal_rotation_deg,
                       radius_m=IGNORE_RADIUS_M):
    """Where one episode ended relative to start, original goal, rotated goal and reference end.

    Progress terms project the start-to-final displacement on the original and rotated goal
    directions, normalized by the start-goal distance: a goal-blind adapter under rotation
    scores about (1, 0), a goal-following one about (0, 1).
    """
    rotation = 0.0 if goal_rotation_deg is None else float(goal_rotation_deg)
    start = np.asarray(task["start_xyz"], dtype=np.float64)[:2]
    goal = np.asarray(task["goal_xyz"], dtype=np.float64)[:2]
    rotated = np.asarray(rotate_goal_about_start(task["start_xyz"], task["goal_xyz"], rotation))[:2]
    final = np.asarray(final_root_xyz, dtype=np.float64)[:2]
    reference = np.asarray(reference_endpoint_xyz, dtype=np.float64)[:2]
    length = float(np.linalg.norm(goal - start))
    if length < 1e-6:
        raise ValueError("Goal coincides with the start; goal use is undefined")
    along, across = (goal - start) / length, (rotated - start) / length
    displacement = final - start
    separation = float(np.linalg.norm(rotated - goal))
    moved = float(np.linalg.norm(displacement))
    return dict(
        goal_rotation_deg=rotation,
        start_goal_xy_m=length,
        goal_separation_xy_m=separation,
        discriminative=separation > 2 * radius_m,
        final_xy=final.tolist(),
        final_xy_distance_to_reference_endpoint_m=float(np.linalg.norm(final - reference)),
        final_xy_distance_to_original_goal_m=float(np.linalg.norm(final - goal)),
        final_xy_distance_to_rotated_goal_m=float(np.linalg.norm(final - rotated)),
        final_xy_distance_to_start_m=moved,
        near_reference_endpoint=bool(np.linalg.norm(final - reference) <= radius_m),
        near_rotated_goal=bool(np.linalg.norm(final - rotated) <= radius_m),
        near_start=bool(moved <= radius_m),
        progress_along_original=float(displacement @ along / length),
        progress_along_rotated=float(displacement @ across / length),
        displacement_heading_vs_original_deg=(
            _wrap_deg(_heading_deg(displacement) - _heading_deg(goal - start))
            if moved >= MIN_DISPLACEMENT_M
            else None
        ),
    )


def goal_use_decision(rows, baseline_rows=None, radius_m=IGNORE_RADIUS_M,
                      fraction=IGNORE_FRACTION):
    """Apply the Phase 0.4 rule to per-task rows keyed by task id.

    The registered rule is reported unchanged. Rollouts at a fixed seed have been
    bit-deterministic (seed 91260; roadmap 0.3 re-checks), so a goal-blind adapter would
    reproduce the unrotated baseline and the rule can only fire if the baseline itself ends
    near the reference endpoint in at least ``fraction`` of tasks. With a baseline, that
    attainability and the paired shifts are reported as a separate sensitivity check; they
    never change the registered verdict. Final-XY shifts measure sensitivity to the goal
    input (physics divergence inflates them); heading turns toward the rotated goal
    measure goal following.
    """
    if not rows:
        raise ValueError("No completed tasks to analyze")
    n = len(rows)
    near = sum(r["near_reference_endpoint"] for r in rows.values())
    headings = [
        r["displacement_heading_vs_original_deg"]
        for r in rows.values()
        if r["displacement_heading_vs_original_deg"] is not None
    ]
    toward = sum(
        abs(_wrap_deg(r["displacement_heading_vs_original_deg"] - r["goal_rotation_deg"]))
        <= HEADING_TOLERANCE_DEG
        for r in rows.values()
        if r["displacement_heading_vs_original_deg"] is not None
    )
    result = dict(
        rule=(
            f"adapter ignores the goal if final XY stays within {radius_m} m of the reference "
            f"endpoint in >= {fraction:.0%} of tasks"
        ),
        radius_m=radius_m,
        fraction_threshold=fraction,
        tasks=n,
        near_reference_endpoint=near,
        near_reference_endpoint_fraction=near / n,
        near_rotated_goal=sum(r["near_rotated_goal"] for r in rows.values()),
        near_start=sum(r["near_start"] for r in rows.values()),
        discriminative_tasks=sum(r["discriminative"] for r in rows.values()),
        adapter_ignores_goal=near / n >= fraction,
        median_progress_along_original=float(
            np.median([r["progress_along_original"] for r in rows.values()])
        ),
        median_progress_along_rotated=float(
            np.median([r["progress_along_rotated"] for r in rows.values()])
        ),
        median_displacement_heading_vs_original_deg=(
            float(np.median(headings)) if headings else None
        ),
        heading_tasks=len(headings),
        heading_toward_rotated_goal=toward,
    )
    if baseline_rows is None:
        result["sensitivity"] = None
        return result
    paired = sorted(set(rows) & set(baseline_rows))
    if not paired:
        raise ValueError("Ablation and baseline panels share no completed task")
    base_near = [t for t in paired if baseline_rows[t]["near_reference_endpoint"]]
    turns, turned = [], 0
    for t in paired:
        ablated = rows[t]["displacement_heading_vs_original_deg"]
        base = baseline_rows[t]["displacement_heading_vs_original_deg"]
        if ablated is not None and base is not None:
            turns.append(_wrap_deg(ablated - base))
            turned += (
                abs(_wrap_deg(turns[-1] - rows[t]["goal_rotation_deg"])) <= HEADING_TOLERANCE_DEG
            )
    base_fraction = len(base_near) / len(paired)
    shifts = [
        float(np.linalg.norm(np.subtract(rows[t]["final_xy"], baseline_rows[t]["final_xy"])))
        for t in paired
    ]
    unchanged = sum(s <= radius_m for s in shifts)
    result["sensitivity"] = dict(
        paired_tasks=len(paired),
        baseline_near_reference_endpoint=len(base_near),
        baseline_near_reference_endpoint_fraction=base_fraction,
        rule_attainable=base_fraction >= fraction,
        ablated_near_reference_given_baseline_near=(
            sum(rows[t]["near_reference_endpoint"] for t in base_near) if base_near else 0
        ),
        median_final_xy_shift_vs_baseline_m=float(np.median(shifts)),
        median_heading_turn_vs_baseline_deg=float(np.median(turns)) if turns else None,
        heading_turn_tasks=len(turns),
        heading_turned_with_goal=turned,
        final_xy_within_radius_of_baseline=unchanged,
        final_xy_within_radius_of_baseline_fraction=unchanged / len(paired),
        # Proposed paired analogue, NOT the registered rule: compare each rotated run with
        # its own unrotated run instead of with the reference endpoint.
        paired_rule_proposed=dict(
            rule=(
                f"(proposed, not registered) adapter ignores the goal if the rotated run ends "
                f"within {radius_m} m of the unrotated run's final XY (same checkpoint and seed) "
                f"in >= {fraction:.0%} of tasks"
            ),
            adapter_ignores_goal=unchanged / len(paired) >= fraction,
        ),
        note=(
            "A goal-blind adapter reproduces the baseline at the same seed. If "
            "rule_attainable is false, the registered rule cannot report 'ignores the goal' "
            "for any adapter, so a negative verdict is uninformative on its own."
        ),
    )
    return result
