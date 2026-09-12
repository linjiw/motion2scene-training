"""Pruned scene queries must preserve exact primitive eligibility decisions."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_timed_scene_screen import beam_clearance

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance


def test_pruned_clearance_matches_full_exact_query_with_world_offsets():
    rng = np.random.default_rng(12231)
    starts = rng.uniform([-4, -4, 0.1], [4, 4, 2], (37, 11, 3))
    ends = starts + rng.normal(0, 0.17, starts.shape)
    radii = rng.uniform(0.01, 0.13, (11,))
    beam = dict(
        center_xy_m=[0.27, -0.13],
        underside_m=1.29,
        thickness_m=0.1,
        length_m=0.75,
        width_m=1.2,
        yaw_rad=0.21,
    )
    for offset in ([0, 0, 0, 0], [0.02, -0.02, 0.005, 0.02], [-0.02, 0.02, -0.005, -0.02]):
        yaw = beam["yaw_rad"] + offset[3]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center = np.r_[beam["center_xy_m"], beam["underside_m"] + 0.05] + offset[:3]
        half = np.array([0.75, 1.2, 0.1]) / 2
        expected = capsule_box_clearance(
            (starts - center) @ rotation, (ends - center) @ rotation, radii, -half, half
        ).min(initial=0.2)
        actual = beam_clearance((starts, ends, radii, None), beam, np.asarray(offset))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_far_capsules_return_explicit_cap():
    starts = np.full((3, 2, 3), 10.0)
    beam = dict(
        center_xy_m=[0, 0], underside_m=1.3, thickness_m=0.1, length_m=0.1, width_m=1.2, yaw_rad=0.0
    )
    assert (
        beam_clearance((starts, starts + 1, np.array([0.03, 0.08]), None), beam, np.zeros(4)) == 0.2
    )
