"""Sensor-only geometry, temporal registration, occlusion and timing regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
    capture_ray_fan,
    rays_from_overhang_packet,
    transition_timing_eligibility,
)


def ray_to(point, normal=None, origin=(0, 0, 0)):
    displacement = np.asarray(point) - origin
    distance = np.linalg.norm(displacement)
    return SensorRay(origin, displacement / distance, 5, distance, normal)


def history(max_age_s=2):
    return FloorCeilingHistory(
        HistoryGrid(lower_m=(-2, -2, -2), upper_m=(3, 3, 3), resolution_m=0.5, max_age_s=max_age_s)
    )


def snapshot(mapper, now=0, root=(0, 0, 0), quat=(1, 0, 0, 0)):
    return mapper.snapshot(root, quat, now)


def test_floor_and_underside_survive_in_same_cell_without_filling_vertical_interval():
    mapper = history()
    mapper.push([ray_to((1.1, 0.1, -1), (0, 0, 1)), ray_to((1.1, 0.1, 1), (0, 0, -1))], 0)
    result = snapshot(mapper)
    i, j = 6, 4
    assert result["floor_observed"][i, j] and result["ceiling_observed"][i, j]
    assert result["floor_height_m"][i, j] == -1
    assert result["ceiling_height_m"][i, j] == 1
    assert result["vertical_gap_m"][i, j] == 2
    # Two measured boundary samples do not observe the intervening vertical column.
    assert result["unknown"][i, j, 4]
    assert result["floor_unknown"][0, 0] and result["ceiling_unknown"][0, 0]


def test_front_face_and_missing_normals_do_not_invent_a_ceiling_or_floor():
    mapper = history()
    mapper.push([ray_to((1.1, 0.1, 1)), ray_to((1.1, 0.1, -1), (-1, 0, 0))], 0)
    result = snapshot(mapper)
    assert result["occupied"].sum() == 2
    assert not result["floor_observed"].any()
    assert not result["ceiling_observed"].any()


def test_world_registered_memory_moves_with_root_yaw_and_translation():
    mapper = history()
    mapper.push([ray_to((1.1, 0.1, 1), (0, 0, -1))], 0)
    before = snapshot(mapper)
    assert before["ceiling_observed"][6, 4]
    # World point minus root = (1.1, .1, .5); at +90 yaw local point=(.1,-1.1,.5).
    after = snapshot(mapper, now=1, root=(0, 0, 0.5), quat=(2**-0.5, 0, 0, 2**-0.5))
    assert after["ceiling_observed"][4, 1]
    assert after["ceiling_height_m"][4, 1] == pytest.approx(0.5)
    assert after["ceiling_capture_time_s"][4, 1] == 0
    assert not after["ceiling_observed"][6, 4]


def test_expiry_causality_and_episode_reset():
    mapper = history(max_age_s=1)
    mapper.push([ray_to((1.1, 0.1, 1))], 0.1, delivered_time_s=0.3)
    assert snapshot(mapper, 0.3)["occupied"].any()
    with pytest.raises(ValueError, match="capture times"):
        mapper.push([], 0.4, delivered_time_s=0.3)
    with pytest.raises(ValueError, match="capture times"):
        mapper.push([], 0.1, delivered_time_s=0.4)
    expired = snapshot(mapper, 1.2)
    assert expired["unknown"].all() and expired["retained_frames"] == 0
    with pytest.raises(ValueError, match="monotonic"):
        snapshot(mapper, 1.1)
    mapper.reset()
    mapper.push([], 0)
    assert snapshot(mapper)["unknown"].all()


def test_already_expired_delivered_packet_is_not_reintroduced():
    mapper = history(max_age_s=1)
    mapper.push([ray_to((1, 0, 0))], 0, delivered_time_s=2)
    assert snapshot(mapper, 2)["unknown"].all()


def test_no_return_observes_only_sampled_ray_and_stops_at_occlusion():
    mapper = history()
    mapper.push([SensorRay((0, 0, 0), (1, 0, 0), 3, 1.1)], 0)
    result = snapshot(mapper)
    assert result["free_sampled"][4, 4, 4]
    assert result["occupied"][6, 4, 4]
    assert result["unknown"][8, 4, 4]  # Behind the first return.
    assert result["unknown"][4, 7, 4]  # Away from the ray.
    mapper.push([SensorRay((0, 0, 0), (1, 0, 0), 3)], 0.2)
    retained = snapshot(mapper, 0.2)
    assert retained["occupied"][6, 4, 4] and not retained["free_sampled"][6, 4, 4]
    assert retained["free_sampled"][8, 4, 4]
    assert np.all(retained["occupied"] | retained["free_sampled"] | retained["unknown"])


def test_sensor_ray_and_grid_are_immutable_and_validated():
    origin = [0, 0, 0]
    ray = SensorRay(origin, [1, 0, 0], 3)
    origin[0] = 99
    assert ray.origin_w == (0, 0, 0)
    for kwargs in ({"direction_w": (2, 0, 0)}, {"hit_distance_m": 4}, {"range_m": -1}):
        params = {"origin_w": (0, 0, 0), "direction_w": (1, 0, 0), "range_m": 3}
        params.update(kwargs)
        with pytest.raises(ValueError):
            SensorRay(**params)
    with pytest.raises(ValueError, match="normal"):
        SensorRay((0, 0, 0), (1, 0, 0), 3, hit_normal_w=(0, 0, 1))
    with pytest.raises(ValueError, match="grid"):
        HistoryGrid(max_age_s=-1)
    with pytest.raises(ValueError, match="integer multiples"):
        HistoryGrid(resolution_m=0.7)


def test_adapter_ignores_identity_and_does_not_treat_uncaptured_lower_rays_as_free():
    packet = {
        "origin": [0, 0, 0],
        "rays": [
            {
                "direction": [1, 0, 0],
                "hit": {"distance": 1, "position": [1, 0, 0], "path": "arbitrary"},
                "upper_candidate": True,
                "lower_rays": [],
            }
        ],
    }
    rays = rays_from_overhang_packet(packet)
    assert len(rays) == 1 and rays[0].hit_normal_w is None
    packet["rays"][0]["hit"]["path"] = "different"
    assert rays_from_overhang_packet(packet) == rays
    packet["rays"][0]["hit"]["position"] = [2, 0, 0]
    with pytest.raises(ValueError, match="disagree"):
        rays_from_overhang_packet(packet)


class Query:
    def __init__(self, malformed=False):
        self.origins = []
        self.malformed = malformed

    def raycast_all(self, origin, direction, distance, callback):
        self.origins.append(origin)
        if self.malformed:
            callback(object())
            return
        for path, hit_distance in (
            ("/World/envs/env_0/Robot/torso", 0.1),
            ("/World/far", 3),
            ("/World/unknown_object", 1),
        ):
            callback(
                SimpleNamespace(
                    collision=path,
                    position=np.asarray(origin) + np.asarray(direction) * hit_distance,
                    distance=hit_distance,
                    normal=-np.asarray(direction),
                )
            )


def test_body_mounted_rays_use_full_pose_and_nearest_occluder():
    query = Query()
    rays = capture_ray_fan(
        (0, 0, 1),
        (2**-0.5, 0, 0, 2**-0.5),
        query,
        elevations_deg=(0, 30),
        azimuths_deg=(0,),
    )
    assert len(rays) == 2
    assert all(np.allclose(origin, (0, 0.2, 1.4)) for origin in query.origins)
    assert np.allclose(rays[0].direction_w, (0, 1, 0))
    assert all(ray.hit_distance_m == 1 for ray in rays)
    assert all(ray.hit_normal_w is not None for ray in rays)
    with pytest.raises(RuntimeError, match="sensor callback"):
        capture_ray_fan((0, 0, 0), (1, 0, 0, 0), Query(malformed=True))


def timing(seen=0.1, arrival=1, **kwargs):
    params = {
        "sensing_latency_s": 0.1,
        "transition_duration_s": 0.5,
        "margin_s": 0.1,
        "legal_entry_times_s": [0.2, 0.3, 0.4],
    }
    params.update(kwargs)
    return transition_timing_eligibility(seen, arrival, **params)


def test_visibility_requires_enough_time_and_a_future_legal_entry():
    assert timing()["eligible"]
    assert timing(arrival=0.8)["slack_s"] == pytest.approx(0)
    assert not timing(arrival=0.79)["eligible"]
    assert timing(seen=None)["reason"] == "unobserved"
    assert timing(seen=0.31)["reason"] == "no_legal_entry_after_observation"
    assert timing(decision_time_s=0.31)["entry_time_s"] == 0.4
    assert not timing(decision_time_s=0.5)["eligible"]
    with pytest.raises(ValueError, match="timing inputs"):
        timing(legal_entry_times_s=[0.3, 0.2])
    with pytest.raises(ValueError, match="timing inputs"):
        timing(transition_duration_s=float("nan"))
