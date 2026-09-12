"""Audit post-command sensor packets against pre-command physical recordings.

Isaac records physics first, then advances the reference and collects this
sensor packet. A final command reset can therefore invalidate only the sensor
packet while leaving the same-index physical/contact row valid.
"""

import numpy as np


def audit_sensor_alignment(payload, observations, *, reference_frames):
    """Return first-episode packet eligibility without deleting physical rows.

    Modern packets verify all recorded root and joint fields exactly. Legacy
    packets without those fields retain an explicitly weaker phase-only audit;
    no pose agreement is inferred from ray origins or obstacle identities.
    """
    fps = float(payload["fps"])
    times = np.asarray(payload["motion_time_s"], dtype=float)
    if (
        times.ndim != 1
        or not len(times)
        or len(observations) != len(times)
        or not np.isfinite(times).all()
        or np.any(times < 0)
        or not np.isfinite(fps)
        or fps <= 0
        or type(reference_frames) is not int
        or reference_frames < 2
    ):
        raise ValueError("requires aligned finite physical and sensor capture counts")
    reset_indices = np.flatnonzero(np.diff(times) <= 0) + 1
    first_end = int(reset_indices[0]) if len(reset_indices) else len(times)
    dt, final_phase = 1 / fps, (reference_frames - 1) / fps
    if not np.allclose(np.diff(times[:first_end]), dt, atol=1e-8, rtol=0):
        raise ValueError("first physical episode does not follow its recorded reference clock")
    packets = []
    for index, packet in enumerate(observations):
        phase = float(packet["time_s"])
        reasons = []
        if index >= first_end:
            reasons.append("after_physical_first_episode")
        if not np.isfinite(phase) or not np.isclose(phase, times[index] + dt, atol=1e-8, rtol=0):
            reasons.append("phase_not_next_reference_tick")
        if not np.isfinite(phase) or not 0 <= phase <= final_phase + 1e-8:
            reasons.append("reference_horizon_exceeded")
        elapsed = packet.get("capture_elapsed_s")
        if elapsed is not None and (
            not np.isfinite(elapsed) or not np.isclose(elapsed, index / fps, atol=1e-8, rtol=0)
        ):
            reasons.append("elapsed_clock_mismatch")
        state = packet.get("state", {})
        checked, mismatched = [], []
        for name, observed in (
            ("root_pos_w", packet.get("root_pos_w")),
            ("root_quat_w", packet.get("root_quat_w")),
            ("dof_pos", state.get("dof_pos")),
            ("dof_vel", state.get("dof_vel")),
            ("projected_gravity_b", state.get("projected_gravity_b")),
        ):
            if observed is None or name not in payload:
                continue
            checked.append(name)
            actual = np.asarray(observed)
            expected = np.asarray(payload[name][index])
            if (
                actual.shape != expected.shape
                or not np.isfinite(actual).all()
                or not np.array_equal(actual, expected)
            ):
                mismatched.append(name)
        if mismatched:
            reasons.append("robot_state_mismatch")
        full_pose = {"root_pos_w", "root_quat_w", "dof_pos"}.issubset(checked)
        confidence = (
            "exact_recorded_pose_and_phase"
            if full_pose
            else "partial_recorded_state_and_phase" if checked else "phase_only_pose_unavailable"
        )
        packets.append(
            {
                "index": index,
                "eligible": not reasons,
                "confidence": confidence,
                "checked_state_fields": checked,
                "mismatched_state_fields": mismatched,
                "reasons": reasons,
            }
        )
    return {
        "schema": "motion2scene_sensor_physics_alignment_v1",
        "physical_capture_frames": len(times),
        "physical_first_episode_frames": first_end,
        "reference_frames": reference_frames,
        "expected_sensor_phase_offset_s": dt,
        "packet_eligible": [row["eligible"] for row in packets],
        "eligible_first_episode_packets": sum(row["eligible"] for row in packets),
        "pose_unavailable_packets": sum(
            row["confidence"] == "phase_only_pose_unavailable" for row in packets
        ),
        "packets": packets,
        "scope": (
            "sensor eligibility only; every original physical row is retained; "
            "a command reset after recording need not appear in physical reset_count"
        ),
    }
