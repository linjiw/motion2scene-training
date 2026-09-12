"""Executable options supported by the installed phase-aligned SONIC interface.

An option is a request, not a feasibility certificate. Qualification requires an
actual approach/entry/maintenance/return rollout with the frozen tracker.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MotionOption:
    option_id: str
    reference: str
    entry_time_s: float = 0.3
    return_request_time_s: float = 3.3

    def __post_init__(self):
        if not self.option_id or not self.reference:
            raise ValueError("option and reference identifiers are required")
        tick = self.entry_time_s * 50
        if (
            not np.isfinite(tick)
            or not 0.2 <= self.entry_time_s <= 0.4
            or abs(tick - round(tick)) > 1e-8
        ):
            raise ValueError("entry requires a legal 50 Hz tick from 0.20 to 0.40 seconds")
        if self.return_request_time_s != 3.3:
            raise ValueError("installed runtime only requests return at 3.30 seconds")

    @property
    def action(self):
        return int(self.reference != "neutral")

    def as_dict(self):
        return {**asdict(self), "action": self.action}


def qualification_predicates(option, interface, passage):
    """Separate successful physical passage from execution of the requested option."""
    switches = interface["switches"]
    flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
    entry = [s for s in switches if s["to"] == 1]
    returns = [s for s in switches if s["to"] == 0]
    legal = all(
        all(s[k] for k in flags)
        and s["joint_reference_jump_rad"] <= 0.05
        and s["root_reference_jump_m"] <= 0.01
        and (0.2 <= s["time_s"] <= 0.4 if s["to"] else 3.3 <= s["time_s"] <= 3.5)
        for s in switches
    )
    observations = interface["observations"]
    no_refusal = all(
        not o["transition"]["attempted"] or o["transition"]["allowed"] for o in observations
    )
    state_intact = all(all(o["transition"][k] for k in flags) for o in observations)
    correct_entry = (
        len(entry) == 1 and abs(entry[0]["time_s"] - option.entry_time_s) < 1e-8
        if option.action
        else not switches
    )
    return_complete = len(returns) == 1 if option.action else not switches
    return {
        "requested_entry_executed": bool(correct_entry),
        "return_completed": bool(return_complete),
        "legal_transitions": bool(legal and state_intact and observations),
        "no_transition_refusal": bool(no_refusal),
        "physical_passage": bool(passage["pass"]),
        "no_reset": passage["reset_count"] == 0,
        "no_fall": not passage["fall_observed"],
    }


def mechanical_work(torque_nm, velocity_rad_s, physics_dt_s):
    """Rectangular quadrature of sampled actuator work; not battery energy."""
    torque = np.asarray(torque_nm, dtype=np.float64)
    velocity = np.asarray(velocity_rad_s, dtype=np.float64)
    if (
        torque.ndim != 2
        or torque.shape != velocity.shape
        or 0 in torque.shape
        or not np.isfinite(torque).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(physics_dt_s)
        or physics_dt_s <= 0
    ):
        raise ValueError("finite synchronized nonempty actuator samples and positive dt required")
    power = torque * velocity
    return {
        "positive_mechanical_work_j": float(np.maximum(power, 0).sum() * physics_dt_s),
        "absolute_mechanical_work_j": float(np.abs(power).sum() * physics_dt_s),
        "net_mechanical_work_j": float(power.sum() * physics_dt_s),
        "samples": len(torque),
        "measurement": "applied actuator effort times post-step joint velocity; rectangular quadrature",
    }
