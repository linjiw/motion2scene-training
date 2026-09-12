"""Opt-in finite-horizon motion schedules and evidence gates, without simulation.

An experimental request is never an online registry. Each action identifies a
whole entry/maintenance/return schedule; independently mixing tested ticks is
not permitted. Current K5 registries and phase policies do not use this module.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

REQUEST_SCHEMA = "motion2scene_timed_option_request_v1"
REGISTRY_SCHEMA = "motion2scene_timed_option_registry_v1"
EVIDENCE_SCHEMA = "motion2scene_timed_option_evidence_v1"
FPS = 50
HEIGHT_MEASUREMENT = "executed_outer_collision_height_above_support_plane"
REQUIRED_CHECKS = (
    "matched_approach_prefix",
    "all_contacts_audited",
    "whole_episode_stable",
    "full_horizon",
    "no_reset",
    "no_wrap",
    "source_bank_bound",
    "controller_bound",
    "sensor_clock_aligned",
    "reference_guard_verified",
    "recovery_verified",
    "no_refusals",
)


def definition_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _artifact(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("SHA-bound artifact path required")
    if not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute():
        raise ValueError("artifact path must be absolute")
    if (
        not isinstance(ref["sha256"], str)
        or re.fullmatch(r"sha256:[a-f0-9]{64}", ref["sha256"]) is None
    ):
        raise ValueError("canonical SHA-256 required")


def checked_artifact(ref):
    _artifact(ref)
    path = Path(ref["path"])
    if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def _tick(value):
    return type(value) is int and value >= 0


def seconds_to_tick(seconds):
    value = float(seconds) * FPS
    if not np.isfinite(value) or value < 0 or abs(value - round(value)) > 1e-8:
        raise ValueError("transition times must be exact nonnegative 50 Hz ticks")
    return int(round(value))


@dataclass(frozen=True)
class TimedOptionBank:
    definition: dict
    online_verified: bool

    @property
    def request(self):
        return self.definition["request"] if self.online_verified else self.definition

    @property
    def option_ids(self):
        return ("neutral", *(option["option_id"] for option in self.request["options"]))

    @property
    def frame_count(self):
        return self.request["expected_loaded_frames"]


def validate_request(request):
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("split") != "development"
        or request.get("max_entries_per_episode") != 1
        or request.get("reference_fps") != FPS
        or not _tick(request.get("expected_loaded_frames"))
        or request["expected_loaded_frames"] < 4
        or not isinstance(request.get("request_id"), str)
        or not request["request_id"]
        or request.get("height_measurement") != HEIGHT_MEASUREMENT
        or not np.isfinite(request.get("support_plane_z_m", np.nan))
    ):
        raise ValueError(
            "explicit development request, 50 Hz finite horizon and one entry required"
        )
    _artifact(request["controller"])
    references, options = request["references"], request["options"]
    if (
        not isinstance(references, list)
        or len(references) < 2
        or not isinstance(options, list)
        or not options
    ):
        raise ValueError("neutral and at least one alternate/schedule required")
    ids = [reference["reference_id"] for reference in references]
    if ids[0] != "neutral" or len(ids) != len(set(ids)):
        raise ValueError("distinct reference names with neutral first required")
    for reference in references:
        _artifact(reference["motion"])
        if reference["expected_loaded_frames"] != request["expected_loaded_frames"]:
            raise ValueError("options require equal complete reference horizons")
        if reference.get("construction") not in (
            "generated",
            "authored_local_crouch",
            "authored_prior_projection_splice",
        ):
            raise ValueError("reference construction ancestry must be explicit")
        if reference["construction"] == "authored_local_crouch":
            _artifact(reference["parent_motion"])
            if reference["parent_motion"] != references[0]["motion"]:
                raise ValueError("authored profiles must bind the stated neutral parent")
        if reference["construction"] == "authored_prior_projection_splice":
            for key in (
                "parent_motion",
                "prior_motion",
                "derivation_registration",
                "derivation_result",
                "boundary_diagnostic",
            ):
                _artifact(reference.get(key))
            if reference["parent_motion"] != references[0]["motion"]:
                raise ValueError("projected prior splice must bind the stated neutral parent")
    option_ids = [option["option_id"] for option in options]
    if len(option_ids) != len(set(option_ids)) or "neutral" in option_ids:
        raise ValueError("each complete schedule needs a distinct non-neutral option identifier")
    for option in options:
        entry, ret, recovered = (
            option[k] for k in ("entry_tick", "return_tick", "recovery_end_tick")
        )
        maintenance = option["maintenance"]
        start, end = maintenance["start_tick"], maintenance["end_tick"]
        if (
            option["reference_id"] not in ids[1:]
            or not all(_tick(v) for v in (entry, ret, recovered, start, end))
            or not 1 <= entry <= start <= end < ret < recovered < request["expected_loaded_frames"]
            or not isinstance(option.get("profile_label"), str)
            or not option["profile_label"]
        ):
            raise ValueError(
                "entry, maintenance, exact return and recovery must fit one reference pass"
            )
        maximum = maintenance["maximum_body_height_m"]
        if maximum is not None and (not np.isfinite(maximum) or maximum <= 0):
            raise ValueError("maintenance height bound must be positive or explicitly absent")
    # Copy through JSON so mutation of the caller's request cannot alter this bank.
    return TimedOptionBank(json.loads(json.dumps(request, allow_nan=False)), False)


def assert_loaded_bank(bank, reference_ids, frame_counts, fps):
    """Call after actual loading; anticipated source resampling is not evidence."""
    expected = [reference["reference_id"] for reference in bank.request["references"]]
    if list(reference_ids) != expected or fps != FPS or len(frame_counts) != len(expected):
        raise ValueError("actual loaded reference identity/rate differs from the request")
    if any(not _tick(count) or count != bank.frame_count for count in frame_counts):
        raise ValueError("actual loaded bank is not the registered equal-duration bank")


def guard_before_reference_advance(current_tick, frame_count):
    if not _tick(current_tick) or not _tick(frame_count) or frame_count < 2:
        raise ValueError("finite reference clock required")
    if current_tick + 1 >= frame_count:
        raise RuntimeError(
            "reference exhausted: abort capture before wrap; this is not a protective stop"
        )


def validate_evidence(request, evidence):
    """Check a SHA-bound physical audit's full timeline and schedule predicates.

    Contact, pose, prefix and geometry audits are upstream measurements. This
    validator checks their binding and completeness; it does not rerun physics.
    """
    bank = validate_request(request)
    if (
        evidence.get("schema") != EVIDENCE_SCHEMA
        or evidence.get("request_digest") != definition_digest(request)
        or evidence.get("controller") != request["controller"]
        or evidence.get("reference_motions") != [ref["motion"] for ref in request["references"]]
        or evidence.get("option_id") not in bank.option_ids
        or evidence.get("height_measurement") != request["height_measurement"]
        or evidence.get("support_plane_z_m") != request["support_plane_z_m"]
    ):
        raise ValueError("physical evidence does not bind this request/controller/reference bank")
    for key in (
        "trajectory",
        "interface",
        "physics_contacts",
        "clock_audit",
        "prefix_audit",
        "geometry_audit",
    ):
        _artifact(evidence["artifacts"][key])
    checks = evidence["checks"]
    if any(checks.get(key) is not True for key in REQUIRED_CHECKS):
        raise ValueError("online qualification requires every whole-episode physical audit")
    assert_loaded_bank(
        bank, evidence["loaded_reference_ids"], evidence["loaded_reference_frames"], evidence["fps"]
    )
    ticks = np.asarray(evidence["command_ticks"])
    if ticks.dtype.kind not in "iu" or not np.array_equal(ticks, np.arange(1, bank.frame_count)):
        raise ValueError("qualification requires every aligned pre-wrap command packet")
    if evidence["recorded_physics_steps"] != 4 * (bank.frame_count - 1):
        raise ValueError("whole-episode 200 Hz physical steps are incomplete")
    active = np.asarray(evidence["active_option_ids"])
    if active.shape != ticks.shape:
        raise ValueError("every command tick needs its actual active option")
    option_id = evidence["option_id"]
    expected = np.full(len(ticks), "neutral", dtype=object)
    expected_switches = []
    if option_id != "neutral":
        option = next(value for value in request["options"] if value["option_id"] == option_id)
        expected[(ticks >= option["entry_tick"]) & (ticks < option["return_tick"])] = option_id
        expected_switches = [
            (option["entry_tick"], "neutral", option_id),
            (option["return_tick"], option_id, "neutral"),
        ]
        if evidence["matched_prefix_frames"] < option["entry_tick"]:
            raise ValueError("paired physical approach does not reach the requested entry state")
        height = np.asarray(evidence["executed_body_height_m"], dtype=float)
        if height.shape != ticks.shape or not np.isfinite(height).all() or (height <= 0).any():
            raise ValueError("full executed collision-envelope height trace required")
        maintenance = option["maintenance"]
        window = (ticks >= maintenance["start_tick"]) & (ticks <= maintenance["end_tick"])
        maximum = maintenance["maximum_body_height_m"]
        if maximum is not None and np.any(height[window] > maximum):
            raise ValueError("executed maintenance height exceeds the preregistered bound")
    if not np.array_equal(active, expected):
        raise ValueError(
            "entry, maintenance or recovery timeline differs from the complete schedule"
        )
    switches = evidence["switches"]
    actual = [(switch["tick"], switch["from"], switch["to"]) for switch in switches]
    if actual != expected_switches:
        raise ValueError("requires exact paired entry/return ticks and no repeat or k-to-j switch")
    for switch in switches:
        joint, root = switch["joint_jump_rad"], switch["root_jump_m"]
        if (
            not np.isfinite([joint, root]).all()
            or not 0 <= joint <= 0.05
            or not 0 <= root <= 0.01
            or any(
                switch.get(key) is not True
                for key in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
            )
        ):
            raise ValueError("physical switch violated reference-jump or unchanged-state guards")
    return {
        "option_id": option_id,
        "qualified": True,
        "frames": len(ticks),
        "physical_steps": evidence["recorded_physics_steps"],
    }


def make_verified_registry(request, evidence_artifacts):
    """Promote only complete SHA-checked physical evidence, including neutral."""
    bank = validate_request(request)
    receipts = {}
    for ref in evidence_artifacts:
        evidence = json.loads(checked_artifact(ref).read_text())
        receipt = validate_evidence(request, evidence)
        if receipt["option_id"] in receipts:
            raise ValueError("duplicate qualification evidence")
        for bound in evidence["artifacts"].values():
            checked_artifact(bound)
        receipts[receipt["option_id"]] = ref
    if set(receipts) != set(bank.option_ids):
        raise ValueError("neutral and every online schedule require complete physical evidence")
    for ref in [request["controller"], *(value["motion"] for value in request["references"])]:
        checked_artifact(ref)
    for reference in request["references"]:
        if reference["construction"] == "authored_prior_projection_splice":
            for key in (
                "parent_motion",
                "prior_motion",
                "derivation_registration",
                "derivation_result",
                "boundary_diagnostic",
            ):
                checked_artifact(reference[key])
    return {
        "schema": REGISTRY_SCHEMA,
        "request": request,
        "evidence": receipts,
        "scope": (
            "finite qualified reference schedules in audited approach contexts; "
            "no arbitrary duration/reentry guarantee"
        ),
    }


def load_verified_registry(path, expected_sha256):
    value = json.loads(
        checked_artifact({"path": str(Path(path).resolve()), "sha256": expected_sha256}).read_text()
    )
    if value.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("experimental qualification requests are not verified online registries")
    rebuilt = make_verified_registry(value["request"], list(value["evidence"].values()))
    if rebuilt != value:
        raise ValueError("verified registry differs from its complete evidence")
    return TimedOptionBank(rebuilt, True)


@dataclass(frozen=True)
class TimedOptionState:
    active: str = "neutral"
    entries: int = 0
    last_tick: int = 0


def legal_timed_actions(bank, state, tick, joint_jumps, root_jumps, *, qualification_only=False):
    if not bank.online_verified and not qualification_only:
        raise ValueError("experimental request may only drive explicit forced qualification")
    if not _tick(tick) or tick <= state.last_tick or tick >= bank.frame_count:
        raise ValueError("non-repeated, in-horizon command tick required; clip wrap is forbidden")
    ids = bank.option_ids
    if (
        state.active not in ids
        or state.entries not in (0, 1)
        or (state.active != "neutral" and state.entries != 1)
    ):
        raise ValueError("invalid one-entry option state")
    joint, root = np.asarray(joint_jumps, dtype=float), np.asarray(root_jumps, dtype=float)
    if (
        joint.shape != (len(ids),)
        or root.shape != joint.shape
        or not np.isfinite(joint).all()
        or not np.isfinite(root).all()
        or (joint < 0).any()
        or (root < 0).any()
    ):
        raise ValueError("finite current-state reference jumps required for every schedule")
    mask = np.zeros(len(ids), dtype=bool)
    mask[ids.index(state.active)] = True
    mandatory = None
    for index, option in enumerate(bank.request["options"], 1):
        if state.active == "neutral" and state.entries == 0 and tick == option["entry_tick"]:
            mask[index] = joint[index] <= 0.05 and root[index] <= 0.01
        if state.active == option["option_id"]:
            if tick > option["return_tick"]:
                raise RuntimeError(
                    "qualified return was missed; abort and retain incomplete episode"
                )
            if tick == option["return_tick"]:
                mandatory = 0
                mask[0] = joint[0] <= 0.05 and root[0] <= 0.01
    return mask, mandatory


def apply_timed_request(
    bank, state, tick, requested, joint_jumps, root_jumps, *, qualification_only=False
):
    mask, mandatory = legal_timed_actions(
        bank, state, tick, joint_jumps, root_jumps, qualification_only=qualification_only
    )
    if requested not in bank.option_ids:
        raise ValueError("unknown requested schedule")
    index = bank.option_ids.index(requested)
    allowed = bool(mask[index] and (mandatory is None or index == mandatory))
    entries = state.entries + int(allowed and state.active == "neutral" and requested != "neutral")
    next_state = TimedOptionState(requested if allowed else state.active, entries, tick)
    return next_state, {
        "requested": requested,
        "allowed": allowed,
        "mandatory_return": mandatory is not None,
        "reason": None if allowed else "unqualified_tick_transition_repeat_or_reference_jump",
    }
