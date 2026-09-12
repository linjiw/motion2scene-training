"""Fail-closed record schema for Motion2Scene-DB.

The schema deliberately permits partial E0 records while refusing to infer missing evidence.
It also prevents a compatible scene from being labeled critical without a disclosed matched
alternative set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FrozenModel(BaseModel):
    """Immutable base model so validated records cannot drift in memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceLevel(StrEnum):
    """Evidence tiers remain distinct; higher tiers never imply missing lower records."""

    MOTION = "M"
    KINEMATICS = "Q0"
    PENETRATION = "Q1"
    SUPPORT_DYNAMICS = "Q2"
    TRACKING_SINGLE_SEED = "Q3"
    TRACKING_MULTI_SEED = "Q4"
    EXACT_GEOMETRY = "G"
    OBSTACLE_ABSENT_TRACKING = "T"
    OBSTACLE_PRESENT_PHYSICS = "P"


class EvidenceStatus(StrEnum):
    NOT_MEASURED = "not_measured"
    PASSED = "passed"
    FAILED = "failed"
    UNEVALUABLE = "unevaluable"


class MotionSourceType(StrEnum):
    GENERATED = "generated"
    RETARGETED = "retargeted"
    REPAIRED = "repaired"
    CAPTURED = "captured"


class SemanticStatus(StrEnum):
    """Ordered functional evidence; values are names, not interchangeable booleans."""

    NOT_MEASURED = "not_measured"
    ABSENT = "absent"
    ELICITED = "elicited"
    LOCALIZED = "localized"
    ROUTE_ALIGNED = "route_aligned"
    CONTROLLER_RETAINED = "controller_retained"


class RouteValidityClass(StrEnum):
    VALID_STRAIGHT = "valid_straight"
    VALID_GENTLE_LEFT = "valid_gentle_left"
    VALID_GENTLE_RIGHT = "valid_gentle_right"
    OVER_TURN = "over_turn"
    LOOPING_REVERSAL = "looping_reversal"
    INSUFFICIENT_PROGRESS = "insufficient_progress"


class LadderFamily(StrEnum):
    DUCK = "duck"
    ARM_TUCK = "arm_tuck"
    SHOULDER_TURN = "shoulder_turn"
    STEP_OVER = "step_over"


class MonotonicityStatus(StrEnum):
    NOT_MEASURED = "not_measured"
    ORDERED = "ordered"
    NOT_ORDERED = "not_ordered"


class Q4Classification(StrEnum):
    NOT_MEASURED = "not_measured"
    FAILED = "failed"
    ROBUST = "robust_2_of_3"
    STRICT = "strict_3_of_3"


class FeasibilityOutcome(StrEnum):
    NOT_MEASURED = "not_measured"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNEVALUABLE = "unevaluable"


class InterventionCondition(StrEnum):
    OBSTACLE_ABSENT = "obstacle_absent"
    OBSTACLE_PRESENT = "obstacle_present"


class InterventionMotionRole(StrEnum):
    TARGET = "target"
    WEAKER = "weaker"


class SelectionRule(StrEnum):
    FIXED_ENERGY = "fixed_energy"
    ORDINAL_MINIMAL_FEASIBLE = "ordinal_minimal_feasible"


class ArtifactRef(FrozenModel):
    path: NonEmpty
    sha256: Sha256
    media_type: NonEmpty | None = None
    size_bytes: int | None = Field(default=None, ge=0)


class EvidenceRecord(FrozenModel):
    level: EvidenceLevel
    status: EvidenceStatus
    method: NonEmpty
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    failure_reasons: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def require_failure_reason(self) -> EvidenceRecord:
        if self.status in {EvidenceStatus.FAILED, EvidenceStatus.UNEVALUABLE}:
            if not self.failure_reasons:
                raise ValueError("failed or unevaluable evidence requires a reason")
        elif self.failure_reasons:
            raise ValueError("failure reasons are only valid for failed/unevaluable evidence")
        return self


class MotionIdentity(FrozenModel):
    motion_id: NonEmpty
    source: NonEmpty
    robot: NonEmpty
    qpos: ArtifactRef
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=2)
    prompt: NonEmpty | None = None
    generator_model: NonEmpty | None = None
    generator_seed: int | None = None
    base_motion_id: NonEmpty
    prompt_design_version: NonEmpty | None = None
    prompt_cell_id: NonEmpty | None = None
    generation_seed: int | None = None
    matched_seed_group_id: NonEmpty | None = None
    source_model: NonEmpty | None = None
    source_checkpoint: NonEmpty | None = None
    source_type: MotionSourceType | None = None

    @model_validator(mode="after")
    def reject_conflicting_seed_aliases(self) -> MotionIdentity:
        if (
            self.generator_seed is not None
            and self.generation_seed is not None
            and self.generator_seed != self.generation_seed
        ):
            raise ValueError("legacy and versioned generation seeds disagree")
        return self


class MotionDescriptor(FrozenModel):
    skill: NonEmpty
    route_style: NonEmpty
    speed_style: NonEmpty
    intensity: float | None = None
    event_progress: float | None = Field(default=None, ge=0.0, le=1.0)
    event_phase: NonEmpty | None = None
    lead_side: Annotated[str, StringConstraints(pattern=r"^(left|right|none|unknown)$")] = "unknown"


class EventInterval(FrozenModel):
    onset_progress: float = Field(ge=0.0, le=1.0)
    recovery_progress: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_positive_duration(self) -> EventInterval:
        if self.onset_progress >= self.recovery_progress:
            raise ValueError("event onset must precede recovery")
        return self


class SemanticAssessment(FrozenModel):
    semantic_predicate_version: NonEmpty
    semantic_status: SemanticStatus
    continuous_metrics: dict[str, float] = Field(default_factory=dict)
    reference_event_interval: EventInterval | None = None
    achieved_event_interval: EventInterval | None = None
    matched_walk_motion_id: NonEmpty | None = None
    notes: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def require_evidence_for_advanced_statuses(self) -> SemanticAssessment:
        if (
            self.semantic_status
            in {
                SemanticStatus.LOCALIZED,
                SemanticStatus.ROUTE_ALIGNED,
                SemanticStatus.CONTROLLER_RETAINED,
            }
            and self.reference_event_interval is None
        ):
            raise ValueError("localized-or-higher semantics require a reference event interval")
        if (
            self.semantic_status == SemanticStatus.CONTROLLER_RETAINED
            and self.achieved_event_interval is None
        ):
            raise ValueError("controller-retained semantics require an achieved event interval")
        return self


class RouteAssessment(FrozenModel):
    route_predicate_version: NonEmpty
    net_displacement_m: float = Field(ge=0.0)
    path_length_m: float = Field(ge=0.0)
    net_to_path_ratio: float = Field(ge=0.0, le=1.0)
    signed_heading_change_rad: float
    total_absolute_curvature_rad: float = Field(ge=0.0)
    monotonic_progress_fraction: float = Field(ge=0.0, le=1.0)
    maximum_lateral_departure_m: float = Field(ge=0.0)
    self_intersection: bool
    reversal: bool
    validity_class: RouteValidityClass


class PhysicsSeedOutcome(FrozenModel):
    physics_seed: int
    tracker_survived: bool
    semantic_retained: bool | None = None
    fell: bool
    route_progress: float | None = Field(default=None, ge=0.0)
    metrics: dict[str, float] = Field(default_factory=dict)


class QualificationRecord(FrozenModel):
    q0_status: EvidenceStatus = EvidenceStatus.NOT_MEASURED
    q1_status: EvidenceStatus = EvidenceStatus.NOT_MEASURED
    q2_status: EvidenceStatus = EvidenceStatus.NOT_MEASURED
    q3_status: EvidenceStatus = EvidenceStatus.NOT_MEASURED
    q4_status: EvidenceStatus = EvidenceStatus.NOT_MEASURED
    q4_classification: Q4Classification = Q4Classification.NOT_MEASURED
    per_joint_limit_slack_rad: dict[str, float] = Field(default_factory=dict)
    sustained_saturation_duration_s: dict[str, float] = Field(default_factory=dict)
    floor_penetration_m: float | None = Field(default=None, ge=0.0)
    support_statistics: dict[str, float] = Field(default_factory=dict)
    contact_statistics: dict[str, float] = Field(default_factory=dict)
    physics_seed_outcomes: tuple[PhysicsSeedOutcome, ...] = ()
    reference_to_achieved_envelope_loss: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_q4_classification(self) -> QualificationRecord:
        retained = sum(
            outcome.tracker_survived and outcome.semantic_retained is True
            for outcome in self.physics_seed_outcomes
        )
        if self.q4_classification == Q4Classification.STRICT and (
            len(self.physics_seed_outcomes) != 3 or retained != 3
        ):
            raise ValueError("Q4-strict requires 3/3 tracker-and-semantic-retained seeds")
        if self.q4_classification == Q4Classification.ROBUST and (
            len(self.physics_seed_outcomes) != 3 or retained != 2
        ):
            raise ValueError("Q4-robust requires exactly 2/3 retained seeds")
        if (
            self.q4_classification in {Q4Classification.ROBUST, Q4Classification.STRICT}
            and self.q4_status != EvidenceStatus.PASSED
        ):
            raise ValueError("a robust Q4 classification requires passed Q4 status")
        return self


class LadderMembership(FrozenModel):
    base_carrier_id: NonEmpty
    ladder_group_id: NonEmpty
    ladder_family: LadderFamily
    ladder_level: int = Field(ge=0)
    adaptation_intensity: float = Field(ge=0.0)
    matched_alternative_ids: tuple[NonEmpty, ...] = ()
    monotonicity_status: MonotonicityStatus = MonotonicityStatus.NOT_MEASURED
    admitted: bool = False


class AlternativeRole(StrEnum):
    NEUTRAL = "neutral"
    WEAKER = "weaker"
    TARGET = "target"
    STRONGER = "stronger"
    ALTERNATIVE_SKILL = "alternative_skill"


class AlternativeMember(FrozenModel):
    motion_id: NonEmpty
    role: AlternativeRole
    fixed_effort_cost: float | None = Field(default=None, ge=0.0)
    ladder_level: int | None = Field(default=None, ge=0)


class AlternativeSet(FrozenModel):
    alternative_set_id: NonEmpty
    members: tuple[AlternativeMember, ...] = Field(min_length=2)
    selection_rule: SelectionRule = SelectionRule.FIXED_ENERGY

    @model_validator(mode="after")
    def require_one_target_and_unique_motion_ids(self) -> AlternativeSet:
        target_count = sum(member.role == AlternativeRole.TARGET for member in self.members)
        if target_count != 1:
            raise ValueError("alternative set requires exactly one target")
        ids = [member.motion_id for member in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("alternative-set motion IDs must be unique")
        if self.selection_rule == SelectionRule.FIXED_ENERGY:
            if any(member.fixed_effort_cost is None for member in self.members):
                raise ValueError("fixed-energy selection requires every effort cost")
        else:
            levels = [member.ladder_level for member in self.members]
            if any(level is None for level in levels) or len(levels) != len(set(levels)):
                raise ValueError("ordinal selection requires unique ladder levels")
            target_level = next(
                member.ladder_level
                for member in self.members
                if member.role == AlternativeRole.TARGET
            )
            assert target_level is not None
            for member in self.members:
                if member.ladder_level is None:
                    continue
                if member.role in {AlternativeRole.NEUTRAL, AlternativeRole.WEAKER} and (
                    member.ladder_level >= target_level
                ):
                    raise ValueError("neutral/weaker ordinal levels must precede the target")
                if member.role == AlternativeRole.STRONGER and member.ladder_level <= target_level:
                    raise ValueError("stronger ordinal levels must follow the target")
        return self


class CriticalToken(FrozenModel):
    token_id: NonEmpty
    constraint_type: NonEmpty
    body_region: NonEmpty
    route_progress: float = Field(ge=0.0, le=1.0)
    event_phase: NonEmpty | None = None
    normal_xyz: tuple[float, float, float]
    geometry_interval_m: tuple[float, float]
    presence_probability: float = Field(ge=0.0, le=1.0)
    exact_target_margin_m: float | None = None
    exact_weaker_deficit_m: float | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> CriticalToken:
        lower, upper = self.geometry_interval_m
        if lower >= upper:
            raise ValueError("critical geometry interval must have positive width")
        if sum(component * component for component in self.normal_xyz) <= 1e-12:
            raise ValueError("constraint normal must be nonzero")
        return self


class GeometryJitterOutcome(FrozenModel):
    jitter_id: NonEmpty
    target_outcome: FeasibilityOutcome
    weaker_outcome: FeasibilityOutcome


class CriticalSceneAssessment(FrozenModel):
    critical_token_id: NonEmpty
    criticality_margin_m: float = Field(gt=0.0)
    scene_realization_id: NonEmpty
    nuisance_seed: int
    target_outcome: FeasibilityOutcome
    weaker_outcome: FeasibilityOutcome
    stronger_outcome: FeasibilityOutcome = FeasibilityOutcome.NOT_MEASURED
    obstacle_removal_weaker_outcome: FeasibilityOutcome
    geometry_jitter_outcomes: tuple[GeometryJitterOutcome, ...] = Field(min_length=2)
    exact_verifier_agrees: bool

    @model_validator(mode="after")
    def require_counterfactual_reversal(self) -> CriticalSceneAssessment:
        if self.target_outcome != FeasibilityOutcome.FEASIBLE:
            raise ValueError("critical scene requires a feasible target")
        if self.weaker_outcome != FeasibilityOutcome.INFEASIBLE:
            raise ValueError("critical scene requires an infeasible weaker motion")
        if self.obstacle_removal_weaker_outcome != FeasibilityOutcome.FEASIBLE:
            raise ValueError("critical obstacle removal must make the weaker motion feasible")
        if any(
            outcome.target_outcome != FeasibilityOutcome.FEASIBLE
            or outcome.weaker_outcome != FeasibilityOutcome.INFEASIBLE
            for outcome in self.geometry_jitter_outcomes
        ):
            raise ValueError("every registered geometry jitter must preserve preference reversal")
        if not self.exact_verifier_agrees:
            raise ValueError("critical-scene labels require exact-verifier agreement")
        return self


class PhysicsIntervention(FrozenModel):
    execution_pair_id: NonEmpty
    condition: InterventionCondition
    motion_role: InterventionMotionRole
    physics_seed: int
    contact: bool | None = None
    local_crossing: bool | None = None
    full_route: bool | None = None
    evidence_tier: EvidenceLevel = EvidenceLevel.OBSTACLE_PRESENT_PHYSICS
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def match_condition_to_evidence_tier(self) -> PhysicsIntervention:
        expected = (
            EvidenceLevel.OBSTACLE_ABSENT_TRACKING
            if self.condition == InterventionCondition.OBSTACLE_ABSENT
            else EvidenceLevel.OBSTACLE_PRESENT_PHYSICS
        )
        if self.evidence_tier != expected:
            raise ValueError("intervention condition and evidence tier disagree")
        return self


class Provenance(FrozenModel):
    source_repository: NonEmpty
    source_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    code_artifacts: tuple[ArtifactRef, ...]
    model_artifacts: tuple[ArtifactRef, ...] = ()
    simulator: NonEmpty | None = None
    controller: NonEmpty | None = None


class Motion2SceneRecord(FrozenModel):
    schema_version: Annotated[str, StringConstraints(pattern=r"^motion2scene_record_v\d+$")]
    group_id: NonEmpty
    motion: MotionIdentity
    descriptor: MotionDescriptor
    evidence: dict[EvidenceLevel, EvidenceRecord]
    alternative_set: AlternativeSet | None = None
    critical_token: CriticalToken | None = None
    critical_scene: CriticalSceneAssessment | None = None
    semantics: SemanticAssessment | None = None
    route: RouteAssessment | None = None
    qualification: QualificationRecord | None = None
    ladder: LadderMembership | None = None
    physics_interventions: tuple[PhysicsIntervention, ...] = ()
    corpus_role: NonEmpty = "development"
    dataset_eligible: bool = False
    split_group: NonEmpty
    split_name: NonEmpty = "unassigned"
    provenance: Provenance
    annotations: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_evidence_and_criticality_contract(self) -> Motion2SceneRecord:
        for key, record in self.evidence.items():
            if key != record.level:
                raise ValueError(f"evidence key {key} does not match record level {record.level}")
        if self.split_group != self.motion.base_motion_id:
            raise ValueError("split group must equal the base carrier ID")
        if self.ladder is not None and self.ladder.base_carrier_id != self.motion.base_motion_id:
            raise ValueError("ladder and motion base-carrier IDs disagree")
        if self.critical_scene is not None:
            if self.critical_token is None:
                raise ValueError("critical-scene assessment requires a critical token")
            if self.critical_scene.critical_token_id != self.critical_token.token_id:
                raise ValueError("critical-scene and critical-token IDs disagree")
        if self.critical_token is not None:
            if self.alternative_set is None:
                raise ValueError("critical token requires a matched alternative set")
            tracking = (
                self.evidence.get(EvidenceLevel.TRACKING_SINGLE_SEED),
                self.evidence.get(EvidenceLevel.TRACKING_MULTI_SEED),
            )
            if not any(
                record is not None and record.status == EvidenceStatus.PASSED for record in tracking
            ):
                raise ValueError("critical token requires passed Q3 or Q4 target tracking")
            geometry = self.evidence.get(EvidenceLevel.EXACT_GEOMETRY)
            if geometry is None or geometry.status != EvidenceStatus.PASSED:
                raise ValueError("critical token requires passed exact-geometry evidence")
            targets = [
                member.motion_id
                for member in self.alternative_set.members
                if member.role == AlternativeRole.TARGET
            ]
            if targets != [self.motion.motion_id]:
                raise ValueError("record motion must be the target in its alternative set")
        if self.dataset_eligible:
            if self.critical_token is None or self.critical_scene is None:
                raise ValueError("dataset-eligible records require a verified critical scene")
            q4 = self.evidence.get(EvidenceLevel.TRACKING_MULTI_SEED)
            if q4 is None or q4.status != EvidenceStatus.PASSED:
                raise ValueError("dataset-eligible critical scenes require passed Q4 evidence")
            if self.semantics is None or (
                self.semantics.semantic_status != SemanticStatus.CONTROLLER_RETAINED
            ):
                raise ValueError("dataset-eligible records require controller-retained semantics")
            if self.ladder is None or not self.ladder.admitted:
                raise ValueError("dataset-eligible records require an admitted matched ladder")
            if self.ladder.monotonicity_status != MonotonicityStatus.ORDERED:
                raise ValueError("dataset-eligible records require an ordered ladder")
            if self.alternative_set is None or (
                self.alternative_set.selection_rule != SelectionRule.ORDINAL_MINIMAL_FEASIBLE
            ):
                raise ValueError("dataset-eligible records require ordinal minimal feasibility")
            if self.qualification is None or (
                self.qualification.q4_classification != Q4Classification.STRICT
            ):
                raise ValueError("dataset-eligible records require Q4-strict qualification")
        return self


def validate_split_isolation(records: Iterable[Motion2SceneRecord]) -> None:
    """Fail if one carrier, critical token, or realization crosses named splits."""

    assignments: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        assignments[("carrier", record.motion.base_motion_id)].add(record.split_name)
        if record.ladder is not None:
            assignments[("ladder", record.ladder.ladder_group_id)].add(record.split_name)
        if record.critical_token is not None:
            assignments[("token", record.critical_token.token_id)].add(record.split_name)
        if record.critical_scene is not None:
            assignments[("scene", record.critical_scene.scene_realization_id)].add(
                record.split_name
            )
    crossed = {key: groups for key, groups in assignments.items() if len(groups) > 1}
    if crossed:
        details = ", ".join(
            f"{kind}:{identity}->{sorted(groups)}" for (kind, identity), groups in crossed.items()
        )
        raise ValueError(f"split leakage detected: {details}")
