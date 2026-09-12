from __future__ import annotations

import pytest
from pydantic import ValidationError

from motion2scene.dataset.schema import (
    AlternativeMember,
    AlternativeRole,
    AlternativeSet,
    ArtifactRef,
    CriticalSceneAssessment,
    CriticalToken,
    EventInterval,
    EvidenceLevel,
    EvidenceRecord,
    EvidenceStatus,
    FeasibilityOutcome,
    GeometryJitterOutcome,
    LadderFamily,
    LadderMembership,
    MonotonicityStatus,
    Motion2SceneRecord,
    MotionDescriptor,
    MotionIdentity,
    PhysicsSeedOutcome,
    Provenance,
    Q4Classification,
    QualificationRecord,
    SelectionRule,
    SemanticAssessment,
    SemanticStatus,
    validate_split_isolation,
)

HASH = "sha256:" + "a" * 64
COMMIT = "b" * 40


def artifact(path: str = "/data/motion.csv") -> ArtifactRef:
    return ArtifactRef(path=path, sha256=HASH)


def base_record(**updates) -> dict:
    payload = {
        "schema_version": "motion2scene_record_v1",
        "group_id": "carrier-001/unmatched",
        "motion": MotionIdentity(
            motion_id="motion-001",
            source="kimodo",
            robot="g1-29dof",
            qpos=artifact(),
            fps=30.0,
            frame_count=120,
            prompt="A person walks forward",
            generator_model="kimodo-g1-rp",
            generator_seed=1000,
            base_motion_id="carrier-001",
        ),
        "descriptor": MotionDescriptor(skill="walk", route_style="straight", speed_style="steady"),
        "evidence": {
            EvidenceLevel.KINEMATICS: EvidenceRecord(
                level=EvidenceLevel.KINEMATICS,
                status=EvidenceStatus.PASSED,
                method="joint-limit saturation screen",
            )
        },
        "split_group": "carrier-001",
        "provenance": Provenance(
            source_repository="groot-wbc-sonic-sim-trackb",
            source_commit=COMMIT,
            code_artifacts=(artifact("scripts/screen.py"),),
        ),
    }
    payload.update(updates)
    return payload


def critical_evidence() -> dict[EvidenceLevel, EvidenceRecord]:
    return {
        EvidenceLevel.TRACKING_SINGLE_SEED: EvidenceRecord(
            level=EvidenceLevel.TRACKING_SINGLE_SEED,
            status=EvidenceStatus.PASSED,
            method="SONIC screen-empty rollout",
        ),
        EvidenceLevel.EXACT_GEOMETRY: EvidenceRecord(
            level=EvidenceLevel.EXACT_GEOMETRY,
            status=EvidenceStatus.PASSED,
            method="capsule-to-beam signed distance",
        ),
    }


def test_partial_e0_record_does_not_require_unmeasured_evidence() -> None:
    record = Motion2SceneRecord.model_validate(base_record())
    assert set(record.evidence) == {EvidenceLevel.KINEMATICS}
    assert record.critical_token is None


def test_failed_evidence_requires_reason() -> None:
    with pytest.raises(ValidationError, match="requires a reason"):
        EvidenceRecord(
            level=EvidenceLevel.TRACKING_SINGLE_SEED,
            status=EvidenceStatus.FAILED,
            method="SONIC rollout",
        )


def test_critical_token_requires_matched_alternatives() -> None:
    token = CriticalToken(
        token_id="beam-001",
        constraint_type="beam",
        body_region="head_torso",
        route_progress=0.5,
        normal_xyz=(0.0, 0.0, -1.0),
        geometry_interval_m=(1.1, 1.2),
        presence_probability=1.0,
    )
    with pytest.raises(ValidationError, match="requires a matched alternative set"):
        Motion2SceneRecord.model_validate(base_record(critical_token=token))


def test_critical_record_target_must_match_alternative_set() -> None:
    token = CriticalToken(
        token_id="beam-001",
        constraint_type="beam",
        body_region="head_torso",
        route_progress=0.5,
        normal_xyz=(0.0, 0.0, -1.0),
        geometry_interval_m=(1.1, 1.2),
        presence_probability=1.0,
    )
    alternatives = AlternativeSet(
        alternative_set_id="ladder-001",
        members=(
            AlternativeMember(
                motion_id="other-target",
                role=AlternativeRole.TARGET,
                fixed_effort_cost=1.0,
            ),
            AlternativeMember(
                motion_id="motion-001",
                role=AlternativeRole.NEUTRAL,
                fixed_effort_cost=0.0,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="record motion must be the target"):
        Motion2SceneRecord.model_validate(
            base_record(
                critical_token=token,
                alternative_set=alternatives,
                evidence=critical_evidence(),
            )
        )


def test_critical_token_requires_passed_tracking_evidence() -> None:
    token = CriticalToken(
        token_id="beam-001",
        constraint_type="beam",
        body_region="head_torso",
        route_progress=0.5,
        normal_xyz=(0.0, 0.0, -1.0),
        geometry_interval_m=(1.1, 1.2),
        presence_probability=1.0,
    )
    alternatives = AlternativeSet(
        alternative_set_id="ladder-001",
        members=(
            AlternativeMember(
                motion_id="motion-001", role=AlternativeRole.TARGET, fixed_effort_cost=1.0
            ),
            AlternativeMember(
                motion_id="neutral", role=AlternativeRole.NEUTRAL, fixed_effort_cost=0.0
            ),
        ),
    )
    geometry_only = {
        EvidenceLevel.EXACT_GEOMETRY: EvidenceRecord(
            level=EvidenceLevel.EXACT_GEOMETRY,
            status=EvidenceStatus.PASSED,
            method="capsule-to-beam signed distance",
        )
    }

    with pytest.raises(ValidationError, match="passed Q3 or Q4"):
        Motion2SceneRecord.model_validate(
            base_record(
                critical_token=token,
                alternative_set=alternatives,
                evidence=geometry_only,
            )
        )


def test_critical_token_requires_passed_exact_geometry() -> None:
    token = CriticalToken(
        token_id="beam-001",
        constraint_type="beam",
        body_region="head_torso",
        route_progress=0.5,
        normal_xyz=(0.0, 0.0, -1.0),
        geometry_interval_m=(1.1, 1.2),
        presence_probability=1.0,
    )
    alternatives = AlternativeSet(
        alternative_set_id="ladder-001",
        members=(
            AlternativeMember(
                motion_id="motion-001", role=AlternativeRole.TARGET, fixed_effort_cost=1.0
            ),
            AlternativeMember(
                motion_id="neutral", role=AlternativeRole.NEUTRAL, fixed_effort_cost=0.0
            ),
        ),
    )
    tracking_only = {
        EvidenceLevel.TRACKING_SINGLE_SEED: EvidenceRecord(
            level=EvidenceLevel.TRACKING_SINGLE_SEED,
            status=EvidenceStatus.PASSED,
            method="SONIC screen-empty rollout",
        )
    }

    with pytest.raises(ValidationError, match="exact-geometry"):
        Motion2SceneRecord.model_validate(
            base_record(
                critical_token=token,
                alternative_set=alternatives,
                evidence=tracking_only,
            )
        )


def test_critical_interval_must_have_positive_width() -> None:
    with pytest.raises(ValidationError, match="positive width"):
        CriticalToken(
            token_id="beam-001",
            constraint_type="beam",
            body_region="head_torso",
            route_progress=0.5,
            normal_xyz=(0.0, 0.0, -1.0),
            geometry_interval_m=(1.2, 1.1),
            presence_probability=1.0,
        )


def test_hashes_are_fail_closed() -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(path="motion.csv", sha256="not-a-hash")


def test_generation_seed_aliases_cannot_disagree() -> None:
    with pytest.raises(ValidationError, match="seeds disagree"):
        MotionIdentity(
            motion_id="motion-001",
            source="kimodo",
            robot="g1-29dof",
            qpos=artifact(),
            fps=30.0,
            frame_count=120,
            generator_seed=1,
            generation_seed=2,
            base_motion_id="carrier-001",
        )


def test_controller_retained_semantics_require_both_event_intervals() -> None:
    reference = EventInterval(onset_progress=0.3, recovery_progress=0.7)
    with pytest.raises(ValidationError, match="achieved event"):
        SemanticAssessment(
            semantic_predicate_version="semantic_v2",
            semantic_status=SemanticStatus.CONTROLLER_RETAINED,
            reference_event_interval=reference,
        )


def test_q4_strict_is_a_three_of_three_semantic_retention_contract() -> None:
    outcomes = tuple(
        PhysicsSeedOutcome(
            physics_seed=seed,
            tracker_survived=True,
            semantic_retained=True,
            fell=False,
        )
        for seed in (1, 2, 3)
    )
    record = QualificationRecord(
        q4_status=EvidenceStatus.PASSED,
        q4_classification=Q4Classification.STRICT,
        physics_seed_outcomes=outcomes,
    )
    assert record.q4_classification == Q4Classification.STRICT


def test_q4_robust_rejects_one_of_three_retained() -> None:
    outcomes = tuple(
        PhysicsSeedOutcome(
            physics_seed=seed,
            tracker_survived=True,
            semantic_retained=seed == 1,
            fell=False,
        )
        for seed in (1, 2, 3)
    )
    with pytest.raises(ValidationError, match="exactly 2/3"):
        QualificationRecord(
            q4_status=EvidenceStatus.PASSED,
            q4_classification=Q4Classification.ROBUST,
            physics_seed_outcomes=outcomes,
        )


def test_ordinal_alternative_set_requires_unique_ladder_levels() -> None:
    with pytest.raises(ValidationError, match="unique ladder levels"):
        AlternativeSet(
            alternative_set_id="ladder-001",
            selection_rule=SelectionRule.ORDINAL_MINIMAL_FEASIBLE,
            members=(
                AlternativeMember(
                    motion_id="motion-001",
                    role=AlternativeRole.TARGET,
                    ladder_level=1,
                ),
                AlternativeMember(
                    motion_id="neutral",
                    role=AlternativeRole.NEUTRAL,
                    ladder_level=1,
                ),
            ),
        )


def test_critical_scene_requires_removal_reversal() -> None:
    with pytest.raises(ValidationError, match="removal"):
        CriticalSceneAssessment(
            critical_token_id="beam-001",
            criticality_margin_m=0.01,
            scene_realization_id="scene-001",
            nuisance_seed=5,
            target_outcome=FeasibilityOutcome.FEASIBLE,
            weaker_outcome=FeasibilityOutcome.INFEASIBLE,
            obstacle_removal_weaker_outcome=FeasibilityOutcome.INFEASIBLE,
            geometry_jitter_outcomes=(
                GeometryJitterOutcome(
                    jitter_id="lower",
                    target_outcome=FeasibilityOutcome.FEASIBLE,
                    weaker_outcome=FeasibilityOutcome.INFEASIBLE,
                ),
                GeometryJitterOutcome(
                    jitter_id="upper",
                    target_outcome=FeasibilityOutcome.FEASIBLE,
                    weaker_outcome=FeasibilityOutcome.INFEASIBLE,
                ),
            ),
            exact_verifier_agrees=True,
        )


def test_dataset_eligible_requires_q4_controller_retention_and_ordered_ladder() -> None:
    token = CriticalToken(
        token_id="beam-001",
        constraint_type="beam",
        body_region="head_torso",
        route_progress=0.5,
        normal_xyz=(0.0, 0.0, -1.0),
        geometry_interval_m=(1.1, 1.2),
        presence_probability=1.0,
    )
    alternatives = AlternativeSet(
        alternative_set_id="ladder-001",
        selection_rule=SelectionRule.ORDINAL_MINIMAL_FEASIBLE,
        members=(
            AlternativeMember(motion_id="motion-001", role=AlternativeRole.TARGET, ladder_level=1),
            AlternativeMember(motion_id="neutral", role=AlternativeRole.NEUTRAL, ladder_level=0),
        ),
    )
    scene = CriticalSceneAssessment(
        critical_token_id="beam-001",
        criticality_margin_m=0.01,
        scene_realization_id="scene-001",
        nuisance_seed=5,
        target_outcome=FeasibilityOutcome.FEASIBLE,
        weaker_outcome=FeasibilityOutcome.INFEASIBLE,
        obstacle_removal_weaker_outcome=FeasibilityOutcome.FEASIBLE,
        geometry_jitter_outcomes=(
            GeometryJitterOutcome(
                jitter_id="lower",
                target_outcome=FeasibilityOutcome.FEASIBLE,
                weaker_outcome=FeasibilityOutcome.INFEASIBLE,
            ),
            GeometryJitterOutcome(
                jitter_id="upper",
                target_outcome=FeasibilityOutcome.FEASIBLE,
                weaker_outcome=FeasibilityOutcome.INFEASIBLE,
            ),
        ),
        exact_verifier_agrees=True,
    )
    with pytest.raises(ValidationError, match="passed Q4"):
        Motion2SceneRecord.model_validate(
            base_record(
                dataset_eligible=True,
                critical_token=token,
                critical_scene=scene,
                alternative_set=alternatives,
                ladder=LadderMembership(
                    base_carrier_id="carrier-001",
                    ladder_group_id="ladder-001",
                    ladder_family=LadderFamily.DUCK,
                    ladder_level=1,
                    adaptation_intensity=0.1,
                    monotonicity_status=MonotonicityStatus.ORDERED,
                    admitted=True,
                ),
                evidence=critical_evidence(),
            )
        )


def test_split_validator_rejects_token_crossing_named_splits() -> None:
    first = Motion2SceneRecord.model_validate(base_record(split_name="development"))
    second_payload = base_record(split_name="motion_ood")
    second_payload["motion"] = second_payload["motion"].model_copy(
        update={"motion_id": "motion-002", "base_motion_id": "carrier-002"}
    )
    second_payload["split_group"] = "carrier-002"
    second = Motion2SceneRecord.model_validate(second_payload)
    # Carrier identities do not cross; one shared ladder identity still must not cross.
    first = first.model_copy(
        update={
            "ladder": LadderMembership(
                base_carrier_id="carrier-001",
                ladder_group_id="ladder-shared",
                ladder_family=LadderFamily.DUCK,
                ladder_level=0,
                adaptation_intensity=0.0,
            )
        }
    )
    second = second.model_copy(
        update={
            "ladder": LadderMembership(
                base_carrier_id="carrier-002",
                ladder_group_id="ladder-shared",
                ladder_family=LadderFamily.DUCK,
                ladder_level=1,
                adaptation_intensity=0.1,
            )
        }
    )
    with pytest.raises(ValueError, match="split leakage"):
        validate_split_isolation([first, second])
