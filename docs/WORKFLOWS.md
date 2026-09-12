# Design and implementation map

Read the current SONIC plan first: [repaired teacher and navigation design](sonic/motion2scene/REPAIRED_TEACHER_NAVIGATION_PLAN_20260912.md), [BFM implementation](sonic/motion2scene/BFM_DISTILLATION_IMPLEMENTATION_20260912.md), [coverage study](sonic/motion2scene/BFM_SCENE_NAVIGATION_COVERAGE_20260912.md), and [curriculum smoke](sonic/motion2scene/BFM_CURRICULUM_SMOKE_20260912.md).

The separate project's governing question and evidence remain in [Motion2Scene design](../vendor/motion2scene/docs/DESIGN_PLAN.md) and [claim/evidence map](../vendor/motion2scene/docs/CLAIM_EVIDENCE_MAP.md). Those historical project states are preserved; their outcomes are not silently merged into newer SONIC results.

| Work | Implementation under `vendor/sonic/` | Evidence or limit |
|---|---|---|
| Motion repair inputs and teacher fit | `gear_sonic/research/hindsight_training/` | 89 screened train / 20 development; offline repair is not policy qualification |
| BFM foundation and command curriculum | `gear_sonic/research/scene_distillation/{foundation,commands,curriculum,train}.py` | Full/root/navigation control masks; privileged future only during posterior training |
| Teacher querying and DAgger | `gear_sonic/research/scene_distillation/collect.py` | Teacher-bound actions, valid support masks, interventions logged |
| Goal, trajectory and obstacles | `gear_sonic/research/scene_distillation/tasks.py` | Body-frame public inputs; future trajectory is a training label |
| Physical scene qualification | `gear_sonic/research/scene_distillation/{scene_env,scene_qualification,scene_teacher}.py` | Contact, goal-hold and tracking gates; proposal is not qualified data |
| Scene/goal recurrent student | `gear_sonic/research/scene_distillation/{policy,navigation,online,navigation_runtime}.py` | Public observations only; prior qualification gate |
| Bounded residual improvement | `gear_sonic/research/scene_distillation/residual_rl.py` | Frozen base, bounded residual, PPO; no successful physical residual result claimed |
| Motion-to-scene inverse construction | `gear_sonic/dataset_generation/hallucination/motion2scene_inverse.py` and neighboring modules | Full research implementation retained |
| Learned scene / obstacle studies | `gear_sonic/dataset_generation/hallucination/` and `scripts/research/` | Retractions and failed validation remain authoritative; `learned_hallucinator.py` includes a retraction notice |
| Videos and dataset galleries | `scripts/research/render_*`, `gear_sonic/research/scene_distillation/render_student_comparison.py` | Existing artifacts included; choose correct tracked/reference roles |

Next research stages: qualify the repaired teacher; collect complete walking/turning/stopping supervision with original phase; test same-state command controllability; evaluate sparse-command stability; qualify scene/goal continuations; then compare bounded residual task learning against residual-disabled control. Do not train scene-navigation imitation on failed qualification probes or call low offline action error navigation success.
