# Design and implementation map

Start with the [full distillation research report](DISTILLATION_RESEARCH.md). Its implementation follow-ups connect the original [BFM design](sonic/motion2scene/BFM_DISTILLATION_IMPLEMENTATION_20260912.md) to the [motor recovery result](sonic/motion2scene/BFM_MOTOR_RECOVERY_RESULTS_20260913.md), [navigation pilot](sonic/motion2scene/NAVIGATION_MOTOR_PILOT_20260913.md), [terminal-control audit](sonic/motion2scene/NAVIGATION_TERMINAL_CONTROL_20260913.md), and [supported recovery protocol](sonic/motion2scene/NAVIGATION_SUPPORTED_RECOVERY_20260913.md).

The separate project's governing question and evidence remain in [Motion2Scene design](../vendor/motion2scene/docs/DESIGN_PLAN.md) and [claim/evidence map](../vendor/motion2scene/docs/CLAIM_EVIDENCE_MAP.md). Those historical project states are preserved; their outcomes are not silently merged into newer SONIC results.

| Work | Implementation under `vendor/sonic/` | Evidence or limit |
|---|---|---|
| Motion repair inputs and teacher fit | `gear_sonic/research/hindsight_training/` | 89 screened train / 20 development; offline repair is not policy qualification |
| BFM foundation and command curriculum | `gear_sonic/research/scene_distillation/{foundation,commands,curriculum,train}.py` | Full/root/navigation control masks; privileged future only during posterior training |
| Preserved motor recovery | `gear_sonic/research/scene_distillation/{anticipatory_motor,motor_training,online_motor,motor_runtime}.py` | 88/89 selected train completion and 10/20 development; frozen teacher encoder/decoder retained |
| Teacher querying and DAgger | `gear_sonic/research/scene_distillation/collect.py` | Teacher-bound actions, valid support masks, interventions logged |
| Goal, trajectory and obstacles | `gear_sonic/research/scene_distillation/tasks.py` | Body-frame public inputs; future trajectory is a training label |
| Physical scene qualification | `gear_sonic/research/scene_distillation/{scene_env,scene_qualification,scene_teacher}.py` | Contact, goal-hold and tracking gates; proposal is not qualified data |
| Scene/goal recurrent student | `gear_sonic/research/scene_distillation/{policy,navigation,online,navigation_runtime}.py` | Public observations only; prior qualification gate |
| Navigation command adapter | `gear_sonic/research/scene_distillation/{navigation_motor,navigation_motor_runtime,navigation_data,navigation_localization}.py` | Goal/map input with frozen motor; current results remain unreliable and in-sample |
| Supported navigation recovery | `gear_sonic/research/scene_distillation/{navigation_queries,navigation_takeover,navigation_recovery,navigation_recovery_data}.py` | Exact-motor demonstrations and qualification-bound takeovers; registered physical result pending |
| Bounded residual improvement | `gear_sonic/research/scene_distillation/residual_rl.py` | Frozen base, bounded residual, PPO; no successful physical residual result claimed |
| Motion-to-scene inverse construction | `gear_sonic/dataset_generation/hallucination/motion2scene_inverse.py` and neighboring modules | Full research implementation retained |
| Learned scene / obstacle studies | `gear_sonic/dataset_generation/hallucination/` and `scripts/research/` | Retractions and failed validation remain authoritative; `learned_hallucinator.py` includes a retraction notice |
| Videos and dataset galleries | `scripts/research/render_*`, `gear_sonic/research/scene_distillation/render_student_comparison.py` | Existing artifacts included; choose correct tracked/reference roles |

Next research stages: finish the registered supported-recovery panel; expand successful approach, braking, hold and recovery coverage across independent motion families; compare command completion with direct reference/token generation at matched budgets; construct counterfactual obstacle pairs; then transfer the known-map student to a separately validated perception adapter. Keep failed takeovers as diagnostics, and do not call low offline action error or assisted recovery autonomous navigation success.
