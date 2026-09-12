# E0-Q3 Fresh Kimodo Motion-Bank Result

Manifest: `sha256:8e656064c04a973ab4cafd35afaf07f8d5154c8e57458f86fdf2327a01750425`  
Run record: `sha256:3e6d4aab0f068615fa702dba5832794f9a9c19bd2539294bed4022aa9f4ba9b5`

The gradeable V8 batch completed **5/6 accepted (83.3%)**, with **0 N external collision force in every cell**. This is Q3 single-seed feasibility, not Q4 robustness.

## Registered-prediction reconciliation

| prediction | result | observation |
|---|---:|---|
| `P1_q3_acceptance_and_zero_external` | **pass** | 5/6 accepted; zero_external=True |
| `P2_route_heading_signs` | **pass** | left positive and right negative |
| `P3_duck_depth_and_walk_contrast` | **miss** | duck_drop=0.112483 m; walk_minus_duck_root_min=0.030062 m |
| `P4_narrowing_semantics` | **pass** | arm_tuck=0.232541 m; shoulder_turn=0.090567 m |
| `P5_action_token_diversity` | **miss** | pooled=4.603329; between=2.749274 |
| `P6_infrastructure_contract` | **pass** | run_status=completed; all_fps_50=True |

## Prompt and executed distribution

| index | body mode | route | exact prompt | Q3 | heading (rad) | root min (m) | semantic |
|---:|---|---|---|---|---:|---:|---|
| 000 | `walk` | `straight` | A person walks at a steady pace in a straight line | **accepted** | -0.277 | 0.688 | not measured |
| 001 | `walk` | `gentle_left` | A person walks at a steady pace curving gently to the left | **accepted** | 2.457 | 0.725 | not measured |
| 005 | `duck_under` | `gentle_right` | A person walks at a steady pace curving gently to the right and ducks down low to pass under an obstacle | **accepted** | -1.693 | 0.658 | pass |
| 006 | `arm_tuck` | `straight` | A person walks at a steady pace in a straight line holding both arms tight against the body to fit through a narrow gap | **rejected** | -1.501 | 0.676 | pass |
| 011 | `shoulder_turn` | `gentle_right` | A person walks at a steady pace curving gently to the right, turning one shoulder forward and twisting the torso sideways to slip through a narrow opening | **accepted** | -2.005 | 0.733 | pass |
| 016 | `carry_walk` | `gentle_left` | A person walks at a steady pace curving gently to the left while carrying something in both hands | **accepted** | 2.022 | 0.706 | not measured |

## Action-token distribution

| subset | episodes | pooled rank | between-episode rank | mean within-episode rank |
|---|---:|---:|---:|---:|
| all evaluated | 6 | 4.603 | 2.749 | 2.930 |
| Q3 accepted | 5 | 3.653 | 1.860 | 2.474 |

The achieved Q3 bank is broad enough for a duck/shoulder pilot but not yet a balanced critical-motion ladder: arm-tuck failed tracking, step-over had zero reference-semantic yield, and accepted-only action-token rank fell below the registered diversity target.

## Resource result

The completed six-cell batch used **0.0438 contended GPU-hours**. Trajectory-only process samples
ranged from roughly 0.36 to 3.32 GiB while the other training job remained resident. The 6,000 MiB
pre-launch floor was sufficient; a 9,000 MiB floor was not required for this Q3 capture mode.
This does not authorize concurrent SONIC cells or a lower launch floor: another generator started
mid-cell and total free memory temporarily fell to roughly 4.7 GiB.

## Frozen artifacts

- evaluation: `sha256:dc8fcd6cd762c003192f9fc5d0cc4e02b6ef7f655bfb3ad435a4178c5ad94d9a`
- diversity (all): `sha256:2cc80670cfbf522eb883ea8b85c175e1319b1713d16b91b6f05c24ec6e57d47b`
- diversity (Q3 accepted): `sha256:4d8d720fdfbcc631db7ddd75a9c4da2b56408ab39edb42f672493b368eecd017`
- versioned corpus snapshot: `sha256:58f93db540c0e785872749c737e96590480574e5e3fedfd749a155ce7ca5012a`
- evaluator source: commit `4db946b7e28bffc30e1d6dd3a160195ad357581e`, file
  `sha256:847b325835f910a4413d1d4eedaf1174a30fdcf753fbf6375b3ed9c753d3f767`
