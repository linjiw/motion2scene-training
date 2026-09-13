# Direct masked scene/navigation distillation

This addendum adopts the user's clarification: existing per-motion navigation and scene annotations can condition the student just as detailed BFM command annotations do. A separate navigation director is not required. The previous hierarchy remains a useful comparison and debugging tool.

## Model and supervision

Train a single public prior:

`a_t = decoder(prior(history_t, masked_motion_commands_t, navigation_context_t, scene_context_t))`.

The teacher supplies same-state action targets and optional tokens. A training posterior can additionally receive the detailed reference/future. The public prior must have scene/navigation-only training and execution modes in which detailed motion commands, reference phase, motion ID and future reference are absent. At deployment, the user supplies navigation/scene context; robot history is sensed continuously. Internal behavior latents may still exist without an externally supplied waypoint or command sequence.

Treat scene and navigation annotations as conditioning labels; teacher actions remain the supervised action targets. Start/goal, terminal requirements and map geometry can be legitimate task inputs. If a reference trajectory is also recorded, retain it as teacher/posterior supervision or an explicitly optional path-command mode. It cannot silently remain an input in a claimed goal-and-scene-only mode.

The [BFM masked-interface approach](https://arxiv.org/html/2509.13780v1) motivates unifying different behavior specifications. Extending its interface to scene/goal tokens is our proposed adaptation, not a result demonstrated by the paper. Coarse context permits many valid motions, so scene-mode evaluation should score task success and physical constraints rather than demand exact reference-pose reproduction.

## Training modes to compare

1. Detailed motion commands plus optional context: establish motor/action reconstruction and shared representation.
2. Partial root/posture commands plus scene/navigation context: bridge detailed and abstract conditioning.
3. Navigation context and scene only, with robot history: explicitly hide every detailed reference command. Include substantial training and student-driven rollout coverage in this exact mode.
4. Later, camera-derived scene belief and navigation context: replace perfect-map inputs with deployable observations and explicit unknown/visibility information.

Keep required goals and collision information present in the principal navigation mode. Missing geometry is a distinct observation condition, not a claim of free space. Start with known maps; camera partial observability is a separate modality-distillation stage.

## What the paired dataset supports

A clip with a corresponding start/goal and compatible scene gives useful nominal training examples. There is no reason to discard these annotations or require counterfactual data before running a direct-context baseline. However, one scene/context per motion can support a shortcut: identify the associated motion from its context or initial history without learning how geometry constrains motion. This is a generalization concern, not a proof that paired training cannot work.

Use provenance tiers. Physically executed, scene-qualified demonstrations support strong action imitation for that task. Geometry-compatible but unexecuted scene/reference pairs can support weaker context/trajectory objectives and candidate generation; do not describe their teacher labels as proven collision-free actions. Our current scene probes still fail full task qualification, so their role remains explicit.

Improve coverage with changed scenes requiring different continuations, changed goals in the same scene, and multiple valid behaviors for a context. Hold out scene/motion families. During DAgger, query an expert capable of the requested scene/goal on the student's current state. A fixed reference tracker can label supported nominal/recovery regions; its competence does not automatically extend to arbitrary route changes.

## Minimal decisive ablation

Use the same teacher/data/compute budget for (A) the present motion-command student, (B) direct scene/navigation-conditioned masked student, and (C) the hierarchical scene director with the same motor backend. For B, compare context-only episodes against a model that always sees detailed commands; this tests whether masking actually teaches the intended deployment interface. Evaluate scene/goal dependence with held-out task families and context changes, plus goal/hold success, falls, contacts and recovery. Include command responsiveness and motor checks to separate conditional-planning failures from motor failures.

The current transformer experiment does not implement B: its attention tokens cover native history and motion commands, with no scene or final-goal tokens. Larger-model or DAgger improvements in that run therefore do not establish scene awareness. No active source or experiment configuration was changed by this design addendum.
