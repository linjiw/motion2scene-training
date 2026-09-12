# Hindsight → SONIC tracking → scene-conditioned student

For the active **8,000-iteration / 512-environment continuation**, completion checks,
and restart instructions, read
[the current handoff](../../../docs/motion2scene/SONIC_DISTILLATION_HANDOFF_20260911.md).
The stages below describe the original 200-iteration packet. The preferred downstream
design is now [navigation through a frozen motion foundation](../scene_distillation/FOUNDATION_NAVIGATION.md);
the route-conditioned offline student below is an earlier prototype.

This is a separate engineering training packet dated 11 September 2026. It does
not alter Motion2Scene's registered acquisition/evaluation pilot, frozen tracker,
protected layouts, or original utility tests. Existing proxy outcomes were already
accessible. The frozen hindsight dataset v1 remains unchanged.

## Current executable stages

1. `prepare.py` verifies the dataset manifest, converts all 120 reference motions
   to native SONIC motion files, retains the 100 training / 20 development split,
   copies the released checkpoint, and writes the exact training configuration.
   Source poses are not clamped or relabeled as successful executions.
2. `validate.py` loads every motion through SONIC's actual motion library on CPU
   and checks finite states at the beginning, midpoint, and end. This verifies
   ingestion, not physical trackability or joint-limit qualification.
3. `launch.py` launches one bounded, recorded Isaac Lab attempt: 200 PPO updates,
   16 environments, 24 steps per update, seed 91143, 200 Hz physics, 50 Hz actions.
   Actor and critic load strictly from the release; the optimizer starts fresh.
   The G1 encoder is selected; cross-modal auxiliary terms and synthetic motion
   augmentations are disabled. The release's G1 reconstruction auxiliary remains.
   This is obstacle-free tracking with cameras disabled. All 100 training motions
   are resident and sampled uniformly. The 20 development motions are not trained.
4. `student.py`, `observations.py`, `records.py`, and `distill.py` implement a
   prototype offline student path. No executed scene-teacher shards exist in the
   frozen v1 dataset, so no student fit has been launched from it.

Example preparation and launch (the existing packet has already used this run ID;
do not run these commands against it again):

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.hindsight_training.prepare --dataset DATASET --output NEW_PACKET
.venv_isaaclab/bin/python -m gear_sonic.research.hindsight_training.validate NEW_PACKET
.venv_isaaclab/bin/python -m gear_sonic.research.hindsight_training.launch NEW_PACKET
```

## GPU correction

On this host NVML reports kernel/library mismatch, but CUDA allocation and matrix
multiplication succeed. NVML output is diagnostic. Launch decisions use actual
CUDA execution and `cudaMemGetInfo` through PyTorch. The tracking floor is 2 GiB;
the separate image renderer retains its 4 GiB floor. These are preflight limits,
not measured peak requirements. No drivers, services, or other jobs are changed.

## Student contract

The student consumes the same 930-element causal proprioceptive history as SONIC,
start and goal positions expressed relative to the current body, up to five
obstacles, and up to sixteen ordered route command waypoints. A supplied route
is a command available at deployment; future executed reference poses cannot
silently supply it. `observations.py` uses explicit wxyz quaternions and a common
world frame to prevent the earlier trajectory/obstacle alignment error.

Obstacle rows contain center xyz, full bounding dimensions xyz, the first two
rotation columns, and a box/cylinder/other indicator. A masked shared encoder
makes obstacle order irrelevant. Mesh collision validation still uses actual
geometry, not these bounding features. Route order is retained.

The student produces 64 post-FSQ tokens on the release lattice [-1, 15/16] in
increments of 1/16. A decoder loaded strictly from the bound SONIC checkpoint
maps these tokens plus proprioception to 29 action means. The loss is token MSE
plus action-mean MSE through the frozen decoder. Teacher action means must agree
with that decoder on the same measured state. Future-timestamp observations,
development motion IDs, missing receipts, and reference-only labels are rejected.
Hashes bind supplied records; they do not independently authenticate execution.

The optional depth encoder accepts depth/validity tensors at 160×120. It has only
synthetic interface tests. No camera is instantiated by this packet. The proposed
70° field of view still needs a separate sensor/timing specification and matched
noise, dropout, delay, and memory measurements before camera training.

## Gates before scene teaching

First evaluate the trained tracker and original release on a fixed, matched
development panel; retain failures and check complete-motion tracking, contact,
ground support, joint limits, and recovery. The development set is already known,
not a fresh generalization test. Do not call the training curve that evaluation.

Next lock a separate scene-teacher collection design with cloned obstacle physics,
privileged scene observations, task rewards, stopping/censoring rules, bounded
assignments, and 200 Hz contact records. The current USD replay robot is visual;
it is not an articulated training environment. The native scene-USD path is
single-environment and needs validated cloning before batched collection.

An obstacle-blind tracking teacher can supply a motion prior and supervise
tracking-compatible scenes. It cannot by itself establish obstacle avoidance,
goal navigation, or recovery. Those behaviors require an effective scene-aware
teacher, then on-policy student-state querying/data aggregation and fresh labels.
Offline imitation loss alone cannot pass a physical navigation utility gate.

Record start/goal, optional route command, scene and checkpoint hashes, measured
robot history, decision/observation timestamps, teacher tokens and action means,
200 Hz contact evidence, termination, and all collection attempts. Bind each
student shard to those original receipts. No deployment or hardware claim follows
from this packet.
