# E0-Q3 Pilot V7 Recorder Fix

Registered: 2026-09-04T13:29:56-04:00, before V7 execution.

V6 passed camera-free Isaac startup, loaded the SONIC policy, created the environment, and entered
`env.step()`. The combined `dataset` recorder then attempted to read `ego_camera`, which was
correctly absent in trajectory-only mode. It stopped on the first step and produced no complete
trajectory or scientific outcome.

The repository already contains a dedicated `manager_env/recorders=trajectory` profile with the
same trajectory term and no camera term. The rollout driver now selects that profile whenever
`--trajectory-only` is registered. A focused test verifies the profile contains `trajectory` and
does not contain `render_envs`.

V7 changes only the runner hash, source commit, and output lineage. All scientific variables and
resource rules remain frozen.
