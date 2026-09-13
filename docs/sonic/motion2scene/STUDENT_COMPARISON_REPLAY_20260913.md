# Matched student motion replay

The rendered comparison preserves the recorded 8/8 full-command versus 0/8
navigation pilot result. Its three examples are 00908 corridor, 00413 clear,
and 00265 corridor: the full-command student completes all three; the goal/map
student completes none under the unchanged goal, speed, contact and 50-tick hold
criteria. This is an in-sample illustration, not a generalization benchmark.

The video is `/home/linjiw/research-data/m2s-student-comparison-videos-20260913/videos-final/student-comparison.mp4`.
It is 32.6 seconds, 1920×864, 25 fps. Full-command execution is on the left;
goal/map execution is on the right. Overlays show measured distance, speed,
and consecutive hold. A green ring projects the goal region onto the floor;
the scorer uses three-dimensional root distance. End-frame freezes are labeled.

The frames render measured Isaac physics trajectories using MuJoCo meshes and
recorded link transforms. No MuJoCo physics is used to synthesize a different
trajectory. Enabling the Isaac RTX camera crashed during startup, before any
rollout; that failed attempt and its log remain in the external packet.
All six newly captured root, speed and prohibited-contact traces are exactly
equal to the original scored runs. The packet binds the source and video hashes.

Re-render the existing measured poses into a new directory:

```bash
MUJOCO_GL=egl .venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.render_navigation_replay \
  --packet /home/linjiw/research-data/m2s-student-comparison-videos-20260913/physics \
  --output /tmp/m2s-student-replay
```

The packet's per-run `command.json` files reproduce Isaac capture. Capture uses
`RecordFullMotorCallback` and `RecordNavigationMotorCallback` in
`render_navigation_comparison.py`. The companion renderer verifies pose/scorer
alignment and maps measured body transforms to the visual mesh geometry.

Validation: six native rollouts with exact trace parity, pose/scorer row and root
alignment, visual inspection, full MP4 decoding with ffmpeg, and Ruff/Black on
both rendering modules. Compact receipts are in
`evidence/student-comparison-replay-20260913/`; datasets and videos remain external.
