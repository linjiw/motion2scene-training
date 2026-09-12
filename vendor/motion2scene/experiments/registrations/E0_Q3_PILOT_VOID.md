# E0-Q3 Pilot V1 — Void Before Spend

- Manifest SHA-256: `d843a442d1bd8add1b2faf45de82e9651acc24d28965888090a1d5a06596e9cf`
- Physics cells started: 0
- Contended GPU-hours spent: 0.0
- Recorded yield: 8,193 MiB free versus the 9,000 MiB launch gate
- Void trigger: the subsequently run full reference gate showed that selected indices 007
  (`arm_tuck`), 009 (`shoulder_turn`), and 014 (`step_over`) did not contain the behavior named by
  their prompts.
- Trigger artifact: `/home/linjiw/research-data/groot-wbc/cg-wbc-v1-pilot/reference_gate.json`,
  SHA-256 `08a5f04eba17485aad17eb950183c0eb8a8c9b051a2d56e9418e1b0aa029e0b4`.

The manifest and its prediction file remain in Git as an audit record. V2 changes only eligibility:
it samples six clips from the ten references that the pre-physics gate says are worth a rollout.
No V1 outcome was observed and no V1 prediction is adjudicated.

