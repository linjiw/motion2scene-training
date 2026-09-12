# E1 crouch-ladder Q3 predictions

Registered before either edited motion is launched in SONIC.

## Fixed cohort

The carrier is fresh Kimodo motion index 000, whose nominal motion was accepted in the
single-seed E0-Q3 V8 run. The two alternatives are deterministic local-crouch edits of that exact
carrier: `d040` and `d055`. Both preserve the reference root route and foot height. This run tests
obstacle-absent execution only; it does not establish beam-scene feasibility or robustness.

## Predictions

1. Both cells will produce a complete, gradeable 50 Hz trajectory with zero external collision
   force in the `screen_empty` obstacle-absent proxy.
2. The 40 mm edit will satisfy the frozen reference-trackability acceptance policy.
3. At least one of the two edits will be accepted. The 55 mm edit has no directional acceptance
   prediction because the larger joint excursion may exceed the controller's tracking envelope.
4. No result will be promoted beyond Q3 single-seed evidence. A rejected edit remains a scientific
   rejection and will not be tuned or retried in this batch.

## Resource prediction

Each trajectory-only cell should remain below the measured 4.9 GiB process-memory envelope and
finish in substantially less than the registered 375-second contended ceiling. Launch requires at
least 6,000 MiB free VRAM and cells are strictly serial. The 9,000 MiB gate is not required for this
camera-free experiment.
