# E0-Q3 Isaac Runtime Diagnostic

Registered: 2026-09-04T13:19:42-04:00, before execution.

Prediction: a minimal camera-free Isaac Lab simulation will initialize, reset, take ten steps,
emit `ISAACLAB_RUNTIME_SMOKE_SUCCESS steps=10`, and exit cleanly. A pass isolates the V4 failure
to the camera/RTX-renderer startup path sufficiently to justify a separately registered
trajectory-only SONIC probe. A failure redirects work to environment or driver repair before any
new scientific batch.

This diagnostic does not evaluate a motion and cannot change any scientific denominator. It runs
once, serially, under the same 6,000 MiB launch floor and a 0.05 GPU-hour ceiling.
