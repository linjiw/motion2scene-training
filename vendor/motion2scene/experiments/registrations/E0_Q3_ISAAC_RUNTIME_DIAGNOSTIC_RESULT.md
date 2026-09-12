# E0-Q3 Isaac Runtime Diagnostic Result

Observed: 2026-09-04T13:23:28-04:00.

The strict registered prediction missed: the process did not exit cleanly and the 180-second
timeout returned status 124. The completed portions were:

- camera-free Isaac Lab application initialization;
- simulation-context reset;
- ten physics steps;
- `ISAACLAB_RUNTIME_SMOKE_SUCCESS steps=10` marker.

The process then hung inside `simulation_app.close()` until the registered timeout reclaimed it.
The output log is preserved with SHA-256
`dee439ef13ca2626f6f15021c3a1b25c0d273aa01d845f95cb17582a88ae8895`.

This is not recorded as a pass. It narrows the fault: the camera-free runtime can execute physics,
while V4 crashed during RTX renderer initialization. The production SONIC evaluator writes its
success manifest and then calls `os._exit(0)`, so its registered success contract does not depend
on `simulation_app.close()`. The next repair is a trajectory-only SONIC mode that retains physics
and the 50 Hz trajectory recorder while removing the unnecessary Q3 camera path.

No motion was evaluated by this diagnostic, so it contributes no scientific sample.
