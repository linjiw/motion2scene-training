# E0-Q3 Pilot V5 Trajectory-Only Design Delta

Registered: 2026-09-04T13:24:30-04:00, before V5 execution.

V4 entered Isaac Sim but crashed in `librtx.scenedb.plugin.so` while bringing up the camera-enabled
headless-rendering experience. It produced no trajectory or scientific outcome. A separately
registered diagnostic then initialized the camera-free experience, reset the simulation, and took
ten steps. Its strict clean-exit prediction missed because `simulation_app.close()` hung, and that
miss remains reported.

E0-Q3 asks whether SONIC can track the motions and preserve trajectory semantics. It does not test
visual perception and its predictions require only the 50 Hz trajectory. V5 therefore disables
ego rendering but preserves every scientific variable: motions, controller, seeds, plane scene,
prompts, acceptance policy, and registered predictions. The SONIC evaluator's normal success path
writes the manifest and exits with `os._exit(0)`, bypassing the diagnostic's close hang.

This change reduces memory pressure but does not justify lowering the 6,000 MiB launch floor. V5
remains serial and stops on its first infrastructure failure.
