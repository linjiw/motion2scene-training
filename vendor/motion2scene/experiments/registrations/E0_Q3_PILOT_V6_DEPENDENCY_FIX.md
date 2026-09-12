# E0-Q3 Pilot V6 Dependency Fix

Registered: 2026-09-04T13:27:47-04:00, before V6 execution.

V5 confirmed that trajectory-only Isaac startup bypasses the V4 RTX-renderer crash. It then
stopped during policy construction with `ModuleNotFoundError: tensordict`, before environment
creation, trajectory capture, or any scientific motion outcome.

The established Isaac environment on this machine contains `tensordict==0.14.0`. That version is
now installed in the isolated repository environment, declared in `gear_sonic[training]`, and
checked by `check_environment.py --training`. The full training preflight passes, including a real
CUDA check and all declared imports.

V6 changes only the source/environment provenance and output lineage. It preserves V5's
trajectory-only capture, the V2 predictions, all six motions, simulator seeds, controller,
acceptance policy, 6,000 MiB gate, serial rule, and cost ceiling.
