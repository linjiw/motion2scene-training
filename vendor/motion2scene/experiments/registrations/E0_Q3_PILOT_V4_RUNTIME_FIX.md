# E0-Q3 Pilot V4 Runtime Fix

Registered: 2026-09-04T13:16:14-04:00.

V3 passed manifest preflight and the 6,000 MiB memory gate, then failed in 0.004 seconds because
the manifest driver supplied a stale developer-machine Python path. Isaac did not start, no motion
was evaluated, and no scientific result was observed. The immutable V3 run record is hash-pinned
in the V4 manifest.

The source driver now uses `implementation.python` from the manifest unless an explicit CLI
override is supplied, fails closed if that interpreter is absent, and has three focused tests for
manifest selection, explicit override, and missing-environment behavior.

V4 changes only the driver hash, source commit, cell identifiers, and output lineage. It preserves
the V2 predictions, sample, motions, simulator seeds, checkpoint, verdict policy, 6,000 MiB launch
threshold, serial scheduling rule, and cost ceiling.
