# E0 Q3 eligible-expansion V2 memory amendment

Registered before retrying any V1 infrastructure-only cell.

V1 produced one accepted scientific outcome (walk-right 002), then stopped when straight-duck 003
failed during cuBLAS initialization. The failed cell emitted no success marker and has no
scientific outcome; cells 015 and 017 never started.

V2 contains only those three unmeasured cells and writes to fresh output directories. Motions,
seeds, scene, controller, recorder, scoring policy, aggregate scientific predictions, and serial
scheduling are unchanged. The only execution-policy change is raising the free-VRAM launch floor
from 6,000 to 7,500 MiB, based on the observed contrast:

- completed cells launched with approximately 7,700–8,100 MiB free;
- the failed cell launched with 6,825 MiB free while another process was allocating;
- the trajectory-only workload itself has remained below the earlier measured 4.9 GiB envelope.

This does not claim that 7,500 MiB reserves the GPU or makes concurrent jobs safe. It is a
conservative shared-machine launch guard below 9,000 MiB. If memory is below the floor, V2 yields;
if another allocation still causes an infrastructure failure, V2 stops without further retry.

The original aggregate prediction remains: at least three of the four expansion motions are
accepted when the one valid V1 result and three V2 cells are combined. No new cell-level outcome
prediction is introduced after observing V1.
