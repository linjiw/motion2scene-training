# E0 Q3 eligible-expansion V1 result

Status: **stopped on infrastructure failure after 1/4 scientific cells**.

The first cell, walk-right 002, completed and was accepted with zero external force, 0.082 m
endpoint error, and 0.093 m schedule error p95. The recorded phase share of schedule error was
0.0368. It is valid Q3 single-seed evidence.

The runner then yielded when free VRAM fell to 4,254 MiB. After free VRAM returned above the
registered 6,000 MiB gate, it resumed. The straight-duck 003 cell began with 6,825 MiB free, but an
unrelated job allocated memory during startup and cuBLAS failed to create its handle. The cell has
no success marker or gradeable trajectory and therefore has **no scientific outcome**. Per the
manifest stop rule, carry-straight 015 and carry-right 017 were not started.

| Cell | Status | Scientific outcome |
| --- | --- | --- |
| walk-right 002 | completed | accepted |
| duck-straight 003 | infrastructure failure | not measured |
| carry-straight 015 | not started | not measured |
| carry-right 017 | not started | not measured |

The aggregate 3/4 acceptance prediction cannot yet be adjudicated. The all-cells-complete resource
prediction was interrupted by shared-GPU contention, not falsified by a scientific motion result.

The memory lesson is narrower than “9,000 MiB required.” Completed trajectory-only cells still use
less than that and have launched successfully around 7,700–8,100 MiB free. On this shared machine,
however, 6,000 MiB observed at one instant is insufficient protection when another process can
allocate mid-startup. A retry, if separately registered, should raise the launch floor to 7,500 MiB
and retain serial scheduling; it still cannot guarantee coordination with an uncooperative job.

Frozen run record:
`/home/linjiw/research-data/groot-wbc/cg-wbc-v1-pilot/e0_q3_expansion_v1/run_record.json`

SHA-256: `53bd182d058bd7a7e027e7876cf771cc1fdb8c716076c7a1d135fa60382bf1b1`
