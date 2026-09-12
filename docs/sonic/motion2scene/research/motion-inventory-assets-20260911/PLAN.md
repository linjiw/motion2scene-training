# Motion inventory, probabilities, and indoor asset expansion

2026-09-11. Read-only inventory and illustrative sampling. No retraining, new geometry labels, physics episodes, or changes to the registered pilot.

The current critical-location model trained on seven executed schedules: neutral walking; early short and sustained adaptations; early and middle prior-splice adaptations; late short and sustained adaptations. Their identifiers and exact counts are in inventory.json. They belong to one walking/crouch bank, not seven independent source skills. Each has 298 recorded frames at 50 Hz. Training uses 1,152 placements per target; the other 384 are coordinate interpolation within the same bank, now already inspected. The renderer's mesh replay does not demonstrate crawling or new skill qualification.

The historical reference gate contains 150 entries over 14 named body modes; 94 are recorded as worth a rollout, not physically qualified. The recorded source directory /data/robotixx/groot-wbc-kimodo-m0/sweepcf_release/motions/clips is absent. Targeted searches in the checkout, research-data, dataset, kimodo, and Downloads did not establish a relocated copy of that named pool. This is not an exhaustive search of every storage device.

A separate local AMASS-derived candidate bank has 900 CSV files and 900 conversion records marked OK. Conversion metadata includes depth-first (mjlab/MuJoCo) order; this must not be assumed to match SONIC's required ordering. Local README says these are licensed source assets with separate project provenance and cannot be published or shared. No raw motion was copied. Successful conversion is not measured SONIC trackability or Motion2Scene eligibility. Counts here are files and records, not independent source-ancestry counts.

Use the broader corpus through an explicit admission ledger: source/license identity; original ancestry and any existing split assignment; joint and quaternion conventions; timebase and units; kinematic/self-intersection checks; existing frozen-tracker evidence or qualification still needed. Group all descendants of a source before allocating training and testing. Keep all failure/exclusion rows. Reference-only motions can supervise a separately identified geometry-pretraining study but cannot be presented as executed-motion evidence. Include feasible background/no-adaptation examples; training only on critical contrasts cannot teach when ordinary walking is appropriate.

Do not put all available motions into training: preserve a source-disjoint test set and avoid reusing another project's held-out motions without an explicit design. Sideways and turning motions also need route-aligned obstacle coordinates and appropriate alternative actions; adding them to the current world-axis overhead grid does not solve those representation constraints. A new training design should compare the same unconditional and motion-conditioned controls with equal data and compute before increasing architecture size.

## Probability to obstacle

placement-probabilities.csv exports all 29,568 model/seed/target/cell records, with both raw and constrained probabilities, target-clearance bounds and geometric critical flags. These are probabilities of selecting a candidate location, not probabilities that the robot will pass.

For a sampled cell (x, h), the existing beam has center (x, 0, h+0.1) m, full dimensions (0.3, 2.0, 0.2) m and zero yaw. h is the underside, not the center height. A categorical draw selects one cell; no interpolation or jitter is introduced because unsampled geometry would require a fresh check. sampled-beams.json contains five reproducible draws with replacement at seed 20260911. All draws are retained, and physical_passage is null. These illustrative draws are not new independent evaluation results.

The first draw selects cell 244 for sustained_e015_r255 under contrast-301: raw probability 0.1112079546, constrained probability 0.1403276275, center (3.2421875, 0, 1.3640624046) m, and lower modeled capsule gap 0.0204518214 m. Its geometric critical label does not establish physical passage.

The existing offline viewer already shows these fields on click, model/seed controls, raw/constrained probabilities, and a sampled beam. It displays all 384 previously inspected candidate cells; no favorable subset is selected.

## Real indoor assets: proposed implementation contract

Isaac Lab supports USD-file spawning using UsdFileCfg; its official tutorial includes a table asset. This establishes an available integration mechanism, not local asset availability or valid colliders for our study. [Official spawning tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/spawn_prims.html).

Factor generation as q(constraint | motion) p(asset, pose | constraint). The learned stage proposes a binding surface or gap. The realization stage chooses an asset whose authored collision geometry satisfies that constraint, then checks the entire placed asset against the motion. A tabletop may realize an overhead underside, but all legs, braces, and neighboring geometry must also be checked. A shelving opening requires lateral as well as overhead clearance. Do not scale or move an arbitrary visual mesh and inherit the beam's label.

First integration should retain the exact checked beam collider with an explicitly marked visualization-only appearance. This tests display, not realistic asset collision behavior. The next separately designed study should use native asset collision geometry: bind the USD and referenced dependencies, units/up-axis, pose/scale, collision approximation, and physical material; align the intended binding face; re-evaluate all geometry and later registered closed-loop labels. Report realized-asset acceptance and rejected fits, and retain a primitive control with the same nominal critical constraint.

The current scene_asset_preflight accepts a narrow axis-aligned Cube/Plane subset and explicitly limits asset provenance. USD references and richer geometry need a separately versioned asset preflight; do not bypass it or insert them into the frozen pilot. No actual indoor USD asset was downloaded, imported, or physically evaluated in this session. The eighteen reserved layouts remain outside this work.

## Receipt and next permitted work

Completed: bound inventory metadata, exported every displayed probability, and sampled five illustrative beam placements. Training and physics steps: zero. Existing physics outcome status unchanged. BLOCKED/UNTESTED: all-motion SONIC qualification, source-disjoint generalization, native indoor-asset physical utility. The next permitted step is an ancestry-aware candidate ledger and separately bounded geometry-pretraining/asset-preflight design. This does not authorize an expansion of the registered physical pilot.
