# Native runtime review snapshot and proposed enforcement

`m2s-native-runtime-freeze-v2` is a CPU review snapshot. It preserves v1 and does not adopt an evaluation protocol or change the running 35-branch development batch. Its 222 additional artifact identities include 60 native assets, 125 configuration files, and the additional dynamic Python sources. The environment receipt now records distribution `direct_url.json`, six editable source locations, their Git commits, and hashes of staged, unstaged, and nonignored untracked source modifications. Deleted files remain explicit. The local gear package has 41 changed source files in this snapshot; the five editable IsaacLab packages have no changes within their declared source subdirectories.

The installed USD libraries were used in a separate CPU process to traverse the converted robot's composed dependencies. All five required USD layers resolve and are already in the cache snapshot. Any unresolved non-MDL dependency is rejected. `OmniPBR.mdl` remains an external renderer dependency and is explicitly outside the physical-geometry and ideal PhysX range/contact reproduction claim. Material rendering, ignored source files, and binary-identical operating-system reproduction are not claimed.

The snapshot does **not** enforce the runtime environment. Before primary execution, the smallest collector change would be:

1. Bind the exact `environment.json` artifact explicitly alongside `runtime_assets.json`; do not infer an environment declaration from a filename or copy expected settings from the evaluation protocol.
2. Immediately before each new rollout, resolve the launching process's effective `ISAACLAB_USD_CACHE_DIR`, using the runtime's existing default `~/.cache/isaaclab/usd`, and require its canonical path to equal the recorded `native_usd_cache` parent. Unset and explicitly equivalent values are the same effective configuration. The current SHA-bound shell runner inherits this value and does not override it.
3. Recheck the declared native/cache artifact hashes before each rollout. Recompute the editable-source identity from the declared source directory and compare commit, changed paths, deletions, and content hashes with the receipt. This catches a previously clean source becoming dirty after the snapshot. A package-version string alone cannot establish editable-source identity.
4. Store the actual effective-cache check and source-identity check in the attempt receipt. A mismatch is an unlaunched configuration failure; it is never a humanoid traversal failure or an invented teacher label.

This is a proposed preflight hook, not an implemented collector gate. A final snapshot should be created after learner/runtime editing ends and before primary registration. The current collection's source freeze stays unchanged.

Focused tests cover missing source meshes, XML includes, external composed USD geometry, unresolved non-MDL dependencies, explicit MDL exclusions, actual root-layer identity, editable commit/deletion/modification hashes, and direct-URL targets distinct from metadata locations.
