# Fixed six-context development learner extension

One unchanged phase-conditioned ridge learner was fitted at the previously selected penalty **10.0** to the original five contexts plus the separately audited complementary context. No model or penalty search was performed. All 42 branches were independently re-audited: 50,064 existing physics steps, 18 available neutral-phase packets, 18 complete legal tables, and 17 consequential targets. The early-beam final phase has no successful legal continuation and remains excluded from fitting.

The extended model now waits at 0.30 and 1.00 seconds and chooses sustained adaptation at 1.40 seconds on the complementary passage. It retains the prior response at 1.00 seconds on the early beam. Exact pre-entry matches support a six-of-six recorded-branch proxy, versus five-of-six for the original model on this expanded development panel. These are offline branch proxies, not new policy rollouts.

**Measured cost regressions remain:** the extended model also waits for sustained adaptation on the original short and long beams. Their selected branch times increase from 3.62 to 4.06 seconds and from 4.48 to 4.64 seconds. Empty, early-beam, and course branch choices remain unchanged. The empty-scene model still requests prior adaptation at 1.00 seconds.

## Every recorded neutral-phase readout

Regret uses the measured complete teacher continuation for the selected action. WAIT can refer to a later successful adaptation. The table includes hypothetical neutral phases after an earlier model commitment; those are not actual model visits. An em dash means no consequential target because all legal continuations fail, not zero regret.

| Context | Phase (s) | Original-five action | Teacher regret | Extended-six action | Teacher regret |
| --- | ---: | --- | ---: | --- | ---: |
| Empty | 0.30 | WAIT | 0.000000 | WAIT | 0.000000 |
| Empty | 1.00 | Prior 1.00 | 0.003717 | Prior 1.00 | 0.003717 |
| Empty | 1.40 | Sustained 1.40 | 0.094595 | Sustained 1.40 | 0.094595 |
| Short beam | 0.30 | WAIT | 0.000000 | WAIT | 0.000000 |
| Short beam | 1.00 | Prior 1.00 | 0.000000 | WAIT | 0.103960 |
| Short beam | 1.40 | Sustained 1.40 | 0.004926 | Sustained 1.40 | 0.004926 |
| Long beam | 0.30 | Prior .30 | 0.000000 | WAIT | 0.004310 |
| Long beam | 1.00 | Prior 1.00 | 0.000000 | WAIT | 0.030172 |
| Long beam | 1.40 | Sustained 1.40 | 0.000000 | Sustained 1.40 | 0.000000 |
| Early beam | 0.30 | WAIT | 0.000000 | WAIT | 0.000000 |
| Early beam | 1.00 | Prior 1.00 | 0.000000 | Prior 1.00 | 0.000000 |
| Early beam | 1.40 | Sustained 1.40 | — | Sustained 1.40 | — |
| Two-beam course | 0.30 | WAIT | 0.000000 | WAIT | 0.000000 |
| Two-beam course | 1.00 | Prior 1.00 | 0.000000 | Prior 1.00 | 0.000000 |
| Two-beam course | 1.40 | Sustained 1.40 | 0.084071 | Sustained 1.40 | 0.084071 |
| Complementary | 0.30 | Prior .30 | 1.000000 | WAIT | 0.000000 |
| Complementary | 1.00 | Prior 1.00 | 1.000000 | WAIT | 0.000000 |
| Complementary | 1.40 | Sustained 1.40 | 0.000000 | Sustained 1.40 | 0.000000 |

Across the same 17 consequential recorded phases, mean selected physical teacher regret changes from **0.128665** to **0.019162**. The old model selects a failed continuation at two hypothetical phases of the complementary context; the extended model selects no failed consequential continuation. These are decision-table counts, not two separate failed episodes. The always-fail early-beam final phase remains physically known and unsupervised for both models.

## Immutable artifacts and scope

`/home/linjiw/research-data/groot-wbc/m2s-six-context-development-policy-v1/policy.npz` has SHA-256 `648887bf3534fba205eb97192f79aca8350975b339cac18ff45025f1fd97b3a5`. The prefit plan `m2s-six-context-development-policy-plan-v1/specification.json` (SHA-256 `97f4610a298219080137ab44d337469bf22fe43bdc8c13fe077afa075b052022`) binds the complementary audit, exact six inputs, original model/selection, fixed penalty, and single-trial rule.

`readout_audit.json` has SHA-256 `99d6f5edc9b0acd35f9701bef57a2d4a118f0ecee59841d935f0c101496b65a6`. It retains every phase and first-commit proof, verifies all stored fit choices/regrets against the actual runtime loader, and confirms the original model hash is unchanged. Its audit code and import closure are retained under `readout_source_snapshot`.

The original five-context tuning, selected baselines, model, and registered 20-episode comparison remain unchanged. This is an **in-sample development data-extension check**. It does not establish generalization, an independent curriculum advantage, or actual extended-policy traversal performance. No GPU or new physics was used.
