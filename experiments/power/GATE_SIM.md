# Registered-gate simulation for nav8192-confirm-v1

**Simulated statistics [S]; no physics was run.** September 23, 2026. This is roadmap Phase 0.1, item "simulate the actual gate on CPU and size Stage 2".

- Script: `experiments/power/gate_cmh.py`, RNG seed 20260923.
- Gate code: `registered_gate` in `scripts/experiments/eval_stats.py`, the same function that will read out Stage 1 and Stage 2.
- Full table: `workspace/phase0/stats/gate_sim.json` (gitignored). It records the sha256 of both scripts and of `registration.json`.

## Bottom line

- **Recommended Stage 2 size: 12 seeds in total, so Stage 2 adds 9 (92604 onward).**
  - If approach-c2's true held-out rate is 6/19 (uniform-c2 2/19, nav-v2-recovery 4/19), the base model gives P(confirm) = **0.847** (Monte Carlo SE 0.003).
  - Across the sensitivity grid, 12 seeds give 0.80–0.89. In the σ_task = 1.0, σ_inst = 0.5 cell the value is just under 0.80.
  - 15 seeds reach ≥ 0.86 in every cell.
- **The registered seed block caps the total at 9 seeds.** `experiments/seeds.yaml` sets `confirmation_nav8192: [92601, 92609]`, and 92610 onward is the dev block.
  - At 9 seeds, P(confirm | 6/19) = 0.74 (0.69–0.80 across the grid).
  - Running 12 or 15 seeds needs a dated amendment before Stage 2 launches, taking the extra seeds from an unallocated range. For example, 92800–92899 is not allocated in `seeds.yaml` [P]. **This is a decision for the user.**
- **False confirmation is rare.** In the null (approach = recovery = 4/19, uniform 2/19), P(confirm) is 0.3% at 12 seeds.
  - It is ≤ 1.0% in every cell and at every seed count, against the registered α = 0.025.
  - With the arm×task interaction switched off (the sharp null), it is also 0.3%.
  - The mean ≥ 5/19 clause and the futility rule do most of this work. In the base model, the CMH test against recovery on its own rejects in 0.7–1.3% of null runs, so the test is conservative, including under the interaction null.
- **Futility stop at Stage 1**, by true approach rate (range across the grid):

  | True rate | 7/19 | 6/19 | 5/19 | 4/19 | 3/19 |
  |---|---|---|---|---|---|
  | P(futility stop) | 0.6% (0.2–1.5) | 5.5% (3.0–6.9) | 23.5% (19.6–24.5) | 57% (56–60) | 88% (87–92) |

  At the plausible rates, almost all stops come from the discordance clause (wins over recovery ≤ losses). At 6/19 the mean clause alone fires in 0.1% of runs.
- **The gate tests "true rate above 5/19", not "at least 5/19".** At a true rate of exactly 5/19, P(confirm) stays ≤ 0.38 even at 20 seeds, because the mean ≥ 5/19 clause passes only about 53% of the time.
- **At the in-sample 7/19**, which is probably optimistic, 8 seeds already give 0.97 (0.94–0.98 across the grid).

## Model [A]

Every parameter is an assumption. This extends `sim_indep.py` to three arms.

| Component | Setting |
|---|---|
| Tasks | 19 feasible tasks. The 6 long tasks fail with probability exactly 0 in every arm |
| Success probability | expit(a_k + u_t + w_kt + v_ts) on the 13 short tasks (arm k, task t, physics seed s) |
| Task effect u_t | N(0, σ_task), shared by all arms. Base σ_task = 1.5; sensitivity {1, 2} |
| Arm×task interaction w_kt | N(0, σ_int/√2) per arm, so any two arms differ with SD σ_int = 0.5 (the `sim_indep.py` convention). The sharp null uses σ_int = 0 |
| Instance effect v_ts | N(0, σ_inst), shared by all arms at a (task, seed): same seed, same physics draw. Base σ_inst = 1.0; sensitivity {0.5, 1, 2} |
| Outcomes | Independent Bernoulli given (u, w, v) |
| Calibration | In each replicate, a_k is set so that expected successes per panel equal the scenario rate × 19. The instance effect is integrated out with 40-node Gauss–Hermite quadrature. `sim_indep.py` ignored v here |
| Seeds | Nested: the S-seed gate uses the first S seeds, and seeds 1–3 are Stage 1 |
| Replicates | 20,000 per base cell and 8,000 per sensitivity cell. Monte Carlo SE ≤ 0.004 (base) and ≤ 0.006 (sensitivity) |

**Rules, as registered.**
- **Futility** after 3 seeds: stop if approach's mean is ≤ 3/19, or if its discordant wins over recovery are ≤ its losses.
- **Gate at S seeds:** the one-sided exact task-stratified CMH test rejects against both uniform-c2 and nav-v2-recovery after Holm at α = 0.025, and approach's mean is ≥ 5/19.
- P(confirm) = P(not futile and gate passes). It assumes the Phase 0.4 goal-use ablation passes.

## Results: base model (σ_task 1.5, σ_inst 1.0, σ_int 0.5)

P(confirm) in %, futility included. Uniform is 2/19 and recovery 4/19 in every row.

| Scenario | P(futility stop) | S=3 | S=5 | S=8 | S=9 | S=10 | S=12 | S=15 | S=20 |
|---|---|---|---|---|---|---|---|---|---|
| approach 3/19 | 88.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| approach 4/19 | 56.9 | 0.6 | 0.7 | 0.5 | 0.4 | 0.3 | 0.3 | 0.2 | 0.1 |
| approach 5/19 | 23.5 | 5.6 | 10.8 | 17.0 | 19.0 | 20.8 | 24.2 | 28.1 | 33.2 |
| **approach 6/19** | 5.5 | 24.7 | 46.1 | 68.9 | 74.4 | 78.6 | **84.7** | 90.0 | 93.4 |
| approach 7/19 | 0.6 | 55.8 | 83.7 | 96.7 | 97.8 | 98.6 | 99.2 | 99.4 | 99.4 |
| null: approach = recovery = 4/19 | 57.2 | 0.5 | 0.6 | 0.4 | 0.4 | 0.3 | 0.3 | 0.2 | 0.1 |
| sharp null (σ_int = 0) | 57.2 | 0.6 | 0.7 | 0.5 | 0.4 | 0.4 | 0.3 | 0.2 | 0.1 |

"approach 4/19" and "null" define the same model with independent RNG streams. They agree within Monte Carlo error, which serves as a consistency check.

**Where the 6/19 power goes**, in %:

| Quantity | S=3 | S=5 | S=8 | S=9 | S=10 | S=12 | S=15 | S=20 |
|---|---|---|---|---|---|---|---|---|
| P(confirm), futility included | 24.7 | 46.1 | 68.9 | 74.4 | 78.6 | 84.7 | 90.0 | 93.4 |
| Gate passes, futility ignored | 24.7 | 46.1 | 69.8 | 75.9 | 80.7 | 87.7 | 94.2 | 98.5 |
| Both CMH tests pass after Holm | 24.7 | 46.3 | 70.1 | 76.1 | 80.9 | 88.0 | 94.4 | 98.6 |
| CMH vs recovery, p ≤ 0.025 | 25.5 | 46.4 | 70.1 | 76.1 | 80.9 | 88.0 | 94.4 | 98.6 |
| CMH vs uniform, p ≤ 0.025 | 89.3 | 99.1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Approach mean ≥ 5/19 | 90.3 | 94.1 | 97.3 | 98.0 | 98.4 | 98.9 | 99.5 | 99.9 |
| Same gate with pooled McNemar (not registered) | 28.1 | 50.1 | 72.5 | 77.7 | 81.8 | 86.9 | 91.2 | 93.8 |

What this shows:
- The binding comparison is against nav-v2-recovery.
- Holm costs almost nothing: the uniform p-value is ≤ 0.0125 in nearly every run where the recovery test passes.
- From 12 seeds on, the futility rule costs about 3–5 points. Those are runs where 3 unlucky seeds stop a true 6/19 policy.

## Sensitivity (σ_int = 0.5; 8,000 replicates per cell except the base row)

P(confirm) at a true approach rate of 6/19, in %:

| σ_task | σ_inst | P(futility stop) | S=3 | S=5 | S=8 | S=9 | S=10 | S=12 | S=15 | S=20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 1.0 (base) | 5.5 | 24.7 | 46.1 | 68.9 | 74.4 | 78.6 | 84.7 | 90.0 | 93.4 |
| 1.0 | 0.5 | 6.9 | 22.9 | 41.6 | 64.1 | 68.7 | 73.8 | 80.0* | 86.2 | 91.0 |
| 1.0 | 1.0 | 6.8 | 22.1 | 41.1 | 63.3 | 68.9 | 72.9 | 80.4 | 86.7 | 91.1 |
| 1.0 | 2.0 | 4.4 | 19.1 | 38.4 | 63.3 | 69.9 | 75.5 | 83.3 | 89.7 | 94.1 |
| 2.0 | 0.5 | 5.0 | 29.3 | 53.0 | 74.9 | 79.6 | 83.4 | 88.3 | 92.4 | 94.4 |
| 2.0 | 1.0 | 5.0 | 27.7 | 50.6 | 74.4 | 79.5 | 83.1 | 88.3 | 92.2 | 94.5 |
| 2.0 | 2.0 | 3.0 | 22.7 | 45.7 | 71.4 | 77.5 | 82.4 | 89.1 | 94.0 | 96.4 |

\* Just under 0.80, which is why the smallest total that reaches 0.80 in this cell is 15.

P(false confirm) in the null (approach = recovery = 4/19), in %:

| σ_task | σ_inst | P(futility stop) | S=3 | S=5 | S=8 | S=9 | S=10 | S=12 | S=15 | S=20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 1.0 (base) | 57.2 | 0.5 | 0.6 | 0.4 | 0.4 | 0.3 | 0.3 | 0.2 | 0.1 |
| 1.0 | 0.5 | 56.8 | 0.9 | 1.0 | 0.8 | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 |
| 1.0 | 1.0 | 57.4 | 0.4 | 0.6 | 0.5 | 0.5 | 0.5 | 0.3 | 0.2 | 0.1 |
| 1.0 | 2.0 | 58.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.1 | 0.1 |
| 2.0 | 0.5 | 56.8 | 0.9 | 1.0 | 0.4 | 0.5 | 0.3 | 0.2 | 0.1 | 0.0 |
| 2.0 | 1.0 | 56.7 | 0.6 | 0.6 | 0.4 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 |
| 2.0 | 2.0 | 59.8 | 0.2 | 0.2 | 0.1 | 0.2 | 0.2 | 0.1 | 0.1 | 0.0 |

The JSON also holds the 7/19 and 5/19 sensitivity rows:
- At 7/19, 8 seeds give 93.7–98.3%.
- At 5/19, 12 seeds give 18.3–28.4%, and 20 seeds give 29.0–37.5%.

## Notes and caveats

- **CMH versus McNemar.** The registered CMH test ignores which seed an outcome came from, so it cannot use the instance effect the arms share. A pooled McNemar test inside the same gate would gain:
  - 0–4 points in the base model;
  - up to 17 points (at 5 seeds) when σ_inst = 2.

  In this model the McNemar variant's false-confirmation rate is also ≤ 1.0%. The registered test stays CMH; this is a note for future registrations.
- **The rates are assumptions.** 7/19 is in-sample. nav-v2-recovery scored 4/19 in-sample at 91260 but 1/19 out-of-sample at 91262 [M]. If recovery's true held-out rate is below 4/19, power goes up.
- **The binary model is a stand-in.** Rollouts are bit-deterministic given the seed, so the Bernoulli term stands for an arm's idiosyncratic response to a physics draw. The variance components are not estimated. Stage 1 data could estimate σ_inst once it is complete.
- **Stage 2 cost** [C] (roadmap figures: best case about 11 minutes per 19-task panel, typical 40–60 minutes). Every panel covers 3 arms × 19 tasks per new seed.

  | Total seeds | New seeds | Panels | Best case | Typical, serial |
  |---|---|---|---|---|
  | 9 | 6 | 18 | about 3.4 h | 12–18 h |
  | 12 | 9 | 27 | about 5 h | 18–27 h |
  | 15 | 12 | 36 | about 6.8 h | 24–36 h |

  Two chains roughly halve the wall time.
- **Sanity check [M].** `eval_stats.py` reproduces the exact two-sided McNemar p = 1/8 = 0.125 for approach-c2 against uniform-c2 at 91260 (6 vs 1 discordant tasks). With one seed, the one-sided CMH p equals the one-sided McNemar p (1/16), as it should. The loader's checkpoint sha256s match `registration.json` for all three arms. Outputs: `workspace/phase0/stats/sanity_91260_*.json` (in-sample diagnostic).
- **No interim look.** None of this reads the live `workspace/confirm-v1/` panels.

## Reproduce

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= .venv_native/bin/python experiments/power/gate_cmh.py \
    --reps 20000 --sens-reps 8000 --workers 4 --out workspace/phase0/stats/gate_sim.json   # about 2 min
# Stage readout (after Stage 1 completes; S = all seeds present):
.venv/bin/python scripts/experiments/eval_stats.py gate \
    --registration experiments/nav8192-confirm-v1/registration.json --stage 1 \
    --approach workspace/confirm-v1/eval/approach-c2-9260{1,2,3} \
    --uniform  workspace/confirm-v1/eval/uniform-c2-9260{1,2,3} \
    --recovery workspace/confirm-v1/eval/recovery-9260{1,2,3}
```
