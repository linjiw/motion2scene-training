# Token-space diagnostics for a policy acting in SONIC's FSQ tokens

Roadmap Phase 0.8, run September 23, 2026. **Verdict: the token action space passes the pre-registered viability gate on release SONIC.** The gate was: ±1-bin token noise changes decoded actions by <0.15 RMS. The measured change is 0.095 on the lattice and 0.117 off the lattice.

These are offline properties of the frozen networks on recorded states. No physics was run, so closed-loop behaviour of a token policy is still untested.

## Setup

**Data.** 30 evenly spaced episodes (10,015 rows) from the 8192×500 teacher's collection at seed 91400. That is `workspace/distill-8192/collect/seed-91400/metrics/episode-*.npz`, covering training motions only. Each row contains proprioception (930-D), the teacher's 640-D future reference, and the recorded teacher tokens and actions.

**Networks.** Frozen release SONIC (`workspace/models/sonic_release.pt`, sha `e6bdab…`) and the 8192×500 fine-tune (sha `26ca1ed…`). Each provides:
- the G1 encoder (640→64);
- FSQ with 2×32 dimensions at 32 levels, so one bin is 1/16 in code space and lattice values are k/16 for k ∈ [−16, 15];
- the `g1_dyn` decoder (64 tokens + 930 → 29);
- the `g1_kin` decoder (64 → 640 future reference).

**How release SONIC was evaluated here.** Its tokens were computed by encoding the recorded future references. This needed no GPU dump.

**Parity.** The 8192×500 encoder and decoder reproduce the recorded tokens exactly (maximum difference 0) and the recorded actions to within 1.9e-5. The offline pipeline is therefore faithful to what the simulator executed.

**Units.** Actions are the decoder's raw action means, before the environment's action scaling. For scale: release's natural per-step action change is 0.130 RMS, and its policy std averages 0.38 (range 0.30–0.50).

Code is in `vendor/sonic/gear_sonic/research/goal_teacher/diag_token_space.py`, with tests in `vendor/sonic/decoupled_wbc/tests/test_goal_teacher_token_diag.py`. The full output is at [`evidence/token-diag-20260923/diag-seed91400-30ep.json`](evidence/token-diag-20260923/diag-seed91400-30ep.json). Command:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=vendor/sonic .venv_native/bin/python -m gear_sonic.research.goal_teacher.diag_token_space \
  --shards workspace/distill-8192/collect/seed-91400/metrics/episode-*.npz --max-episodes 30 \
  --release workspace/models/sonic_release.pt --teacher workspace/distill-8192/teacher/model_step_000500.pt \
  --teacher-sha256 26ca1ed70c2287c1acb6243820649db4b8513a5d9b8b16abc115f890d87b6bff --output <out.json>
```

## Results

Values for release SONIC, with the 8192×500 fine-tune in parentheses.

| Probe | Result | Reading |
|---|---|---|
| All 64 dims shifted uniformly by −1, 0 or +1 bin (mean \|shift\| 0.67 bin) | Action change 0.095 RMS (0.072) | Below the 0.15 gate; about one natural control step (0.130) |
| Gaussian noise, σ = 1 bin, left off the lattice | 0.117 (0.088) | The decoder behaves smoothly between lattice points |
| Gaussian noise, σ = 2 bins, and ±2-bin lattice noise | 0.238 and 0.166 (0.178 and 0.125) | Noise of this size is still well below the PPO policy std of 0.38 |
| One dimension shifted by +1 bin | 0.012–0.018 per dimension, mean 0.015 | No single dimension dominates |
| Token change per 50 Hz step (recorded motion) | 20% of dims change; mean 0.20 bin; p95 1 bin; max 3 bins | Tokens move slowly, so an action-rate penalty in token space has a natural scale |
| Levels used per dim | Mean 19/32 (min 14, max 27); range −14…12 | Roughly the outer quarter of the code range is never visited by tracked motion |
| Midpoint blend between tokens 10 / 25 / 50 steps apart | Nonlinearity 0.055 / 0.191 / 0.101, against endpoint gaps of 0.48 / 0.93 / 0.66; snapping the blend to the lattice costs about 0.04 | Interpolation in token space is benign over short gaps |
| A token held for 1 / 2 / 5 / 10 steps while proprioception evolves | Error vs teacher 0.067 / 0.108 / 0.240 / 0.485, against the teacher's own change of 0.130 / 0.225 / 0.451 / 0.687 | The decoder covers about half the change from proprioception alone. At 25 Hz (hold 2) the cost is modest; at 10 Hz it is large |
| Cycle residual (token → `g1_kin` → encoder → token) on recorded tokens | Median 0.11 bin per sample, p99 0.66 | Recorded tokens are near the encoder's image |
| Same, ±1 / ±2 / ±4-bin random perturbations | Above the corpus p99: 6.7% / 98% / 100% | Useful off-manifold detector. It tolerates 1-bin noise and flags ≥2-bin incoherent noise |
| Same, uniformly random lattice tokens | Median 6.7 bins; 100% above p99 | — |
| Release vs 8192×500 actions on the same states | 0.318 RMS | Fine-tuning changed the decoder mapping a lot. Token-space data from one tracker cannot be reused with the other |

In the fine-tune, the `g1_kin` cycle is noisier (median 0.34 bin, p99 0.95). Its `g1_kin` decoder likely drifted from the fine-tuned encoder. Use release for the cycle criterion.

## Consequences for Phase 3 [P]

- **Action parametrization.**
  - Act in bounded code space with linear bins (§7 of the roadmap).
  - An initial policy std of about 1 bin (1/16) adds action noise of about 0.1, which is well inside the decoder's natural range.
  - An std of 2 bins is still moderate.
- **Off-manifold kill criterion.** The per-sample cycle residual above the release corpus p99 (0.66 bin) is the proposed test. It is sensitive to ≥2-bin incoherent noise and tolerant of 1-bin noise. Re-estimate the threshold on the full 89-clip corpus before locking it.
- **Token rate.** Updating tokens at 50 Hz (as SONIC does) is the default. A 25 Hz token policy costs about 0.11 RMS against the teacher's action.
- **Tracker choice.** The 8192×500 checkpoint's decoder differs substantially from release. Phase 3 arms therefore standardize on release, and B3 behaviour-cloning data must be logged with release tokens.

Caveats:
- Training motions only.
- Proprioception comes from 8192×500 rollouts, not release rollouts.
- Sensitivity is measured on recorded states, so it does not show how errors compound in closed loop.
