# Power simulations for navigation panel comparisons

CPU-only simulations behind the power figures in [the roadmap §10](../../docs/ROADMAP_20260923.md#10-evaluation-protocol). They are simulated statistics, not measurements.

- `sim_indep.py`: logistic model of 19 paired tasks. It uses task effects, an arm×task interaction and per-instance effects shared across arms (same seed, same physics draw). The six long tasks are fixed at failure. The test is a pooled two-sided exact McNemar. `run(S, pA, pB, ...)` returns rejection rates for S instances per task.
- `gate.py`: sweeps S for 7/19 vs 4/19 and other rate pairs.
- `gate_cmh.py`: simulates the registered nav8192-confirm-v1 gate (Stage-1 futility rule, then the one-sided exact task-stratified CMH test against both comparators with Holm at α = 0.025, plus mean ≥ 5/19) with three arms and calibrated rates. It calls the same `registered_gate` that reads out the real panels (`scripts/experiments/eval_stats.py`). Results and the Stage 2 sizing are in [GATE_SIM.md](GATE_SIM.md).

The variance components are assumptions, not estimates. 7/19 is an in-sample rate. The registered gate (one-sided task-stratified CMH, Holm over two comparators, mean ≥5/19) is simulated in `gate_cmh.py`; see [GATE_SIM.md](GATE_SIM.md). Requires numpy and scipy.
