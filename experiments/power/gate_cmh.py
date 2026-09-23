"""Simulate the registered nav8192-confirm-v1 gate (roadmap Phase 0.1) on CPU.

Model (extends sim_indep.py to three arms; every parameter is an assumption):
- 19 feasible tasks; 6 long tasks fail with probability exactly 0 for every arm.
- Short task t, arm k, physics seed s: P = expit(a_k + u_t + w_kt + v_ts).
  u_t ~ N(0, sig_task) shared by all arms; w_kt ~ N(0, sig_int / sqrt(2)) per arm, so any
  two arms' task interaction differs with SD sig_int (the sim_indep convention);
  v_ts ~ N(0, sig_inst) shared by all arms at a seed (same seed = same physics draw).
  Outcomes are independent Bernoulli given (u, w, v).
- Per replicate, a_k is calibrated so that the expected successes per 19-task panel,
  averaged over instance effects (Gauss-Hermite), equal the scenario's true rate exactly.
  sim_indep.py calibrated without v; this version integrates it out.
- Seeds are nested: the S-seed gate uses the first S seeds, and Stage 1 is seeds 1-3.

Rules (as registered in experiments/nav8192-confirm-v1/registration.json):
- Futility after Stage 1 (3 seeds): stop if approach mean <= 3/19, or if approach's
  discordant wins over nav-v2-recovery do not exceed its losses.
- Gate at S total seeds: the one-sided exact task-stratified CMH test (approach > comparator)
  rejects against BOTH uniform-c2 and nav-v2-recovery after Holm at alpha = 0.025, AND the
  approach mean is >= 5/19. The gate code is eval_stats.registered_gate (fast path).
- P(confirm) = P(not futile AND gate passes at S). The Phase 0.4 goal-use ablation, which
  also conditions Stage 2, is assumed to pass.
- For comparison only: the same gate with pooled one-sided exact McNemar tests.

Usage: gate_cmh.py [--reps N] [--sens-reps N] [--workers 4] [--out gate_sim.json]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from numpy.polynomial.hermite_e import hermegauss  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import expit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import eval_stats  # noqa: E402

T, N_LONG = 19, 6
S_GRID = (3, 5, 8, 9, 10, 12, 15, 20)
STAGE1 = 3
ALPHA, MIN_MEAN, FUTILITY_MEAN = 0.025, 5.0, 3.0
BASE = dict(sig_task=1.5, sig_inst=1.0, sig_int=0.5)
SENSITIVITY = [
    dict(sig_task=st, sig_inst=si, sig_int=0.5) for st in (1.0, 2.0) for si in (0.5, 1.0, 2.0)
]
SCENARIOS = [
    *(
        {"name": f"approach_{k}", "approach": k / T, "uniform": 2 / T, "recovery": 4 / T}
        for k in (3, 4, 5, 6, 7)
    ),
    {"name": "null_4_4", "approach": 4 / T, "uniform": 2 / T, "recovery": 4 / T},
    {
        "name": "sharp_null_4_4",
        "approach": 4 / T,
        "uniform": 2 / T,
        "recovery": 4 / T,
        "sig_int": 0.0,
    },
]
GH_X, GH_W = hermegauss(40)
GH_W = GH_W / GH_W.sum()  # E[f(Z)] for Z ~ N(0, 1)
_MCN_TAIL = {}


def calibrate(target, eff, sig_inst):
    """Intercept a: sum over short tasks of E_v[expit(a + eff_t + v)] = target * T."""
    if not 0 < target * T < len(eff):
        raise ValueError("target outside what the short tasks can reach")
    z = sig_inst * GH_X[None, :]

    def f(a):
        return float((expit(a + eff[:, None] + z) @ GH_W).sum()) - target * T

    return brentq(f, -40, 40, xtol=1e-10)


def simulate_rep(rng, rates, *, sig_task, sig_inst, sig_int, s_max):
    """One replicate: dict arm -> T x s_max outcome matrix (long tasks are rows 0..5)."""
    n_short = T - N_LONG
    u = rng.normal(0.0, sig_task, n_short)
    w = rng.normal(0.0, sig_int / math.sqrt(2), (3, n_short))
    v = rng.normal(0.0, sig_inst, (n_short, s_max))
    out = {}
    for k, arm in enumerate(("approach", "uniform", "recovery")):
        eff = u + w[k]
        a = calibrate(rates[arm], eff, sig_inst)
        p = expit(a + eff[:, None] + v)
        y = (rng.random((n_short, s_max)) < p).astype(float)
        out[arm] = np.vstack([np.zeros((N_LONG, s_max)), y])
    return out


def mcnemar_greater_float(b, c):
    n = b + c
    if n not in _MCN_TAIL:
        tot = 2.0**n
        tail, acc = [0.0] * (n + 2), 0
        for i in range(n, -1, -1):
            acc += math.comb(n, i)
            tail[i] = acc / tot
        _MCN_TAIL[n] = tail
    return min(1.0, _MCN_TAIL[n][b])


def mcnemar_gate(app, uni, rec):
    p = [mcnemar_greater_float(*eval_stats.discordant(app, y)) for y in (uni, rec)]
    tests = all(eval_stats.holm(p, ALPHA)["reject"])
    return tests and eval_stats.arm_mean(app) >= MIN_MEAN


def run_config(job):
    """All S for one (variance config, scenario) cell; returns counts."""
    cfg, scen, reps, seed = job["cfg"], job["scenario"], job["reps"], job["seed"]
    params = dict(cfg)
    if "sig_int" in scen:
        params["sig_int"] = scen["sig_int"]
    rng = np.random.default_rng(seed)
    s_max = max(S_GRID)
    n = {"futility": 0, "fut_mean": 0, "fut_disc": 0}
    per_s = {
        s: dict(confirm=0, gate=0, tests=0, mean=0, p_uni=0, p_rec=0, mcn_confirm=0) for s in S_GRID
    }
    for _ in range(reps):
        y = simulate_rep(rng, scen, s_max=s_max, **params)
        fut = eval_stats.futility(
            y["approach"], y["recovery"], max_mean=FUTILITY_MEAN, seeds=STAGE1
        )
        n["futility"] += fut["stop"]
        n["fut_mean"] += fut["approach_mean"] <= FUTILITY_MEAN
        n["fut_disc"] += fut["wins"] <= fut["losses"]
        for s in S_GRID:
            app, uni, rec = (y[k][:, :s] for k in ("approach", "uniform", "recovery"))
            g = eval_stats.registered_gate(app, uni, rec, alpha=ALPHA, min_mean=MIN_MEAN, fast=True)
            r = per_s[s]
            r["gate"] += g["confirmed"]
            r["confirm"] += g["confirmed"] and not fut["stop"]
            r["tests"] += g["tests_pass"]
            r["mean"] += g["mean_pass"]
            r["p_uni"] += g["p_vs_uniform"] <= ALPHA
            r["p_rec"] += g["p_vs_recovery"] <= ALPHA
            r["mcn_confirm"] += mcnemar_gate(app, uni, rec) and not fut["stop"]
    return {
        "cfg": cfg,
        "effective": params,
        "scenario": scen,
        "reps": reps,
        "seed": seed,
        "counts": n,
        "per_s": per_s,
    }


def _rate(k, reps):
    p = k / reps
    return {"p": round(p, 4), "mc_se": round(math.sqrt(p * (1 - p) / reps), 4)}


def summarize(raw):
    reps = raw["reps"]
    c = raw["counts"]
    out = {
        "variance": raw["cfg"],
        "effective_variance": raw["effective"],
        "scenario": {
            k: (round(v * T, 3) if isinstance(v, float) and k != "sig_int" else v)
            for k, v in raw["scenario"].items()
        },
        "reps": reps,
        "rng_seed": raw["seed"],
        "p_futility_stop": _rate(c["futility"], reps),
        "p_futility_mean_rule": _rate(c["fut_mean"], reps),
        "p_futility_discordance_rule": _rate(c["fut_disc"], reps),
        "by_total_seeds": {},
    }
    for s, r in raw["per_s"].items():
        out["by_total_seeds"][str(s)] = {
            "p_confirm": _rate(r["confirm"], reps),
            "p_gate_ignoring_futility": _rate(r["gate"], reps),
            "p_holm_tests_pass": _rate(r["tests"], reps),
            "p_mean_ge_5": _rate(r["mean"], reps),
            "p_cmh_vs_uniform_le_alpha": _rate(r["p_uni"], reps),
            "p_cmh_vs_recovery_le_alpha": _rate(r["p_rec"], reps),
            "p_confirm_mcnemar_variant": _rate(r["mcn_confirm"], reps),
            "p_confirm_given_not_futile": (
                round(r["confirm"] / (reps - c["futility"]), 4) if reps > c["futility"] else None
            ),
        }
    return out


def recommend(results):
    def cell(cfg, name):
        return next(r for r in results if r["variance"] == cfg and r["scenario"]["name"] == name)

    def smallest(r, key="p_confirm", thr=0.8):
        for s in S_GRID:
            if r["by_total_seeds"][str(s)][key]["p"] >= thr:
                return s
        return None

    base6 = cell(BASE, "approach_6")
    per_cfg = {
        json.dumps(r["variance"]): smallest(r)
        for r in results
        if r["scenario"]["name"] == "approach_6"
    }
    s_rec = smallest(base6)
    null = cell(BASE, "null_4_4")
    sharp = cell(BASE, "sharp_null_4_4")
    at = str(s_rec) if s_rec else str(max(S_GRID))
    null_max = max(
        r["by_total_seeds"][at]["p_confirm"]["p"]
        for r in results
        if r["scenario"]["name"] == "null_4_4"
    )
    return {
        "criterion": "smallest total S in the grid with P(confirm) >= 0.80 when approach's "
        "true rate is 6/19 (uniform 2/19, recovery 4/19), base variance config",
        "recommended_total_seeds": s_rec,
        "stage2_new_seeds": (s_rec - STAGE1) if s_rec else None,
        "smallest_S_reaching_0.8_per_variance_config": per_cfg,
        "p_confirm_at_recommended_S_base": base6["by_total_seeds"][at]["p_confirm"],
        "p_false_confirm_null_base": null["by_total_seeds"][at]["p_confirm"],
        "p_false_confirm_sharp_null_base": sharp["by_total_seeds"][at]["p_confirm"],
        "p_false_confirm_null_max_over_configs": null_max,
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reps", type=int, default=10000, help="replicates per base-config cell")
    p.add_argument("--sens-reps", type=int, default=4000, help="replicates per sensitivity cell")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    cfgs = [(BASE, a.reps)] + [(c, a.sens_reps) for c in SENSITIVITY]
    seeds = np.random.SeedSequence(a.seed).spawn(len(cfgs) * len(SCENARIOS))
    jobs = []
    for i, (cfg, reps) in enumerate(cfgs):
        for j, scen in enumerate(SCENARIOS):
            ss = seeds[i * len(SCENARIOS) + j]
            jobs.append(dict(cfg=cfg, scenario=scen, reps=reps, seed=int(ss.generate_state(1)[0])))
    t0 = time.time()
    with Pool(min(a.workers, 4)) as pool:
        raw = pool.map(run_config, jobs)
    results = [summarize(r) for r in raw]
    reg = ROOT / "experiments" / "nav8192-confirm-v1" / "registration.json"
    doc = {
        "schema": "nav8192_confirm_v1_gate_sim_v1",
        "kind": "simulated statistics [S]; no physics was run",
        "script": {"path": "experiments/power/gate_cmh.py", "sha256": sha(__file__)},
        "eval_stats": {
            "path": "scripts/experiments/eval_stats.py",
            "sha256": sha(eval_stats.__file__),
        },
        "registration": {"path": str(reg.relative_to(ROOT)), "sha256": sha(reg)}
        if reg.exists()
        else None,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": sys.version.split()[0],
        },
        "model": {
            "tasks": T,
            "long_tasks_fixed_at_failure": N_LONG,
            "linear_predictor": "a_k + u_t + w_kt + v_ts on the 13 short tasks",
            "u_task": "N(0, sig_task), shared across arms",
            "w_arm_task": "N(0, sig_int/sqrt(2)) per arm; pairwise difference SD = sig_int",
            "v_instance": "N(0, sig_inst), shared across arms at a (task, seed)",
            "calibration": "per replicate, expected successes per panel (integrating v by "
            "40-node Gauss-Hermite) equal the scenario rate x 19",
            "seeds_nested": True,
            "base_variance": BASE,
            "sensitivity_variance": SENSITIVITY,
        },
        "rules": {
            "stage1_seeds": STAGE1,
            "futility": f"approach mean <= {FUTILITY_MEAN}/19 over {STAGE1} seeds OR discordant "
            "wins over recovery <= losses",
            "gate": f"one-sided exact task-stratified CMH vs uniform and vs recovery, Holm at "
            f"alpha={ALPHA}, both rejected, AND approach mean >= {MIN_MEAN}/19",
            "p_confirm": "P(not futile AND gate passes at total S); Phase 0.4 assumed to pass",
            "mcnemar_variant": "same gate with pooled one-sided exact McNemar (comparison only)",
        },
        "total_seed_grid": list(S_GRID),
        "results": results,
        "recommendation": recommend(results),
        "runtime_s": round(time.time() - t0, 1),
    }
    text = json.dumps(doc, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(json.dumps(doc["recommendation"], indent=1))
    print(f"runtime {doc['runtime_s']} s")


if __name__ == "__main__":
    main()
