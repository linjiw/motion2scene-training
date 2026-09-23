"""Paired statistics for navigation stage panels (roadmap §10; Phase 0.1 and 1.7).

Routines:
- ``wilson``: Wilson score interval for one success rate.
- ``mcnemar_exact``: exact McNemar (binomial on discordant pairs), two- or one-sided.
- ``cmh_exact`` / ``cmh_exact_counts``: exact task-stratified CMH test, i.e. the
  conditional test of a common odds ratio of 1. Within each task, arm A and arm B are
  the two rows of a 2x2 table over the S physics seeds; conditioning on each task's
  total successes gives a hypergeometric count per task, and the null distribution of
  the summed arm-A successes is the convolution of those hypergeometrics. p-values are
  exact rationals (integer arithmetic); ``cmh_p_float`` is the fast float path the
  power simulation uses.
- ``cluster_bootstrap_diff``: task-cluster percentile bootstrap for the success-rate
  difference (seeds stay together inside a resampled task).
- ``holm``: Holm step-down adjustment.
- ``load_stage`` / ``outcome_matrix``: read ``run_stage.sh`` outputs.
- ``registered_gate`` / ``futility``: the nav8192-confirm-v1 rules, as registered.

Ideas carried over from vendor/sonic/scripts/research/motion2scene_evaluation_statistics.py:
seeded, reproducible bootstrap draws resampling whole units; unknown outcomes stay
assigned (NaN) and bound claims instead of being imputed. NumPy and the standard library
only; SciPy is used only by the tests.

Usage:
  eval_stats.py summary STAGE_DIR...
  eval_stats.py compare --a DIR... --b DIR... [--tasks-from REG.json | --tasks ID...]
  eval_stats.py gate --registration REG.json --approach DIR... --uniform DIR... \
      --recovery DIR... [--stage 1|2]
"""

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np

DRAWS = 20000
RANDOM_SEED = 202609231
ALTERNATIVES = ("two-sided", "greater", "less")

# ------------------------------------------------------------------ intervals / tests


def wilson(k, n, conf=0.95):
    """Wilson score interval for k successes in n trials."""
    if n < 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if n == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def _check_alternative(alternative):
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")


def mcnemar_exact(b, c, alternative="two-sided", *, as_fraction=False):
    """Exact McNemar test. b = pairs where only A succeeds, c = only B succeeds.

    Under H0 b ~ Binomial(b + c, 1/2). 'greater' tests A > B: P(X >= b).
    Two-sided: min(1, 2 * P(X <= min(b, c))).
    """
    _check_alternative(alternative)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    total = 2**n

    def tail_ge(k):
        return Fraction(sum(math.comb(n, i) for i in range(max(k, 0), n + 1)), total)

    if alternative == "greater":
        p = tail_ge(b)
    elif alternative == "less":
        p = tail_ge(c)
    else:
        p = min(Fraction(1), 2 * tail_ge(max(b, c)))
    return p if as_fraction else float(p)


def _as_pair_arrays(ya, yb):
    ya, yb = np.asarray(ya, dtype=float), np.asarray(yb, dtype=float)
    if ya.ndim == 1:
        ya, yb = ya[:, None], yb[:, None]
    if ya.shape != yb.shape or ya.ndim != 2:
        raise ValueError("outcome matrices must be task x seed and of equal shape")
    for y in (ya, yb):
        if np.any(~(np.isnan(y) | (y == 0) | (y == 1))):
            raise ValueError("outcomes must be 0, 1 or NaN (unknown)")
    known = ~(np.isnan(ya) | np.isnan(yb))
    return ya, yb, known


def discordant(ya, yb):
    """(b, c) over pairwise-complete (task, seed) pairs."""
    ya, yb, known = _as_pair_arrays(ya, yb)
    b = int(((ya == 1) & (yb == 0) & known).sum())
    c = int(((ya == 0) & (yb == 1) & known).sum())
    return b, c


def _hypergeom_weights(n_a, n_b, m):
    """Integer weights C(n_a, x) C(n_b, m - x) for x = lo..hi; they sum to C(n_a + n_b, m)."""
    lo, hi = max(0, m - n_b), min(n_a, m)
    return lo, [math.comb(n_a, x) * math.comb(n_b, m - x) for x in range(lo, hi + 1)]


def _poly_mul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return out


def cmh_exact_counts(a_succ, n_a, b_succ, n_b, alternative="greater", *, as_fraction=False):
    """Exact conditional CMH test from per-stratum counts.

    a_succ[t] of n_a[t] arm-A trials succeed and b_succ[t] of n_b[t] arm-B trials succeed.
    Statistic: T = sum_t a_succ[t]. Null: common odds ratio 1, so given each stratum's
    total successes m_t, a_succ[t] is hypergeometric and T is their convolution.
    'greater' (A > B) -> P(T >= t_obs); two-sided sums outcomes no more likely than the
    observed one (the rule of R's mantelhaen.test(exact=TRUE) and of Fisher's test).
    """
    _check_alternative(alternative)
    a_succ, n_a, b_succ, n_b = (list(map(int, v)) for v in (a_succ, n_a, b_succ, n_b))
    if not len(a_succ) == len(n_a) == len(b_succ) == len(n_b):
        raise ValueError("per-stratum arrays must have equal length")
    poly, offset, denom, t_obs = [1], 0, 1, 0
    informative = 0
    mh_num = mh_den = 0.0
    for a, na, b, nb in zip(a_succ, n_a, b_succ, n_b, strict=True):
        if not (0 <= a <= na and 0 <= b <= nb):
            raise ValueError("successes must lie in [0, trials]")
        if na + nb == 0:
            continue
        m = a + b
        lo, w = _hypergeom_weights(na, nb, m)
        poly = _poly_mul(poly, w)
        offset += lo
        denom *= math.comb(na + nb, m)
        t_obs += a
        informative += len(w) > 1
        n_tot = na + nb
        mh_num += a * (nb - b) / n_tot
        mh_den += (na - a) * b / n_tot
    idx = t_obs - offset
    if alternative == "greater":
        num = sum(poly[idx:])
    elif alternative == "less":
        num = sum(poly[: idx + 1])
    else:
        obs = poly[idx]
        num = sum(w for w in poly if w <= obs)
    p = Fraction(num, denom)
    expected = sum(
        Fraction(na * (a + b), na + nb)
        for a, na, b, nb in zip(a_succ, n_a, b_succ, n_b, strict=True)
        if na + nb
    )
    mh_or = mh_num / mh_den if mh_den > 0 else None  # None: undefined or infinite
    return {
        "p": p if as_fraction else float(p),
        "p_fraction": f"{p.numerator}/{p.denominator}",
        "alternative": alternative,
        "statistic": t_obs,
        "null_expectation": float(expected),
        "support": [offset, offset + len(poly) - 1],
        "informative_strata": informative,
        "strata": len(a_succ),
        "mantel_haenszel_odds_ratio": mh_or,
    }


def cmh_exact(ya, yb, alternative="greater", *, as_fraction=False):
    """Exact task-stratified CMH test on task x seed outcome matrices (NaN = unknown).

    Pairs where either arm is unknown are dropped, so both arms of a stratum use the same
    seeds. The test ignores which seed each outcome came from; with instance effects shared
    across arms it is conservative relative to a test that uses the pairing.
    """
    ya, yb, known = _as_pair_arrays(ya, yb)
    n = known.sum(axis=1).astype(int)
    a = np.where(known, ya, 0).sum(axis=1).astype(int)
    b = np.where(known, yb, 0).sum(axis=1).astype(int)
    out = cmh_exact_counts(a, n, b, n, alternative, as_fraction=as_fraction)
    out["complete_pairs"] = int(known.sum())
    out["unknown_pairs"] = int((~known).sum())
    return out


@lru_cache(maxsize=None)
def _pmf_float(n_a, n_b, m):
    lo, w = _hypergeom_weights(n_a, n_b, m)
    tot = math.comb(n_a + n_b, m)
    return lo, np.array([x / tot for x in w])


def cmh_p_float(a_succ, b_succ, n_a, n_b, alternative="greater"):
    """Float CMH p-value for equal per-stratum sizes n_a, n_b (the simulation path).

    a_succ and b_succ are integer arrays of per-stratum successes. Agrees with
    ``cmh_exact_counts`` to floating-point precision (tested).
    """
    pmf, offset, t_obs = np.ones(1), 0, 0
    for a, b in zip(a_succ, b_succ, strict=True):
        m = int(a) + int(b)
        t_obs += int(a)
        if m == 0 or m == n_a + n_b:  # degenerate stratum: x is 0 or n_a
            offset += n_a if m else 0
            continue
        lo, w = _pmf_float(n_a, n_b, m)
        pmf = np.convolve(pmf, w)
        offset += lo
    idx = t_obs - offset
    if alternative == "greater":
        return float(min(1.0, pmf[idx:].sum()))
    if alternative == "less":
        return float(min(1.0, pmf[: idx + 1].sum()))
    return float(min(1.0, pmf[pmf <= pmf[idx] * (1 + 1e-7)].sum()))


def holm(pvalues, alpha=0.05):
    """Holm step-down: adjusted p-values and reject flags, in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return {"adjusted": adjusted.tolist(), "reject": (adjusted <= alpha).tolist(), "alpha": alpha}


def cluster_bootstrap_diff(ya, yb, *, draws=DRAWS, seed=RANDOM_SEED, conf=0.95):
    """Task-cluster percentile bootstrap of mean(A - B) over tasks.

    Each task contributes the mean paired difference over its pairwise-complete seeds;
    tasks are resampled with replacement and their seeds travel together. Returns the
    difference in per-(task, seed) success probability and the same scaled to the panel
    (x number of tasks, i.e. successes per panel per seed).
    """
    ya, yb, known = _as_pair_arrays(ya, yb)
    keep = known.any(axis=1)
    if not keep.any():
        raise ValueError("no pairwise-complete outcomes")
    diff = np.where(known, ya - yb, 0.0).sum(axis=1)[keep] / known.sum(axis=1)[keep]
    t = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(t, size=(draws, t))
    boot = diff[idx].mean(axis=1)
    q = (1 - conf) / 2
    lo, hi = np.quantile(boot, [q, 1 - q], method="linear")
    return {
        "mean_difference": float(diff.mean()),
        "ci": [float(lo), float(hi)],
        "per_panel_difference": float(diff.mean() * ya.shape[0]),
        "per_panel_ci": [float(lo * ya.shape[0]), float(hi * ya.shape[0])],
        "share_draws_nonpositive": float((boot <= 0).mean()),
        "tasks_used": int(t),
        "draws": draws,
        "seed": seed,
        "confidence": conf,
        "rng": "numpy.random.default_rng / PCG64; tasks resampled, seeds kept together",
    }


# ------------------------------------------------------------------ run_stage.sh loader

RESULT_LINE = re.compile(
    r"^(?P<task>\S+) (?P<res>success|FAIL) hold (?P<hold>-?\d+) reach (?P<reach>True|False)"
    r" force (?P<force>\S+) fell (?P<fell>True|False) steps (?P<steps>-?\d+)"
    r" stop (?P<stop>\S+)$"
)
FAILED_LINE = re.compile(r"^(?P<task>\S+) PROCESS_FAILED exit=(?P<code>-?\d+)$")
HYDRA_SEED = re.compile(r"^\+*seed=(\d+)$")
DIR_SEED = re.compile(r"-(\d{5,6})$")


@dataclass
class Outcome:
    task_id: str
    status: str  # success | fail | infra_failed
    success: bool | None
    source: str
    hold: int | None = None
    reach: bool | None = None
    force: float | None = None
    fell: bool | None = None
    steps: int | None = None
    stop: str | None = None
    infra_failures: int = 0
    student_sha256: str | None = None


@dataclass
class Stage:
    path: str
    seed: int | None
    outcomes: dict = field(default_factory=dict)
    student_sha256: str | None = None
    notes: list = field(default_factory=list)


def parse_results_txt(path):
    """Parse results.txt; the last terminal line per task wins over earlier PROCESS_FAILED.

    Conflicting terminal lines for one task (a policy failure rerun) raise ValueError.
    """
    outcomes = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = RESULT_LINE.match(line)
        if m:
            g = m.groupdict()
            new = Outcome(
                task_id=g["task"],
                status="success" if g["res"] == "success" else "fail",
                success=g["res"] == "success",
                source=f"{path}:{n}",
                hold=int(g["hold"]),
                reach=g["reach"] == "True",
                force=float(g["force"]),
                fell=g["fell"] == "True",
                steps=int(g["steps"]),
                stop=g["stop"],
            )
            old = outcomes.get(new.task_id)
            if old is not None and old.status != "infra_failed":
                if old.success != new.success:
                    raise ValueError(f"conflicting terminal results for {new.task_id} in {path}")
                new.infra_failures = old.infra_failures
            elif old is not None:
                new.infra_failures = old.infra_failures
            outcomes[new.task_id] = new
            continue
        m = FAILED_LINE.match(line)
        if m:
            task = m["task"]
            old = outcomes.get(task)
            if old is None or old.status == "infra_failed":
                count = (old.infra_failures if old else 0) + 1
                outcomes[task] = Outcome(
                    task, "infra_failed", None, f"{path}:{n}", infra_failures=count
                )
            else:  # a terminal result already exists; keep it and count the failure
                old.infra_failures += 1
            continue
        raise ValueError(f"unrecognized line {path}:{n}: {line!r}")
    return outcomes


def _command_seed(task_dir):
    f = task_dir / "command.json"
    if not f.exists():
        return None
    try:
        args = json.loads(f.read_text())
    except json.JSONDecodeError:
        return None
    for x in args if isinstance(args, list) else []:
        m = HYDRA_SEED.match(str(x))
        if m:
            return int(m.group(1))
    return None


def load_stage(stage_dir):
    """Load one run_stage.sh stage directory (one arm at one physics seed).

    task/task-result.json is authoritative when present; results.txt supplies
    PROCESS_FAILED rows and is cross-checked against the JSON.
    """
    stage_dir = Path(stage_dir)
    results = stage_dir / "results.txt"
    outcomes = parse_results_txt(results) if results.exists() else {}
    stage = Stage(path=str(stage_dir), seed=None)
    seeds, shas = set(), set()
    for f in sorted(stage_dir.glob("*/task/task-result.json")):
        task_dir = f.parent.parent
        task = task_dir.name
        r = json.loads(f.read_text())
        success = bool(r["navigation_success"])
        prior = outcomes.get(task)
        if prior is not None and prior.success is not None and prior.success != success:
            raise ValueError(f"{f} disagrees with results.txt for {task}")
        outcomes[task] = Outcome(
            task_id=task,
            status="success" if success else "fail",
            success=success,
            source=str(f),
            hold=r.get("max_hold_ticks"),
            reach=r.get("goal_ever_reached"),
            force=r.get("max_undesired_force_n"),
            fell=r.get("fell"),
            steps=r.get("control_steps"),
            stop=r.get("stop_reason"),
            infra_failures=prior.infra_failures if prior else 0,
            student_sha256=r.get("student_sha256"),
        )
        shas.add(r.get("student_sha256"))
        seed = _command_seed(task_dir)
        if seed is not None:
            seeds.add(seed)
    if len(seeds) > 1:
        raise ValueError(f"{stage_dir} mixes physics seeds {sorted(seeds)}")
    dir_seed = DIR_SEED.search(stage_dir.name)
    dir_seed = int(dir_seed.group(1)) if dir_seed else None
    if seeds:
        stage.seed = seeds.pop()
        if dir_seed is not None and dir_seed != stage.seed:
            raise ValueError(f"{stage_dir}: directory seed {dir_seed} != command seed {stage.seed}")
    else:
        stage.seed = dir_seed
        stage.notes.append(
            "seed taken from the directory name (no command.json)"
            if dir_seed is not None
            else "physics seed unknown (no command.json, no -SEED directory suffix)"
        )
    shas.discard(None)
    if len(shas) > 1:
        raise ValueError(f"{stage_dir} mixes student checkpoints")
    stage.student_sha256 = shas.pop() if shas else None
    stage.outcomes = outcomes
    return stage


def outcome_matrix(stages, tasks):
    """Task x seed matrix (1 success, 0 fail, NaN unknown) ordered by seed."""
    stages = sorted(stages, key=lambda s: (s.seed is None, s.seed))
    seeds = [s.seed for s in stages]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds within one arm: {seeds}")
    y = np.full((len(tasks), len(stages)), np.nan)
    for j, s in enumerate(stages):
        for i, t in enumerate(tasks):
            o = s.outcomes.get(t)
            if o is not None and o.success is not None:
                y[i, j] = float(o.success)
    return y, seeds


def stage_summary(stage, tasks=None):
    ids = list(tasks) if tasks is not None else sorted(stage.outcomes)
    rows = [stage.outcomes.get(t) for t in ids]
    k = sum(1 for o in rows if o is not None and o.success)
    n_known = sum(1 for o in rows if o is not None and o.success is not None)
    infra = sum(1 for o in rows if o is not None and o.status == "infra_failed")
    missing = sum(1 for o in rows if o is None)
    lo, hi = wilson(k, n_known) if n_known else (None, None)
    return {
        "stage": stage.path,
        "seed": stage.seed,
        "student_sha256": stage.student_sha256,
        "assigned": len(ids),
        "success": k,
        "known": n_known,
        "infra_failed": infra,
        "missing": missing,
        "rate": k / n_known if n_known else None,
        "wilson95": [lo, hi],
        "max_infra_reruns": max((o.infra_failures for o in rows if o is not None), default=0),
        "notes": stage.notes,
    }


# ------------------------------------------------------------------ registered rules


def align(arms):
    """Restrict {name: (Y, seeds)} to the seeds every arm has, in a common order."""
    common = sorted(set.intersection(*(set(s) for _, s in arms.values())))
    out = {}
    for name, (y, seeds) in arms.items():
        cols = [seeds.index(s) for s in common]
        out[name] = y[:, cols]
    return out, common


def arm_mean(y):
    """Mean successes per panel per seed (NaN counted as unknown -> error)."""
    if np.isnan(y).any():
        raise ValueError("arm_mean needs complete outcomes")
    return float(y.sum(axis=0).mean())


def futility(app, rec, *, max_mean=3.0, seeds=3):
    """Registered Stage-1 futility rule on the first ``seeds`` seeds.

    Stop if the approach mean <= max_mean successes per panel, or if approach's
    discordant wins over recovery do not exceed its losses.
    """
    a, r = app[:, :seeds], rec[:, :seeds]
    mean = arm_mean(a)
    b, c = discordant(a, r)
    reasons = []
    if mean <= max_mean:
        reasons.append(f"approach mean {mean:.3f} <= {max_mean}")
    if b <= c:
        reasons.append(f"discordant wins {b} <= losses {c} vs recovery")
    return {
        "stop": bool(reasons),
        "approach_mean": mean,
        "wins": b,
        "losses": c,
        "reasons": reasons,
        "seeds_used": int(a.shape[1]),
    }


def registered_gate(app, uni, rec, *, alpha=0.025, min_mean=5.0, fast=False):
    """nav8192-confirm-v1 confirmation gate on complete task x seed matrices.

    Confirmed only if the one-sided exact task-stratified CMH test (approach > comparator)
    rejects against BOTH uniform-c2 and nav-v2-recovery after Holm at ``alpha``, AND
    approach's mean successes per panel per seed is >= ``min_mean``.
    """
    s = app.shape[1]
    if fast:
        p = [
            cmh_p_float(app.sum(1).astype(int), y.sum(1).astype(int), s, s, "greater")
            for y in (uni, rec)
        ]
    else:
        p = [cmh_exact(app, y, "greater")["p"] for y in (uni, rec)]
    h = holm(p, alpha)
    mean = arm_mean(app)
    tests = all(h["reject"])
    return {
        "p_vs_uniform": p[0],
        "p_vs_recovery": p[1],
        "holm_adjusted": h["adjusted"],
        "tests_pass": tests,
        "approach_mean": mean,
        "mean_pass": mean >= min_mean,
        "confirmed": bool(tests and mean >= min_mean),
        "alpha": alpha,
        "min_mean": min_mean,
        "seeds": int(s),
    }


def _bounded(fn, app, uni, rec, key, **kw):
    """Evaluate a rule on complete data, or on the two worst/best-case imputations."""
    if not (np.isnan(app).any() or np.isnan(uni).any() or np.isnan(rec).any()):
        out = fn(app, uni, rec, **kw)
        return {"status": "complete", "decision": out[key], "result": out}
    pess = fn(np.nan_to_num(app, nan=0), np.nan_to_num(uni, nan=1), np.nan_to_num(rec, nan=1), **kw)
    opt = fn(np.nan_to_num(app, nan=1), np.nan_to_num(uni, nan=0), np.nan_to_num(rec, nan=0), **kw)
    decided = pess[key] == opt[key]
    return {
        "status": "bounded",
        "decision": pess[key] if decided else "undetermined",
        "unfavourable_imputation": pess,
        "favourable_imputation": opt,
    }


# ------------------------------------------------------------------ CLI


def _tasks_from(args, stages):
    if getattr(args, "tasks", None):
        return list(args.tasks)
    reg = getattr(args, "tasks_from", None) or getattr(args, "registration", None)
    if reg:
        return list(json.loads(Path(reg).read_text())["tasks"][args.task_set])
    return sorted(set().union(*(s.outcomes for s in stages)))


def _pair_report(name_a, ya, name_b, yb, seeds, draws, seed):
    b, c = discordant(ya, yb)
    return {
        "a": name_a,
        "b": name_b,
        "seeds": seeds,
        "discordant_a_only": b,
        "discordant_b_only": c,
        "mcnemar_two_sided": mcnemar_exact(b, c, "two-sided"),
        "mcnemar_one_sided_a_gt_b": mcnemar_exact(b, c, "greater"),
        "cmh_one_sided_a_gt_b": cmh_exact(ya, yb, "greater"),
        "cmh_two_sided": cmh_exact(ya, yb, "two-sided"),
        "task_cluster_bootstrap": cluster_bootstrap_diff(ya, yb, draws=draws, seed=seed),
    }


def _arm_report(name, y, seeds):
    known = ~np.isnan(y)
    k = int(np.nansum(y))
    n = int(known.sum())
    lo, hi = wilson(k, n) if n else (None, None)
    return {
        "arm": name,
        "seeds": seeds,
        "per_seed_success": [int(np.nansum(y[:, j])) for j in range(y.shape[1])],
        "per_seed_known": [int(known[:, j].sum()) for j in range(y.shape[1])],
        "pooled_success": k,
        "pooled_known": n,
        "pooled_rate": k / n if n else None,
        "wilson95": [lo, hi],
        "mean_per_panel": float(np.nansum(y) / y.shape[1]) if y.shape[1] else None,
    }


def cmd_summary(args):
    stages = [load_stage(d) for d in args.stages]
    tasks = _tasks_from(args, stages) if (args.tasks or args.tasks_from) else None
    return {"stages": [stage_summary(s, tasks) for s in stages]}


def cmd_compare(args):
    sa, sb = [load_stage(d) for d in args.a], [load_stage(d) for d in args.b]
    tasks = _tasks_from(args, sa + sb)
    (ya, seeds_a), (yb, seeds_b) = outcome_matrix(sa, tasks), outcome_matrix(sb, tasks)
    aligned, common = align({"a": (ya, seeds_a), "b": (yb, seeds_b)})
    return {
        "tasks": tasks,
        "arms": [
            _arm_report(args.name_a, aligned["a"], common),
            _arm_report(args.name_b, aligned["b"], common),
        ],
        "comparison": _pair_report(
            args.name_a, aligned["a"], args.name_b, aligned["b"], common, args.draws, args.seed
        ),
        "unpaired_seeds": sorted(set(seeds_a) ^ set(seeds_b)),
    }


def cmd_gate(args):
    reg = json.loads(Path(args.registration).read_text())
    tasks = list(reg["tasks"][args.task_set])
    arms = {}
    shas = {}
    for name, dirs in (
        ("approach", args.approach),
        ("uniform", args.uniform),
        ("recovery", args.recovery),
    ):
        stages = [load_stage(d) for d in dirs]
        arms[name] = outcome_matrix(stages, tasks)
        shas[name] = sorted({s.student_sha256 for s in stages if s.student_sha256})
    expected = {
        "approach": "dag-approach-c2",
        "uniform": "dag-uniform-c2",
        "recovery": "nav-v2-recovery",
    }
    provenance = {}
    for name, arm in expected.items():
        want = reg["arms"][arm]["sha256"]
        provenance[name] = {
            "registered_sha256": want,
            "observed": shas[name],
            "match": shas[name] == [want],
        }
    aligned, common = align(arms)
    app, uni, rec = aligned["approach"], aligned["uniform"], aligned["recovery"]
    stage1 = [s for s in common if s in reg["seeds"]["stage1"]]
    out = {
        "registration": str(args.registration),
        "task_set": args.task_set,
        "seeds": common,
        "unpaired_seeds_dropped": sorted(set().union(*(s for _, s in arms.values())) - set(common)),
        "provenance": provenance,
        "arms": [_arm_report(n, aligned[n], common) for n in ("approach", "uniform", "recovery")],
    }
    if stage1:
        cols = [common.index(s) for s in stage1]
        out["futility_stage1"] = _bounded(
            lambda a, u, r: futility(a, r, seeds=len(cols)),
            app[:, cols],
            uni[:, cols],
            rec[:, cols],
            "stop",
        )
        if len(stage1) < len(reg["seeds"]["stage1"]):
            out["futility_stage1"]["warning"] = "not all Stage-1 seeds present"
    out["confirmation_gate"] = _bounded(
        registered_gate, app, uni, rec, "confirmed", alpha=args.alpha, min_mean=args.min_mean
    )
    out["secondary"] = {
        "approach_vs_uniform": _pair_report(
            "approach", app, "uniform", uni, common, args.draws, args.seed
        ),
        "approach_vs_recovery": _pair_report(
            "approach", app, "recovery", rec, common, args.draws, args.seed
        ),
    }
    if args.stage == 1:
        out["note"] = "Stage-1 readout: the futility rule is binding; the gate is informational."
    return out


def _json_default(x):
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Outcome):
        return asdict(x)
    raise TypeError(type(x))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--tasks", nargs="+", help="task ids (default: all tasks found)")
        sp.add_argument("--tasks-from", type=Path, help="registration.json supplying the task list")
        sp.add_argument("--task-set", default="feasible_19")
        sp.add_argument("--out", type=Path, help="also write the JSON here (must not exist)")

    s = sub.add_parser("summary", help="per-stage counts and Wilson intervals")
    s.add_argument("stages", nargs="+", type=Path)
    common(s)
    c = sub.add_parser("compare", help="paired comparison of two arms over seeds")
    c.add_argument(
        "--a", nargs="+", type=Path, required=True, help="arm A stage dirs (one per seed)"
    )
    c.add_argument(
        "--b", nargs="+", type=Path, required=True, help="arm B stage dirs (one per seed)"
    )
    c.add_argument("--name-a", default="A")
    c.add_argument("--name-b", default="B")
    c.add_argument("--draws", type=int, default=DRAWS)
    c.add_argument("--seed", type=int, default=RANDOM_SEED)
    common(c)
    g = sub.add_parser("gate", help="nav8192-confirm-v1 futility rule and confirmation gate")
    g.add_argument("--registration", type=Path, required=True)
    g.add_argument("--approach", nargs="+", type=Path, required=True)
    g.add_argument("--uniform", nargs="+", type=Path, required=True)
    g.add_argument("--recovery", nargs="+", type=Path, required=True)
    g.add_argument("--stage", type=int, choices=(1, 2), default=2)
    g.add_argument("--alpha", type=float, default=0.025)
    g.add_argument("--min-mean", type=float, default=5.0)
    g.add_argument("--draws", type=int, default=DRAWS)
    g.add_argument("--seed", type=int, default=RANDOM_SEED)
    g.add_argument("--task-set", default="feasible_19")
    g.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    result = {"summary": cmd_summary, "compare": cmd_compare, "gate": cmd_gate}[a.cmd](a)
    text = json.dumps(result, indent=1, default=_json_default)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x") as handle:
            handle.write(text + "\n")
    sys.stdout.write(text + "\n")
    return result


if __name__ == "__main__":
    main()
