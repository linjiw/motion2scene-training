import numpy as np
from scipy import stats
from scipy.special import expit, logit
from scipy.optimize import brentq
rng = np.random.default_rng(0)
T = 19

def calib(target, u):
    return brentq(lambda a: expit(a + u).mean() - target, -20, 20)

def mcnemar_p(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))

def run(S, pA=7/19, pB=4/19, sig_task=1.5, sig_int=0.5, sig_inst=1.0, n_long_zero=6, reps=2000, alpha=0.05):
    res = dict(mcn=0, task_t=0, task_wil=0, boot=0, gate_pass=0)
    for r in range(reps):
        # task effects
        u = rng.normal(0, sig_task, T)
        if n_long_zero:
            u[:n_long_zero] = -8.0  # effectively always-fail long tasks for both arms
        inter = rng.normal(0, sig_int, T); inter[:n_long_zero] = 0
        aA = calib(pA, u + inter/2); aB = calib(pB, u - inter/2)
        # instances shared across arms (same seed -> same physics draw)
        v = rng.normal(0, sig_inst, (T, S))
        PA = expit(aA + (u + inter/2)[:, None] + v)
        PB = expit(aB + (u - inter/2)[:, None] + v)
        # couple via common uniform (same instance) -> positive correlation
        YA = (rng.random((T, S)) < PA).astype(int); YB = (rng.random((T, S)) < PB).astype(int)
        b = int(((YA == 1) & (YB == 0)).sum()); c = int(((YA == 0) & (YB == 1)).sum())
        if mcnemar_p(b, c) < alpha and b > c: res['mcn'] += 1
        d = (YA - YB).mean(1)
        if np.all(d == d[0]):
            pt = 1.0
        else:
            pt = stats.ttest_1samp(d, 0).pvalue
        if pt < alpha and d.mean() > 0: res['task_t'] += 1
        nz = d[d != 0]
        if len(nz) >= 1:
            pw = stats.wilcoxon(nz).pvalue if len(nz) > 0 and np.any(nz != 0) else 1.0
        else: pw = 1.0
        if pw < alpha and d.mean() > 0: res['task_wil'] += 1
        # task-cluster bootstrap CI of mean difference
        idx = rng.integers(0, T, (1000, T))
        bm = d[idx].mean(1)
        if np.quantile(bm, 0.025) > 0: res['boot'] += 1
        # roadmap gate: mean over seeds of arm A >= 5/19
        if YA.sum(0).mean() >= 5: res['gate_pass'] += 1
    return {k: v / reps for k, v in res.items()}

if __name__ == '__main__':
    import sys
    scen = {
      'A_struct(6 long=0, sig_task1.5, inst1.0)': dict(sig_task=1.5, sig_int=0.5, sig_inst=1.0, n_long_zero=6),
      'B_highinst(6 long=0, task1.0, inst2.0)': dict(sig_task=1.0, sig_int=0.5, sig_inst=2.0, n_long_zero=6),
      'C_lowinst(6 long=0, task2.0, inst0.5)': dict(sig_task=2.0, sig_int=0.5, sig_inst=0.5, n_long_zero=6),
    }
    for name, kw in scen.items():
        print('==', name)
        for S in [1, 3, 5, 8, 10, 15, 20]:
            r = run(S, reps=1000, **kw)
            print(f"S={S:2d} mcnemar={r['mcn']:.2f} task_t={r['task_t']:.2f} task_wilcoxon={r['task_wil']:.2f} task_boot={r['boot']:.2f}")
