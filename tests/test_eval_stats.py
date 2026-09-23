import importlib.util
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


es = _load("eval_stats", ROOT / "scripts/experiments/eval_stats.py")


# ------------------------------------------------------------------ Wilson / McNemar


def test_wilson_hand_values():
    lo, hi = es.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.27753, abs=1e-5)
    lo, hi = es.wilson(7, 19)
    # p=7/19, z=1.959964: centre 0.3906, half-width 0.1991
    assert (lo, hi) == pytest.approx((0.19150, 0.58960), abs=1e-4)
    assert es.wilson(5, 5)[1] == 1.0
    with pytest.raises(ValueError):
        es.wilson(6, 5)


def test_wilson_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for n in (1, 7, 19, 57, 190):
        for k in {0, 1, n // 3, n // 2, n - 1, n}:
            ci = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
            assert es.wilson(k, n) == pytest.approx((ci.low, ci.high), abs=1e-12)


def test_mcnemar_hand_values():
    assert es.mcnemar_exact(6, 1, "two-sided", as_fraction=True) == Fraction(1, 8)
    assert es.mcnemar_exact(6, 1, "greater", as_fraction=True) == Fraction(1, 16)
    assert es.mcnemar_exact(6, 1, "less", as_fraction=True) == Fraction(127, 128)
    assert es.mcnemar_exact(0, 0) == 1.0
    assert es.mcnemar_exact(3, 3) == 1.0


def test_mcnemar_matches_scipy_binomtest():
    stats = pytest.importorskip("scipy.stats")
    for b in range(0, 13):
        for c in range(0, 13):
            n = b + c
            if n == 0:
                continue
            for alt in ("two-sided", "greater", "less"):
                ref = stats.binomtest(b, n, 0.5, alternative=alt).pvalue
                assert es.mcnemar_exact(b, c, alt) == pytest.approx(ref, rel=1e-10, abs=1e-15)


# ------------------------------------------------------------------ exact CMH


def _brute_cmh(a_succ, n_a, b_succ, n_b, alternative):
    """Enumerate every table set with the observed margins; exact rationals."""
    supports = []
    for a, na, b, nb in zip(a_succ, n_a, b_succ, n_b):
        m = a + b
        tot = math.comb(na + nb, m)
        supports.append(
            [
                (x, Fraction(math.comb(na, x) * math.comb(nb, m - x), tot))
                for x in range(max(0, m - nb), min(na, m) + 1)
            ]
        )
    dist = {}
    for combo in itertools.product(*supports):
        t = sum(x for x, _ in combo)
        pr = math.prod((p for _, p in combo), start=Fraction(1))
        dist[t] = dist.get(t, 0) + pr
    t_obs = sum(a_succ)
    if alternative == "greater":
        return sum(p for t, p in dist.items() if t >= t_obs)
    if alternative == "less":
        return sum(p for t, p in dist.items() if t <= t_obs)
    return sum(p for p in dist.values() if p <= dist[t_obs])


def test_cmh_hand_computed_two_strata():
    # Stratum 1: 2 vs 2 trials, 2 successes total (A has both): P(a=2) = 1/6.
    # Stratum 2: 2 vs 2 trials, 1 success total (A has it): P(a=1) = 1/2.
    # T = 3 is the maximum: P = 1/12; two-sided adds T = 0 (also 1/12) -> 1/6.
    r = es.cmh_exact_counts([2, 1], [2, 2], [0, 0], [2, 2], "greater", as_fraction=True)
    assert r["p"] == Fraction(1, 12) and r["statistic"] == 3 and r["support"] == [0, 3]
    r = es.cmh_exact_counts([2, 1], [2, 2], [0, 0], [2, 2], "two-sided", as_fraction=True)
    assert r["p"] == Fraction(1, 6)
    assert r["null_expectation"] == pytest.approx(1.5)


def test_cmh_rabbits_example_from_r_documentation():
    # R ?mantelhaen.test "Rabbits" data (Mantel 1963): exact two-sided p = 0.040,
    # one-sided ("greater") p = 0.020, statistic S = 16. Arm A = no delay, success = cured.
    a_succ, n_a = [0, 3, 6, 5, 2], [6, 6, 6, 6, 2]
    b_succ, n_b = [0, 0, 2, 6, 5], [5, 6, 6, 6, 5]
    two = es.cmh_exact_counts(a_succ, n_a, b_succ, n_b, "two-sided")
    one = es.cmh_exact_counts(a_succ, n_a, b_succ, n_b, "greater")
    assert two["statistic"] == 16
    assert round(two["p"], 3) == 0.040
    assert round(one["p"], 3) == 0.020
    # Mantel-Haenszel common odds ratio by hand: (18/12 + 24/12) / (6/12) = 7.
    assert two["mantel_haenszel_odds_ratio"] == pytest.approx(7.0)


def test_cmh_single_stratum_equals_fisher():
    stats = pytest.importorskip("scipy.stats")
    for na, nb in ((3, 3), (5, 5), (4, 7), (8, 8)):
        for a in range(na + 1):
            for b in range(nb + 1):
                for alt in ("greater", "less", "two-sided"):
                    ref = stats.fisher_exact([[a, na - a], [b, nb - b]], alternative=alt)[1]
                    got = es.cmh_exact_counts([a], [na], [b], [nb], alt)["p"]
                    assert got == pytest.approx(ref, rel=1e-9, abs=1e-14)


def test_cmh_matches_brute_force_enumeration():
    rng = np.random.default_rng(7)
    for _ in range(60):
        k = int(rng.integers(1, 5))
        n_a = rng.integers(1, 5, k).tolist()
        n_b = rng.integers(1, 5, k).tolist()
        a = [int(rng.integers(0, x + 1)) for x in n_a]
        b = [int(rng.integers(0, x + 1)) for x in n_b]
        for alt in ("greater", "less", "two-sided"):
            got = es.cmh_exact_counts(a, n_a, b, n_b, alt, as_fraction=True)["p"]
            assert got == _brute_cmh(a, n_a, b, n_b, alt)


def test_cmh_float_path_matches_exact():
    rng = np.random.default_rng(11)
    for _ in range(80):
        s = int(rng.integers(1, 9))
        ya = (rng.random((19, s)) < rng.random((19, 1))).astype(float)
        yb = (rng.random((19, s)) < rng.random((19, 1))).astype(float)
        for alt in ("greater", "less", "two-sided"):
            exact = es.cmh_exact(ya, yb, alt)["p"]
            fast = es.cmh_p_float(ya.sum(1).astype(int), yb.sum(1).astype(int), s, s, alt)
            assert fast == pytest.approx(exact, rel=1e-9, abs=1e-15)


def test_cmh_with_one_seed_is_mcnemar():
    rng = np.random.default_rng(3)
    for _ in range(40):
        ya = (rng.random(19) < 0.4).astype(float)
        yb = (rng.random(19) < 0.3).astype(float)
        b, c = es.discordant(ya, yb)
        assert es.cmh_exact(ya, yb, "greater")["p"] == pytest.approx(
            es.mcnemar_exact(b, c, "greater")
        )
        assert es.cmh_exact(ya, yb, "two-sided")["p"] == pytest.approx(es.mcnemar_exact(b, c))


def test_cmh_drops_incomplete_pairs_and_degenerate_strata():
    ya = np.array([[1, 1, np.nan], [0, 0, 0], [1, 1, 1]], float)
    yb = np.array([[0, 0, 1], [0, 0, 0], [1, 1, 1]], float)
    r = es.cmh_exact(ya, yb, "greater", as_fraction=True)
    assert r["unknown_pairs"] == 1 and r["complete_pairs"] == 8
    assert r["informative_strata"] == 1
    assert r["p"] == Fraction(1, 6)  # stratum 1: 2 vs 2, both successes on A
    with pytest.raises(ValueError):
        es.cmh_exact(np.array([[2.0]]), np.array([[0.0]]))


# ------------------------------------------------------------------ Holm / bootstrap


def test_holm_hand_values():
    h = es.holm([0.01, 0.04, 0.03], alpha=0.05)
    assert h["adjusted"] == pytest.approx([0.03, 0.06, 0.06])
    assert h["reject"] == [True, False, False]
    # Registered gate: two tests at 0.025 -> smaller p must be <= 0.0125, larger <= 0.025.
    assert es.holm([0.012, 0.024], 0.025)["reject"] == [True, True]
    assert es.holm([0.013, 0.02], 0.025)["reject"] == [False, False]


def test_cluster_bootstrap():
    ya = np.array([[1, 1], [1, 0], [0, 0], [1, 1]], float)
    yb = np.array([[0, 0], [0, 0], [0, 0], [1, 1]], float)
    r = es.cluster_bootstrap_diff(ya, yb, draws=4000, seed=5)
    assert r["mean_difference"] == pytest.approx((1 + 0.5 + 0 + 0) / 4)
    assert r["per_panel_difference"] == pytest.approx(1.5)
    assert r["ci"][0] <= r["mean_difference"] <= r["ci"][1]
    assert r == es.cluster_bootstrap_diff(ya, yb, draws=4000, seed=5)  # reproducible
    same = es.cluster_bootstrap_diff(np.ones((3, 2)), np.zeros((3, 2)), draws=100)
    assert same["ci"] == [1.0, 1.0]


# ------------------------------------------------------------------ loader


def _task(stage, task, success, seed, sha="ab" * 32, hold=50):
    d = stage / task
    (d / "task").mkdir(parents=True)
    (d / "command.json").write_text(json.dumps(["python", "x.py", f"++seed={seed}"]))
    (d / "task/task-result.json").write_text(
        json.dumps(
            {
                "navigation_success": success,
                "goal_ever_reached": success,
                "fell": False,
                "max_hold_ticks": hold,
                "max_undesired_force_n": 0.0,
                "control_steps": 300,
                "stop_reason": "goal_hold" if success else "deadline",
                "student_sha256": sha,
                "teacher_mode": False,
            }
        )
    )


def _line(task, success, hold=50):
    res = "success" if success else "FAIL"
    stop = "goal_hold" if success else "deadline"
    return f"{task} {res} hold {hold} reach {success} force 0.0 fell False steps 300 stop {stop}"


def test_load_stage_reruns_and_infra_failures(tmp_path):
    stage = tmp_path / "approach-c2-92601"
    _task(stage, "t1", True, 92601)
    _task(stage, "t2", False, 92601, hold=0)
    (stage / "results.txt").write_text(
        "\n".join(
            [
                "t1 PROCESS_FAILED exit=1",
                "t3 PROCESS_FAILED exit=137",
                _line("t1", True),
                _line("t2", False, hold=0),
                "t3 PROCESS_FAILED exit=1",
            ]
        )
        + "\n"
    )
    s = es.load_stage(stage)
    assert s.seed == 92601 and s.student_sha256 == "ab" * 32
    assert s.outcomes["t1"].success is True and s.outcomes["t1"].infra_failures == 1
    assert s.outcomes["t2"].success is False
    assert s.outcomes["t3"].status == "infra_failed" and s.outcomes["t3"].infra_failures == 2
    summary = es.stage_summary(s, ["t1", "t2", "t3", "t4"])
    assert (summary["success"], summary["known"], summary["infra_failed"], summary["missing"]) == (
        1,
        2,
        1,
        1,
    )
    y, seeds = es.outcome_matrix([s], ["t1", "t2", "t3", "t4"])
    assert seeds == [92601]
    assert y[:2, 0].tolist() == [1.0, 0.0] and np.isnan(y[2:, 0]).all()


def test_load_stage_rejects_inconsistent_inputs(tmp_path):
    a = tmp_path / "a-92601"
    _task(a, "t1", True, 92601)
    (a / "results.txt").write_text(_line("t1", False, hold=0) + "\n")
    with pytest.raises(ValueError, match="disagrees"):
        es.load_stage(a)
    b = tmp_path / "b-92601"
    _task(b, "t1", True, 92601)
    _task(b, "t2", True, 92602)
    with pytest.raises(ValueError, match="mixes physics seeds"):
        es.load_stage(b)
    c = tmp_path / "c-92605"
    _task(c, "t1", True, 92601)
    with pytest.raises(ValueError, match="directory seed"):
        es.load_stage(c)
    d = tmp_path / "d-92601"
    d.mkdir()
    (d / "results.txt").write_text(_line("t1", True) + "\n" + _line("t1", False, hold=0) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        es.load_stage(d)
    e = tmp_path / "e-92601"
    e.mkdir()
    (e / "results.txt").write_text("garbage line\n")
    with pytest.raises(ValueError, match="unrecognized"):
        es.load_stage(e)


def test_results_txt_only_stage_takes_seed_from_directory(tmp_path):
    s = tmp_path / "recovery-92603"
    s.mkdir()
    (s / "results.txt").write_text(_line("t1", True) + "\n")
    stage = es.load_stage(s)
    assert stage.seed == 92603 and stage.outcomes["t1"].success


# ------------------------------------------------------------------ registered rules


def _arm(rate_per_task, s):
    return np.repeat(np.asarray(rate_per_task, float)[:, None], s, axis=1)


def test_futility_rule():
    tasks = 19
    app = np.zeros((tasks, 3))
    rec = np.zeros((tasks, 3))
    app[:3] = 1  # mean 3 -> futile by the mean rule
    f = es.futility(app, rec)
    assert f["stop"] and f["approach_mean"] == 3.0 and f["wins"] == 9
    app[3] = 1  # mean 4
    rec[5:9] = 1  # 12 losses vs 12 wins -> futile by the discordance rule
    f = es.futility(app, rec)
    assert f["stop"] and f["wins"] == 12 and f["losses"] == 12 and len(f["reasons"]) == 1
    rec[8] = 0
    assert not es.futility(app, rec)["stop"]


def test_registered_gate():
    s = 10
    app = _arm([1] * 7 + [0] * 12, s)
    uni = _arm([0] * 19, s)
    rec = _arm([1] * 2 + [0] * 17, s)
    g = es.registered_gate(app, uni, rec)
    assert g["confirmed"] and g["approach_mean"] == 7.0
    fast = es.registered_gate(app, uni, rec, fast=True)
    assert fast["confirmed"] == g["confirmed"] and fast["tests_pass"] == g["tests_pass"]
    assert fast["p_vs_recovery"] == pytest.approx(g["p_vs_recovery"], rel=1e-9)
    low = _arm([1] * 4 + [0] * 15, s)  # mean 4 < 5: tests pass, mean fails
    g = es.registered_gate(low, uni, _arm([0] * 19, s))
    assert g["tests_pass"] and not g["mean_pass"] and not g["confirmed"]
    g = es.registered_gate(app, uni, app.copy())  # equal to recovery: p = 1
    assert g["p_vs_recovery"] == 1.0 and not g["confirmed"]


def test_bounded_gate_with_unknowns():
    s = 3
    app = _arm([1] * 7 + [0] * 12, s)
    uni = _arm([0] * 19, s)
    rec = _arm([0] * 19, s)
    app[0, 0] = np.nan
    out = es._bounded(es.registered_gate, app, uni, rec, "confirmed")
    assert out["status"] == "bounded"
    assert (
        out["unfavourable_imputation"]["approach_mean"]
        < out["favourable_imputation"]["approach_mean"]
    )


# ------------------------------------------------------------------ CLI and real data


def test_cli_gate_and_compare(tmp_path, capsys):
    reg = {
        "arms": {
            "dag-approach-c2": {"sha256": "aa" * 32},
            "dag-uniform-c2": {"sha256": "bb" * 32},
            "nav-v2-recovery": {"sha256": "cc" * 32},
        },
        "tasks": {"feasible_19": ["t1", "t2", "t3"]},
        "seeds": {"stage1": [92601, 92602, 92603]},
    }
    (tmp_path / "reg.json").write_text(json.dumps(reg))
    dirs = {}
    for arm, sha, wins in (("approach", "aa", 3), ("uniform", "bb", 0), ("recovery", "cc", 1)):
        dirs[arm] = []
        for sd in (92601, 92602, 92603):
            d = tmp_path / f"{arm}-{sd}"
            for i, t in enumerate(("t1", "t2", "t3")):
                _task(d, t, i < wins, sd, sha=sha * 32)
            dirs[arm].append(str(d))
    out = es.main(
        [
            "gate",
            "--registration",
            str(tmp_path / "reg.json"),
            "--approach",
            *dirs["approach"],
            "--uniform",
            *dirs["uniform"],
            "--recovery",
            *dirs["recovery"],
            "--draws",
            "200",
            "--stage",
            "1",
            "--out",
            str(tmp_path / "gate.json"),
        ]
    )
    assert all(v["match"] for v in out["provenance"].values())
    assert out["futility_stage1"]["decision"] is True  # mean 3 successes per panel <= 3
    assert out["confirmation_gate"]["status"] == "complete"
    assert json.loads((tmp_path / "gate.json").read_text())["seeds"] == [92601, 92602, 92603]
    out = es.main(["compare", "--a", *dirs["approach"], "--b", *dirs["recovery"], "--draws", "200"])
    assert out["comparison"]["discordant_a_only"] == 6
    assert out["comparison"]["discordant_b_only"] == 0
    capsys.readouterr()


NAV_EVAL = Path("/home/robotixx/motion2scene-training/workspace/nav-8192") / "eval"


@pytest.mark.skipif(
    not (NAV_EVAL / "dag-approach-c2-91260").exists(), reason="workspace data absent"
)
def test_reproduces_in_sample_mcnemar_on_91260_panels():
    reg = json.loads((ROOT / "experiments/nav8192-confirm-v1/registration.json").read_text())
    tasks = reg["tasks"]["feasible_19"]
    ya, _ = es.outcome_matrix([es.load_stage(NAV_EVAL / "dag-approach-c2-91260")], tasks)
    yb, _ = es.outcome_matrix([es.load_stage(NAV_EVAL / "dag-uniform-c2-91260")], tasks)
    assert (ya.sum(), yb.sum()) == (7, 2)
    assert es.discordant(ya, yb) == (6, 1)
    assert es.mcnemar_exact(6, 1, "two-sided", as_fraction=True) == Fraction(1, 8)
    assert es.cmh_exact(ya, yb, "two-sided")["p"] == 0.125
