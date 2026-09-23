import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sim = _load("gate_cmh", ROOT / "experiments/power/gate_cmh.py")


def test_calibration_hits_target_rate_with_instance_effects():
    rng = np.random.default_rng(0)
    rates = {"approach": 6 / 19, "uniform": 2 / 19, "recovery": 4 / 19}
    totals = {k: [] for k in rates}
    for _ in range(400):
        y = sim.simulate_rep(rng, rates, sig_task=1.5, sig_inst=2.0, sig_int=0.5, s_max=10)
        for k in rates:
            assert not y[k][: sim.N_LONG].any()  # long tasks always fail
            totals[k].append(y[k].sum(0).mean())
    for k, r in rates.items():
        mean = np.mean(totals[k])
        se = np.std(totals[k]) / np.sqrt(len(totals[k]))
        assert abs(mean - r * 19) < 4 * se + 0.02, (k, mean)


def test_calibrate_root_and_bounds():
    eff = np.linspace(-2, 2, 13)
    a = sim.calibrate(5 / 19, eff, 1.0)
    z = sim.GH_X[None, :]
    expected = (1 / (1 + np.exp(-(a + eff[:, None] + z))) @ sim.GH_W).sum()
    assert expected == pytest.approx(5.0, abs=1e-8)
    with pytest.raises(ValueError):
        sim.calibrate(14 / 19, eff, 1.0)  # only 13 short tasks can succeed


def test_run_config_small_is_deterministic():
    job = dict(cfg=sim.BASE, scenario=sim.SCENARIOS[3], reps=20, seed=123)
    a, b = sim.run_config(job), sim.run_config(job)
    assert a["per_s"] == b["per_s"] and a["counts"] == b["counts"]
    for s in sim.S_GRID:
        r = a["per_s"][s]
        assert r["confirm"] <= r["gate"] <= min(r["tests"], r["mean"])
