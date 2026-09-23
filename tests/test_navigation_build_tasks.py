"""build_tasks.py start/goal clearance on a synthetic motion (CPU; needs the vendored SONIC)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "sonic"

np = pytest.importorskip("numpy")
joblib = pytest.importorskip("joblib")
pytest.importorskip("scipy")
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
adapter = pytest.importorskip("gear_sonic.dataset_generation.kimodo_motion_adapter")


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "nav_build_tasks", ROOT / "scripts" / "navigation_distill" / "build_tasks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packet(tmp_path):
    """One 2 s motion whose path bows 0.465 m sideways: the +side wall's inner face then sits
    0.235 m from both start and goal, like the legacy 00399-corridor wall (0.231 m)."""
    s = np.linspace(0, 1, 61)
    q = np.zeros((61, 36))
    q[:, 0] = 0.6 * s
    q[:, 1] = -0.465 * np.sin(np.pi * s)
    q[:, 2] = 0.74
    q[:, 3] = 1.0
    entry = adapter.qpos_to_sonic_motion_entry(
        q, source_fps=30.0, canonicalize_horizontal_origin=False
    )
    motions = tmp_path / "motions"
    motions.mkdir()
    pkl = motions / "hindsight_00001.pkl"
    joblib.dump({"hindsight_00001": entry}, pkl, compress=3)
    ledger = tmp_path / "ledger.json"
    row = dict(
        id="00001",
        split="train",
        group="g",
        category="navigation",
        pkl_sha256=hashlib.sha256(pkl.read_bytes()).hexdigest(),
    )
    ledger.write_text(json.dumps([row]))
    body = tmp_path / "body.npz"
    np.savez(body, body_names=np.array(["pelvis"] + [f"link_{i}" for i in range(29)]))
    return tmp_path, ["--motions", str(motions), "--ledger", str(ledger),
                      "--body-reference", str(body), "--ids", "00001"]


def run(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["build_tasks.py", *args])
    load_builder().main()


def test_default_build_fails_loudly_on_a_crowding_wall(packet, monkeypatch):
    root, args = packet
    with pytest.raises(ValueError, match=r"00001-stop-corridor: .*obstacle 1 \(box\) start 0.235"):
        run(monkeypatch, args + ["--output", str(root / "fail")])
    assert not (root / "fail" / "00001" / "corridor.usda").exists()
    # The clear variant has no obstacles and still builds on its own.
    run(monkeypatch, args + ["--variants", "clear", "--output", str(root / "clear")])
    manifest = json.loads((root / "clear" / "manifest.json").read_text())
    assert [t["task_id"] for t in manifest["tasks"]] == ["00001-stop-clear"]
    assert manifest["clearance_fix"] == "none" and manifest["min_start_goal_clearance_m"] == 0.35
    # A lower declared margin accepts the legacy geometry unchanged.
    run(monkeypatch, args + ["--min-clearance-m", "0.2", "--output", str(root / "low")])
    task = json.loads((root / "low" / "00001" / "corridor.json").read_text())
    assert "clearance_amendments" not in task
    assert task["obstacles"][1]["center_xyz"][1] == pytest.approx(0.335)


def test_shift_out_moves_only_the_offending_wall_outward(packet, monkeypatch):
    root, args = packet
    out = root / "fixed"
    run(monkeypatch, args + ["--variants", "corridor", "--clearance-fix", "shift-out",
                             "--output", str(out)])
    task = json.loads((out / "00001" / "corridor.json").read_text())
    assert not (out / "00001" / "clear.json").exists()
    (amendment,) = task["clearance_amendments"]
    assert amendment["obstacle_index"] == 1 and amendment["shift_m"] == pytest.approx(0.12)
    assert amendment["clearance_before_m"]["start"] == pytest.approx(0.235)
    assert amendment["clearance_after_m"]["start"] == pytest.approx(0.355)
    assert amendment["direction_xy"] == pytest.approx([0.0, 1.0], abs=1e-12)
    walls = task["obstacles"]
    assert walls[0]["center_xyz"] == pytest.approx([0.3, -1.265, 0.75])
    assert walls[1]["center_xyz"] == pytest.approx([0.3, 0.455, 0.75])
    assert walls[1]["full_dimensions_xyz"] == [1.0, 0.2, 1.5]
    assert task["min_start_goal_clearance_m"] == 0.35
    usda = (out / "00001" / "corridor.usda").read_text()
    assert f"({walls[1]['center_xyz'][0]}, {walls[1]['center_xyz'][1]}, 0.75)" in usda
    proposal = json.loads((out / "00001" / "corridor-proposal.json").read_text())
    assert proposal["clearance_amendments"] == task["clearance_amendments"]
    assert min(min(r["start_m"], r["goal_m"]) for r in proposal["start_goal_clearance_m"]) >= 0.35
