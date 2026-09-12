import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def collector(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts/research"))
    spec = importlib.util.spec_from_file_location(
        "multi_option_collector", root / "scripts/research/motion2scene_collect_multi_options.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_template_must_match_qualified_source_motion_and_controller(collector, tmp_path):
    checkpoint, motion = tmp_path / "controller", tmp_path / "neutral"
    checkpoint.write_bytes(b"frozen controller")
    motion.write_bytes(b"qualified reference")
    registry = {
        "source": 41002,
        "controller": collector.artifact(checkpoint),
        "references": [{"motion": collector.artifact(motion)}],
    }
    parent = {"implementation": {"checkpoint": collector.artifact(checkpoint)}}
    cell = {"generation_seed": 41002, "motion": collector.artifact(motion)}
    collector.validate_template_bindings(parent, [cell], registry)
    with pytest.raises(ValueError, match="source"):
        collector.validate_template_bindings(parent, [{**cell, "generation_seed": 41003}], registry)
    other = tmp_path / "other"
    other.write_bytes(b"different reference")
    with pytest.raises(ValueError, match="neutral"):
        collector.validate_template_bindings(
            parent, [{**cell, "motion": collector.artifact(other)}], registry
        )
    with pytest.raises(ValueError, match="checkpoint"):
        collector.validate_template_bindings(
            {"implementation": {"checkpoint": collector.artifact(other)}}, [cell], registry
        )


def test_clock_audit_exposes_final_resample_and_same_row_state_alignment(collector):
    roots = np.array([[0.0, 0, 1], [1.0, 0, 1], [2.0, 0, 1]])
    payload = {"motion_time_s": np.array([0.0, 0.02, 0.04]), "root_pos_w": roots}
    packets = [
        {"time_s": 0.02, "root_pos_w": roots[0].tolist()},
        {"time_s": 0.04, "root_pos_w": roots[1].tolist()},
        {"time_s": 0.02, "root_pos_w": roots[0].tolist()},
    ]
    audit = collector.sensor_clock_audit(packets, payload)
    assert audit["phase_wrap_indices"] == [2]
    assert audit["pre_wrap_rows"] == 2
    assert audit["pre_wrap_root_state_exact_match"]
    packets[-1] = {"time_s": 0.06, "root_pos_w": roots[-1].tolist()}
    audit = collector.sensor_clock_audit(packets, payload)
    assert audit["phase_wrap_indices"] == []
    assert audit["pre_wrap_rows"] == 3
    assert audit["pre_wrap_root_state_exact_match"]
