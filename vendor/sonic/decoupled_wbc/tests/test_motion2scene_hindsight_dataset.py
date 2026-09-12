"""Reference-scene records cannot silently become physical teacher supervision."""

import json

import numpy as np
import pytest

from scripts.research.lflh_next.dataset.adapter import load_reference_scene


def test_reference_adapter_rejects_missing_teacher_labels(tmp_path):
    (tmp_path / "scenes").mkdir()
    record = {
        "teacher_eligible": False,
        "physical_labels": {"teacher_actions": None},
        "reference": "../reference.npz",
    }
    (tmp_path / "scenes/test.json").write_text(json.dumps(record))
    np.savez(tmp_path / "reference.npz", qpos=np.ones((3, 36)))
    with pytest.raises(ValueError, match="Executed teacher labels unavailable"):
        load_reference_scene(tmp_path, "test")
    scene, arrays = load_reference_scene(tmp_path, "test", require_executed_teacher=False)
    assert scene["teacher_eligible"] is False
    assert arrays["qpos"].shape == (3, 36)


def test_reference_adapter_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="identifier"):
        load_reference_scene(tmp_path, "../outside")
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes/test.json").write_text(json.dumps({"reference": "../../outside.npz"}))
    with pytest.raises(ValueError, match="escaped"):
        load_reference_scene(tmp_path, "test", require_executed_teacher=False)
