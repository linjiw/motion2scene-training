import numpy as np
import pytest

from scripts.research.motion2scene_export_traversal_dataset import (
    decision_alignment,
    digest,
    history_alignment,
)


def payload():
    return {"fps": 50, "motion_time_s": np.arange(3) / 50}


def test_reference_limit_comes_from_hash_bound_loaded_bank(tmp_path):
    path = tmp_path / "bank.npz"
    np.savez(path, root_xyz=np.zeros((2, 3, 3)), joint_pos=np.zeros((2, 3, 29)), fps=50)
    full = {"bank": {"path": str(path), "sha256": digest(path)}}
    result = history_alignment(payload(), [{"time_s": t} for t in [0.02, 0.04, 0.02]], full)
    assert result["reference_frames"] == 3
    assert result["packet_eligible"] == [True, True, False]
    assert result["pose_unavailable_packets"] == 3
    assert result["reference_evidence"]["kind"] == "measured_loaded_reference_bank"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        history_alignment(payload(), [], full)


def test_unknown_bank_never_assumes_199_or_admits_history():
    result = history_alignment(payload(), [{"time_s": t} for t in [0.02, 0.04, 0.06]], {})
    assert result["reference_frames"] is None
    assert result["packet_eligible"] == [False] * 3


def test_audited_first_decision_can_survive_unverified_history():
    alignment = history_alignment(payload(), [{"time_s": 0.02}] * 3, {})
    decision = {"capture_frame": 0, "phase_s": 0.02, "features": [0.0] * 214}
    full = {
        "predicates": dict.fromkeys(
            ["decision_contract", "features_exact", "state_bound", "ray_origin_bound"], True
        )
    }
    result = decision_alignment(decision, payload(), full, alignment)
    assert result["student_input_eligible"] and not result["history_packet_eligible"]
    full["predicates"]["state_bound"] = False
    assert not decision_alignment(decision, payload(), full, alignment)["student_input_eligible"]
