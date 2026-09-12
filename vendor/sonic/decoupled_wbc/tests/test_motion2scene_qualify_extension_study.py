import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_option_extension_study import specification
from motion2scene_qualify_extension_study import build_request, validate_request
from motion2scene_timing_diagnostic import artifact


def fixture_request(tmp_path):
    parent = tmp_path / "parent.json"
    parent.write_text("{}")
    ref = artifact(parent)
    rows, raw = [], {}
    for arm in ("generated", "authored"):
        for i in range(4):
            name = f"{arm}_{i:02d}"
            path = tmp_path / (name + ".pkl")
            path.write_bytes(b"test-only placeholder, never executed")
            Path(str(path) + ".manifest.json").write_text(json.dumps(dict(test=True)))
            rows.append(dict(candidate_id=name, arm=arm, motion=artifact(path)))
            raw[name] = ref
    request = build_request(
        dict(specification=specification()),
        dict(rows=rows),
        ref,
        ref,
        raw,
        plan_ref=ref,
        candidates_ref=ref,
    )
    return request


def test_full_bank_has_equal_schedule_budget_and_honest_ancestry(tmp_path):
    request = fixture_request(tmp_path)
    bank = validate_request(request)
    assert len(bank.option_ids) == 17 and len(request["references"]) == 9
    for arm in ("generated", "authored"):
        assert sum(o["option_id"].startswith(arm + "_") for o in request["options"]) == 8
    for ref in request["references"][1:]:
        assert ref["parent_motion"] == request["references"][0]["motion"]
        assert "same native hinge-box" in ref["common_repair"]
        if ref["reference_id"].startswith("generated_"):
            assert ref["construction"] == "authored_prior_projection_splice"
            assert all(
                k in ref
                for k in (
                    "prior_motion",
                    "derivation_registration",
                    "derivation_result",
                    "boundary_diagnostic",
                )
            )


def test_repaired_sample_is_not_an_online_bank(tmp_path):
    bank = validate_request(fixture_request(tmp_path))
    assert not bank.online_verified
    for option in bank.request["options"]:
        assert option["entry_tick"] in (15, 50)
        assert option["return_tick"] == 265 and option["recovery_end_tick"] == 297
        assert option["maintenance"]["maximum_body_height_m"] is None


def test_missing_generated_ancestry_is_rejected(tmp_path):
    request = copy.deepcopy(fixture_request(tmp_path))
    del request["references"][1]["prior_motion"]
    with pytest.raises(ValueError):
        validate_request(request)
