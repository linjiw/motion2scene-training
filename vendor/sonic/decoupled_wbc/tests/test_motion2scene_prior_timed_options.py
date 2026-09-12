"""Explicit ancestry and immutable parent provenance for authored prior derivatives."""

import copy
import json

import pytest
from test_motion2scene_timed_options import artifact, evidence, fixture

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    make_verified_registry,
    validate_request,
)


def test_projected_prior_requires_explicit_provenance_and_neutral_parent(tmp_path):
    request = fixture(tmp_path)
    ref = request["references"][1]
    ref["construction"] = "authored_prior_projection_splice"
    for key in (
        "prior_motion",
        "derivation_registration",
        "derivation_result",
        "boundary_diagnostic",
    ):
        path = tmp_path / (key + ".fixture")
        path.write_text("synthetic test provenance only: " + key)
        ref[key] = artifact(path)
    validate_request(request)
    missing = copy.deepcopy(request)
    del missing["references"][1]["prior_motion"]
    with pytest.raises(ValueError):
        validate_request(missing)
    wrong_parent = copy.deepcopy(request)
    wrong_parent["references"][1]["parent_motion"] = request["references"][2]["motion"]
    with pytest.raises(ValueError, match="neutral parent"):
        validate_request(wrong_parent)
    receipts = []
    for name in validate_request(request).option_ids:
        path = tmp_path / (name + ".evidence.json")
        path.write_text(json.dumps(evidence(request, name)))
        receipts.append(artifact(path))
    make_verified_registry(request, receipts)
    (tmp_path / "prior_motion.fixture").write_text("changed after provenance binding")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        make_verified_registry(request, receipts)
