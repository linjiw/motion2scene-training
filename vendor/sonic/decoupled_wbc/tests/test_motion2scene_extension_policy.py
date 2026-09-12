"""Arm-specific models retain their full-bank and training-fold bindings."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_extension_policy import load_extension_policy
from motion2scene_extension_teaching import arm_view
from motion2scene_fit_extension_teaching import fit_fold
from test_motion2scene_extension_teaching import bank
from test_motion2scene_fit_extension_teaching import training_fold

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (
    artifact,
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import definition_digest


def fixture(tmp_path):
    b = bank()
    fold, training = training_fold(b)
    model, analysis = fit_fold(b, "generated", fold, training)
    registry = write_new(tmp_path / "registry.json", dict(request=b.request))
    prepared = write_new(
        tmp_path / "prepared.json",
        dict(
            tasks=[
                dict(task_id=name, collection=dict(path=f"/{name}/manifest.json", sha256="test"))
                for name in fold["training_task_ids"] + fold["evaluation_task_ids"]
            ]
        ),
    )
    study = write_new(
        tmp_path / "study.json",
        dict(
            schema="motion2scene_extension_equivalent_teaching_v1",
            registry=registry,
            prepared=prepared,
            folds=[fold],
            arms=["generated", "authored"],
            l2=10.0,
        ),
    )
    np.savez(tmp_path / "policy.npz", **model)
    result = dict(
        registry=registry,
        study=study,
        arm="generated",
        analysis=analysis,
        logical_request_digest=definition_digest(arm_view(b, "generated")[0].request),
        policy=artifact(tmp_path / "policy.npz"),
    )
    return b, model, result


def test_loaded_policy_exactly_matches_common_ridge_archive(tmp_path):
    b, original, result = fixture(tmp_path)
    ref = write_new(tmp_path / "result.json", result)
    model, metadata = load_extension_policy(ref["path"], ref["sha256"], b)
    for key in model:
        np.testing.assert_array_equal(model[key], original[key])
    assert metadata == result
    assert len(b.option_ids) == 5 and len(model["option_ids"]) == 3


@pytest.mark.parametrize(
    "change", ["arm", "logical_bank", "training_task", "parent_bank", "archive"]
)
def test_policy_binding_mismatches_are_rejected(tmp_path, change):
    b, _, result = fixture(tmp_path)
    if change == "arm":
        result["arm"] = "authored"
    elif change == "logical_bank":
        result["logical_request_digest"] = "different"
    elif change == "training_task":
        result["analysis"]["training_collections"][0]["path"] = "/test_0/result.json"
    elif change == "parent_bank":
        b.request["max_entries_per_episode"] = 2
    else:
        with (tmp_path / "policy.npz").open("ab") as stream:
            stream.write(b"changed")
    ref = write_new(tmp_path / "result.json", result)
    with pytest.raises(ValueError):
        load_extension_policy(ref["path"], ref["sha256"], b)
