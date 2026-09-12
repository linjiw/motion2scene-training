"""A selected proposal must bind the real frozen geometry and earlier prefix."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_prepare_complementary_pilot as pilot


def save(path, value):
    path.write_text(json.dumps(value))
    return pilot.artifact(path)


def pool(tmp_path):
    minimum = [[-0.1, 0.03]]
    counts = [[1, 81]]
    negative = [[-0.02, 0.04]]
    data = dict(
        rows=[
            dict(
                candidate_id="c0",
                minimum_evaluated_positive_clearance_m=minimum[0],
                positive_offset_counts=counts[0],
                nominal_negative_clearance_m=negative[0],
                beams=[],
            )
        ]
    )
    outer = np.full((1, 81, 2), np.nan)
    outer[:, 0, 0] = -0.1
    outer[:, :, 1] = 0.03
    geometry = tmp_path / "geometry.npz"
    np.savez(
        geometry,
        candidate_ids=["c0"],
        option_ids=["neutral", "adapt"],
        outer_clearance_by_offset_m=outer,
        evaluated_offset_counts=counts,
        nominal_inner_clearance_m=negative,
        outer_beam_queries=[[1, 81]],
        inner_beam_queries=[[1, 1]],
    )
    return dict(
        candidates=save(tmp_path / "candidates.json", data), geometry=pilot.artifact(geometry)
    )


def test_geometry_loader_matches_frozen_array_values_and_charges_queries(tmp_path):
    rows, options, minimum, counts, negative, queries = pilot.read_geometry(pool(tmp_path))
    assert options == ["neutral", "adapt"]
    assert minimum.tolist() == [[-0.1, 0.03]]
    assert counts.tolist() == [[1, 81]]
    assert negative.tolist() == [[-0.02, 0.04]]
    assert queries.tolist() == [84]
    assert rows[0]["geometry_key"].startswith("sha256:")


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_id", "changed"),
        ("minimum_evaluated_positive_clearance_m", [-0.1, 0.05]),
        ("positive_offset_counts", [1, 1]),
        ("nominal_negative_clearance_m", [0.02, 0.04]),
    ],
)
def test_candidate_metadata_cannot_relabel_geometry(tmp_path, field, value):
    refs = pool(tmp_path)
    path = Path(refs["candidates"]["path"])
    data = json.loads(path.read_text())
    data["rows"][0][field] = value
    refs["candidates"] = save(path, data)
    with pytest.raises(ValueError, match="differ|disagree"):
        pilot.read_geometry(refs)


def prefix(tmp_path):
    registry = dict(path="bank", sha256="bank_hash")
    teachers = [
        save(
            tmp_path / f"teacher{i}.json",
            dict(
                rows=[
                    dict(
                        forced_option_id="neutral",
                        outcome=dict(task_outcome="failure"),
                        costs=dict(passage_time_s=None),
                    ),
                    dict(
                        forced_option_id="adapt",
                        outcome=dict(task_outcome="pass"),
                        costs=dict(passage_time_s=4.0),
                    ),
                ]
            ),
        )
        for i in range(3)
    ]
    registration = save(
        tmp_path / "registration.json",
        dict(
            collections=teachers,
            registry=registry,
            l2=10.0,
            allow_measured_tie_initialization=True,
            replay_weights=None,
        ),
    )
    policy = save(tmp_path / "policy.json", dict(frozen=True))
    model = save(
        tmp_path / "model.json",
        dict(
            status="complete",
            registration=registration,
            policy=policy,
            replay_audit=None,
        ),
    )
    corpus = dict(
        seed=93201,
        checkpoint=model,
        source_results=[dict(teacher=t) for t in teachers],
        tasks=[dict(outcomes=dict(neutral="failure", adapt="pass")) for _ in range(2)],
        physical_cost=dict(total_recorded_steps=27416),
    )
    return corpus, registry


def test_response_is_selected_only_from_bound_earlier_teacher_outcomes(tmp_path):
    corpus, registry = prefix(tmp_path)
    result = pilot.earlier_response(corpus, ["neutral", "adapt"], registry)
    assert result["response"]["assigned_tasks"] == 2
    assert result["response"]["selected_fixed_schedule"] == "adapt"
    assert result["original_model"] == corpus["checkpoint"]


def test_wrong_prefix_cannot_be_substituted_for_original_m2(tmp_path):
    corpus, registry = prefix(tmp_path)
    corpus["source_results"].reverse()
    with pytest.raises(ValueError, match="exact completed"):
        pilot.earlier_response(corpus, ["neutral", "adapt"], registry)


def test_audit_and_source_outcomes_must_agree(tmp_path):
    corpus, registry = prefix(tmp_path)
    corpus["tasks"][0]["outcomes"]["neutral"] = "pass"
    with pytest.raises(ValueError, match="independent M2 audit"):
        pilot.earlier_response(corpus, ["neutral", "adapt"], registry)


def test_completed_selection_is_never_silently_replaced(tmp_path):
    save(tmp_path / "declaration.json", dict(implementation=[]))
    save(tmp_path / "selection.json", dict(original=True))
    with pytest.raises(ValueError, match="already exists"):
        pilot.select(tmp_path)
    assert json.loads((tmp_path / "selection.json").read_text()) == dict(original=True)
