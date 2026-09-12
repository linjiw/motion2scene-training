"""Split leakage and display-only simplification safeguards."""

import numpy as np
import pytest

from scripts.research.lflh_next.navigation import study
from scripts.research.lflh_next.navigation.render import cluster_mesh
from scripts.research.lflh_next.navigation.study import prompt_groups


def test_prompt_components_include_transitive_shared_prompts():
    rows = [
        {"id": "a", "prompts": ["Walk", "Turn"]},
        {"id": "b", "prompts": ["  TURN  ", "Duck"]},
        {"id": "c", "prompts": ["duck"]},
        {"id": "d", "prompts": ["Jump"]},
    ]
    groups = prompt_groups(rows)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["a"] != groups["d"]
    assert groups == prompt_groups(list(reversed(rows)))


def test_missing_prompt_ancestry_fails_closed():
    with pytest.raises(ValueError, match="missing prompt"):
        prompt_groups([{"id": "a", "prompts": []}])


def test_existing_study_is_not_owned_by_new_attempt(tmp_path, monkeypatch):
    marker = tmp_path / "receipt.json"
    marker.write_text("original")
    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(study, "_CREATED_THIS_ATTEMPT", True)
    with pytest.raises(FileExistsError):
        study.main()
    assert not study._CREATED_THIS_ATTEMPT
    assert marker.read_text() == "original"
    assert list(tmp_path.iterdir()) == [marker]


def test_display_clustering_removes_degenerate_faces_without_mutation():
    v = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    before = v.copy()
    small, faces = cluster_mesh(v, f)
    assert len(faces) == 1 and len(small) == 3
    assert len(set(faces[0])) == 3
    np.testing.assert_array_equal(v, before)
