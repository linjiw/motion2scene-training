"""Reserved layouts retain dimensions and splits when mapped onto a carrier."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_materialize_evaluation import place_beams, reserved_scene_text


def specification():
    return {
        "route_progress_fraction": 0.25,
        "underside_m": 1.234,
        "length_m": 0.678,
        "width_m": 1.456,
        "thickness_m": 0.123,
        "lateral_offset_m": 0.1,
        "yaw_offset_rad": 0.2,
    }


def test_arc_length_and_positive_left_for_rotated_route():
    spec = specification()
    beam = place_beams([[0, 0], [0, 0], [0, 1], [0, 4]], [spec])[0]
    assert beam["center_xy_m"] == pytest.approx([-0.1, 1.0])
    assert beam["yaw_rad"] == pytest.approx(np.pi / 2 + 0.2)
    assert all(beam[key] == value for key, value in spec.items())


def test_scene_keeps_full_dimensions_and_reserved_metadata():
    template = """#usda 1.0
def Xform "World" {
    custom string g1Dataset:sceneId = "old_scene"
    custom string g1Dataset:splitGroup = "development"
    def Cube "CounterfactualBeam" { double size = 1 }
}
"""
    beam = place_beams([[0, 0], [4, 0]], [specification()])[0]
    text = reserved_scene_text(template, [beam], "locked_v2_single_01")
    assert "(0.678, 1.456, 0.123)" in text
    assert "motion2scene_reserved_evaluation_v2" in text
    assert '"locked_v2_single_01"' in text
    assert "development" not in text
    with pytest.raises(ValueError, match="reserved layout"):
        reserved_scene_text(template, [beam], "scratch")


def test_closed_or_nonfinite_route_refused():
    with pytest.raises(ValueError, match="net heading"):
        place_beams([[0, 0], [1, 0], [0, 0]], [specification()])
    with pytest.raises(ValueError, match="finite locked"):
        place_beams([[0, 0], [4, 0]], [{**specification(), "lateral_offset_m": np.nan}])
