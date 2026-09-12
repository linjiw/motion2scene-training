"""Fixed capability tasks cannot be screened or replaced after generation."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_extension_tasks import fixed_tasks


def plan():
    tasks = []
    for x in (1.7, 2.25, 2.8):
        for length in (0.1, 0.75):
            for height in (1.24, 1.30):
                tasks.append(
                    dict(
                        task_id=f"fixed_{len(tasks)}",
                        center_xy_m=[x, -0.1],
                        length_m=length,
                        underside_m=height,
                        width_m=1.2,
                        thickness_m=0.1,
                        yaw_rad=0.0,
                    )
                )
    return dict(specification=dict(independent_capability_tasks=tasks))


def test_all_tasks_preserved_in_original_order_without_geometry_screen():
    p = plan()
    rows = fixed_tasks(p)
    assert len(rows) == 12 and [r[0] for r in rows] == [f"fixed_{i}" for i in range(12)]
    assert all("task_id" not in beam for _, beam in rows)
    assert "task_id" in p["specification"]["independent_capability_tasks"][0]


@pytest.mark.parametrize("change", ["drop", "duplicate", "move", "resize"])
def test_post_generation_panel_changes_are_rejected(change):
    p = plan()
    tasks = p["specification"]["independent_capability_tasks"]
    if change == "drop":
        tasks.pop()
    elif change == "duplicate":
        tasks[-1] = tasks[0].copy()
    elif change == "move":
        tasks[0]["center_xy_m"][0] += 0.05
    else:
        tasks[0]["length_m"] = 0.15
    with pytest.raises(ValueError):
        fixed_tasks(p)
