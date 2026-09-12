"""Publication figures retain assigned denominators and explicit unknown bounds."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_plot_evaluation as plotting  # noqa: E402
from test_motion2scene_evaluation_statistics import receipt  # noqa: E402


@pytest.fixture(scope="module")
def results():
    source = receipt()
    for row in source["report"]["rows"]:
        if row["acquisition_seed"] is not None:
            row["acquisition_actual_physics_steps"] -= (row["acquisition_seed"] - 93201) * 100
    complete = copy.deepcopy(plotting.statistics.summarize(source))
    source["report"]["rows"][0].update(
        status="not_run", successful_passage_time_s=None, recorded_physics_steps=None
    )
    source["status"] = "paused"
    source["report"]["complete"] = False
    return complete, plotting.statistics.summarize(source)


def write_input(tmp_path, value):
    folder = tmp_path / "synthetic_statistics"
    folder.mkdir(exist_ok=True)
    path = folder / "result.json"
    path.write_text(json.dumps(value))
    return path


def test_corpus_curves_use_actual_steps_and_equal_weight_mean(results):
    data = plotting.make_plot_data(results[0])
    assert [p["acquisition_seed"] for p in data["panels"]] == [93201, 93202, 93203, None]
    points = [p["series"][0]["points"][0] for p in data["panels"]]
    assert [p["actual_steps"] for p in points[:3]] == [27316, 27216, 27116]
    assert points[-1]["actual_steps"] == 27216
    assert [p["assigned"] for p in points] == [36, 36, 36, 108]
    assert all(r["assigned"] == 36 for r in data["baselines"].values())
    assert len(data["comparisons"]) == 34
    assert all(
        r["unique_right_episodes"] == 36 for r in data["comparisons"] if r["shared_right_baseline"]
    )


def test_unknown_has_no_mean_and_keeps_bounds_in_mean_panel(results):
    data = plotting.make_plot_data(results[1])
    individual = data["panels"][0]["series"][0]["points"][0]
    aggregate = data["panels"][3]["series"][0]["points"][0]
    assert individual["mean"] is aggregate["mean"] is None
    assert individual["lower"] == 35 / 36 and individual["upper"] == 1
    assert aggregate["lower"] == np.mean([35 / 36, 1, 1])
    assert aggregate["measured"] == 107 and aggregate["assigned"] == 108
    unknown = [r for r in data["comparisons"] if r["unknown_matched_pairs"]]
    assert unknown
    assert all(r["completion"]["mean_difference"] is None for r in unknown)
    assert all(r["completion"]["crossed_95_percentile"] is None for r in unknown)


@pytest.mark.parametrize(
    "change", ["summary", "curve", "comparison", "unknown_interval", "bootstrap"]
)
def test_inconsistent_display_data_rejected(tmp_path, results, change):
    value = copy.deepcopy(results[1])
    if change == "summary":
        value["policy_summaries"]["always_walk"]["assigned"] = 108
    elif change == "curve":
        value["per_corpus_learning_curves"][0]["acquisition_actual_physics_steps"] = 12
    elif change == "comparison":
        value["paired_comparisons"][0]["unique_right_episodes"] = 1
    elif change == "unknown_interval":
        row = next(r for r in value["paired_comparisons"] if r["unknown_matched_pairs"])
        row["completion"]["crossed_95_percentile"] = [-0.1, 0.1]
    else:
        value["bootstrap"]["draws"] = 5000
    with pytest.raises(ValueError):
        plotting.load_statistics(write_input(tmp_path, value))


def test_input_hash_is_enforced(tmp_path, results):
    path = write_input(tmp_path, results[0])
    with pytest.raises(ValueError, match="hash differs"):
        plotting.load_statistics(path, "0" * 64)
    loaded, ref = plotting.load_statistics(path, plotting.statistics.file_ref(path)["sha256"])
    assert loaded == results[0] and ref == plotting.statistics.file_ref(path)


def test_synthetic_watermark_and_actual_budget_not_clipped(results):
    data = plotting.make_plot_data(results[0])
    data["panels"][0]["series"][0]["points"][0]["actual_steps"] = 4000
    figure, caption = plotting.learning_figure(data, True, "synthetic hash test")
    assert figure.axes[0].get_xlim()[0] <= 4
    assert any("SYNTHETIC — NOT PAPER EVIDENCE" in text.get_text() for text in figure.texts)
    assert "36 unique" in caption and "equal-weight" in caption
    plt.close(figure)
    figure, caption = plotting.comparison_figure(data, True, "synthetic hash test")
    assert len(figure.axes[0].get_yticklabels()) == 34
    assert "positive favors left" in figure.axes[0].get_xlabel()
    assert "No time/cost effect" in caption
    plt.close(figure)


def test_export_has_both_formats_and_source_hashes(tmp_path, results):
    source = write_input(tmp_path, results[1])
    args = SimpleNamespace(
        input=source, out=tmp_path / "plots", expected_sha256=None, data_kind="synthetic"
    )
    plotting.run(args)
    result = json.loads((args.out / "result.json").read_text())
    registration = json.loads((args.out / "registration.json").read_text())
    assert set(result["figures"]) == {
        "learning_curves.pdf",
        "learning_curves.png",
        "paired_completion.pdf",
        "paired_completion.png",
    }
    assert result["data_kind"] == "synthetic"
    assert (
        result["displayed_intervals_recomputed"] is False
        and result["cost_or_time_plotted"] is False
    )
    assert registration["statistics"] == plotting.statistics.file_ref(source)
    assert registration["plot_source"] == plotting.statistics.file_ref(plotting.__file__)
    for name, ref in result["figures"].items():
        path = args.out / name
        assert ref == plotting.statistics.file_ref(path)
        assert path.read_bytes().startswith(b"%PDF" if name.endswith("pdf") else b"\x89PNG")
    with pytest.raises(FileExistsError):
        plotting.run(args)


def test_synthetic_cannot_be_silently_presented_as_reported(tmp_path, results):
    source = write_input(tmp_path, results[0])
    args = SimpleNamespace(
        input=source, out=tmp_path / "plots", expected_sha256=None, data_kind="reported"
    )
    with pytest.raises(ValueError, match="synthetic label"):
        plotting.run(args)
    assert not args.out.exists()


def test_output_cannot_mutate_statistics_folder(tmp_path, results):
    source = write_input(tmp_path, results[0])
    args = SimpleNamespace(
        input=source, out=source.parent / "plots", expected_sha256=None, data_kind="synthetic"
    )
    with pytest.raises(ValueError, match="separate output"):
        plotting.run(args)
    assert not args.out.exists()
