import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gear_sonic.dataset_generation.hallucination.motion2scene_execution_proposal import (  # noqa: E402
    execution_summary,
    fit_proposal,
    proposal_condition,
)


def test_condition_contains_both_executions_and_sensor_context():
    state = {"starts": np.zeros((16, 1, 3)), "ends": np.ones((16, 1, 3)), "radii": [0.1]}
    lowered = {**state, "ends": state["ends"] * [1, 1, 0.5]}
    assert not np.array_equal(execution_summary(state), execution_summary(lowered))
    first = proposal_condition(state, lowered, [0.3, 0.8], [1, 2])
    second = proposal_condition(lowered, state, [0.3, 0.8], [1, 2])
    assert not np.array_equal(first, second)
    assert first[-4:].tolist() == [0.3, 0.8, 1, 2]


def test_likelihood_fit_and_samples_are_finite_and_bounded():
    torch.set_num_threads(1)
    conditions = [[0, 1], [1, 0]]
    targets = [np.array([[0.3, 1.2], [0.4, 1.22]]), np.array([[0.6, 1.3], [0.7, 1.32]])]
    model, history = fit_proposal(conditions, targets, [0.1, 1.1], [0.9, 1.45], steps=60)
    assert history[-1]["negative_log_likelihood"] < history[0]["negative_log_likelihood"]
    with torch.no_grad():
        values = model.sample(
            torch.tensor(conditions[0], dtype=torch.float64),
            100,
            [0.1, 1.1],
            [0.9, 1.45],
            torch.Generator().manual_seed(7),
        ).numpy()
    assert np.isfinite(values).all()
    assert (values > [0.1, 1.1]).all() and (values < [0.9, 1.45]).all()
    with pytest.raises(ValueError):
        model.sample(torch.tensor(conditions[0]).double(), 2, [0, 0], [np.inf, 1], None)
