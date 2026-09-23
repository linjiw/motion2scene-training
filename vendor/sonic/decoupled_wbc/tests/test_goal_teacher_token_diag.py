import torch

from gear_sonic.research.goal_teacher.diag_token_space import (
    BIN,
    FrozenSonic,
    cycle_study,
    lag_pairs,
    snap,
    to_levels,
    token_dynamics,
)
from gear_sonic.research.hindsight_training.student import mlp


def fake_state():
    state = {}
    for prefix, dims in (
        ("actor_module.encoders.g1.module.", [640, 2048, 1024, 512, 512, 64]),
        ("actor_module.decoders.g1_dyn.module.", [994, 2048, 2048, 1024, 1024, 512, 512, 29]),
        ("actor_module.decoders.g1_kin.module.", [64, 2048, 1024, 512, 512, 640]),
    ):
        state.update({prefix + k: v for k, v in mlp(dims).state_dict().items()})
    return state


def test_snap_maps_to_32_level_lattice():
    x = torch.tensor([-2.0, -1.0, -0.03, 0.03, 0.94, 5.0])
    levels = to_levels(snap(x))
    assert levels.min() >= -16 and levels.max() <= 15
    assert torch.allclose(snap(x) / BIN, levels)
    assert torch.equal(snap(torch.tensor([0.031])), torch.tensor([0.0]))


def test_lag_pairs_stay_within_episode():
    episode = torch.tensor([0, 0, 0, 1, 1, 2])
    i, j = lag_pairs(episode, 1)
    assert torch.equal(i, torch.tensor([0, 1, 3]))
    assert torch.equal(j, torch.tensor([1, 2, 4]))
    assert torch.equal(episode[i], episode[j])


def test_token_dynamics_counts_changes_in_bins():
    tokens = torch.zeros(4, 64)
    tokens[1, 0] = BIN
    tokens[2, 0] = 3 * BIN
    stats = token_dynamics(tokens, torch.tensor([0, 0, 0, 1]))
    assert stats["per_step_max_abs_change_bins"] == 2.0
    assert abs(stats["per_step_mean_abs_change_bins"] - 3 / 128) < 1e-6


def test_frozen_sonic_encode_outputs_lattice_tokens_and_cycle_runs():
    model = FrozenSonic(fake_state())
    with torch.no_grad():
        tokens = model.encode(torch.randn(8, 640))
        assert tokens.shape == (8, 64)
        assert torch.allclose(snap(tokens), tokens, atol=1e-6)
        assert model.act(tokens, torch.randn(8, 930)).shape == (8, 29)
        out = cycle_study(model, tokens, torch.Generator().manual_seed(0))
    assert out["random_lattice_tokens"]["fraction_above_corpus_p99"] >= 0.0
