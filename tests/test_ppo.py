from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan import ppo, train  # noqa: E402
from catan.actions import ActionType, space_for  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import _seat, static_graph  # noqa: E402
from catan.game import start  # noqa: E402
from catan.model import CatanNet, ModelConfig, packing  # noqa: E402
from catan.policy import NetworkPolicy  # noqa: E402
from catan.ppo import PPOConfig, advantages, assemble, rotate, update  # noqa: E402
from catan.selfplay import Collector  # noqa: E402


def a_policy(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = CatanNet(space_for(game), graph, players, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space_for(game), packing(graph, players))


def some_episodes(policy, games: int = 4, seed: int = 0, lanes: int = 8):
    return Collector(policy, lanes=lanes, seed=seed, action_cap=3000).collect(games)


def test_the_discount_is_one_and_is_not_something_a_run_can_change():
    # The reward is zero-sum, so about half of all terminal values are negative
    # and a discount below 1 makes a late loss cheaper than an early one. That
    # pays a losing policy to stall. It is a constant, not a knob.
    assert ppo.GAMMA == 1.0

    fields = {f.name for f in dataclasses.fields(PPOConfig)}
    assert "gamma" not in fields
    # Anti-vacuity: an empty or renamed config would satisfy the line above for
    # the wrong reason.
    assert {"lam", "clip", "epochs"} <= fields

    parser_flags = train.main.__doc__ or ""
    assert "--gamma" not in parser_flags
    with pytest.raises(SystemExit):
        train.main(["--gamma", "0.99", "--iterations", "0"])


def test_gae_at_lambda_one_is_just_the_outcome_minus_the_estimate():
    # With gamma 1 and a terminal-only reward, GAE(1) telescopes to the Monte
    # Carlo advantage. Anything else means the bootstrap terms are wrong.
    values = np.array([0.1, -0.2, 0.4, 0.0], dtype=np.float32)
    out = advantages(values, terminal=0.6, lam=1.0)
    assert out == pytest.approx(0.6 - values, abs=1e-6)


def test_gae_below_one_leans_on_the_value_head_instead_of_the_outcome():
    values = np.array([0.1, -0.2, 0.4, 0.0], dtype=np.float32)
    monte_carlo = advantages(values, terminal=0.6, lam=1.0)
    shrunk = advantages(values, terminal=0.6, lam=0.5)

    # The last step is identical either way: there is no future to discount.
    assert shrunk[-1] == pytest.approx(monte_carlo[-1], abs=1e-6)
    # Earlier steps are pulled towards the value head, so they must differ.
    assert not np.allclose(shrunk[:-1], monte_carlo[:-1])


def test_a_single_step_trajectory_is_the_whole_payoff():
    assert advantages(np.array([0.25], dtype=np.float32), 1.0, 0.95) == pytest.approx(
        [0.75], abs=1e-6
    )


def test_rotate_agrees_with_the_rotation_the_encoder_applies():
    # The value head's outputs have to line up with its inputs. `_seat` is the
    # encoder's own function, so this pins the two together rather than
    # restating the arithmetic.
    payoffs = (0.4, -0.1, -0.2, -0.1)
    for perspective in range(4):
        seen = rotate(payoffs, perspective)
        assert seen[0] == payoffs[perspective]
        for board_seat in range(4):
            assert seen[_seat(board_seat, perspective, 4)] == payoffs[board_seat]


def test_assemble_keeps_every_transition_of_every_episode():
    policy = a_policy(seed=1)
    episodes = some_episodes(policy, games=3, seed=1)
    batch = assemble(episodes, policy.layout, PPOConfig())

    assert len(batch) == sum(len(e) for e in episodes)
    # Anti-vacuity: one episode with one seat acting would not exercise the
    # per-seat demultiplexing this walks over.
    assert len(episodes) >= 3
    assert all(sum(1 for s in e.trajectories if s) >= 2 for e in episodes)


def test_every_position_in_a_game_carries_that_game_s_terminal_outcome():
    # The value head is trained on terminal outcomes, never on a bootstrap, so
    # a seat's target is the same vector at its first decision and its last.
    policy = a_policy(seed=2)
    episodes = some_episodes(policy, games=2, seed=2)
    batch = assemble(episodes[:1], policy.layout, PPOConfig())

    episode = episodes[0]
    from catan.rewards import reward

    expected = {
        seat: rotate(reward(episode.outcome), seat)
        for seat, trajectory in enumerate(episode.trajectories)
        if trajectory
    }
    assert len(expected) >= 2

    rows = 0
    for seat, trajectory in enumerate(episode.trajectories):
        for _ in trajectory:
            target = tuple(batch.value_target[rows].tolist())
            assert target == pytest.approx(expected[seat], abs=1e-6)
            rows += 1
    assert rows == len(batch)


def test_the_value_targets_of_one_game_sum_to_zero_across_the_table():
    policy = a_policy(seed=3)
    episodes = some_episodes(policy, games=2, seed=3)
    batch = assemble(episodes, policy.layout, PPOConfig())
    # Each row is one seat's view of a zero-sum outcome, so every row sums to
    # zero on its own.
    assert batch.value_target.sum(dim=1).abs().max() < 1e-5


def test_a_trading_transition_carries_its_offer_mask_into_the_batch():
    policy = a_policy(seed=4)
    episodes = some_episodes(policy, games=4, seed=4, lanes=16)
    batch = assemble(episodes, policy.layout, PPOConfig())

    trades = [
        t
        for e in episodes
        for seat in e.trajectories
        for t in seat
        if t.action.type is ActionType.PROPOSE_TRADE
    ]
    assert trades, "no trade was proposed, so the mask path went unchecked"
    assert all(isinstance(t.aux, np.ndarray) for t in trades)
    assert int(batch.pair.sum()) >= len(trades)


def test_an_update_returns_a_ratio_of_one_before_it_has_changed_anything():
    # At the first minibatch of the first epoch the parameters are the ones that
    # collected the data, so the KL must be zero. A non-zero reading here means
    # `evaluate` and `act` disagree, which is the failure that trains happily on
    # the wrong distribution.
    policy = a_policy(seed=5)
    episodes = some_episodes(policy, games=3, seed=5)
    batch = assemble(episodes, policy.layout, PPOConfig())
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=0.0)

    stats = update(policy, optimiser, batch, PPOConfig(epochs=1, minibatch=len(batch)))
    assert stats.positions == len(batch)
    assert stats.approx_kl == pytest.approx(0.0, abs=1e-5)
    assert stats.clip_fraction == pytest.approx(0.0, abs=1e-6)


def test_an_update_moves_the_value_head_towards_the_outcome_it_was_shown():
    policy = a_policy(seed=6)
    episodes = some_episodes(policy, games=3, seed=6)
    batch = assemble(episodes, policy.layout, PPOConfig())
    config = PPOConfig(epochs=1, minibatch=len(batch), entropy_coefficient=0.0)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    first = update(policy, optimiser, batch, config)
    for _ in range(20):
        last = update(policy, optimiser, batch, config)

    assert last.value_loss < first.value_loss
    # Anti-vacuity: a value loss that started at zero would satisfy nothing.
    assert first.value_loss > 1e-4


def test_the_clipped_surrogate_stops_rewarding_a_ratio_past_the_clip():
    # Hand-built rather than collected, so the arithmetic is checkable: with a
    # positive advantage and a ratio well past 1 + clip, the loss must be the
    # clipped branch and so independent of how far past it went.
    advantage = torch.tensor([1.0])
    clip = 0.2
    for log_ratio in (1.0, 2.0, 5.0):
        ratio = torch.tensor([log_ratio]).exp()
        clamped = ratio.clamp(1 - clip, 1 + clip) * advantage
        loss = -torch.min(ratio * advantage, clamped)
        assert loss.item() == pytest.approx(-(1 + clip))


def test_learning_from_no_episodes_at_all_is_an_error_rather_than_a_silent_pass():
    policy = a_policy(seed=7)
    with pytest.raises(ValueError):
        assemble([], policy.layout, PPOConfig())


def test_minibatches_partition_the_batch_and_never_yield_a_single_row():
    """A one-row trailing minibatch is a nan bomb, so it must not be emitted.

    `advantage.std()` on one row divides by `n - 1 == 0` and returns nan, which
    the caller's `+ 1e-8` cannot rescue; the loss goes nan, `clip_grad_norm_`
    propagates it and `optimiser.step()` writes nan into every parameter while
    the run keeps logging happily. `positions` varies per iteration, so this was
    a ~1-in-`minibatch` chance every iteration — around 40% over a 500-iteration
    run. The smallest remainder in the 592 logged iterations under `runs/` is 3,
    so it never fired. That was luck.
    """
    for size in range(2, 40):
        for minibatch in range(2, 12):
            chunks = list(
                ppo._minibatches(size, minibatch, torch.Generator().manual_seed(0))
            )
            assert chunks, f"no minibatches for size={size} minibatch={minibatch}"
            for chunk in chunks:
                assert len(chunk) >= 2, (
                    f"size={size} minibatch={minibatch} yielded {len(chunk)} row(s)"
                )
                assert len(chunk) <= minibatch + 1, "a fold may add one row, not more"
            # Still an exact partition of every row, each appearing once.
            assert sorted(torch.cat(chunks).tolist()) == list(range(size))


def test_the_kl_gauge_cannot_report_a_negative_divergence():
    """`(r - 1) - log r` is non-negative in exact arithmetic and was not in
    float32: `exp(x)` rounds to 1.0 below the ~1.2e-7 epsilon, so the estimator
    collapsed to `-log_ratio`. On-policy batches sit exactly there — cohort
    collection leaves log-ratios around 1e-9 from reduction order alone — and
    the gauge reported -5.8e-10 for a divergence that is really ~1.7e-19.
    """
    # A batch the current weights actually played: the log-ratios are not zero,
    # they are reduction-order noise around zero with a tiny drift. Symmetric
    # values would let the naive form's errors cancel and hide this.
    log_ratio = torch.tensor([1e-9, 2e-9, 5e-10, 3e-9, 1e-9], dtype=torch.float32)

    naive = ((log_ratio.exp() - 1) - log_ratio).mean()
    stable = (torch.expm1(log_ratio) - log_ratio).mean()

    assert naive < 0, "the defect this guards no longer reproduces"
    assert stable >= 0


def test_the_kl_gauge_is_unchanged_where_it_was_already_right():
    """Every reading on record sits above a log-ratio of ~1e-3, where the two
    forms agree — so no recorded figure is affected by the fix."""
    log_ratio = torch.tensor([0.01, -0.02, 0.05, 0.2], dtype=torch.float32)

    naive = ((log_ratio.exp() - 1) - log_ratio).mean()
    stable = (torch.expm1(log_ratio) - log_ratio).mean()

    assert stable == pytest.approx(float(naive), rel=1e-3)
