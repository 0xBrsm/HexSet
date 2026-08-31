# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from benchmarks.floor import Branching, Sampling, Snapshot, collect, pool, split
from hexset.actions import apply
from hexset.selfplay import Choice

from test_mcts import a_game
from test_sibling import a_request


class Valued:
    """A policy that plays the first option and claims a value, as a net would."""

    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.seen = []

    def act(self, requests):
        self.seen.extend(request.game for request in requests)
        return [
            Choice(action=request.options[0], value=(self.value, 0.0, 0.0, 0.0))
            for request in requests
        ]


class Silent:
    """A scripted policy: no value, so nothing to compare a rollout against."""

    def act(self, requests):
        return [Choice(action=request.options[0]) for request in requests]


def test_the_error_splits_into_floor_and_bias_exactly():
    returns = np.array([0.4, -0.2, 0.1, 0.7, -0.5])
    prediction = 0.15
    floor, bias = split(returns, prediction)
    assert floor + bias == pytest.approx(float(((returns - prediction) ** 2).mean()))


def test_a_prediction_at_the_mean_leaves_only_the_floor():
    returns = np.array([0.4, -0.2, 0.1, 0.7, -0.5])
    floor, bias = split(returns, float(returns.mean()))
    assert bias == pytest.approx(0.0)
    assert floor == pytest.approx(float(returns.var()))


def test_pool_recovers_the_mean_and_the_share_over_all_rows():
    rows = [
        {"floor": 0.1, "bias_squared": 0.01},
        {"floor": 0.2, "bias_squared": 0.02},
    ]
    got = pool(rows)
    assert got["mean_floor"] == pytest.approx(0.15)
    assert got["mean_bias_squared"] == pytest.approx(0.015)
    assert got["mean_squared_error"] == pytest.approx(0.165)
    assert got["irreducible_share"] == pytest.approx(0.15 / 0.165, abs=1e-4)


def test_pool_weights_by_row_not_by_shard():
    """What a merge across shards of different sizes relies on.

    Concatenating a 1-row shard and a 3-row shard and pooling once must weight
    3:1 by position; averaging the two shards' own means first would give 0.5
    instead of the 0.75 the union actually has.
    """
    small_shard = [{"floor": 0.0, "bias_squared": 0.0}]
    big_shard = [{"floor": 1.0, "bias_squared": 0.0}] * 3
    got = pool(small_shard + big_shard)
    assert got["mean_floor"] == pytest.approx(0.75)


def test_a_sampled_decision_keeps_a_copy_and_not_the_live_position():
    policy = Sampling(Valued(), rate=1.0, rng=random.Random(0))
    game = a_game()
    options = [a for a in _options(game)]
    (choice,) = policy.act([a_request(game, tuple(options))])
    assert isinstance(choice.aux, Snapshot)
    assert choice.aux.prediction == 0.25
    turns_before = choice.aux.game.turns
    for _ in range(3):
        apply(game, _options(game)[0])
    assert choice.aux.game is not game
    assert choice.aux.game.turns == turns_before


def test_a_policy_with_no_value_is_never_sampled():
    policy = Sampling(Silent(), rate=1.0, rng=random.Random(0))
    game = a_game()
    (choice,) = policy.act([a_request(game, tuple(_options(game)))])
    assert choice.aux is None


def test_every_branched_lane_starts_from_the_given_position():
    game = a_game()
    for _ in range(6):
        apply(game, _options(game)[0])
    policy = Valued()
    branch = Branching(
        policy, game, rng=random.Random(0), lanes=4, players=4, seed=1, deal=4
    )
    branch.tick()
    assert len(policy.seen) == 4
    assert {g.turns for g in policy.seen} == {game.turns}
    assert {g.current_player for g in policy.seen} == {game.current_player}


def test_branching_leaves_the_position_it_was_given_alone():
    game = a_game()
    for _ in range(6):
        apply(game, _options(game)[0])
    before = (game.turns, game.current_player, game.state.deck[:])
    branch = Branching(
        Valued(), game, rng=random.Random(0), lanes=4, players=4, seed=1, deal=4
    )
    for _ in range(5):
        branch.tick()
    assert (game.turns, game.current_player, game.state.deck[:]) == before


def test_collect_finds_the_snapshots_and_dates_them():
    class Outcome:
        actions = 21

    class Transition:
        def __init__(self, aux, step):
            self.aux = aux
            self.step = step

    class Episode:
        pass

    kept = Snapshot(game=None, seat=2, prediction=-0.1)
    episode = Episode()
    episode.outcome = Outcome()
    episode.trajectories = [
        [Transition(kept, 10), Transition(None, 11)],
        [Transition("not a snapshot", 3)],
    ]
    (found,) = collect([episode])
    assert found[0] is kept
    assert found[1] == pytest.approx(0.5)


def _options(game):
    from hexset.actions import legal_actions

    return list(legal_actions(game))


def test_dump_returns_leaves_the_default_output_bit_identical(tmp_path, capsys):
    """`--dump-returns` is a side channel, not a mode switch.

    The flag defaults off, and every recorded floor run was made with it off,
    so the only thing this change may do to an unflagged run is add a file --
    stdout, and everything that fed it, must come out exactly as before. This
    runs the real CLI twice, once with the flag and once without, on the same
    seed, and diffs stdout byte for byte rather than trusting that reading the
    diff by eye would have caught a shift.
    """
    pytest.importorskip("torch", reason="PyTorch runs on the training box only")
    from hexset.netbot import load

    from benchmarks.floor import main
    from test_netbot import a_checkpoint

    checkpoint = tmp_path / "latest.pt"
    a_checkpoint(checkpoint, max_offers=1, seed=0)
    load.cache_clear()

    argv = [
        "--checkpoint",
        str(checkpoint),
        "--seed-games",
        "6",
        "--positions",
        "2",
        "--rollouts",
        "3",
        "--action-cap",
        "200",
        "--seed",
        "0",
        "--json",
    ]

    assert main(list(argv)) == 0
    without_flag = _drop_timings(capsys.readouterr().out)

    load.cache_clear()
    dump_path = tmp_path / "dump.json"
    assert main(argv + ["--dump-returns", str(dump_path)]) == 0
    with_flag = _drop_timings(capsys.readouterr().out)

    assert with_flag == without_flag
    dumped = json.loads(dump_path.read_text())
    assert len(dumped["positions"]) == without_flag["positions"]
    for position in dumped["positions"]:
        assert set(position) == {"progress", "seat", "prediction", "returns"}
        assert len(position["returns"]) == without_flag["rollouts_each"]
    load.cache_clear()


def _drop_timings(payload_json: str) -> dict:
    """Strip the two wall-clock fields, which vary run to run by construction
    and are not part of what "bit-identical" means here."""
    payload = json.loads(payload_json)
    payload.pop("seconds", None)
    payload.pop("seed_seconds", None)
    return payload
