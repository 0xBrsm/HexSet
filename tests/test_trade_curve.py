# SPDX-License-Identifier: GPL-3.0-only
"""Protect the experiment against trade/RNG confounding and wrong pairing."""
import random
from types import SimpleNamespace

import pytest

from hexset.arena import deal_board, spawn
from hexset.bench.trade_curve import (
    WillingBot, blend_weights, comparisons, lineup, register_curve,
)
from hexset.bots.heximax import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, heximax
from hexset.game import start


def test_blending_changes_only_focal_move_weights():
    assert blend_weights(0) == NO_TRADE_WEIGHTS
    assert blend_weights(1) == TRADING_WEIGHTS
    register_curve()
    board = deal_board(44, 0)
    for alpha in (0, 0.5, 1):
        entrants = lineup(alpha, 0.2, 0.7)
        focal = spawn(entrants[0], board, random.Random(9))
        assert focal.bot.evaluator.weights == blend_weights(alpha)
        assert focal.gate.evaluator.weights == TRADING_WEIGHTS
        assert focal.initiate == focal.respond == 1
        for entrant in entrants[1:]:
            opponent = spawn(entrant, board, random.Random(9))
            assert opponent.bot.evaluator.weights == TRADING_WEIGHTS
            assert opponent.gate.evaluator.weights == TRADING_WEIGHTS
            assert (opponent.initiate, opponent.respond) == (0.2, 0.7)


def test_willingness_is_stable_monotone_and_role_specific():
    gate = SimpleNamespace(trade_floor=0.01)
    low = WillingBot(None, gate, "seed", 0.2, 0.7)
    high = WillingBot(None, gate, "seed", 0.8, 0.9)
    for active in (False, True):
        draws = [low.willing(t, active) for t in range(2000)]
        assert draws == [low.willing(t, active) for t in range(2000)]
        assert all(not value or high.willing(t, active) for t, value in enumerate(draws))
        expected = 0.2 if active else 0.7
        assert abs(sum(draws) / len(draws) - expected) < 0.04
    shut = WillingBot(None, gate, "seed", 0, 1)
    assert not shut.willing(10, True)
    assert shut.willing(10, False)


def test_declining_cannot_be_overcome_by_repeated_candidate_batches():
    calls = []
    gate = SimpleNamespace(trade_floor=0.01,
                           gains_many=lambda *args: calls.append(args) or [0.1])
    bot = WillingBot(None, gate, "seed", 0, 1)
    bot.game = SimpleNamespace(turns=5, current_player=0)
    view = SimpleNamespace(perspective=0)
    for _ in range(10):
        assert bot.gains_many(view, [(1, -1)], [1]) == [-1.0]
    assert not calls
    view.perspective = 1
    assert bot.gains_many(view, [(1, -1)], [0]) == [0.1]


def test_unrestricted_wrapper_preserves_policy_rng_and_opening():
    register_curve()
    board = deal_board(33, 0)
    wrapped = spawn(lineup(1, 1, 1)[0], board, random.Random(17))
    plain = heximax(board, random.Random(17))
    assert wrapped.bot.rng.getstate() == plain.rng.getstate()
    game = start(board, 4, random.Random(1))
    assert wrapped.choose(game) == plain.choose(game)
    assert wrapped.bot.rng.getstate() == plain.rng.getstate()


def test_comparison_pairs_same_games_then_clusters_by_board():
    def cell(alpha, winners):
        return dict(alpha=alpha, initiate=0.5, respond=0.5,
                    experiment={"outcomes": [dict(winner=w, points=[v, 0, 0, 0])
                                              for w, v in zip(winners, [10, 8, 7, 9])]})
    baseline = cell(1, [0, 1, 1, 1])
    challenger = cell(0.5, [0, 0, 0, 1])
    result, = comparisons([baseline, challenger])
    assert result["win_rate_difference"]["mean"] == 0.5
    assert result["win_rate_difference"]["samples"] == 2
    assert result["focal_vp_difference"]["mean"] == 0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_blends(value):
    with pytest.raises(ValueError):
        blend_weights(value)


def test_zero_floor_is_the_default_for_every_player():
    register_curve()
    board = deal_board(44, 0)
    for entrant in lineup(0.5, 0.2, 0.7):
        bot = spawn(entrant, board, random.Random(9))
        assert bot.trade_floor == bot.gate.trade_floor == 0
        assert bot.gate.evaluator.weights == TRADING_WEIGHTS


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_diagnostic_floor(value):
    register_curve()
    with pytest.raises(ValueError):
        spawn(lineup(1, 1, 1, gate_floor=value)[0], deal_board(44, 0), random.Random(9))


@pytest.mark.slow
def test_zero_floor_wrapper_preserves_full_game_and_trades():
    from hexset.arena import _play_and_record
    register_curve()
    board = deal_board(410000, 0)
    entrants = lineup(1, 1, 1)
    wrapped = [spawn(e, board, random.Random(f"410000:0:{i}"))
               for i, e in enumerate(entrants)]
    plain = [heximax(board, random.Random(f"410000:0:{i}")) for i in range(4)]
    for bot in plain:
        bot.trade_floor = 0
    _, wrapped_record, wrapped_trades = _play_and_record(wrapped, board, 410000, 0, 20000)
    _, plain_record, plain_trades = _play_and_record(plain, board, 410000, 0, 20000)
    assert plain_trades
    assert wrapped_record == plain_record
    assert wrapped_trades == plain_trades
