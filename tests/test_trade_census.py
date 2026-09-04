# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

from hexset.arena import base_name, lineup_from_names
from hexset.bench.trade_census import (
    CensusResult,
    TradeRecord,
    _bundle_category,
    _resource_split,
    run_census,
    summarize,
)


def _record(**overrides) -> TradeRecord:
    base = dict(
        game=0,
        turn=1,
        phase="MAIN",
        seat_a=0,
        seat_b=1,
        name_a="alice",
        name_b="bob",
        given_a=(0, 0, 0, 0, 0),
        given_b=(0, 0, 0, 0, 0),
        hand_before_a=3,
        hand_before_b=3,
        surplus_a=0.1,
        surplus_b=0.05,
        larger_surplus="a",
    )
    base.update(overrides)
    return TradeRecord(**base)


def test_resource_split_reads_the_signed_bundle_towards_a():
    given_a, given_b = _resource_split((-1, 0, 2, 0, 0))
    assert given_a == (1, 0, 0, 0, 0)
    assert given_b == (0, 0, 2, 0, 0)


def test_bundle_category_is_symmetric_in_the_two_sides():
    assert _bundle_category(1, 1) == "1:1"
    assert _bundle_category(2, 1) == "2:1"
    assert _bundle_category(1, 2) == "2:1"
    assert _bundle_category(2, 2) == "2:2"
    assert _bundle_category(3, 1) == "3+:1"
    assert _bundle_category(1, 4) == "3+:1"
    assert _bundle_category(3, 2) == "other"


def test_summarize_reorients_each_trade_per_side_on_a_synthetic_list():
    # alice gives 1 card and gets 3 back every trade -- an obviously
    # imbalanced, value-positive-for-alice pattern -- so the summary should
    # show alice's mean_received > mean_given and bob's the reverse, with one
    # trade side row per participant per trade.
    trades = [
        _record(given_a=(1, 0, 0, 0, 0), given_b=(0, 0, 0, 3, 0)),
        _record(given_a=(0, 1, 0, 0, 0), given_b=(0, 0, 3, 0, 0), hand_before_a=9),
    ]
    result = CensusResult(games=1, trades=trades, winners=[0], turns=[10])
    summaries = summarize(result, ["alice", "bob"])

    alice, bob = summaries["alice"], summaries["bob"]
    assert alice.trade_sides == 2
    assert bob.trade_sides == 2
    assert alice.mean_given == pytest.approx(1.0)
    assert alice.mean_received == pytest.approx(3.0)
    assert bob.mean_given == pytest.approx(3.0)
    assert bob.mean_received == pytest.approx(1.0)
    # both trades are 3-for-1: bulk on the bob side, 3+:1 bucket throughout.
    assert alice.bundle_distribution == {"3+:1": 2}
    assert alice.bulk_share == pytest.approx(1.0)
    # one of alice's two trades started with a 9-card hand.
    assert alice.dump_share == pytest.approx(0.5)
    assert bob.dump_share == pytest.approx(0.0)
    # alice nets +2 cards at the 4:1 bank rate each trade: (3-1)/4.
    assert alice.mean_value_swing == pytest.approx(0.5)
    assert bob.mean_value_swing == pytest.approx(-0.5)


def test_summarize_keeps_a_bot_present_with_zero_trades():
    result = CensusResult(games=1, trades=[], winners=[None], turns=[5])
    summaries = summarize(result, ["nobody"])
    assert summaries["nobody"].trade_sides == 0
    assert summaries["nobody"].mean_given == 0.0
    assert summaries["nobody"].bundle_distribution == {}


@pytest.mark.slow
def test_four_game_smoke_against_heximax():
    """A real, short census: heximax self-play, 4 games (one full rotation).

    Not a behaviour-preservation test -- it just exercises the whole
    instrumented play loop (trade harvesting before and after `apply`, the
    per-turn reset on `end_turn`) against the real engine, and checks the
    census comes back internally consistent rather than empty or malformed.
    """
    entrants = lineup_from_names(["heximax", "heximax", "heximax", "heximax"])
    result = run_census(entrants, games=4, seed=7, workers=1)

    assert result.games == 4
    assert len(result.winners) == 4
    assert len(result.turns) == 4
    assert len(result.points) == 4
    assert all(len(row) == 4 for row in result.points)
    assert result.trades, "heximax self-play produced no trades at all -- unexpected"

    for t in result.trades:
        assert t.seat_a != t.seat_b
        assert sum(t.given_a) >= 1 or sum(t.given_b) >= 1
        assert t.hand_before_a >= sum(t.given_a)
        assert t.hand_before_b >= sum(t.given_b)
        assert t.larger_surplus in ("a", "b", "tie")

    names = sorted({base_name(e.name) for e in entrants})
    summaries = summarize(result, names)
    assert summaries["heximax"].trade_sides == 2 * len(result.trades)
