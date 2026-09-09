# SPDX-License-Identifier: GPL-3.0-only
"""Research-tool accounting, with fixed outcomes and bounded engine steps."""
from types import SimpleNamespace

import pytest

from hexset.arena import ClearedTrade, Entrant
from hexset.bench import trade_census
from hexset.bench.profile_heximax import TimedBot, play_one_game


def test_census_maps_rotated_seats_and_preserves_resource_direction(monkeypatch):
    trade = ClearedTrade(
        step=12, turn=3, phase="MAIN", a=0, b=1,
        received=(-2, 1, 0, 0, 0), hand_a=8, hand_b=4,
        gain_a=0.1, gain_b=0.2,
    )
    def compete(*args, **kwargs):
        return SimpleNamespace(
            winners=(1,), turns=(10,), points=((8, 10),), records=("record",),
            cleared=((trade,),), seating=((1, 0),),
        )

    monkeypatch.setattr(trade_census, "compete", compete)
    result = trade_census.run_census(
        [Entrant("first"), Entrant("second")], 1, records=True,
    )
    row, = result.trades
    assert (row.name_a, row.name_b) == ("second", "first")
    assert row.given_a == (2, 0, 0, 0, 0)
    assert row.given_b == (0, 1, 0, 0, 0)
    assert result.records == ["record"]
    summary = trade_census.summarize(result, ["first", "second"])
    assert summary["second"].dump_share == 1.0
    assert summary["first"].mean_value_swing == pytest.approx(0.25)
    assert summary["second"].trades_per_turn == pytest.approx(0.1)


def test_profiling_preserves_the_trade_gate_and_obeys_the_action_cap():
    class Gate:
        trade_floor = 0.03

        def gains_many(self, view, received, counterparties):
            return [0.1] * len(received)

    wrapped = TimedBot(Gate(), [])
    assert wrapped.trade_floor == 0.03
    assert wrapped.gains_many(None, [(1, 0, 0, 0, 0)], [1]) == [0.1]
    game, times = play_one_game("random", seed=3, action_cap=12)
    assert len(times) == 12
    assert all(t >= 0 for t in times)
    assert len(game.gates) == 4
