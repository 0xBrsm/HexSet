# SPDX-License-Identifier: GPL-3.0-only
"""A seat's reading of the hidden cards reaches what draws worlds outside its
search: the robber deals' beliefs (`ValueDealer._beliefs`) draw by the
network's own `hold`, and a declared trader with no reading of its own takes
the seat's (`hexset.arena.traded`). Without a reading, both are what they
were, draw for draw."""

from __future__ import annotations

import random
from types import SimpleNamespace

from hexset.arena import Entrant, register_entrant_kind, register_preset, traded
from hexset.bots import TradesBy
from hexset.deals import STEAL_SAMPLES, ValueDealer, _Plan
from hexset.view import View

from test_view_shown import PLAIN, WEIGHED, WOOD, untyped_game


def empty_plan() -> _Plan:
    return _Plan(("test",), {}, [], [], -1, {}, {}, {}, {}, {}, {}, {}, None)


def beliefs(view: View, hold, seed: int) -> _Plan:
    plan = empty_plan()
    ValueDealer(SimpleNamespace(hold=hold), None, random.Random(seed))._beliefs(view, plan, [1, 2])
    return plan


def test_robber_deal_beliefs_without_a_reading_are_the_alike_draws():
    game = untyped_game()
    game.ledger.show(1, [1, 0, 0, 0, 0])            # seat 1 asks for wood
    view = View.from_game(game, 0)
    plan = beliefs(view, None, seed=3)
    rng = random.Random(3)
    alike = [view.sample(rng) for _ in range(STEAL_SAMPLES)]
    assert plan.sample.hands == alike[0].hands
    held = sum(1 for w in alike if w.hands[1][WOOD]) / STEAL_SAMPLES
    assert abs(plan.held.get((1, WOOD), 0.0) - held) < 1e-12


def test_robber_deal_beliefs_draw_by_the_networks_reading():
    game = untyped_game()
    game.ledger.show(1, [1, 0, 0, 0, 0])
    view = View.from_game(game, 0)
    # Seat 1 holds two of the four pool cards; the three pairs with wood
    # weigh 0.25 against three without.
    exact = 3 * 0.25 / (3 * 0.25 + 3)
    runs = 60
    weighed = sum(beliefs(view, WEIGHED, seed).held.get((1, WOOD), 0.0) for seed in range(runs)) / runs
    plain = sum(beliefs(view, PLAIN, seed).held.get((1, WOOD), 0.0) for seed in range(runs)) / runs
    assert abs(weighed - exact) < 0.03
    assert abs(plain - 0.5) < 0.03


class _Reads:
    """A trader that reads hidden cards: `hold` its reading, `None` for none."""

    def __init__(self, hold=None) -> None:
        self.hold = hold


_OWN = object()
register_entrant_kind("test-reads-none", lambda entrant, board, rng: _Reads())
register_preset("test-reads-none", Entrant("test-reads-none", kind="test-reads-none"), listed=False)
register_entrant_kind("test-reads-own", lambda entrant, board, rng: _Reads(_OWN))
register_preset("test-reads-own", Entrant("test-reads-own", kind="test-reads-own"), listed=False)


def gate_of(bot) -> object:
    assert isinstance(bot, TradesBy)
    return bot.trader


def test_a_trader_with_no_reading_takes_the_seats():
    game = untyped_game()
    board, mover = game._state.board, object()
    assert gate_of(traded(mover, "test-reads-none", board, random.Random(0), hold=WEIGHED)).hold is WEIGHED
    assert gate_of(traded(mover, "test-reads-none", board, random.Random(0))).hold is None


def test_a_trader_keeps_its_own_reading():
    game = untyped_game()
    board, mover = game._state.board, object()
    assert gate_of(traded(mover, "test-reads-own", board, random.Random(0), hold=WEIGHED)).hold is _OWN


def test_in_a_duel_the_robber_deal_reads_alike_with_or_without_a_reading():
    """Two seats see every card the other gains, so no resource card is
    untyped and a reading of the hidden cards can move only the other seat's
    development cards. The steal's odds, every drawn hand and what the
    seat's own inputs show of each drawn world (the record, and each duel
    contract's observation) are the same with a reading as without one; only the other seat's development cards differ."""
    from hexset.actions import space_for
    from hexset.contract9 import DUEL_CONTRACTS, observation_row
    from hexset.board.board import random_base_board
    from hexset.game import imagine, is_over, start, to_move
    from hexset.onnx_record import record_from_game
    from hexset.play import step_randomly
    import numpy as np

    positions = devs_moved = 0
    for seed in range(6):
        rng = random.Random(seed)
        game = start(random_base_board(rng), 2, rng)
        space = space_for(game)
        for step in range(600):
            if is_over(game):
                break
            step_randomly(game, rng)
            if step % 7:
                continue
            seat = to_move(game)
            other = 1 - seat
            view = View.from_game(game, seat)
            assert view.shown_weights(PLAIN) is None
            plain = empty_plan()
            ValueDealer(SimpleNamespace(hold=None), None, random.Random(step))._beliefs(view, plain, [other])
            read = empty_plan()
            ValueDealer(SimpleNamespace(hold=PLAIN), None, random.Random(step))._beliefs(view, read, [other])
            assert read.odds == plain.odds and read.held == plain.held
            assert read.holding.keys() == plain.holding.keys()
            for a, b in [(read.sample, plain.sample)] + [(read.holding[k], plain.holding[k]) for k in plain.holding]:
                assert a.hands == b.hands
                devs_moved += a.dev_cards[other] != b.dev_cards[other]
                worlds = []
                for world in (a, b):
                    g = imagine(game, random.Random(0), randomize_deck=False, seat=seat)
                    g.set_state(world)
                    worlds.append(record_from_game(g, seat, space))
                    for contract in DUEL_CONTRACTS:
                        worlds[-1][contract] = observation_row(g, seat, contract)
                assert worlds[0].keys() == worlds[1].keys()
                for field in worlds[0]:
                    assert np.array_equal(worlds[0][field], worlds[1][field]), field
            positions += 1
    assert positions > 100 and devs_moved > 0
