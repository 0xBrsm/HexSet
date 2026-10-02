# SPDX-License-Identifier: GPL-3.0-only
"""The one-event trade mechanic: gates return magnitudes, the table clears the
deal `Game.trade_rule` ranks highest, and nothing is published.
"""

from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.game import Phase, enter_main, end_turn, imagine, start
from dataclasses import replace

from hexset.trading import (
    ENUMERATION_CARDS,
    UNLIMITED,
    Trade,
    TradeParams,
    TradeProtocol,
    bundle,
    execute_agreed,
    install,
    one_for_one,
    trade_event,
    valued,
)
from hexset.trading._engine import _candidates
from helpers import give

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)


def a_game(players: int = 4):
    rng = random.Random(0)
    game = start(random_base_board(rng), players, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


def stocked(*hands: tuple[int, Resource, int]):
    game = a_game()
    for player, resource, count in hands:
        give(game._state, player, resource, count)
    return game


class Trader:
    """Prices every candidate with `gain(received, counterparty)`; the default never
    trades. Candidates land in `asked` in the order the batched call got them.
    """

    trade_floor = 0.0

    def __init__(self, gain=lambda received, counterparty: -1.0):
        self.gain = gain
        self.asked: list[tuple[tuple[int, ...], int]] = []
        self.calls = 0

    def gains_many(self, view, received, counterparties):
        self.calls += 1
        out = []
        for r, c in zip(received, counterparties):
            r = tuple(r)
            self.asked.append((r, c))
            out.append(self.gain(r, c))
        return out


def wants(resource: int, magnitude: float = 1.0):
    """Positive only when the candidate hands the seat more of `resource`; a
    direction-blind gate would ping-pong.
    """
    return lambda received, counterparty: magnitude if received[resource] > 0 else -1.0


def _unused_gate(seat, view, received, other):
    raise AssertionError("the single-ask gate must not be used when game.gates is seated")


def run(game, traders) -> list[Trade]:
    game.gates = tuple(traders)
    return trade_event(game, _unused_gate)


def ab_received(trades: list[Trade]) -> list[tuple[int, int, tuple[int, ...]]]:
    return [(t.a, t.b, t.received) for t in trades]


def _old_hand_multisets(hand):
    """The uncapped brute-force reference the next test filters against."""
    resources = [r for r in range(NUM_RESOURCES) if hand[r] > 0]
    counts = [0] * NUM_RESOURCES

    def walk(idx):
        if idx == len(resources):
            if any(counts):
                yield tuple(counts)
            return
        r = resources[idx]
        for n in range(hand[r] + 1):
            counts[r] = n
            yield from walk(idx + 1)
        counts[r] = 0

    yield from walk(0)


def _old_candidates(state, me, locked):
    give_options = list(_old_hand_multisets(state.hands[me]))
    for them in range(state.num_players):
        if them == me or them in locked:
            continue
        receive_options = list(_old_hand_multisets(state.hands[them]))
        for given in give_options:
            for received in receive_options:
                if any(g and r for g, r in zip(given, received)):
                    continue
                yield them, tuple(r - g for r, g in zip(received, given))


def _sides(received) -> tuple[int, int]:
    given = sum(-n for n in received if n < 0)
    got = sum(n for n in received if n > 0)
    return given, got


def test_candidates_never_exceeds_the_card_cap_but_keeps_every_bundle_under_it():
    game = stocked(
        (0, Resource.WOOD, 5),
        (0, Resource.BRICK, 4),
        (0, Resource.SHEEP, 2),
        (1, Resource.WHEAT, 5),
        (1, Resource.ORE, 4),
    )
    state = game._state
    new = set(_candidates(state, 0, game.locked, ENUMERATION_CARDS))
    assert new, "a rich hand must still find candidates under the cap"
    for _them, received in new:
        given, got = _sides(received)
        assert given <= ENUMERATION_CARDS
        assert got <= ENUMERATION_CARDS

    old_at_or_under_cap = {
        (them, received)
        for them, received in _old_candidates(state, 0, game.locked)
        if all(n <= ENUMERATION_CARDS for n in _sides(received))
    }
    assert new == old_at_or_under_cap


def test_a_deal_both_sides_gain_from_clears():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(wants(ORE)),
        Trader(wants(WOOD)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert ab_received(done) == [(0, 1, one_for_one(WOOD, ORE))]
    assert done[0].gain_a == 1.0 and done[0].gain_b == 1.0
    assert game._state.hands[0][ORE] == 1
    assert game._state.hands[1][WOOD] == 1
    assert game.trades_made == 1
    assert game.trades == done


@pytest.mark.parametrize("zeroed", [1])
def test_either_side_priced_at_zero_or_below_vetoes_the_deal(zeroed):
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    gains = [1.0, 1.0]
    gains[zeroed] = 0.0
    traders = [
        Trader(lambda r, c: gains[0]),
        Trader(lambda r, c: gains[1]),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_only_the_current_player_trades():
    game = stocked((1, Resource.WOOD, 1), (2, Resource.ORE, 1))
    traders = [
        Trader(),
        Trader(lambda r, c: 1.0),
        Trader(lambda r, c: 1.0),
        Trader(),
    ]
    assert run(game, traders) == []


def test_a_locked_seat_is_never_a_counterparty():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.locked = frozenset({1})
    traders = [
        Trader(lambda r, c: 1.0),
        Trader(lambda r, c: 1.0),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_the_ledger_certifies_what_a_trade_moved():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [Trader(wants(ORE)), Trader(wants(WOOD)), Trader(), Trader()]
    run(game, traders)
    assert game.ledger.seats[1].known[WOOD] == 1


def test_the_event_keeps_going_while_anything_clears():
    """Uncapped: the default of one exchange a turn would stop after the first."""
    game = stocked(
        (0, Resource.WOOD, 3), (0, Resource.BRICK, 3), (1, Resource.ORE, 3), (2, Resource.WHEAT, 3)
    )

    def seat0_gain(received, counterparty):
        if counterparty == 1:
            return 1.0 if received == bundle(wood=-3, ore=3) else -1.0
        if counterparty == 2:
            return 1.0 if received == bundle(brick=-3, wheat=3) else -1.0
        return -1.0

    traders = [
        Trader(seat0_gain),
        Trader(lambda r, c: 1.0 if r == bundle(ore=-3, wood=3) else -1.0),
        Trader(lambda r, c: 1.0 if r == bundle(wheat=-3, brick=3) else -1.0),
        Trader(),
    ]
    done = run(game, traders)
    assert ab_received(done) == [
        (0, 1, bundle(wood=-3, ore=3)),
        (0, 2, bundle(brick=-3, wheat=3)),
    ]
    assert game._state.hands[0] == [0, 0, 0, 3, 3]
    assert game._state.hands[1] == [3, 0, 0, 0, 0]
    assert game._state.hands[2] == [0, 3, 0, 0, 0]


def test_a_gate_at_zero_offers_is_the_off_switch():
    """The switch is the participant's, not the table's: a gate that makes no
    offers opens nothing and signs nothing."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(lambda r, c: 1.0),
        Trader(lambda r, c: 1.0),
        Trader(),
        Trader(),
    ]
    traders[0].trade_params = TradeParams(max_offers=0)
    assert run(game, traders) == []
    assert not traders[0].asked, "a switched-off event should not even ask a gate"


def test_a_counterparty_at_zero_offers_signs_nothing_either():
    """The clearing house never deals with a gate that does not trade, even
    as the counterparty to a deal it would price above its floor."""
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
    traders = [Trader(wants(ORE)), Trader(wants(WOOD)), Trader(wants(WOOD)), Trader()]
    traders[1].trade_params = TradeParams(max_offers=0)
    done = run(game, traders)
    assert done and all(trade.b == 2 for trade in done)
    assert not traders[1].asked


def test_a_boolean_gates_tie_clears_the_smallest_exchange():
    """Every candidate both sides want prices the same, so the tie-break
    decides, and it decides for the fewest cards moved."""
    game = stocked((0, Resource.WOOD, 3), (1, Resource.ORE, 3))
    traders = [Trader(wants(ORE)), Trader(wants(WOOD)), Trader(), Trader()]
    traders[0].trade_params = TradeParams(max_offers=1)
    [trade] = run(game, traders)
    assert trade.received == one_for_one(WOOD, ORE)


def test_the_count_and_the_log_reset_with_the_turn():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    run(game, [Trader(wants(ORE)), Trader(wants(WOOD)), Trader(), Trader()])
    assert game.trades_made == 1
    end_turn(game)
    assert game.trades_made == 0
    assert game.trades == []


# --- the trade rule: egalitarian (default), nash, actor ------------------------
#
#
#   counterparty 1: (mine=5,    theirs=5)    egalitarian key 5,   nash 25,   actor 5
#   counterparty 2: (mine=20,   theirs=0.1)  egalitarian key 0.1, nash 2,    actor 20
#   counterparty 3: (mine=4.9,  theirs=100)  egalitarian key 4.9, nash 490,  actor 4.9
#


def _rule_fixture():
    game = stocked(
        (0, Resource.WOOD, 1), (1, Resource.ORE, 1), (2, Resource.ORE, 1), (3, Resource.ORE, 1)
    )
    seat0_gain_by_counterparty = {1: 5.0, 2: 20.0, 3: 4.9}
    counterparty_gain = {1: 5.0, 2: 0.1, 3: 100.0}

    def seat0_gain(received, counterparty):
        return seat0_gain_by_counterparty[counterparty] if received[ORE] > 0 else -1.0

    traders = [
        Trader(seat0_gain),
        Trader(wants(WOOD, counterparty_gain[1])),
        Trader(wants(WOOD, counterparty_gain[2])),
        Trader(wants(WOOD, counterparty_gain[3])),
    ]
    return game, traders


def test_egalitarian_is_the_default_and_maximises_the_smaller_gain():
    game, traders = _rule_fixture()
    assert game.trade_rule == "egalitarian"
    done = run(game, traders)
    assert len(done) == 1
    assert done[0].b == 1
    assert done[0].gain_a == 5.0 and done[0].gain_b == 5.0


def test_actor_rule_maximises_the_current_players_own_gain():
    game, traders = _rule_fixture()
    game.trade_rule = "actor"
    done = run(game, traders)
    assert len(done) == 1
    assert done[0].b == 2
    assert done[0].gain_a == 20.0


def test_nash_rule_maximises_the_product_of_both_gains():
    game, traders = _rule_fixture()
    game.trade_rule = "nash"
    done = run(game, traders)
    assert len(done) == 1
    assert done[0].b == 3
    assert done[0].gain_a == pytest.approx(4.9) and done[0].gain_b == pytest.approx(100.0)


def test_each_seat_is_held_to_its_own_floor_and_no_floor_is_borrowed():
    """A gate declaring no floor is refused: there is no table default."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    strict = Trader(wants(ORE, 1.0))
    strict.trade_floor = 1.5
    easy = Trader(wants(WOOD, 1.0))
    assert run(game, [strict, easy, Trader(), Trader()]) == [], "the actor's own floor refuses its 1.0"

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    done = run(game, [Trader(wants(ORE, 1.0)), Trader(wants(WOOD, 1.0)), Trader(), Trader()])
    assert len(done) == 1, "the same gains clear at floor 0.0"

    class Undeclared:
        def gains_many(self, view, received, counterparties):
            return [1.0] * len(received)

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    with pytest.raises(TypeError, match="declares no trade_floor"):
        run(game, [Undeclared(), Trader(wants(WOOD, 1.0)), Trader(), Trader()])


def test_an_event_that_comes_back_to_a_position_ends_there():
    """Gates read a sampled world, so a revisit is an ordinary way to end."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    done = run(game, [Trader(lambda r, c: 1.0), Trader(lambda r, c: 1.0), Trader(), Trader()])
    assert done, "the first exchange clears"
    assert all(n >= 0 for hand in game._state.hands for n in hand)
    assert sum(sum(hand) for hand in game._state.hands) == 2


def test_a_bot_with_no_trading_surface_never_trades():
    class JustChoose:
        def choose(self, game):  # pragma: no cover -- not exercised here
            raise NotImplementedError

    view = a_game().state(0)
    assert valued(JustChoose(), view, one_for_one(WOOD, ORE), 1) == -1.0

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [JustChoose(), Trader(lambda r, c: 1.0), Trader(), Trader()]
    assert run(game, traders) == []


def test_an_imagined_game_does_not_carry_the_seated_gates():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.gates = tuple(Trader(lambda r, c: 1.0) for _ in range(4))
    child = imagine(game, random.Random(1))
    assert child.gates is None
    child.phase = Phase.MAIN
    enter_main(child)
    assert child.trades == []


def test_pending_is_cleared_at_the_start_of_every_trade_event():
    game = stocked((0, Resource.WOOD, 1))
    game.pending.append(Trade(1, 0, one_for_one(WOOD, ORE)))
    game.gates = tuple(Trader() for _ in range(4))
    trade_event(game, _unused_gate)
    assert game.pending == []


def _seated(game, traders):
    game.gates = tuple(traders)


def put_to(game, proposer, counterparty, received):
    """A bundle `proposer` composed by hand and put to a bot `counterparty`:
    submitting is the proposer's own consent, so only the counterparty's
    gate is asked."""
    return execute_agreed(
        game, proposer, counterparty, received, ask_actor=False, ask_counterparty=True
    )


def test_a_hand_composed_exchange_clears_on_coverage_and_the_counterpartys_gain():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    received = one_for_one(WOOD, ORE)
    trade = put_to(game, 0, 1, received)
    assert trade.a == 0 and trade.b == 1 and trade.received == received
    assert trade.gain_b == 1.0
    assert game._state.hands[0] == [0, 0, 0, 0, 1]
    assert game._state.hands[1] == [1, 0, 0, 0, 0]
    assert game.trades == [trade]
    assert game.trades_made == 1


def test_a_hand_composed_exchange_rejects_whichever_side_cannot_cover_it():
    proposer_short = stocked((1, Resource.ORE, 1))
    _seated(proposer_short, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="seat 0 cannot cover"):
        put_to(proposer_short, 0, 1, one_for_one(WOOD, ORE))

    counterparty_short = stocked((0, Resource.WOOD, 1))
    _seated(counterparty_short, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="seat 1 cannot cover"):
        put_to(counterparty_short, 0, 1, one_for_one(WOOD, ORE))


def test_a_hand_composed_exchange_refuses_a_counterparty_gain_under_a_nonzero_floor():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    fussy = Trader(lambda r, c: 1.0)
    fussy.trade_floor = 2.0
    _seated(game, [Trader(), fussy, Trader(), Trader()])
    with pytest.raises(ValueError, match="does not want"):
        put_to(game, 0, 1, one_for_one(WOOD, ORE))


def test_a_hand_composed_exchange_never_asks_the_proposers_own_gate():

    class Boom:
        def gains_many(self, view, received, counterparties):
            raise AssertionError("the proposer's own gate must never be asked")

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    _seated(game, [Boom(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    trade = put_to(game, 0, 1, one_for_one(WOOD, ORE))
    assert trade.a == 0 and trade.b == 1


def test_a_hand_composed_exchange_rejects_a_seat_that_is_neither_proposer_nor_current_player():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.current_player = 2
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="neither seat"):
        put_to(game, 0, 1, one_for_one(WOOD, ORE))


def test_a_hand_composed_exchange_requires_main_phase():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.phase = Phase.ROLL
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="MAIN"):
        put_to(game, 0, 1, one_for_one(WOOD, ORE))


def test_the_referee_caps_nothing_and_the_counterparty_caps_itself():
    """There is no table card rule. A four-card side is legal if the seat
    being asked will sign it, and refused if that seat declares it will not --
    which is the same answer at the same place as every other limit."""
    four_given = stocked((0, Resource.WOOD, 4), (1, Resource.ORE, 1))
    willing = Trader(lambda r, c: 1.0)
    _seated(four_given, [Trader(), willing, Trader(), Trader()])
    received = [0, 0, 0, 0, 0]
    received[WOOD] = -4
    received[ORE] = 1
    # The counterparty declares no cap, so its own valuation is the whole test.
    assert put_to(four_given, 0, 1, tuple(received)).received == tuple(received)

    narrow = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 4))
    capped = Trader(lambda r, c: 1.0)
    capped.trade_params = replace(UNLIMITED, max_give_cards=3)
    install(capped, TradeProtocol(capped, capped.trade_params, seed=0))
    _seated(narrow, [Trader(), capped, Trader(), Trader()])
    received = [0, 0, 0, 0, 0]
    received[WOOD] = -1
    received[ORE] = 4  # four out of the counterparty, which it will not do
    with pytest.raises(ValueError, match="seat 1 does not want this exchange"):
        put_to(narrow, 0, 1, tuple(received))
