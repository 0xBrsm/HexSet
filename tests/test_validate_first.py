# SPDX-License-Identifier: GPL-3.0-only
"""Engine calls the rules refuse before they change anything: each raises
and leaves the game exactly as it was -- board, hands, bank, deck, ledger,
every flag of the turn and the chance source alike.
"""
from __future__ import annotations

import pickle
import random

import pytest
from helpers import clear_hand, give

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase
from hexset.game import (
    Phase,
    build_settlement,
    buy_development_card,
    discard_one,
    end_turn,
    move_robber_to,
    place_initial_road,
    place_initial_settlement,
    play_monopoly_card,
    play_year_of_plenty_card,
    roll_dice,
    start,
    submit_discard,
    trade_with_bank,
)
from hexset.robber import occupants, victims
from hexset.state import Building

from tests.test_game import free_vertex, run_setup


def refuses(game, call, *args, error=ValueError):
    """`call(game, *args)` raises `error`, and the whole game -- pickled,
    so nothing reachable from it escapes the comparison -- is unchanged."""
    before = pickle.dumps(game)
    with pytest.raises(error):
        call(game, *args)
    assert pickle.dumps(game) == before


def in_main(seed: int = 0, players: int = 3):
    """Seat 0 in `MAIN` with an empty hand, every other seat's hand as dealt."""
    rng = random.Random(seed)
    game = run_setup(start(random_base_board(rng), players, rng))
    game.phase = Phase.MAIN
    clear_hand(game._state, 0)
    return game


def fund(game, purchase, seat: int = 0):
    for resource, count in enumerate(COSTS[purchase]):
        give(game._state, seat, resource, count)


def hold(game, card: DevCard, seat: int = 0) -> None:
    """A matured card, its age kept with it as a purchase would have."""
    game._state.dev_cards[seat][card] += 1
    game._state.dev_ages[seat].append(1)


# --- building ---------------------------------------------------------------


def test_nothing_else_is_done_while_a_free_road_can_still_be_placed():
    game = in_main()
    game.free_roads = 1
    fund(game, Purchase.SETTLEMENT)
    fund(game, Purchase.DEV_CARD)
    give(game._state, 0, Resource.WOOD, 4)
    for call, *args in [
        (build_settlement, free_vertex(game)),
        (trade_with_bank, Resource.WOOD, Resource.ORE),
        (buy_development_card,),
        (end_turn,),
    ]:
        refuses(game, call, *args)


# --- development cards ------------------------------------------------------


def test_a_second_card_in_a_turn_is_refused():
    game = in_main()
    hold(game, DevCard.MONOPOLY)
    hold(game, DevCard.YEAR_OF_PLENTY)
    play_monopoly_card(game, Resource.ORE)
    refuses(game, play_year_of_plenty_card, [Resource.WOOD, Resource.WOOD])


def test_an_unaffordable_card_is_not_bought():
    game = in_main()
    refuses(game, buy_development_card)


def test_an_empty_deck_sells_nothing():
    game = in_main()
    fund(game, Purchase.DEV_CARD)
    game._state.deck = []
    refuses(game, buy_development_card)


# --- bank trade -------------------------------------------------------------


def test_a_bank_trade_short_of_the_rate_is_refused():
    game = in_main()
    give(game._state, 0, Resource.WOOD, 3)
    refuses(game, trade_with_bank, Resource.WOOD, Resource.ORE)


def test_a_bank_trade_the_bank_cannot_pay_is_refused():
    game = in_main()
    give(game._state, 0, Resource.WOOD, 4)
    game._state.bank[Resource.ORE] = 0
    refuses(game, trade_with_bank, Resource.WOOD, Resource.ORE)


# --- the robber -------------------------------------------------------------


def _robber_phase():
    """Seat 0 to move the robber, with a hex where seat 1 can be robbed and
    a hex where nobody can."""
    game = in_main()
    game.phase = Phase.ROBBER
    state = game._state
    robbable = next(h for h in range(state.board.num_hexes)
                    if h != state.robber and 1 in victims(state, h, 0))
    empty = next(h for h in range(state.board.num_hexes)
                 if h != state.robber and not occupants(state, h))
    return game, robbable, empty


def test_a_seat_on_the_hex_has_to_be_robbed():
    game, robbable, _ = _robber_phase()
    refuses(game, move_robber_to, robbable, None)


def test_only_a_seat_on_the_hex_can_be_robbed():
    game, robbable, empty = _robber_phase()
    refuses(game, move_robber_to, empty, 1)
    refuses(game, move_robber_to, robbable, 0)     # the thief itself
    refuses(game, move_robber_to, robbable, 7)     # no such seat


def test_a_seat_with_no_cards_cannot_be_robbed():
    game, robbable, _ = _robber_phase()
    for seat in (1, 2):
        clear_hand(game._state, seat)
    refuses(game, move_robber_to, robbable, 1)
    move_robber_to(game, robbable, None)           # nobody left to rob
    assert game._state.robber == robbable


@pytest.mark.parametrize("where", ["here", "off"])
def test_the_robber_moves_to_another_hex_on_the_board(where):
    game, _, _ = _robber_phase()
    target = game._state.robber if where == "here" else -1
    refuses(game, move_robber_to, target, None)


# --- discards ---------------------------------------------------------------


def _discarding():
    game = in_main()
    for resource in Resource:
        give(game._state, 0, resource, 2)
    game.phase = Phase.ROLL
    roll_dice(game, roll=7)
    assert game.phase is Phase.DISCARD and game.discard_quota[0] == 5
    return game


@pytest.mark.parametrize("cards", [
    [1, 1, 1, 1],            # four
    [3, 2, 0, 0, 0],         # more wood than the hand holds
])
def test_a_discard_names_cards_the_hand_holds(cards):
    game = _discarding()
    refuses(game, submit_discard, 0, cards)


def test_a_discard_by_no_seat_is_refused():
    game = _discarding()
    refuses(game, submit_discard, -1, [1, 1, 1, 1, 1])
    refuses(game, discard_one, -1, Resource.WOOD)


def test_a_discard_of_no_resource_is_refused():
    game = _discarding()
    refuses(game, discard_one, 0, -1)


# --- setup ------------------------------------------------------------------


def test_an_opening_settlement_breaking_the_distance_rule_is_refused():
    rng = random.Random(0)
    game = start(random_base_board(rng), 3, rng)
    place_initial_settlement(game, free_vertex(game))
    place_initial_road(game, next(
        e for e in game._state.board.topology.vertex_edges[game.last_settlement]))
    neighbour = game._state.board.topology.vertex_neighbors[
        game._state.vertex_owner.index(0)][0]
    refuses(game, place_initial_settlement, neighbour)
    refuses(game, place_initial_settlement, -1)


def test_an_opening_road_away_from_the_settlement_is_refused():
    rng = random.Random(0)
    game = start(random_base_board(rng), 3, rng)
    place_initial_settlement(game, free_vertex(game))
    away = next(e for e in range(game._state.board.topology.num_edges)
                if e not in game._state.board.topology.vertex_edges[game.last_settlement])
    refuses(game, place_initial_road, away)
    assert game._state.vertex_building[game.last_settlement] == Building.SETTLEMENT
