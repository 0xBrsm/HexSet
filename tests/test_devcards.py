# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from collections import Counter

import pytest
from helpers import give, mini_board

from hexset.board.terrain import Resource
from hexset.cards import DECK_SIZE, PLAYABLE, DevCard, make_deck
from hexset.devcards import (
    buy,
    can_buy,
    can_play,
    holdings,
    mature,
    play_knight,
    play_monopoly,
    play_road_building,
    play_year_of_plenty,
)
from hexset.economy import COSTS, Purchase, expected_total, total_in_play
from hexset.robber import move_robber
from hexset.state import new_game, place_settlement


def a_game(players: int = 2, seed: int = 0):
    return new_game(mini_board(), players, random.Random(seed))


def fund(state, player, purchase=Purchase.DEV_CARD):
    for resource, count in enumerate(COSTS[purchase]):
        give(state, player, resource, count)


def stack(state, player, card, count=1):
    state.dev_cards[player][card] += count


def test_deck_has_the_official_composition():
    counts = Counter(make_deck())
    assert sum(counts.values()) == DECK_SIZE == 25
    assert counts[DevCard.KNIGHT] == 14
    assert counts[DevCard.VICTORY_POINT] == 5
    assert counts[DevCard.ROAD_BUILDING] == 2
    assert counts[DevCard.YEAR_OF_PLENTY] == 2
    assert counts[DevCard.MONOPOLY] == 2


def test_shuffling_changes_order_but_not_contents():
    ordered = make_deck()
    shuffled = make_deck(random.Random(1))
    assert Counter(ordered) == Counter(shuffled)
    assert ordered != shuffled


def test_buying_costs_and_draws():
    state = a_game()
    fund(state, 0)
    assert can_buy(state, 0)

    card = buy(state, 0)

    assert len(state.deck) == DECK_SIZE - 1
    assert state.new_dev_cards[0][card] == 1
    assert state.hands[0] == [0] * 5
    assert total_in_play(state) == expected_total()


def test_cannot_buy_without_paying():
    state = a_game()
    assert not can_buy(state, 0)
    with pytest.raises(ValueError):
        buy(state, 0)


def test_cannot_buy_from_an_empty_deck():
    state = a_game()
    state.deck.clear()
    fund(state, 0)
    assert not can_buy(state, 0)
    with pytest.raises(ValueError):
        buy(state, 0)


def test_a_card_cannot_be_played_the_turn_it_is_bought():
    state = a_game()
    state.deck = [DevCard.MONOPOLY]
    fund(state, 0)
    buy(state, 0)

    assert not can_play(state, 0, DevCard.MONOPOLY)
    mature(state, 0)
    assert can_play(state, 0, DevCard.MONOPOLY)


def test_holdings_include_cards_bought_this_turn():
    state = a_game()
    state.deck = [DevCard.VICTORY_POINT]
    fund(state, 0)
    buy(state, 0)
    assert holdings(state, 0)[DevCard.VICTORY_POINT] == 1


def test_victory_point_cards_are_never_playable():
    state = a_game()
    stack(state, 0, DevCard.VICTORY_POINT)
    assert DevCard.VICTORY_POINT not in PLAYABLE
    assert not can_play(state, 0, DevCard.VICTORY_POINT)


def test_knight_spends_the_card_and_counts_toward_an_army():
    """`play_knight` no longer takes a target or victim -- moving the robber
    and stealing happen afterwards, through the same robber phase a seven
    uses (`hexset.game.play_knight_card`/`move_robber_to`)."""
    state = a_game()
    stack(state, 0, DevCard.KNIGHT)

    play_knight(state, 0)

    assert state.knights_played[0] == 1
    assert state.dev_cards[0][DevCard.KNIGHT] == 0


def test_robber_must_actually_move():
    """The invariant a knight used to enforce through `play_knight` -- it
    now lives only in `hexset.robber.move_robber`, which both a seven and a
    knight resolve through (`hexset.game.move_robber_to`)."""
    state = a_game()
    with pytest.raises(ValueError):
        move_robber(state, state.robber)


def test_road_building_places_two_free_roads():
    state = a_game()
    stack(state, 0, DevCard.ROAD_BUILDING)
    topology = state.board.topology
    v = next(v for v in range(topology.num_vertices) if len(topology.vertex_edges[v]) == 3)
    place_settlement(state, 0, v, connected=False)
    edges = list(topology.vertex_edges[v])[:2]

    play_road_building(state, 0, edges)

    assert all(state.edge_owner[e] == 0 for e in edges)
    assert state.hands[0] == [0] * 5


def test_road_building_rejects_illegal_placements():
    state = a_game()
    stack(state, 0, DevCard.ROAD_BUILDING)
    with pytest.raises(ValueError):
        play_road_building(state, 0, [0])


def test_year_of_plenty_takes_two_from_the_bank():
    state = a_game()
    stack(state, 0, DevCard.YEAR_OF_PLENTY)

    play_year_of_plenty(state, 0, [Resource.ORE, Resource.ORE])

    assert state.hands[0][Resource.ORE] == 2
    assert total_in_play(state) == expected_total()


def test_year_of_plenty_respects_bank_stock():
    state = a_game()
    stack(state, 0, DevCard.YEAR_OF_PLENTY)
    state.bank[Resource.ORE] = 1

    with pytest.raises(ValueError):
        play_year_of_plenty(state, 0, [Resource.ORE, Resource.ORE])
    assert state.dev_cards[0][DevCard.YEAR_OF_PLENTY] == 1


def test_monopoly_takes_one_resource_from_everyone():
    state = a_game(players=3)
    stack(state, 0, DevCard.MONOPOLY)
    give(state, 1, Resource.SHEEP, 3)
    give(state, 2, Resource.SHEEP, 2)
    give(state, 2, Resource.ORE, 4)

    taken = play_monopoly(state, 0, Resource.SHEEP)

    assert taken == 5
    assert state.hands[0][Resource.SHEEP] == 5
    assert state.hands[1][Resource.SHEEP] == 0
    assert state.hands[2][Resource.ORE] == 4
    assert total_in_play(state) == expected_total()


def test_playing_a_card_you_do_not_hold_is_rejected():
    state = a_game()
    with pytest.raises(ValueError):
        play_monopoly(state, 0, Resource.SHEEP)
