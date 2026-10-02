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
    mature,
    play_knight,
    play_monopoly,
    play_year_of_plenty,
)
from hexset.economy import COSTS, Purchase, expected_total, total_in_play
from hexset.robber import move_robber
from hexset.state import new_game


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


def test_buying_costs_and_draws():
    state = a_game()
    fund(state, 0)
    assert can_buy(state, 0)

    card = buy(state, 0)

    assert len(state.deck) == DECK_SIZE - 1
    assert state.new_dev_cards[0][card] == 1
    assert state.hands[0] == [0] * 5
    assert total_in_play(state) == expected_total()


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


def test_victory_point_cards_are_never_playable():
    state = a_game()
    stack(state, 0, DevCard.VICTORY_POINT)
    assert DevCard.VICTORY_POINT not in PLAYABLE
    assert not can_play(state, 0, DevCard.VICTORY_POINT)


def test_robber_must_actually_move():
    """Enforced in `hexset.robber.move_robber`, which both paths resolve through."""
    state = a_game()
    with pytest.raises(ValueError):
        move_robber(state, state.robber)


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


def test_dev_ages_follow_each_card_from_purchase_to_play():
    """One age per card held, in turns held, oldest first: a purchase adds a
    0, a turn's end ages every card, and a play takes off the youngest card
    that could have been played -- never this turn's purchase."""
    from hexset.state import copy_state

    state = a_game()
    state.deck = [int(DevCard.KNIGHT)] * 3
    for _ in range(2):
        fund(state, 0)
        buy(state, 0)
    assert state.dev_ages[0] == [0, 0]
    mature(state, 0)
    mature(state, 0)
    assert state.dev_ages[0] == [2, 2]
    fund(state, 0)
    buy(state, 0)
    assert state.dev_ages[0] == [2, 2, 0]

    copy = copy_state(state)
    play_knight(state, 0)
    assert state.dev_ages[0] == [2, 0]
    assert copy.dev_ages[0] == [2, 2, 0], "a copy carries its own ages"
    assert state.dev_ages[1] == []


def test_a_state_built_without_ages_keeps_none():
    from hexset.state import GameState

    state = a_game()
    bare = GameState(
        board=state.board, num_players=2, vertex_owner=state.vertex_owner[:],
        vertex_building=state.vertex_building[:], edge_owner=state.edge_owner[:],
        robber=state.robber, hands=[[1, 1, 1, 1, 1], [0] * 5], bank=state.bank[:],
        deck=[int(DevCard.KNIGHT)], dev_cards=[[0] * 5, [0] * 5],
        new_dev_cards=[[0] * 5, [0] * 5], knights_played=[0, 0],
        dev_cards_played=[0] * 5, road_lengths=[0, 0],
    )
    assert bare.dev_ages == [None, None]
    buy(bare, 0)
    mature(bare, 0)
    play_knight(bare, 0)
    assert bare.dev_ages == [None, None]
