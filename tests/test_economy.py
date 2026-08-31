# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest
from helpers import ROLL, a_vertex_touching, give, independent_producers, mini_board

from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.economy import (
    BANK_TRADE_RATIO,
    COSTS,
    Purchase,
    bank_trade,
    can_afford,
    distribute,
    expected_total,
    hand_size,
    pay,
    total_in_play,
)
from hexset.state import BANK_PER_RESOURCE, new_game, place_settlement


def a_game(players: int = 2):
    return new_game(mini_board(), players)


def settle(state, player, vertex):
    place_settlement(state, player, vertex, connected=False)


def test_costs_match_the_rules():
    assert COSTS[Purchase.ROAD] == (1, 1, 0, 0, 0)
    assert COSTS[Purchase.SETTLEMENT] == (1, 1, 1, 1, 0)
    assert COSTS[Purchase.CITY] == (0, 0, 0, 2, 3)
    assert COSTS[Purchase.DEV_CARD] == (0, 0, 1, 1, 1)


def test_bank_starts_full():
    state = a_game()
    assert state.bank == [BANK_PER_RESOURCE] * NUM_RESOURCES
    assert total_in_play(state) == expected_total()


def test_paying_returns_resources_to_the_bank():
    state = a_game()
    give(state, 0, Resource.WOOD)
    give(state, 0, Resource.BRICK)

    assert can_afford(state, 0, Purchase.ROAD)
    pay(state, 0, Purchase.ROAD)

    assert state.hands[0] == [0, 0, 0, 0, 0]
    assert state.bank == [BANK_PER_RESOURCE] * NUM_RESOURCES
    assert total_in_play(state) == expected_total()


def test_cannot_pay_what_you_do_not_have():
    state = a_game()
    give(state, 0, Resource.WOOD)

    assert not can_afford(state, 0, Purchase.ROAD)
    with pytest.raises(ValueError):
        pay(state, 0, Purchase.ROAD)
    assert state.hands[0] == [1, 0, 0, 0, 0]


def test_bank_trade_moves_four_for_one():
    state = a_game()
    give(state, 0, Resource.WOOD, BANK_TRADE_RATIO)

    bank_trade(state, 0, Resource.WOOD, Resource.ORE)

    assert state.hands[0][Resource.WOOD] == 0
    assert state.hands[0][Resource.ORE] == 1
    assert total_in_play(state) == expected_total()


def test_bank_trade_rejects_bad_trades():
    state = a_game()
    give(state, 0, Resource.WOOD, BANK_TRADE_RATIO)

    with pytest.raises(ValueError):
        bank_trade(state, 0, Resource.WOOD, Resource.WOOD)
    with pytest.raises(ValueError):
        bank_trade(state, 0, Resource.BRICK, Resource.ORE)

    state.bank[Resource.ORE] = 0
    with pytest.raises(ValueError):
        bank_trade(state, 0, Resource.WOOD, Resource.ORE)


def test_distribute_pays_from_the_bank():
    state = a_game()
    settle(state, 0, a_vertex_touching(state.board, 2))

    granted = distribute(state, ROLL)

    assert granted[0][Resource.WOOD] == 2
    assert state.hands[0][Resource.WOOD] == 2
    assert state.bank[Resource.WOOD] == BANK_PER_RESOURCE - 2
    assert total_in_play(state) == expected_total()


def test_lone_claimant_takes_what_is_left():
    state = a_game()
    settle(state, 0, a_vertex_touching(state.board, 2))
    state.bank[Resource.WOOD] = 1

    granted = distribute(state, ROLL)

    assert granted[0][Resource.WOOD] == 1
    assert state.bank[Resource.WOOD] == 0


def test_shortage_with_several_claimants_pays_nobody():
    state = a_game()
    a, b = independent_producers(state.board, 2)
    settle(state, 0, a)
    settle(state, 1, b)
    state.bank[Resource.WOOD] = 1

    granted = distribute(state, ROLL)

    assert granted[0][Resource.WOOD] == 0
    assert granted[1][Resource.WOOD] == 0
    assert state.bank[Resource.WOOD] == 1


def test_shortage_is_decided_per_resource():
    """A dry wood bank must not stop an unrelated resource being paid out."""
    state = a_game()
    a, b = independent_producers(state.board, 2)
    settle(state, 0, a)
    settle(state, 1, b)
    give(state, 0, Resource.ORE, 3)
    state.bank[Resource.WOOD] = 0

    granted = distribute(state, ROLL)

    assert granted[0][Resource.WOOD] == 0
    assert state.hands[0][Resource.ORE] == 3


def test_resources_are_conserved_across_a_long_sequence():
    state = a_game()
    for player, v in enumerate(independent_producers(state.board, 2)):
        settle(state, player, v)

    for _ in range(40):
        distribute(state, ROLL)
        for player in range(state.num_players):
            if can_afford(state, player, Purchase.ROAD):
                pay(state, player, Purchase.ROAD)
        assert total_in_play(state) == expected_total()
        assert all(n >= 0 for n in state.bank)
        assert all(n >= 0 for hand in state.hands for n in hand)


def test_hand_size_counts_every_resource():
    state = a_game()
    give(state, 0, Resource.WOOD, 1)
    give(state, 0, Resource.BRICK, 2)
    give(state, 0, Resource.SHEEP, 3)
    assert hand_size(state, 0) == 6
