# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from helpers import independent_vertices, mini_board

from hexset.board.topology import coastal_rings
from hexset.cards import DevCard
from hexset.roads import MIN_LONGEST_ROAD
from hexset.state import NO_OWNER, Building, new_game
from hexset.victory import (
    LARGEST_ARMY_VP,
    LONGEST_ROAD_VP,
    MIN_LARGEST_ARMY,
    WINNING_POINTS,
    public_victory_points,
    update_largest_army,
    update_longest_road,
    victory_points,
    winner,
)


def a_game(players: int = 3):
    return new_game(mini_board(), players, random.Random(0))


def occupy(state, player, vertex, building=Building.SETTLEMENT):
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = building


def chain(state, player, length, start=0):
    """Lay a connected run of roads along the coastline.

    Runs taken from the one ring are contiguous and, given disjoint slices,
    cannot overwrite each other — which a greedy walk could.
    """
    ring = coastal_rings(state.board.topology)[0]
    run = ring[start : start + length]
    assert len(run) == length, "ring too short for that run"
    for e in run:
        assert state.edge_owner[e] == NO_OWNER, "run overlaps an existing road"
        state.edge_owner[e] = player
    return list(run)


def test_an_empty_game_scores_nothing():
    state = a_game()
    assert victory_points(state, 0) == 0
    assert winner(state) is None


def test_settlements_and_cities_score_one_and_two():
    state = a_game()
    occupy(state, 0, 0)
    assert victory_points(state, 0) == 1
    occupy(state, 0, 0, Building.CITY)
    assert victory_points(state, 0) == 2


def test_victory_point_cards_score_but_stay_hidden():
    state = a_game()
    state.dev_cards[0][DevCard.VICTORY_POINT] = 2

    assert victory_points(state, 0) == 2
    assert public_victory_points(state, 0) == 0


def test_a_card_bought_this_turn_can_still_win():
    state = a_game()
    state.new_dev_cards[0][DevCard.VICTORY_POINT] = 1
    assert victory_points(state, 0) == 1


def test_longest_road_needs_five_segments():
    state = a_game()
    chain(state, 0, MIN_LONGEST_ROAD - 1, start=0)
    assert update_longest_road(state) == NO_OWNER

    chain(state, 0, 1, start=MIN_LONGEST_ROAD - 1)
    assert update_longest_road(state) == 0
    assert victory_points(state, 0) == LONGEST_ROAD_VP


def test_a_tie_does_not_take_the_road_card():
    state = a_game()
    chain(state, 0, 5, start=0)
    update_longest_road(state)

    chain(state, 1, 5, start=6)
    assert update_longest_road(state) == 0


def test_beating_the_holder_outright_takes_the_road_card():
    state = a_game()
    chain(state, 0, 5, start=0)
    update_longest_road(state)

    chain(state, 1, 6, start=6)
    assert update_longest_road(state) == 1
    assert victory_points(state, 0) == 0
    assert victory_points(state, 1) == LONGEST_ROAD_VP


def test_the_card_leaves_play_when_challengers_tie_ahead_of_the_holder():
    state = a_game()
    chain(state, 0, 5, start=0)
    update_longest_road(state)

    chain(state, 1, 6, start=6)
    chain(state, 2, 6, start=12)
    assert update_longest_road(state) == NO_OWNER


def test_losing_every_road_drops_the_card():
    state = a_game()
    chain(state, 0, 5)
    update_longest_road(state)

    state.edge_owner = [NO_OWNER] * len(state.edge_owner)
    assert update_longest_road(state) == NO_OWNER


def test_largest_army_needs_three_knights():
    state = a_game()
    state.knights_played[0] = MIN_LARGEST_ARMY - 1
    assert update_largest_army(state) == NO_OWNER

    state.knights_played[0] = MIN_LARGEST_ARMY
    assert update_largest_army(state) == 0
    assert victory_points(state, 0) == LARGEST_ARMY_VP


def test_army_transfers_only_when_beaten_outright():
    state = a_game()
    state.knights_played = [3, 0, 0]
    update_largest_army(state)

    state.knights_played[1] = 3
    assert update_largest_army(state) == 0

    state.knights_played[1] = 4
    assert update_largest_army(state) == 1


def test_a_game_is_won_at_ten_points():
    state = a_game()
    for v in independent_vertices(state.board, 5):
        occupy(state, 0, v, Building.CITY)

    assert victory_points(state, 0) == WINNING_POINTS
    assert winner(state) == 0
