# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from helpers import independent_vertices, mini_board

from hexset.board.board import random_base_board
from hexset.board.topology import coastal_rings
from hexset.cards import DevCard
from hexset.game import is_over, start
from hexset.play import step_randomly
from hexset.roads import MIN_LONGEST_ROAD, road_lengths
from hexset.state import NO_OWNER, Building, copy_state, new_game
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


# --- `road_lengths` cache: `update_longest_road` recomputes only the seat(s)
# --- a placement could have changed rather than every seat from scratch
# --- (`hexset.game`'s four call sites), so these check the cache never
# --- drifts from a full recount, on real games as well as by construction.


def test_cached_road_lengths_never_drift_from_a_from_scratch_recount():
    """30 seeded random 4-seat games, checked after every single action.

    `_award`'s tie-breaking is unchanged code; if the cache it reads
    (`state.road_lengths`) matches a full recount
    (`hexset.roads.road_lengths`) at every step, `state.longest_road_holder`
    is correct by induction from the empty board (where both trivially
    agree) without a separate holder check.
    """
    for seed in range(30):
        rng = random.Random(seed)
        board = random_base_board(rng)
        game = start(board, 4, rng)
        steps = 0
        while not is_over(game) and steps < 2000:
            step_randomly(game, rng)
            steps += 1
            state = game._state
            assert state.road_lengths == road_lengths(state), (
                f"seed {seed}, step {steps}: cache drifted from a full recount"
            )


def test_a_settlement_that_splits_a_route_drops_only_that_seats_length():
    """A foreign building at a through-junction cuts the route passing under
    it (`roads.longest_road`'s `passable` check); the cached length for the
    seat that owned it must drop to match, and nobody else's cached length
    should have been touched by the recompute."""
    state = a_game(players=2)
    path = chain(state, 0, MIN_LONGEST_ROAD + 1, start=0)
    update_longest_road(state)
    assert state.longest_road_holder == 0
    assert state.road_lengths[0] == MIN_LONGEST_ROAD + 1

    # The junction shared by the path's middle two edges splits it into two
    # 3-segment halves once an opponent settles there.
    shared = set(state.board.topology.edges[path[2]]) & set(
        state.board.topology.edges[path[3]]
    )
    break_vertex = shared.pop()
    other_length = state.road_lengths[1]

    occupy(state, 1, break_vertex)
    update_longest_road(state, settlement_vertex=break_vertex, settlement_owner=1)

    assert state.road_lengths == road_lengths(state)
    assert state.road_lengths[0] == 3
    assert state.road_lengths[1] == other_length  # untouched: player 1 owns no road here
    assert state.longest_road_holder == NO_OWNER  # 3 < MIN_LONGEST_ROAD


def test_copy_state_carries_the_cache_and_copies_are_independent():
    state = a_game(players=2)
    chain(state, 0, MIN_LONGEST_ROAD)
    update_longest_road(state)
    assert state.road_lengths[0] == MIN_LONGEST_ROAD

    copy = copy_state(state)
    assert copy.road_lengths == state.road_lengths
    assert copy.road_lengths is not state.road_lengths

    copy.road_lengths[0] = 0
    assert state.road_lengths[0] == MIN_LONGEST_ROAD
