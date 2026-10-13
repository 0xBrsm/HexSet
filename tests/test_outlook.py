# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.economy import COSTS, Purchase
from hexset.game import Phase, is_over, start, to_move
from hexset.outlook import EVENTS, Outlook, Rates, outlook, outlook_of, standing
from hexset.play import step_randomly
from hexset.roads import road_lengths, roads_to_take
from hexset.state import Building
from hexset.view import View

ROAD = COSTS[Purchase.ROAD]
WOOD_OR_BRICK = [r for r, n in enumerate(ROAD) if n]
OTHER = next(r for r, n in enumerate(ROAD) if not n)
RATES = Rates(settlement=0.5, city=0.5, card=0.5, longest_road=0.25, largest_army=0.5,
              lose_longest_road=0.3, lose_largest_army=0.5)
CITY, SETTLEMENT, LONGEST_ROAD = EVENTS.index("city"), EVENTS.index("settlement"), EVENTS.index("longest_road")


def a_route(state, length: int, avoid: set[int]) -> list[int]:
    """A simple chain of `length` edges touching no vertex in `avoid`."""
    topo = state.board.topology

    def walk(path, at, seen):
        if len(path) == length:
            return path
        for e in topo.vertex_edges[at]:
            a, b = topo.edges[e]
            nxt = b if a == at else a
            if e in path or nxt in seen or nxt in avoid:
                continue
            got = walk(path + [e], nxt, seen | {nxt})
            if got:
                return got
        return None

    for v in range(topo.num_vertices):
        if v not in avoid:
            got = walk([], v, {v})
            if got:
                return got
    raise AssertionError("no route")


def an_empty_game(seed: int = 3):
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    st = game._state
    for s in range(4):
        st.hands[s] = [0] * len(ROAD)
    return game


def a_road_race():
    """Seat 1 holds Longest Road on a five-road route; seat 2 has its own five-road route, one road from
    taking it, and holds one of the two cards a road costs. No buildings: no roll produces anything."""
    game = an_empty_game()
    st = game._state
    topo = st.board.topology
    first = a_route(st, 5, set())
    used = {v for e in first for v in topo.edges[e]}
    used |= {u for v in list(used) for u in topo.vertex_neighbors[v]}
    second = a_route(st, 5, used)
    for e in first:
        st.edge_owner[e] = 1
    for e in second:
        st.edge_owner[e] = 2
    st.road_lengths = road_lengths(st)
    st.longest_road_holder = 1
    st.hands[2][WOOD_OR_BRICK[0]] = 1
    assert roads_to_take(st, 2) == 1
    return game


def test_a_position_reads_in_range():
    rng = random.Random(7)
    game = start(random_base_board(rng), 4, rng)
    seen = 0
    for _ in range(4000):
        if is_over(game) or seen == 3:
            break
        if game.phase is Phase.MAIN:
            view = View.from_game(game, to_move(game))
            o = outlook(view, worlds=3, rng=random.Random(1))
            assert isinstance(o, Outlook) and len(o.can) == 4 and len(o.events) == 4
            assert all(0.0 <= p <= 1.0 + 1e-9 for ev in o.events for p in ev)
            assert all(0.0 <= p <= 1.0 + 1e-9 for lo in o.lose for p in lo)
            assert all(c >= -4.0 for c in o.can)
            seen += 1
        step_randomly(game, rng)
    assert seen == 3


def test_the_cards_for_a_city_on_turn_are_a_city():
    game = an_empty_game()
    st = game._state
    v = 0
    st.vertex_owner[v], st.vertex_building[v] = 0, Building.SETTLEMENT
    st.hands[0] = list(COSTS[Purchase.CITY])
    o = outlook_of(st, 0, rolled=True)
    assert o.events[0][CITY] == 1.0
    assert o.can[0] >= 1.0


def test_a_seat_one_road_short_with_both_cards_takes_longest_road_from_its_holder():
    game = a_road_race()
    st = game._state
    short = outlook_of(st, 1, rolled=True)
    assert short.events[2][LONGEST_ROAD] == 0.0 and short.lose[1][0] == 0.0
    st.hands[2][WOOD_OR_BRICK[1]] = 1
    both = outlook_of(st, 1, rolled=True)
    assert both.events[2][LONGEST_ROAD] == pytest.approx(1.0)
    assert both.lose[1][0] == pytest.approx(1.0)
    assert both.can[1] == pytest.approx(-2.0)
    assert both.will(RATES)[2] == pytest.approx(2 * RATES.longest_road)


def test_standing_reads_building_now_and_ending_the_turn_over_the_same_turns():
    game = an_empty_game()
    st = game._state
    st.vertex_owner[0], st.vertex_building[0] = 0, Building.SETTLEMENT
    st.hands[0] = list(COSTS[Purchase.CITY])
    held = standing(st, 0, RATES, mover=0)
    # Ending the turn with the cards: the city is one of the next turn's purchases, scaled by its rate
    # (and what the settlement produces meanwhile may buy more).
    assert held[0] >= 1 + RATES.city
    st.hands[0] = [0] * len(ROAD)
    st.vertex_building[0] = Building.CITY
    built = standing(st, 0, RATES, mover=0)
    assert built[0] >= 2
    assert built[0] > held[0]


def test_standing_counts_the_rival_who_takes_longest_road_before_this_seat_moves_again():
    game = a_road_race()
    st = game._state
    st.hands[2][WOOD_OR_BRICK[1]] = 1
    after = standing(st, 1, RATES, mover=1)
    assert after[2] == pytest.approx(2 * RATES.longest_road)
    assert after[1] == pytest.approx(2 - 2 * RATES.lose_longest_road)
