# SPDX-License-Identifier: GPL-3.0-only
"""Playout endings: what a copy ended with, read without changing any move or score."""

import random

import numpy as np

from test_actions import a_game_owing
from test_rollout import Stub, a_road_or_not, favouring

from hexset.actions import Action, ActionType, legal_actions
from hexset.board.board import pips
from hexset.board.terrain import TERRAIN_RESOURCE
from hexset.endings import (BUILDS, FIELDS, MINE, MINE_CITY, NO_TRACE, OPEN, THEIRS, THEIRS_CITY, TRACE, UNRECORDED,
                            ending, field, follow, holder, upgrade, vertices)
from hexset.rollout import PairedRollouts, RolloutBatch
from hexset.state import Building


def test_an_ending_names_the_seat_and_every_vertex_from_its_side():
    game = a_road_or_not()
    state = game._state
    row = ending(game, 0, game.turns, (3, 1))
    assert row.shape == (len(FIELDS) + len(state.vertex_owner),)
    assert row[field("discarded")] == 3 and row[field("other_discarded")] == 1
    assert row[field("turns")] == 0 and row[field("won")] == 0 and row[field("lost")] == 0
    marks = vertices(row)
    for v, owner in enumerate(state.vertex_owner):
        assert marks[v] == (OPEN if owner < 0 else MINE if owner == 0 else THEIRS)
    assert row[field("settlements")] == sum(1 for o in state.vertex_owner if o == 0)
    assert row[field("other_settlements")] == sum(1 for o in state.vertex_owner if o > 0)
    assert row[field("other_cities")] == 0
    assert [row[field(name)] for name in TRACE] == list(NO_TRACE)


def test_a_city_is_marked_apart_from_a_settlement_and_holder_folds_them():
    game = a_road_or_not()
    state = game._state
    mine = next(v for v, o in enumerate(state.vertex_owner) if o == 0)
    theirs = next(v for v, o in enumerate(state.vertex_owner) if o == 1)
    state.vertex_building[mine] = Building.CITY
    state.vertex_building[theirs] = Building.CITY
    row = ending(game, 0, game.turns)
    marks = vertices(row)
    assert marks[mine] == MINE_CITY and marks[theirs] == THEIRS_CITY
    assert holder(marks)[mine] == MINE and holder(marks)[theirs] == THEIRS
    assert (holder(marks) == np.where(np.asarray(state.vertex_owner) < 0, OPEN,
                                      np.where(np.asarray(state.vertex_owner) == 0, MINE, THEIRS))).all()
    assert row[field("cities")] == 1 and row[field("other_cities")] == 1
    assert row[field("other_settlements")] == sum(1 for o in state.vertex_owner if o > 0) - 1


def test_the_robber_hex_and_the_pips_it_keeps_from_each_side():
    """The robber on a producing hex next to a city of the seat's and a
    settlement of another's: the city's pips twice, the settlement's once,
    and exactly the income the production fields leave out."""
    game = a_road_or_not()
    state = game._state
    board = state.board
    topology = board.topology
    mine = next(v for v, o in enumerate(state.vertex_owner) if o == 0
                and any(board.tokens[h] and TERRAIN_RESOURCE.get(board.terrain[h]) is not None
                        for h in topology.vertex_hexes[v]))
    h = next(h for h in topology.vertex_hexes[mine]
             if board.tokens[h] and TERRAIN_RESOURCE.get(board.terrain[h]) is not None)
    state.vertex_building[mine] = Building.CITY
    other = next((v for v in topology.hex_vertices[h] if state.vertex_owner[v] < 0
                  and all(state.vertex_owner[u] < 0 for u in topology.vertex_neighbors[v])), None)
    if other is not None:
        state.vertex_owner[other] = 2
        state.vertex_building[other] = Building.SETTLEMENT
    away = state.robber
    free = ending(game, 0, game.turns)
    state.robber = h
    row = ending(game, 0, game.turns)
    worth = pips(board.tokens[h])
    expected_mine = sum(state.vertex_building[v] * worth for v in topology.hex_vertices[h] if state.vertex_owner[v] == 0)
    expected_theirs = sum(state.vertex_building[v] * worth for v in topology.hex_vertices[h]
                          if state.vertex_owner[v] > 0)
    assert row[field("robber")] == h and free[field("robber")] == away
    assert row[field("blocked")] == expected_mine >= 2 * worth
    assert row[field("other_blocked")] == expected_theirs
    resource = TERRAIN_RESOURCE[board.terrain[h]]
    name = f"production_{resource.name.lower()}"
    assert away != h and free[field("blocked")] == 0
    assert free[field(name)] - row[field(name)] == expected_mine


def test_the_trace_counts_the_seats_first_build_its_knights_buys_and_robber_and_the_others_first_settlement():
    trace = list(NO_TRACE)
    follow(trace, Action(ActionType.BUILD_SETTLEMENT, 9), False, 0)    # another seat's settlement
    follow(trace, Action(ActionType.BUILD_SETTLEMENT, 11), False, 0)   # only the first is kept
    follow(trace, Action(ActionType.MOVE_ROBBER, 4, 1), False, 1)      # another seat's robber: not the seat's
    follow(trace, Action(ActionType.BUY_DEV_CARD), True, 1)
    follow(trace, Action(ActionType.BUILD_ROAD, 30), True, 2)
    follow(trace, Action(ActionType.PLAY_KNIGHT), True, 3)
    follow(trace, Action(ActionType.MOVE_ROBBER, 7, 1), True, 3)
    follow(trace, Action(ActionType.MOVE_ROBBER, 8, 1), True, 5)
    follow(trace, Action(ActionType.BUY_DEV_CARD), True, 5)
    got = dict(zip(TRACE, trace))
    assert BUILDS[got["first_build"]] == "development card"
    assert got["first_build_at"] == -1 and got["first_build_turn"] == 1
    assert got["other_first_settle"] == 9
    assert got["knights_played"] == 1 and got["cards_bought"] == 2 and got["robber_first"] == 7
    road = list(NO_TRACE)
    follow(road, Action(ActionType.BANK_TRADE, 0, 1), True, 0)
    follow(road, Action(ActionType.BUILD_ROAD, 30), True, 2)
    follow(road, Action(ActionType.BUILD_CITY, 12), True, 4)
    assert dict(zip(TRACE, road))["first_build"] == BUILDS.index("road")
    assert (dict(zip(TRACE, road))["first_build_at"], dict(zip(TRACE, road))["first_build_turn"]) == (30, 2)
    got = dict(zip(TRACE, road))
    assert (BUILDS[got["second_build"]], got["second_build_at"], got["second_build_turn"]) == ("city", 12, 4)
    assert (got["third_build"], got["third_build_at"], got["third_build_turn"]) == (0, -1, -1)


def test_the_trace_keeps_the_seats_first_three_builds_in_order():
    """Road Building's two roads, then the settlement they lead to; a fourth
    build is past the trace."""
    trace = list(NO_TRACE)
    follow(trace, Action(ActionType.PLAY_ROAD_BUILDING), True, 0)
    follow(trace, Action(ActionType.BUILD_ROAD, 40), True, 0)
    follow(trace, Action(ActionType.BUILD_ROAD, 41), True, 0)
    follow(trace, Action(ActionType.BUILD_SETTLEMENT, 28), False, 1)   # another seat's: not traced as a build
    follow(trace, Action(ActionType.BUILD_SETTLEMENT, 30), True, 2)
    follow(trace, Action(ActionType.BUILD_CITY, 30), True, 4)
    got = dict(zip(TRACE, trace))
    builds = [(BUILDS[got[f"{n}_build"]], got[f"{n}_build_at"], got[f"{n}_build_turn"])
              for n in ("first", "second", "third")]
    assert builds == [("road", 40, 0), ("road", 41, 0), ("settlement", 30, 2)]
    assert got["other_first_settle"] == 28


def test_rows_saved_before_a_field_existed_read_back_with_it_unrecorded():
    rng = np.random.default_rng(0)
    nv = 54
    rows = rng.integers(0, 5, (4, 3, len(FIELDS) + nv)).astype(np.int16)
    assert upgrade(rows, FIELDS) is rows
    old = [name for name in FIELDS if not name.startswith(("second_build", "third_build"))]
    saved = np.concatenate([rows[..., [field(name) for name in old]], rows[..., len(FIELDS):]], axis=-1)
    back = upgrade(saved, old)
    assert back.shape == rows.shape
    for name in FIELDS:
        expected = rows[..., field(name)] if name in old else UNRECORDED
        assert (back[..., field(name)] == expected).all()
    assert (vertices(back) == vertices(rows)).all()


def test_a_playouts_trace_is_what_its_copy_did():
    """Each copy's first build, read off the batch's ending rows, is a piece
    of the seat's that stands at the end of that copy, built after the
    playout began; its Knights played add to the Knights it had."""
    game = a_road_or_not()
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
    search = PairedRollouts(stub, worlds=6, top=2, horizon=4, marks=(), det=True, rng=random.Random(5))
    tasks, _ = search.tasks(game, 0)
    moves = tuple(legal_actions(game)[:2])
    batch = RolloutBatch.from_tasks(stub, tasks, moves, 0, game.turns, endings=True)
    rows = batch.advance(4).endings
    before = game._state.knights_played[0]
    for copy, row in zip(batch.games, rows):
        state = copy._state
        kind = BUILDS[int(row[field("first_build")])]
        at = int(row[field("first_build_at")])
        if kind == "road":
            assert state.edge_owner[at] == 0
        elif kind in ("settlement", "city"):
            assert state.vertex_owner[at] == 0
        if kind == "none":
            assert row[field("first_build_turn")] == -1
        else:
            assert 0 <= row[field("first_build_turn")] <= row[field("turns")]
        assert row[field("knights")] == before + row[field("knights_played")]
        assert row[field("robber")] == state.robber


def test_endings_leave_every_score_as_it_was():
    game = a_road_or_not()
    search = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=6, top=2, horizon=3,
                            marks=(), det=True, rng=random.Random(3))
    tasks, _ = search.tasks(game, 0)
    moves = tuple(legal_actions(game)[:2])
    plain = search._lanes(tasks, moves, 0, game.turns)
    kept = search._lanes(tasks, moves, 0, game.turns, endings=True)
    assert plain.endings is None
    np.testing.assert_array_equal(plain.outcomes, kept.outcomes)
    assert kept.endings.shape == (len(tasks) * len(moves), len(FIELDS) + len(game._state.vertex_owner))
    assert (kept.endings[:, field("turns")] >= 0).all()


def test_cards_discarded_to_sevens_are_counted_by_seat():
    """Seats 0 and 3 owe two cards each; seen from seat 0, its own two and
    the other seats' two are counted apart, and the copy plays on as it
    would have."""
    game = a_game_owing()
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
    batch = RolloutBatch(stub, [game], 0, game.turns, endings=True)
    lanes = batch.advance(1)
    assert lanes.endings[0, field("discarded")] >= 2
    assert lanes.endings[0, field("other_discarded")] >= 2


def test_income_hand_ports_and_room_are_read_from_the_seats_side():
    from hexset.board.board import pips
    from hexset.board.terrain import TERRAIN_RESOURCE
    from hexset.endings import RESOURCE_FIELDS

    game = a_road_or_not()
    state = game._state
    board = state.board
    row = ending(game, 0, game.turns)
    income = [0] * 5
    for v, owner in enumerate(state.vertex_owner):
        if owner != 0:
            continue
        for h in board.topology.vertex_hexes[v]:
            r = TERRAIN_RESOURCE.get(board.terrain[h])
            if r is not None and board.tokens[h] and h != state.robber:
                income[r] += state.vertex_building[v] * pips(board.tokens[h])
    assert [row[field(n)] for n in RESOURCE_FIELDS["production"]] == income
    assert [row[field(n)] for n in RESOURCE_FIELDS["hand"]] == list(state.hands[0])
    owned = {port.resource for port in board.ports if any(state.vertex_owner[v] == 0 for v in port.vertices)}
    assert row[field("port_generic")] == float(None in owned)
    assert [row[field(n)] for n in RESOURCE_FIELDS["port"]] == [float(r in owned) for r in range(5)]
    topology = board.topology
    ends = {v for e, o in enumerate(state.edge_owner) if o == 0 for v in topology.edges[e]}
    free = lambda v: state.vertex_owner[v] < 0 and all(state.vertex_owner[u] < 0 for u in topology.vertex_neighbors[v])
    assert row[field("spots_now")] == sum(1 for v in ends if free(v))
    assert row[field("spots_near")] >= 0 and row[field("best_spot")] >= 0
