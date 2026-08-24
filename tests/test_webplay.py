from __future__ import annotations

import math
import random

import pytest

from catan.actions import Action, ActionType, apply, legal_actions
from catan.board.board import random_base_board
from catan.board.coords import Hex, hexagon
from catan.board.maps import BASE_LAYOUT, MINI_LAYOUT
from catan.board.topology import build as build_topology
from catan.bots import RandomBot
from catan.game import Phase, is_over, start, to_move
from catan.record import read as read_records
from catan.record import replay as replay_record
from catan.webplay import (
    SQRT3,
    GameSession,
    action_to_wire,
    board_layout,
    hex_center,
    hex_corner,
    vertex_pixels,
    wire_to_action,
)


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


# --- Hex-to-pixel layout ------------------------------------------------------


def test_hex_center_places_the_origin_hex_at_the_origin():
    assert hex_center(Hex(0, 0, 0), size=42.0) == (0.0, 0.0)


def test_neighbouring_hex_centers_are_size_times_sqrt3_apart():
    topology = build_topology(hexagon(1))
    size = 25.0
    center = topology.hex_index[Hex(0, 0, 0)]
    origin = hex_center(Hex(0, 0, 0), size)
    for neighbor in topology.hex_neighbors[center]:
        other = hex_center(topology.hexes[neighbor], size)
        distance = math.dist(origin, other)
        assert distance == pytest.approx(size * SQRT3)


def test_hex_corners_match_the_pointy_top_formula():
    """`Topology.hex_vertices[h][i]` sits at the textbook angle 60*i - 30."""
    topology = build_topology([Hex(0, 0, 0)])
    size = 10.0
    vpix = vertex_pixels(topology, size)
    for i, v in enumerate(topology.hex_vertices[0]):
        expected = hex_corner((0.0, 0.0), i, size)
        assert vpix[v] == pytest.approx(expected)


@pytest.mark.parametrize("layout", [MINI_LAYOUT, BASE_LAYOUT])
def test_every_edge_measures_exactly_size_between_its_two_vertices(layout):
    """A regular hexagon's edge length equals its circumradius (`size`), so this
    holds for every edge on any board built from hex coordinates, regardless of
    shape — a broad, cheap check on the whole layout at once."""
    topology = build_topology(layout)
    size = 37.0
    vpix = vertex_pixels(topology, size)
    for a, b in topology.edges:
        assert math.dist(vpix[a], vpix[b]) == pytest.approx(size)


def test_vertex_pixels_covers_every_vertex_exactly_once():
    topology = build_topology(BASE_LAYOUT)
    vpix = vertex_pixels(topology, 50.0)
    assert len(vpix) == topology.num_vertices
    assert all(isinstance(p, tuple) and len(p) == 2 for p in vpix)


def test_board_layout_ids_are_internally_consistent():
    board = random_base_board(random.Random(7))
    layout = board_layout(board)
    num_vertices = len(layout["vertices"])
    num_hexes = len(layout["hexes"])

    for hex_entry in layout["hexes"]:
        assert len(hex_entry["vertex_ids"]) == 6
        assert all(0 <= v < num_vertices for v in hex_entry["vertex_ids"])

    for edge in layout["edges"]:
        assert 0 <= edge["v0"] < num_vertices
        assert 0 <= edge["v1"] < num_vertices

    edge_endpoints = {e["id"]: {e["v0"], e["v1"]} for e in layout["edges"]}
    for port in layout["ports"]:
        assert 0 <= port["edge"] < len(layout["edges"])
        assert set(port["vertices"]) == edge_endpoints[port["edge"]]
        assert port["resource"] is None or port["resource"] in layout["resources"]

    assert len(layout["resources"]) == 5
    assert len(layout["year_of_plenty_pairs"]) == 15
    assert num_hexes == board.topology.num_hexes


# --- Wire format ---------------------------------------------------------------


def test_wire_round_trips_across_a_played_out_game():
    game = a_game(seed=11)
    rng = random.Random(99)
    steps = 0
    while not is_over(game) and steps < 500:
        options = legal_actions(game)
        for action in options:
            assert wire_to_action(action_to_wire(action)) == action
        apply(game, rng.choice(options))
        steps += 1
    assert steps > 50  # sanity: the loop actually exercised many phases


def test_action_to_wire_uses_json_friendly_types():
    action = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 1, 0, 0, 0), ask=(2, 3)
    )
    wire = action_to_wire(action)
    assert wire == {
        "type": "PROPOSE_TRADE",
        "a": 0,
        "b": 0,
        "give": [1, 0, 0, 0, 0],
        "want": [0, 1, 0, 0, 0],
        "ask": [2, 3],
    }


def test_wire_to_action_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        wire_to_action({"type": "TELEPORT", "a": 0, "b": 0})


def test_wire_to_action_rejects_a_malformed_payload():
    with pytest.raises(ValueError):
        wire_to_action({"type": "BUILD_ROAD", "a": "not-a-number"})


# --- GameSession ---------------------------------------------------------------


def test_session_rejects_an_action_not_currently_legal():
    game = a_game(seed=2)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    # ROLL is never legal during setup placement.
    forged = action_to_wire(Action(ActionType.ROLL))
    with pytest.raises(ValueError):
        session.apply_human_action(forged)
    assert game.phase is Phase.SETUP_SETTLEMENT
    assert all(owner == -1 for owner in game.state.vertex_owner)


def test_session_rejects_an_out_of_range_target():
    game = a_game(seed=3)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    forged = action_to_wire(Action(ActionType.SETUP_SETTLEMENT, a=999999))
    with pytest.raises(ValueError):
        session.apply_human_action(forged)


def test_session_rejects_when_it_is_not_the_humans_turn():
    game = a_game(seed=4)
    mover = to_move(game)
    other = (mover + 1) % game.state.num_players
    session = GameSession(game=game, human_seat=other, bot=RandomBot())

    # A perfectly legal action for whoever is actually on the move.
    legal_for_mover = action_to_wire(legal_actions(game)[0])
    with pytest.raises(ValueError):
        session.apply_human_action(legal_for_mover)


def test_a_human_trade_with_no_ask_defaults_to_lowest_vp_first():
    """GameSession's own addition on top of catan.game.propose_trade's
    neutral ask=() default (clockwise seat order) — favours whoever's
    behind rather than strict seat order. See GameSession._default_ask_order.
    """
    from catan.board.terrain import Resource
    from catan.state import Building

    game = a_game(seed=7)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state

    # Seat 0 (the proposer) can give wood; seats 1, 2, 3 can all cover the
    # ore ask, so all three are eligible and the ordering is what's under
    # test, not who can respond at all.
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2, 3):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    # Seat 1: a city (2 VP). Seat 3: a settlement (1 VP). Seat 2: nothing (0
    # VP) — expect lowest first: [2, 3, 1].
    state.vertex_owner[0] = 1
    state.vertex_building[0] = Building.CITY
    state.vertex_owner[1] = 3
    state.vertex_building[1] = Building.SETTLEMENT

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    session.apply_human_action(action_to_wire(offer))

    assert game.pending_responders == [2, 3, 1]


def test_a_human_trade_honours_an_explicit_ask_instead_of_the_default():
    from catan.board.terrain import Resource

    game = a_game(seed=8)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state

    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2, 3):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(3, 2, 1)
    )
    session.apply_human_action(action_to_wire(offer))

    assert game.pending_responders == [3, 2, 1]


def test_advance_bots_always_stops_at_the_human_seat_or_game_over():
    game = a_game(seed=5)
    human_seat = 1
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot(rng=random.Random(0)))
    rng = random.Random(6)

    session.advance_bots()
    steps = 0
    while not is_over(session.game) and steps < 300:
        assert to_move(session.game) == human_seat
        options = legal_actions(session.game)
        wire = action_to_wire(rng.choice(options))
        session.apply_human_action(wire)
        session.advance_bots()
        steps += 1
        assert is_over(session.game) or to_move(session.game) == human_seat


def test_state_view_hides_opponent_hands_but_not_the_humans():
    game = a_game(seed=8)
    human_seat = to_move(game)
    other = (human_seat + 1) % game.state.num_players
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    game.state.hands[human_seat][0] = 3
    game.state.hands[other][0] = 5

    view = session.state_view()
    players = {p["seat"]: p for p in view["players"]}
    assert "hand" in players[human_seat]
    assert players[human_seat]["hand"]["Wood"] == 3
    assert "hand" not in players[other]
    assert players[other]["hand_size"] == 5


def test_state_view_reveals_every_hand_once_the_game_is_over():
    game = a_game(seed=9)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())
    game.won_by = (human_seat + 1) % game.state.num_players
    game.phase = Phase.GAME_OVER

    view = session.state_view()
    assert all("hand" in p for p in view["players"])
    assert view["legal_actions"] == []


# --- Recording -------------------------------------------------------------


def test_a_finished_game_is_recorded_and_replays_clean(tmp_path):
    """`GameSession` writes catan.record's own format, not a new one, and what
    it writes has to satisfy that format's own replay check."""
    seed = 42
    # Two independent random.Random(seed) instances, matching catan.record's own
    # convention: replay() rebuilds the board from stored data (no randomness
    # spent) and hands `start` a *fresh* random.Random(seed), so the game's own
    # rng must start from the same untouched state here too.
    board = random_base_board(random.Random(seed))
    game = start(board, 4, random.Random(seed))
    human_seat = to_move(game)
    record_path = tmp_path / "games.jsonl"
    session = GameSession(
        game=game,
        human_seat=human_seat,
        bot=RandomBot(rng=random.Random(1)),
        seed=seed,
        record_path=str(record_path),
    )

    human_rng = random.Random(2)
    session.advance_bots()
    steps = 0
    while not is_over(session.game) and steps < 4000:
        options = legal_actions(session.game)
        session.apply_human_action(action_to_wire(human_rng.choice(options)))
        session.advance_bots()
        steps += 1
    assert is_over(session.game)

    records = list(read_records(str(record_path)))
    assert len(records) == 1
    record = records[0]
    assert record.seed == seed
    assert record.winner == session.game.won_by
    assert record.turns == session.game.turns

    replayed = replay_record(record)  # raises ReplayError if it doesn't match
    assert replayed.won_by == session.game.won_by
    assert replayed.turns == session.game.turns


def test_recording_is_off_by_default():
    game = a_game(seed=13)
    session = GameSession(game=game, human_seat=to_move(game), bot=RandomBot())
    session._apply(session.human_seat, legal_actions(game)[0])
    assert session.record_path is None  # nothing to have written to
