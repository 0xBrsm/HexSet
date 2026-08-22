from __future__ import annotations

import random

import numpy as np
import pytest

from catan.board.board import random_base_board
from catan.board.terrain import Resource
from catan.board.topology import build as build_topology
from catan.board.maps import BASE_LAYOUT, MINI_LAYOUT
from catan.encoding import (
    HEX_FEATURES,
    NUM_BUILDINGS,
    _building_points,
    _seat,
    edge_features,
    encode,
    global_features,
    static_graph,
    vertex_features,
)
from catan.game import is_over, start
from catan.play import step_randomly
from catan.state import NO_OWNER, Building
from catan.victory import building_points


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    return game


def arrays(obs):
    return (obs.hexes, obs.vertices, obs.edges, obs.globals)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_shapes_match_the_declared_widths(players):
    obs = encode(a_game(players=players))
    assert obs.hexes.shape == (19, HEX_FEATURES)
    assert obs.vertices.shape == (54, vertex_features(players))
    assert obs.edges.shape == (72, edge_features(players))
    assert obs.globals.shape == (global_features(players),)


def test_everything_is_finite_and_bounded():
    obs = encode(a_game())
    for array in arrays(obs):
        assert np.isfinite(array).all()
        assert array.min() >= 0.0
        assert array.max() <= 1.0


def test_adjacency_matches_the_topology():
    topology = build_topology(BASE_LAYOUT)
    graph = static_graph(topology)

    assert graph.hex_vertex.shape == (2, 19 * 6)
    assert graph.vertex_edge.shape == (2, 2 * 72)
    assert graph.hex_hex.shape[1] == sum(len(n) for n in topology.hex_neighbors)
    for h, v in graph.hex_vertex.T:
        assert v in topology.hex_vertices[h]


def test_the_graph_is_cached_per_board():
    topology = build_topology(MINI_LAYOUT)
    assert static_graph(topology) is static_graph(build_topology(MINI_LAYOUT))


def test_observations_on_one_board_do_not_share_memory():
    """The board-static template is cached and handed to every encode.

    If a caller ever received the cached array itself rather than a copy, the
    corruption would be silent and would spread to every later position on the
    board, so this pins the copy rather than trusting it.
    """
    game = a_game()
    first = encode(game)
    second = encode(game)
    assert first.hexes is not second.hexes
    assert first.vertices is not second.vertices

    first.hexes.fill(7.0)
    first.vertices.fill(7.0)
    third = encode(game)
    assert np.array_equal(third.hexes, second.hexes)
    assert np.array_equal(third.vertices, second.vertices)


def test_ownership_is_one_hot():
    obs = encode(a_game())
    players = 4
    owner_slice = obs.vertices[:, 3 : 3 + players + 1]
    assert np.allclose(owner_slice.sum(axis=1), 1.0)
    assert np.allclose(obs.edges.sum(axis=1), 1.0)


def test_the_robber_is_marked_on_exactly_one_hex():
    game = a_game()
    obs = encode(game)
    flags = obs.hexes[:, HEX_FEATURES - 1]
    assert flags.sum() == 1.0
    assert flags[game.state.robber] == 1.0


def test_the_mover_is_always_seat_zero():
    game = a_game()
    topology = game.state.board.topology
    owned = [v for v, o in enumerate(game.state.vertex_owner) if o == game.current_player]
    assert owned, "expected the mover to hold something by now"

    obs = encode(game)
    for v in owned:
        assert obs.vertices[v, 3] == 1.0, "own buildings belong in the first owner slot"
    assert topology.num_vertices == obs.vertices.shape[0]


def test_perspective_changes_what_is_seen():
    game = a_game()
    a = encode(game, perspective=0)
    b = encode(game, perspective=1)
    assert not np.array_equal(a.globals, b.globals)
    assert not np.array_equal(a.vertices, b.vertices)


def test_own_hand_is_encoded_exactly():
    game = a_game()
    state = game.state
    before = encode(game, perspective=0).globals.copy()

    state.bank[Resource.ORE] -= 1
    state.hands[0][Resource.ORE] += 1

    assert not np.array_equal(encode(game, perspective=0).globals, before)


def test_opponent_hand_contents_do_not_leak():
    """Swapping cards between two opponents must be invisible to a third player."""
    game = a_game(players=3)
    state = game.state
    for player in (1, 2):
        for resource in range(5):
            state.bank[resource] += state.hands[player][resource]
            state.hands[player][resource] = 0

    state.hands[1][Resource.WOOD] = 1
    state.hands[2][Resource.ORE] = 1
    state.bank[Resource.WOOD] -= 1
    state.bank[Resource.ORE] -= 1
    before = encode(game, perspective=0)

    state.hands[1] = [0, 0, 0, 0, 1]
    state.hands[2] = [1, 0, 0, 0, 0]
    after = encode(game, perspective=0)

    for lhs, rhs in zip(arrays(before), arrays(after)):
        assert np.array_equal(lhs, rhs)


def test_opponent_hand_sizes_are_visible():
    game = a_game(players=3)
    state = game.state
    before = encode(game, perspective=0).globals.copy()

    state.bank[Resource.WHEAT] -= 1
    state.hands[1][Resource.WHEAT] += 1

    assert not np.array_equal(encode(game, perspective=0).globals, before)


def test_opponent_development_cards_show_only_as_a_count():
    from catan.cards import DevCard

    game = a_game(players=3)
    state = game.state
    state.dev_cards[1][DevCard.KNIGHT] = 2
    before = encode(game, perspective=0)

    state.dev_cards[1][DevCard.KNIGHT] = 0
    state.dev_cards[1][DevCard.MONOPOLY] = 2
    after = encode(game, perspective=0)

    for lhs, rhs in zip(arrays(before), arrays(after)):
        assert np.array_equal(lhs, rhs)


def test_ports_are_marked_on_both_of_their_vertices():
    game = a_game()
    obs = encode(game)
    players = game.state.num_players
    port_base = 3 + players + 1

    for port in game.state.board.ports:
        for v in port.vertices:
            flags = obs.vertices[v, port_base : port_base + 6]
            assert flags.sum() >= 1.0


def test_an_unknown_perspective_is_rejected():
    game = a_game(players=3)
    with pytest.raises(ValueError):
        encode(game, perspective=3)


def test_encoding_holds_up_across_a_whole_game():
    rng = random.Random(4)
    game = start(random_base_board(rng), 4, rng)
    while not game.won_by and game.turns < 60:
        step_randomly(game, rng)
        for seat in range(game.state.num_players):
            obs = encode(game, perspective=seat)
            for array in arrays(obs):
                assert np.isfinite(array).all()


def _canonical_vertex_block(state, perspective):
    """Building one-hot then owner one-hot, written the plain way.

    `encode` reaches these same rows by table lookup on a combined key, which
    is faster and is not obviously the same thing, so this is what it is
    pinned to.
    """
    players = state.num_players
    width = NUM_BUILDINGS + players + 1
    out = np.zeros((state.board.topology.num_vertices, width), dtype=np.float32)
    for v in range(out.shape[0]):
        out[v, int(state.vertex_building[v])] = 1.0
        owner = state.vertex_owner[v]
        slot = players if owner == NO_OWNER else _seat(owner, perspective, players)
        out[v, NUM_BUILDINGS + slot] = 1.0
    return out


def _canonical_edges(state, perspective):
    players = state.num_players
    out = np.zeros(
        (state.board.topology.num_edges, edge_features(players)), dtype=np.float32
    )
    for e in range(out.shape[0]):
        owner = state.edge_owner[e]
        slot = players if owner == NO_OWNER else _seat(owner, perspective, players)
        out[e, slot] = 1.0
    return out


def _check_blocks(game, players):
    state = game.state
    for perspective in range(players):
        obs = encode(game, perspective=perspective)
        block = obs.vertices[:, : NUM_BUILDINGS + players + 1]
        assert np.array_equal(block, _canonical_vertex_block(state, perspective))
        assert np.array_equal(obs.edges, _canonical_edges(state, perspective))


@pytest.mark.parametrize("players", [2, 3, 4])
def test_the_table_lookups_agree_with_the_loops(players):
    rng = random.Random(11 + players)
    game = start(random_base_board(rng), players, rng)

    positions = 0
    while not is_over(game) and game.turns < 40:
        step_randomly(game, rng)
        _check_blocks(game, players)
        positions += players
    assert positions > 100

    # Random play this short reaches no cities, so agreeing everywhere would
    # only say the two paths agree on settlements and empty vertices. Every
    # building and owner combination is planted here instead of hoped for.
    state = game.state
    for owner in range(players):
        state.vertex_building[owner] = Building.CITY
        state.vertex_owner[owner] = owner
        state.vertex_building[players + owner] = Building.SETTLEMENT
        state.vertex_owner[players + owner] = owner
        state.edge_owner[owner] = owner
    _check_blocks(game, players)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_building_points_agree_with_the_rules(players):
    rng = random.Random(21 + players)
    game = start(random_base_board(rng), players, rng)
    scored = 0

    while not is_over(game) and game.turns < 40:
        step_randomly(game, rng)
        state = game.state
        for perspective in range(players):
            obs = encode(game, perspective=perspective)
            points = _building_points(obs.vertices, players)
            # Seat-relative: index i is the seat i places after the perspective.
            for i in range(players):
                seat = (perspective + i) % players
                assert points[i] == building_points(state, seat)
                scored += points[i] > 0

    assert scored > 0
