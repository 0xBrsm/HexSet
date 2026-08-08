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
    edge_features,
    encode,
    global_features,
    pips,
    static_graph,
    vertex_features,
)
from catan.game import start
from catan.play import step_randomly


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    return game


def arrays(obs):
    return (obs.hexes, obs.vertices, obs.edges, obs.globals)


@pytest.mark.parametrize(
    ("token", "expected"),
    [(0, 0), (2, 1), (3, 2), (6, 5), (8, 5), (11, 2), (12, 1)],
)
def test_pips_follow_the_dice(token, expected):
    assert pips(token) == expected


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
