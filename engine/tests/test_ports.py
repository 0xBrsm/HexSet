# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from collections import Counter

import pytest
from helpers import give, mini_board

from hexset.board.board import make_board, random_base_board
from hexset.board.maps import BASE_LAYOUT, MINI_LAYOUT
from hexset.board.ports import GENERIC_RATIO, SPECIFIC_RATIO, place_ports
from hexset.board.terrain import Resource, Terrain
from hexset.board.topology import build as build_topology
from hexset.board.topology import coastal_edges, coastal_rings
from hexset.economy import BANK_TRADE_RATIO, bank_trade, trade_ratios
from hexset.state import new_game, place_settlement


def test_every_edge_borders_one_or_two_hexes():
    t = build_topology(BASE_LAYOUT)
    for e in range(t.num_edges):
        assert len(t.edge_hexes[e]) in (1, 2)


def test_hex_edges_and_edge_hexes_agree():
    t = build_topology(BASE_LAYOUT)
    for h, edges in enumerate(t.hex_edges):
        assert len(set(edges)) == 6
        for e in edges:
            assert h in t.edge_hexes[e]


def test_base_board_coastline_is_one_ring_of_thirty():
    t = build_topology(BASE_LAYOUT)
    assert len(coastal_edges(t)) == 30
    rings = coastal_rings(t)
    assert len(rings) == 1
    assert len(rings[0]) == 30


def test_each_island_gets_its_own_ring():
    from hexset.board.coords import Hex
    from hexset.board.maps import islands

    t = build_topology(islands(Hex(0, 0, 0), Hex(9, -9, 0), radius=1))
    rings = coastal_rings(t)
    assert len(rings) == 2
    assert all(len(r) == 18 for r in rings)
    assert sum(len(r) for r in rings) == len(coastal_edges(t))


def test_consecutive_coastal_edges_share_a_vertex():
    t = build_topology(BASE_LAYOUT)
    ring = coastal_rings(t)[0]
    for a, b in zip(ring, ring[1:]):
        assert set(t.edges[a]) & set(t.edges[b])


def test_base_board_has_the_official_port_mix():
    board = random_base_board(random.Random(0))
    assert len(board.ports) == 9
    kinds = Counter(p.resource for p in board.ports)
    assert kinds[None] == 4
    for resource in Resource:
        assert kinds[resource] == 1


def test_ports_sit_on_distinct_coastal_edges():
    board = random_base_board(random.Random(3))
    coastal = set(coastal_edges(board.topology))
    edges = [p.edge for p in board.ports]
    assert len(set(edges)) == len(edges)
    assert all(e in coastal for e in edges)


def test_port_ratios_follow_their_kind():
    board = random_base_board(random.Random(4))
    for port in board.ports:
        expected = GENERIC_RATIO if port.resource is None else SPECIFIC_RATIO
        assert port.ratio == expected


def test_too_many_ports_rejected():
    t = build_topology(MINI_LAYOUT)
    with pytest.raises(ValueError):
        place_ports(t, [None] * 100)


def test_without_a_port_everything_costs_four():
    state = new_game(mini_board(), 2)
    assert trade_ratios(state, 0) == [BANK_TRADE_RATIO] * 5


def _board_with_port(resource):
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    terrain = (Terrain.DESERT,) + (Terrain.FOREST,) * (n - 1)
    tokens = (0,) + (4,) * (n - 1)
    ports = place_ports(topology, [resource])
    return make_board(topology, terrain, tokens, ports)


def test_generic_port_improves_every_resource():
    board = _board_with_port(None)
    state = new_game(board, 2)
    place_settlement(state, 0, board.ports[0].vertices[0], connected=False)

    assert trade_ratios(state, 0) == [GENERIC_RATIO] * 5
    assert trade_ratios(state, 1) == [BANK_TRADE_RATIO] * 5


def test_specific_port_improves_only_its_own_resource():
    board = _board_with_port(Resource.ORE)
    state = new_game(board, 2)
    place_settlement(state, 0, board.ports[0].vertices[1], connected=False)

    ratios = trade_ratios(state, 0)
    assert ratios[Resource.ORE] == SPECIFIC_RATIO
    assert ratios[Resource.WOOD] == BANK_TRADE_RATIO


def test_trading_charges_the_port_rate():
    board = _board_with_port(Resource.WOOD)
    state = new_game(board, 2)
    place_settlement(state, 0, board.ports[0].vertices[0], connected=False)
    give(state, 0, Resource.WOOD, SPECIFIC_RATIO)

    bank_trade(state, 0, Resource.WOOD, Resource.ORE)

    assert state.hands[0][Resource.WOOD] == 0
    assert state.hands[0][Resource.ORE] == 1


def test_port_rate_is_not_available_from_a_distance():
    board = _board_with_port(Resource.WOOD)
    state = new_game(board, 2)
    give(state, 0, Resource.WOOD, SPECIFIC_RATIO)

    with pytest.raises(ValueError):
        bank_trade(state, 0, Resource.WOOD, Resource.ORE)
