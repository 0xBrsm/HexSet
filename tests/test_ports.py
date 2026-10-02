# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from collections import Counter

import pytest
from helpers import give

from hexset.board.board import make_board, random_base_board
from hexset.board.maps import MINI_LAYOUT
from hexset.board.ports import BASE_TRADE_RATIO, GENERIC_RATIO, SPECIFIC_RATIO, place_ports
from hexset.board.terrain import Resource, Terrain
from hexset.board.topology import build as build_topology
from hexset.board.topology import coastal_edges, coastal_rings
from hexset.economy import bank_trade, trade_ratios
from hexset.state import new_game, place_settlement


def test_each_island_gets_its_own_ring():
    from hexset.board.coords import Hex
    from hexset.board.maps import islands

    t = build_topology(islands(Hex(0, 0, 0), Hex(9, -9, 0), radius=1))
    rings = coastal_rings(t)
    assert len(rings) == 2
    assert all(len(r) == 18 for r in rings)
    assert sum(len(r) for r in rings) == len(coastal_edges(t))


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
    assert trade_ratios(state, 1) == [BASE_TRADE_RATIO] * 5


def test_specific_port_improves_only_its_own_resource():
    board = _board_with_port(Resource.ORE)
    state = new_game(board, 2)
    place_settlement(state, 0, board.ports[0].vertices[1], connected=False)

    ratios = trade_ratios(state, 0)
    assert ratios[Resource.ORE] == SPECIFIC_RATIO
    assert ratios[Resource.WOOD] == BASE_TRADE_RATIO


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
