# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from collections import Counter

import pytest

from hexset.board.board import (
    BASE_TERRAIN,
    BASE_TOKENS,
    RED_TOKENS,
    make_board,
    pips,
    random_base_board,
)
from hexset.board.terrain import Terrain
from hexset.board.topology import build as build_topology
from hexset.board.maps import MINI_LAYOUT


@pytest.mark.parametrize(
    ("token", "expected"),
    [(0, 0), (8, 5)],
)
def test_pips_follow_the_dice(token, expected):
    assert pips(token) == expected


def test_official_bags():
    assert len(BASE_TERRAIN) == 19
    assert Counter(BASE_TERRAIN)[Terrain.DESERT] == 1
    assert len(BASE_TOKENS) == 18
    assert 7 not in BASE_TOKENS


def test_random_board_uses_the_official_bags():
    board = random_base_board(random.Random(0))
    assert Counter(board.terrain) == Counter(BASE_TERRAIN)
    assert Counter(t for t in board.tokens if t) == Counter(BASE_TOKENS)


def test_desert_bears_no_token():
    board = random_base_board(random.Random(1))
    for h, terrain in enumerate(board.terrain):
        assert (board.tokens[h] == 0) == (terrain is Terrain.DESERT)


def test_red_numbers_are_never_adjacent():
    for seed in range(5):
        board = random_base_board(random.Random(seed))
        for h, token in enumerate(board.tokens):
            if token in RED_TOKENS:
                neighbours = [board.tokens[n] for n in board.topology.hex_neighbors[h]]
                assert not RED_TOKENS.intersection(neighbours), f"seed {seed}"


def test_red_numbers_may_touch_when_rule_disabled():
    boards = [
        random_base_board(random.Random(s), separate_reds=False) for s in range(30)
    ]
    assert any(
        board.tokens[n] in RED_TOKENS
        for board in boards
        for h, token in enumerate(board.tokens)
        if token in RED_TOKENS
        for n in board.topology.hex_neighbors[h]
    )


def test_make_board_rejects_bad_setups():
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    terrain = (Terrain.DESERT,) + (Terrain.FOREST,) * (n - 1)

    def tokens(token):
        return (0,) + (token,) * (n - 1)

    with pytest.raises(ValueError, match="per hex"):
        make_board(topology, terrain[:-1], tokens(4))
    with pytest.raises(ValueError, match="cannot bear"):
        make_board(topology, terrain, (4,) * n)
    with pytest.raises(ValueError, match="token 7"):
        make_board(topology, terrain, tokens(7))
    with pytest.raises(ValueError, match="out-of-range"):
        make_board(topology, terrain, tokens(13))
    make_board(topology, terrain, tokens(4))


def test_a_board_may_hold_gold_and_no_desert():
    """Seafarers terrain: gold bears a token, and a board needs no desert."""
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    board = make_board(topology, (Terrain.GOLD,) + (Terrain.FOREST,) * (n - 1), (6,) * n)
    assert board.desert_hexes() == ()
