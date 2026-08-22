from __future__ import annotations

import random
from collections import Counter

import pytest

from catan.board.board import (
    BASE_TERRAIN,
    BASE_TOKENS,
    RED_TOKENS,
    make_board,
    pips,
    random_base_board,
)
from catan.board.terrain import Terrain
from catan.board.topology import build as build_topology
from catan.board.maps import MINI_LAYOUT


@pytest.mark.parametrize(
    ("token", "expected"),
    [(0, 0), (2, 1), (3, 2), (6, 5), (8, 5), (11, 2), (12, 1)],
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


@pytest.mark.parametrize("seed", range(20))
def test_red_numbers_are_never_adjacent(seed):
    board = random_base_board(random.Random(seed))
    topology = board.topology
    for h, token in enumerate(board.tokens):
        if token in RED_TOKENS:
            neighbours = [board.tokens[n] for n in topology.hex_neighbors[h]]
            assert not RED_TOKENS.intersection(neighbours)


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


def test_hexes_by_roll_matches_tokens():
    board = random_base_board(random.Random(2))
    for roll, hexes in enumerate(board.hexes_by_roll):
        for h in hexes:
            assert board.tokens[h] == roll
    placed = sum(len(hs) for hs in board.hexes_by_roll)
    assert placed == sum(1 for t in board.tokens if t)
    assert board.hexes_by_roll[7] == ()


def test_same_seed_gives_same_board():
    a = random_base_board(random.Random(7))
    b = random_base_board(random.Random(7))
    assert a.terrain == b.terrain
    assert a.tokens == b.tokens


def test_make_board_rejects_bad_setups():
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    forest = (Terrain.FOREST,) * n

    with pytest.raises(ValueError):
        make_board(topology, forest[:-1], (4,) * n)
    with pytest.raises(ValueError):
        make_board(topology, (Terrain.DESERT,) + forest[1:], (4,) * n)
    with pytest.raises(ValueError):
        make_board(topology, forest, (7,) * n)
    with pytest.raises(ValueError):
        make_board(topology, forest, (13,) * n)
