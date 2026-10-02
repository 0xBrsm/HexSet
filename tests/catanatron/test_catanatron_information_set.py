# SPDX-License-Identifier: GPL-3.0-only
"""The information-set audit: what a seat cannot see must not reach its
encoding, pinned against a *foreign* engine's richer state --
`tests.catanatron.oracle.translate` reads catanatron's full `player_state`.

The permutation must preserve everything genuinely public or a failure means
nothing: hands are pooled and re-dealt keeping each seat's total and the
global multiset fixed, and the deck shuffled but not recomposed.
"""

from __future__ import annotations

import random

import pytest

# A submodule, not bare "catanatron": this directory is named `catanatron`
# too, so with `tests/` on sys.path a bare import can resolve to it as an
# empty namespace package and skip nothing. `catanatron.game` exists only in
# the real distribution.
pytest.importorskip("catanatron.game")

from hexset.encoding import encode
from catanatron.game import Game as CatanatronGame
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color, RandomPlayer

from hexset.catanatron._board import translate_board
from hexset.catanatron._names import DEV_CARD_NAMES, RESOURCE_NAMES
from tests.catanatron.oracle import translate

TRANSLATE_SEED = 1234


def _advance(seed: int, ticks: int):
    random.seed(seed)
    players = [RandomPlayer(c) for c in Color]
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    game = CatanatronGame(players, catan_map=catan_map)
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.execute(game.state.current_player().decide(game, game.playable_actions))
    return game, translate_board(catan_map)


def _observe(game, mapping, perspective: int):
    our_game, _ = translate(game, mapping, random.Random(TRANSLATE_SEED))
    return encode(our_game, perspective)


def _keys(cstate, color):
    return f"P{cstate.color_to_index[color]}"


def _redeal(cstate, colors, names, rng):
    """Re-deal one card family among `colors`, preserving each player's total and
    the global multiset. Returns whether anything actually changed.
    """
    pool = []
    totals = {}
    for color in colors:
        key = _keys(cstate, color)
        held = 0
        for name in names:
            count = cstate.player_state[f"{key}_{name}_IN_HAND"]
            pool.extend([name] * count)
            held += count
        totals[color] = held
    before = {
        (color, name): cstate.player_state[f"{_keys(cstate, color)}_{name}_IN_HAND"]
        for color in colors
        for name in names
    }
    rng.shuffle(pool)
    cursor = 0
    for color in colors:
        key = _keys(cstate, color)
        dealt = pool[cursor : cursor + totals[color]]
        cursor += totals[color]
        for name in names:
            cstate.player_state[f"{key}_{name}_IN_HAND"] = dealt.count(name)
    after = {
        (color, name): cstate.player_state[f"{_keys(cstate, color)}_{name}_IN_HAND"]
        for color in colors
        for name in names
    }
    return before != after


def _identical(a, b) -> bool:
    return (
        (a.hexes == b.hexes).all()
        and (a.vertices == b.vertices).all()
        and (a.edges == b.edges).all()
        and (a.globals == b.globals).all()
    )


def _first_difference(a, b) -> str:
    for field in ("hexes", "vertices", "edges", "globals"):
        left, right = getattr(a, field), getattr(b, field)
        if (left != right).any():
            where = (left != right).nonzero()
            index = tuple(axis[0] for axis in where)
            return (
                f"{field}{list(index)}: {left[index]} -> {right[index]} "
                f"({int((left != right).sum())} cells differ)"
            )
    return "no difference"


@pytest.mark.parametrize("seed, perspective", [(0, 0), (1, 1)])
def test_opponent_hands_and_deck_do_not_reach_the_encoding(seed, perspective):
    game, mapping = _advance(seed, ticks=140)
    cstate = game.state
    before = _observe(game, mapping, perspective)

    colors = list(cstate.colors)
    opponents = [c for i, c in enumerate(colors) if i != perspective]
    rng = random.Random(seed + 9001)

    moved = _redeal(cstate, opponents, RESOURCE_NAMES, rng)
    moved |= _redeal(cstate, opponents, list(DEV_CARD_NAMES.values()), rng)

    deck_before = list(cstate.development_listdeck)
    rng.shuffle(cstate.development_listdeck)
    moved |= list(cstate.development_listdeck) != deck_before
    assert sorted(cstate.development_listdeck) == sorted(deck_before), (
        "the shuffle must not change deck composition, only order"
    )

    if not moved:
        pytest.skip("nothing hidden was actually permutable in this position")

    after = _observe(game, mapping, perspective)
    assert _identical(before, after), (
        f"hidden state reached seat {perspective}'s encoding: "
        f"{_first_difference(before, after)}"
    )


def test_the_audit_can_fail():
    game, mapping = _advance(0, ticks=140)
    cstate = game.state
    perspective = 0
    before = _observe(game, mapping, perspective)

    key = _keys(cstate, list(cstate.colors)[perspective])
    for name in RESOURCE_NAMES:
        cstate.player_state[f"{key}_{name}_IN_HAND"] += 1

    after = _observe(game, mapping, perspective)
    assert not _identical(before, after), (
        "the seat's own hand changed and the encoding did not — the audit "
        "above cannot detect anything"
    )
