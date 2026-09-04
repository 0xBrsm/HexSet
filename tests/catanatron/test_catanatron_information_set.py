# SPDX-License-Identifier: GPL-3.0-only
"""The information-set audit: what a seat cannot see must not reach its encoding.

dev-catan's encoder is documented as information-set correct — "opponents
contribute counts, never contents" — and its own tests assert that against
states dev-catan itself built. That is an argument from code. This pins it
against a *foreign* engine's richer state, which is the only place the claim
can actually fail in the way that would matter: `hexset.catanatron.state.translate`
reads catanatron's full `player_state`, including every opponent's exact hand
and the entire remaining development deck in order, and hands all of it to a
`GameState`. Nothing but the encoder stands between that and the network.

Until this test existed the external bridge numbers were not quotable — see
`agents/outstanding.md` in dev-catan, "the information-set audit for the
catan-bridge numbers, paper-blocking".

**The permutation has to preserve everything that is genuinely public**, or a
failure means nothing. In Catan a seat may see how many cards an opponent
holds, never which. So the resource permutation pools the opponents' cards and
re-deals them, keeping each opponent's *total* fixed and the global multiset
identical — the bank is therefore untouched, and no per-resource count anywhere
in the world changes. Development cards get the same treatment. The deck is
shuffled but not changed in composition, since how many of each card remain is
inferable by counting while the order is not.

What must not move: the perspective seat's own hand (it can see that), played
knights (face up), buildings, roads, the robber, the bank.
"""

from __future__ import annotations

import random

import pytest

# A submodule, not bare "catanatron": this directory is itself named
# `catanatron`, and once pytest's default import mode puts `tests/` on
# sys.path (for the sibling top-level test modules), a bare `catanatron`
# import can resolve to *this directory* as an empty namespace package
# instead of failing -- silently skipping nothing and then blowing up on
# the first real submodule access. `catanatron.game` only exists in the
# real distribution.
pytest.importorskip("catanatron.game")

from hexset.encoding import encode
from catanatron.game import Game as CatanatronGame
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color, RandomPlayer

from hexset.catanatron.board import translate_board
from hexset.catanatron.state import DEV_CARD_NAMES, RESOURCE_NAMES, translate

TRANSLATE_SEED = 1234


def _advance(seed: int, ticks: int):
    """A real catanatron game, played randomly, stopped mid-flight."""
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
    """Encode from one seat, with translation randomness held fixed."""
    our_game, _ = translate(game, mapping, random.Random(TRANSLATE_SEED))
    return encode(our_game, perspective)


def _keys(cstate, color):
    return f"P{cstate.color_to_index[color]}"


def _redeal(cstate, colors, names, rng):
    """Re-deal one card family among `colors`, preserving every invariant a
    seat could legitimately observe: each player's total count, and the global
    multiset (so the bank and the deck stay exactly as they were).

    Returns True if the permutation actually changed anything -- a hand of all
    one resource, or one card between four players, can permute to itself.
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


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("perspective", range(2))
def test_opponent_hands_and_deck_do_not_reach_the_encoding(seed, perspective):
    """Permute everything the seat cannot see; its Observation must not move."""
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


@pytest.mark.parametrize("seed", range(2))
def test_the_audit_can_fail(seed):
    """The control: move something the seat *can* see, and the encoding moves.

    Without this, a test that permutes nothing reachable would pass forever and
    read as proof. Moving the perspective seat's own hand must be visible.
    """
    game, mapping = _advance(seed, ticks=140)
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
