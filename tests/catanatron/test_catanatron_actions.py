# SPDX-License-Identifier: GPL-3.0-only
"""Every dev-catan legal action, at many real positions, must resolve to a
real catanatron playable_action.

`PLAY_KNIGHT` is excluded deliberately: it is resolved across two catanatron
decisions rather than one (see `player.py`) and is covered by
`test_player.py` instead, end to end.
"""

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

from hexset.actions import ActionType as OurActionType, legal_actions

from catanatron.game import Game as CatanatronGame
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color, RandomPlayer

from hexset.catanatron.actions import to_catanatron
from hexset.catanatron.board import translate_board
from hexset.catanatron.state import translate

EXCLUDED = {OurActionType.PLAY_KNIGHT}

# dev-catan's engine does not enforce the physical piece limit (15 roads / 5
# settlements / 4 cities per player) that real Catan and catanatron both do
# -- confirmed by hand: at the one failure this test found, the player had
# built all 15 roads (ROADS_AVAILABLE == 0) and catanatron correctly refused
# a 16th while dev-catan's `legal_actions` still offered it. That is a gap in
# dev-catan's own engine, not a translation bug, and out of scope to fix from
# this repo. It is rare enough that it has likely never been hit by any of
# dev-catan's own recorded games (all of them shorter, better-played games
# than 400 ticks of pure random play), so it is skipped here rather than
# silently swallowed: if the cause is anything other than piece exhaustion,
# this still fails loudly.
_PIECE_AVAILABLE_KEY = {
    OurActionType.BUILD_ROAD: "ROADS_AVAILABLE",
    OurActionType.BUILD_SETTLEMENT: "SETTLEMENTS_AVAILABLE",
    OurActionType.BUILD_CITY: "CITIES_AVAILABLE",
}


def _out_of_pieces(action, catanatron_state) -> bool:
    key = _PIECE_AVAILABLE_KEY.get(action.type)
    if key is None:
        return False
    p = f"P{catanatron_state.color_to_index[catanatron_state.current_color()]}"
    return catanatron_state.player_state[f"{p}_{key}"] <= 0


def _known_limitation(action, catanatron_state) -> bool:
    if _out_of_pieces(action, catanatron_state):
        return True
    # A second gap, the other direction: dev-catan offers PLAY_ROAD_BUILDING
    # whenever the card is held, even with nowhere to build (or no roads left
    # in stock -- the piece-cap gap above, reached through a second path).
    # catanatron only offers it when `road_building_possibilities` is
    # non-empty -- a legal card play that can't achieve anything is filtered
    # out before it is offered at all, rather than being legal-but-useless as
    # dev-catan has it.
    if action.type is OurActionType.PLAY_ROAD_BUILDING:
        color = catanatron_state.current_color()
        key = f"P{catanatron_state.color_to_index[color]}"
        has_roads_available = catanatron_state.player_state[f"{key}_ROADS_AVAILABLE"] > 0
        if not has_roads_available or not catanatron_state.board.buildable_edges(color):
            return True
    # A third, independent gap: catanatron's `apply_end_turn` never resets
    # `is_road_building`/`free_roads_available` when a player ends their turn
    # without placing every free road (e.g. no legal edge left). The flag is
    # then stale -- true again next time that seat is asked to decide,
    # forcing `generate_playable_actions` to offer *only* BUILD_ROAD even
    # before they have rolled. dev-catan has no phase for "must resolve free
    # roads before anything else, including rolling": Phase.ROLL only ever
    # offers ROLL/PLAY_KNIGHT. This is catanatron's own state bug (confirmed
    # by reading `apply_action.py`), not something to route around here.
    return bool(catanatron_state.is_road_building)


@pytest.mark.parametrize("seed", range(3))
def test_every_legal_action_resolves(seed):
    random.seed(seed)
    players = [RandomPlayer(c) for c in Color]
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    game = CatanatronGame(players, catan_map=catan_map)
    mapping = translate_board(catan_map)
    rng = random.Random(seed)

    checked = 0
    for _ in range(400):
        if game.winning_color() is not None:
            break
        our_game, seats = translate(game, mapping, rng)
        options = legal_actions(our_game)
        for action in options:
            if action.type in EXCLUDED:
                continue
            try:
                resolved = to_catanatron(action, our_game, mapping, seats, game.playable_actions)
            except ValueError:
                assert _known_limitation(action, game.state), (
                    f"{action} failed to resolve for an unrecognised reason -- "
                    "this is a real bug, not one of the documented gaps"
                )
                continue
            assert resolved in game.playable_actions
            checked += 1

        action = game.state.current_player().decide(game, game.playable_actions)
        game.execute(action)

    assert checked > 200
