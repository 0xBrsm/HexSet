# SPDX-License-Identifier: GPL-3.0-only
"""Every hexset legal action, at many real positions, must resolve to a real
catanatron playable_action.
"""

import random

import pytest

# A submodule, not bare "catanatron": this directory is named `catanatron`
# too, so with `tests/` on sys.path a bare import can resolve to it as an
# empty namespace package and skip nothing. `catanatron.game` exists only in
# the real distribution.
pytest.importorskip("catanatron.game")

from hexset.actions import ActionType as OurActionType, legal_actions

from catanatron.game import Game as CatanatronGame
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color, RandomPlayer

from hexset.catanatron._actions import to_catanatron
from hexset.catanatron._board import translate_board
from tests.catanatron.oracle import translate

# Known gap, tolerated rather than swallowed: hexset does not enforce the
# physical piece limit (15 roads / 5 settlements / 4 cities) catanatron
# does, so `legal_actions` can offer a build catanatron refuses.
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
    # Second gap, the other direction: hexset offers PLAY_ROAD_BUILDING
    # whenever the card is held; catanatron only when
    # `road_building_possibilities` is non-empty.
    if action.type is OurActionType.PLAY_ROAD_BUILDING:
        color = catanatron_state.current_color()
        key = f"P{catanatron_state.color_to_index[color]}"
        has_roads_available = catanatron_state.player_state[f"{key}_ROADS_AVAILABLE"] > 0
        if not has_roads_available or not catanatron_state.board.buildable_edges(color):
            return True
    # Third gap, catanatron's own: `apply_end_turn` never resets
    # `is_road_building`/`free_roads_available` when a seat ends its turn with
    # free roads unplaced, so the stale flag forces BUILD_ROAD-only options.
    return bool(catanatron_state.is_road_building)


@pytest.mark.parametrize("seed", [2])
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
