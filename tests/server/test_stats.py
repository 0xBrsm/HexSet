"""A game's dice and robber numbers, read from its journal (`_stats`)."""

from __future__ import annotations

import pytest

from test_rollout import a_road_or_not

from hexset.server._stats import WAYS, game_stats, seat_stats


def _board_and_spot():
    game = a_road_or_not()
    board = game.state(0, hidden=False).board
    topo = board.topology
    # A numbered hex that is not the robber's, and a corner of it.
    hexes = [h for h in range(board.num_hexes) if board.tokens[h] in WAYS and h != game._state.robber]
    h = hexes[0]
    return board, h, topo.hex_vertices[h][0], game._state.robber


def _hands(a, b):
    return [[a, 0, 0, 0, 0], [b, 0, 0, 0, 0], [0] * 5, [0] * 5]


def test_rolls_robber_and_takebacks_are_counted_as_played():
    board, h, v, robber = _board_and_spot()
    token = board.tokens[h]
    events = [
        {"kind": "game", "num_players": 4, "locked": [2, 3], "robber": robber},
        {"kind": "action", "step": 0, "round": 0, "actor": 0, "type": "SETUP_SETTLEMENT", "a": v, "hands": _hands(0, 0)},
        # Its number comes up: one card for seat 0.
        {"kind": "action", "step": 1, "round": 1, "actor": 0, "type": "ROLL", "roll": token, "hands": _hands(1, 0)},
        # Taken back, then a 7 and a steal by seat 1 on that hex.
        {"kind": "action", "step": 2, "round": 1, "actor": 0, "type": "ROLL", "roll": token, "hands": _hands(2, 0)},
        {"kind": "undo", "back_to": 2, "hands": _hands(1, 0)},
        {"kind": "action", "step": 2, "round": 1, "actor": 1, "type": "MOVE_ROBBER", "a": h, "b": 0, "hands": _hands(0, 1)},
        # Its number again, now under the robber: blocked.
        {"kind": "action", "step": 3, "round": 2, "actor": 0, "type": "ROLL", "roll": token, "hands": _hands(0, 1)},
    ]
    stats = game_stats(events, board)
    # The corner touches up to three hexes; each pays it on its own number.
    around = [g for g in range(board.num_hexes) if v in board.topology.hex_vertices[g] and board.tokens[g] in WAYS]
    every = sum(WAYS[board.tokens[g]] / 30 for g in around)
    share = WAYS[token] / 30
    assert stats["act"][0] == 1 and stats["blocked"][0] == 1
    assert stats["exp"][0] == pytest.approx(2 * every - share) and stats["exp_all"][0] == pytest.approx(2 * every)
    assert (stats["stolen"][0], stats["gained"][1], stats["on"][0], stats["moves"][0]) == (1, 1, 1, 1)
    assert stats["rounds"] == 2
    mine = seat_stats(events, board, 0)
    # In cards: the dice paid 1 and 1 blocked, against 2 * every on average; the robber blocked 1 and stole 1.
    assert mine["you"]["dice"] == pytest.approx(2 - 2 * every, abs=0.01) and mine["you"]["robber"] == -2
    assert mine["you"]["net"] == pytest.approx(-2 * every, abs=0.01)
    assert mine["them"] == {"dice": 0, "robber": 1, "net": 1, "on": 0, "moves": 0, "discarded": 0}
    assert seat_stats([], board, 0) is None
