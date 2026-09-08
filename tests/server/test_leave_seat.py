# SPDX-License-Identifier: GPL-3.0-only
"""`POST /api/leave` (`Tables.leave_seat`): retiring your own occupied seat,
the one case `close_seat` refuses on purpose. Positions are set directly on
the engine, the same pattern `test_trade_round_api.py` uses.
"""

from __future__ import annotations

import pytest

from hexset.game import Phase, may_act
from hexset.server.api import ApiError
from hexset.server.seating import locked_of
from hexset.server.webplay import _OpenRound
from hexset.trading import Offer

from conftest import new_tables


def _table():
    """A four-seat table (`Tables.create` always deals `MAX_SEATS`): the
    creator at `leaver`, on turn, everyone else still empty. `other` is the
    seat turn rotation actually lands on once `leaver` locks
    (`_next_unlocked`, seat order from `leaver`, wrapping) -- not merely
    some other seat, since the creator can land anywhere `MAX_SEATS` allows."""
    registry = new_tables()
    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    code, token = data["code"], data["token"]
    table = registry.get(code)
    game = table.session.game
    leaver = table.seat_of(token)
    other = (leaver + 1) % game.num_players
    game.phase = Phase.MAIN
    game.current_player = leaver
    return registry, table, code, token, leaver, other


def test_a_closed_seat_can_be_reopened_or_given_a_bot_until_the_first_move():
    registry = new_tables()
    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    code, token = data["code"], data["token"]
    table = registry.get(code)
    me = table.seat_of(token)
    empties = [s for s in range(4) if s != me]
    a, b, c = empties

    view = registry.handle("POST", "/api/close", {"seat": a}, token)
    assert view["locked"] == [a] and view["started"] is False
    view = registry.handle("POST", "/api/open", {"seat": a}, token)
    assert view["locked"] == []
    view = registry.handle("POST", "/api/close", {"seat": a}, token)
    view = registry.handle("POST", "/api/bot", {"seat": a, "model": "heximax"}, token)
    assert view["locked"] == [] and view["seats"][a]["kind"] == "bot"

    # Fill the rest and make the first move: from here the seats are fixed.
    registry.handle("POST", "/api/close", {"seat": b}, token)
    registry.handle("POST", "/api/close", {"seat": c}, token)
    session = table.session
    while session.game.current_player != me:
        break
    state = registry.handle("GET", "/api/state", {}, token)
    assert state["to_move"] == me and state["legal_actions"]
    state = registry.handle("POST", "/api/action", {"action": state["legal_actions"][0]}, token)
    assert state["started"] is True
    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/open", {"seat": b}, token)
    assert refused.value.status == 409
    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/bot", {"seat": b, "model": "heximax"}, token)
    assert "retired" in str(refused.value)


def test_bot_seat_refuses_once_the_game_is_over():
    """`seat_bot`'s "at any point during the game" stops exactly where the
    game does -- the same "already over" `leave_seat` refuses with below,
    so the seat that made every recorded move stays the one the
    journal/log/win banner still name, for good."""
    registry, table, code, token, leaver, other = _table()
    empty = next(s for s in range(4) if table.seats[s].kind.value == "empty")
    registry.handle("POST", "/api/bot", {"seat": empty, "model": "heximax"}, token)
    table.session.game.phase = Phase.GAME_OVER

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/bot", {"seat": empty, "model": "search2"}, token)

    assert "already over" in excinfo.value.args[0]
    assert table.seats[empty].name == "heximax"


def test_leave_locks_the_seat_and_hands_the_turn_on():
    registry, table, code, token, leaver, other = _table()
    game = table.session.game

    data = registry.handle("POST", "/api/leave", {}, token)

    assert leaver in locked_of(game)
    assert not may_act(game, leaver)
    assert game.current_player == other
    assert data["log"] == table.view(leaver)["log"]


def test_leave_is_idempotent():
    registry, table, code, token, leaver, other = _table()
    game = table.session.game

    registry.handle("POST", "/api/leave", {}, token)
    turn_after_first_leave = game.current_player

    registry.handle("POST", "/api/leave", {}, token)  # no-op, does not re-raise

    assert game.current_player == turn_after_first_leave
    assert locked_of(game) == {leaver}


def test_leave_refuses_once_the_game_is_over():
    registry, table, code, token, leaver, other = _table()
    table.session.game.phase = Phase.GAME_OVER

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/leave", {}, token)

    assert "already over" in excinfo.value.args[0]
    assert leaver not in locked_of(table.session.game)


def test_leave_refuses_as_the_open_round_s_actor():
    registry, table, code, token, leaver, other = _table()
    table.session.open_round = _OpenRound(offer=Offer(leaver, [0, 0, 0, 0, 0]), awaiting={other})

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/leave", {}, token)

    assert "open trade round" in excinfo.value.args[0]
    assert leaver not in locked_of(table.session.game)


def test_leave_refuses_as_a_round_s_still_awaiting_responder():
    """The mirror of the actor case: `other` is still owed an answer, so
    `leaver` -- the round's actor here -- must not be the one leaving; it's
    `other` who is guarded, since it's `other`'s answer the round is
    waiting on."""
    registry, table, code, token, leaver, other = _table()
    table.session.open_round = _OpenRound(offer=Offer(other, [0, 0, 0, 0, 0]), awaiting={leaver})

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/leave", {}, token)

    assert "open trade round" in excinfo.value.args[0]
    assert leaver not in locked_of(table.session.game)
