# SPDX-License-Identifier: GPL-3.0-only
"""`POST /api/leave` (`Tables.leave_seat`): retiring your own occupied seat,
the one case `close_seat` refuses on purpose.
"""

from __future__ import annotations

import time

import pytest

from hexset.game import Phase, may_act
from hexset.server.api import ApiError
from hexset.server._webplay import _OpenRound
from hexset.trading import Offer

from conftest import new_tables


def _table():
    """A four-seat table: the creator at `leaver`, on turn, everyone else empty.
    `other` is the seat rotation lands on once `leaver` locks, not merely some
    other seat.
    """
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
    view = registry.handle("POST", "/api/bot", {"seat": a, "model": "test-trader"}, token)
    assert view["locked"] == [] and view["seats"][a]["kind"] == "bot"

    # From the first move on, the seats are fixed.
    registry.handle("POST", "/api/close", {"seat": b}, token)
    registry.handle("POST", "/api/close", {"seat": c}, token)
    # The bot at seat `a` places its opening first when it sits before us.
    deadline = time.monotonic() + 10
    state = registry.handle("GET", "/api/state", {}, token)
    while state["to_move"] != me and time.monotonic() < deadline:
        time.sleep(0.02)
        state = registry.handle("GET", "/api/state", {}, token)
    assert state["to_move"] == me and state["legal_actions"]
    state = registry.handle("POST", "/api/action", {"action": state["legal_actions"][0]}, token)
    assert state["started"] is True
    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/open", {"seat": b}, token)
    assert refused.value.status == 409
    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/bot", {"seat": b, "model": "test-trader"}, token)
    assert "retired" in str(refused.value)


def test_a_bot_can_be_taken_off_its_seat_until_the_first_move(tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    code, token = data["code"], data["token"]
    me = registry.get(code).seat_of(token)
    a, b, c = [s for s in range(4) if s != me]

    registry.handle("POST", "/api/bot", {"seat": a, "model": "test-trader"}, token)
    view = registry.handle("POST", "/api/open", {"seat": a}, token)
    assert view["seats"][a]["kind"] == "empty" and view["locked"] == []
    registry.handle("POST", "/api/bot", {"seat": b, "model": "test-trader"}, token)
    view = registry.handle("POST", "/api/close", {"seat": b}, token)
    assert view["seats"][b]["kind"] == "empty" and view["locked"] == [b]
    assert not registry.get(code).runners

    # A restart reads the journal the same way.
    reopened = new_tables(games_dir=str(tmp_path)).get(code)
    assert [seat.kind.value for seat in reopened.seats] == [
        "player" if s == me else "empty" for s in range(4)
    ]
    assert reopened.session.game.locked == {b}
    assert not reopened.session.bot_names
    reopened.stop_runners()  # not `close`, which would file the game as abandoned

    # Down to one player: the game is the creator's alone. Seat `a`, still
    # open, holds the table while the bot sits at `c`.
    registry.handle("POST", "/api/bot", {"seat": c, "model": "test-trader"}, token)
    registry.handle("POST", "/api/close", {"seat": c}, token)
    registry.handle("POST", "/api/close", {"seat": a}, token)
    state = registry.handle("GET", "/api/state", {}, token)
    assert state["to_move"] == me and state["locked"] == sorted([a, b, c])
    registry.handle("POST", "/api/action", {"action": state["legal_actions"][0]}, token)
    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/open", {"seat": a}, token)
    assert refused.value.status == 409


def test_every_seated_mutation_refuses_once_the_game_is_over():
    """The one gate in `Tables._seated`, across every route it covers,
    including those with no refusal of their own.
    """
    registry, table, _code, token, leaver, _other = _table()
    table.session.game.phase = Phase.GAME_OVER

    routes = [
        ("POST", "/api/name", {"name": "new name"}),
        ("POST", "/api/action", {"action": {"type": "END_TURN"}}),
        ("POST", "/api/undo", {}),
        ("POST", "/api/open", {"seat": (leaver + 1) % 4}),
        ("POST", "/api/close", {"seat": (leaver + 1) % 4}),
    ]
    for method, path, payload in routes:
        with pytest.raises(ApiError) as excinfo:
            registry.handle(method, path, payload, token)
        assert "already over" in excinfo.value.args[0], f"{path} did not refuse"

    # Reads still work: a finished game stays as observable as before.
    registry.handle("GET", "/api/state", {}, token)


def test_leave_locks_the_seat_and_hands_the_turn_on():
    """Leaving mid-turn ends the turn: the next seat starts its own at the
    roll, not the rest of the leaver's."""
    registry, table, _code, token, leaver, other = _table()
    game = table.session.game
    turns = game.turns

    data = registry.handle("POST", "/api/leave", {}, token)

    assert leaver in game.locked
    assert not may_act(game, leaver)
    assert game.current_player == other
    assert game.phase is Phase.ROLL
    assert game.turns == turns + 1
    assert data["log"] == table.view(leaver)["log"]


def test_leave_is_idempotent():
    registry, table, code, token, leaver, other = _table()
    game = table.session.game

    registry.handle("POST", "/api/leave", {}, token)
    turn_after_first_leave = game.current_player

    registry.handle("POST", "/api/leave", {}, token)  # no-op, does not re-raise

    assert game.current_player == turn_after_first_leave
    assert game.locked == {leaver}


def test_leave_refuses_as_the_open_round_s_actor():
    registry, table, code, token, leaver, other = _table()
    table.session.open_round = _OpenRound(offer=Offer(leaver, [0, 0, 0, 0, 0]), awaiting={other})

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/leave", {}, token)

    assert "open trade round" in excinfo.value.args[0]
    assert leaver not in table.session.game.locked


def test_leave_refuses_as_a_round_s_still_awaiting_responder():
    registry, table, code, token, leaver, other = _table()
    table.session.open_round = _OpenRound(offer=Offer(other, [0, 0, 0, 0, 0]), awaiting={leaver})

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/leave", {}, token)

    assert "open trade round" in excinfo.value.args[0]
    assert leaver not in table.session.game.locked


def test_the_last_seat_in_the_game_cannot_leave_it(tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    token = data["token"]
    table = registry.get(data["code"])
    me = table.seat_of(token)
    for seat in range(4):
        if seat != me:
            registry.handle("POST", "/api/close", {"seat": seat}, token)
    path = next(tmp_path.glob("*.jsonl"))
    before = path.read_bytes()

    with pytest.raises(ApiError, match="last seat") as refused:
        registry.handle("POST", "/api/leave", {}, token)

    assert refused.value.status == 409
    assert me not in table.session.game.locked
    assert path.read_bytes() == before


def test_leaving_lets_go_of_a_setup_turn_held_open():
    """A browser seat that placed its setup road and left before ending the
    turn would otherwise hold every other seat for good."""
    registry, table, _code, token, leaver, _other = _table()
    session = table.session
    session.awaiting_confirm = leaver

    view = registry.handle("POST", "/api/leave", {}, token)

    assert session.awaiting_confirm is None
    assert view["awaiting_confirm"] is None
