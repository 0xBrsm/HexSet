"""What `POST /api/games` deals a table as: the board mode and the game type,
their defaults, the duel variant's two fixed seats, and a journal bringing
all of it back -- including journals written before either could be chosen.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from conftest import new_tables

from hexset.actions import Action, ActionType, legal_actions
from hexset.board.board import random_base_board, spiral_base_board
from hexset.game import to_move
from hexset.record import from_events
from hexset.rules import DUEL_VARIANT, DUEL_VARIANT_GAME, STANDARD, STANDARD_GAME
from hexset.server import _journal as journal
from hexset.server.api import ApiError, Config, Seat, SeatKind, build_session
from hexset.server.wire import action_to_wire


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


def deal(registry, **body):
    data = registry.handle("POST", "/api/games", body, None)
    return data, registry.get(data["code"])


def journalled(directory) -> list[dict]:
    return journal.read(next(Path(directory).glob("*.jsonl")))


def drive(session, moves: int, rng: random.Random) -> None:
    for _ in range(moves):
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        session.submit(to_move(session.game), action_to_wire(rng.choice(legal_actions(session.game))))


def test_the_server_deals_a_standard_game_on_a_spiral_board_by_default():
    assert Config().board_mode == "spiral"
    data, table = deal(new_tables(seed=7))
    assert table.session.game._state.board == spiral_base_board(random.Random(7))
    assert (data["board_mode"], data["game_type"], data["seats_fixed"]) == ("spiral", "standard", False)
    assert data["locked"] == [] and data["winning_points"] == 10
    assert table.session.game._state.rules == STANDARD


def test_the_body_names_the_board_and_overrides_the_servers_default():
    registry = new_tables(seed=7)
    _, table = deal(registry, board_mode="random")
    assert table.session.game._state.board == random_base_board(random.Random(7))
    assert table.session.board_mode == "random"
    _, table = deal(new_tables(seed=7, board_mode="random"), board_mode="spiral")
    assert table.session.game._state.board == spiral_base_board(random.Random(7))


@pytest.mark.parametrize("body", [
    {"board_mode": "hexagonal"}, {"board_mode": 7}, {"board_mode": ["spiral"]},
    {"game_type": "duel"}, {"game_type": 2}, {"game_type": {"name": "standard"}},
    {"game_type": "duel-variant", "bots": ["test-trader", "test-trader"]},
])
def test_an_unknown_board_or_game_type_or_too_many_bots_is_a_400(body, tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/games", body, None)
    assert refused.value.status == 400
    assert not list(tmp_path.glob("*.jsonl")), "nothing is dealt"


def test_a_duel_table_plays_two_seats_and_its_other_two_stay_closed():
    registry = new_tables()
    data, table = deal(registry, game_type="duel-variant")
    token = data["token"]
    assert data["seat"] == 0 and data["locked"] == [2, 3]
    assert (data["game_type"], data["seats_fixed"], data["winning_points"]) == ("duel-variant", True, 15)
    assert table.session.game._state.rules == DUEL_VARIANT
    assert table.session.game_type is DUEL_VARIANT_GAME
    assert data["waiting_for"] == [1]

    assert data["fixed_seats"] == [2, 3]
    for route, body in (("/api/open", {"seat": 2}), ("/api/close", {"seat": 3})):
        with pytest.raises(ApiError, match="seats are fixed") as refused:
            registry.handle("POST", route, body, token)
        assert refused.value.status == 409
    with pytest.raises(ApiError, match="seats are fixed") as refused:
        registry.handle("POST", "/api/bot", {"seat": 3, "model": "test-trader"}, token)
    assert refused.value.status == 400
    assert sorted(table.session.game.locked) == [2, 3]

    seat, _ = table.join("Bea")
    assert seat == 1
    with pytest.raises(ApiError) as full:
        table.join("Cy")
    assert full.value.status == 409


def test_a_person_can_practise_a_duel_alone():
    registry = new_tables()
    data, table = deal(registry, game_type="duel-variant")
    token = data["token"]
    view = registry.handle("POST", "/api/close", {"seat": 1}, token)
    assert view["locked"] == [1, 2, 3] and view["to_move"] == 0
    view = registry.handle("POST", "/api/open", {"seat": 1}, token)
    assert view["locked"] == [2, 3] and view["waiting_for"] == [1]
    registry.handle("POST", "/api/close", {"seat": 1}, token)
    # Played on alone, under the duel's rules, past setup and the first rolls.
    rng = random.Random(0)
    state = registry.handle("GET", "/api/state", {}, token)
    for _ in range(300):
        assert state["your_move"] in ("act", "discard"), state["your_move"]
        state = registry.handle("POST", "/api/action", {"action": rng.choice(state["legal_actions"])}, token)
    assert state["round"] > 5 and state["to_move"] == 0


def test_a_standard_table_still_closes_down_to_two_seats():
    registry = new_tables()
    data, table = deal(registry, game_type="standard")
    for seat in (2, 3):
        registry.handle("POST", "/api/close", {"seat": seat}, data["token"])
    view = registry.handle("POST", "/api/open", {"seat": 3}, data["token"])
    assert view["locked"] == [2] and view["seats_fixed"] is False


def test_the_bot_the_creator_seats_plays_a_duel(monkeypatch):
    # The creator at seat 1, so the bot at seat 0 opens the setup snake.
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 1)
    registry = new_tables()
    data, table = deal(registry, game_type="duel-variant", bots=["test-trader"])
    token = data["token"]
    assert data["seat"] == 1 and data["seats"][0]["kind"] == "bot" and data["waiting_for"] == []
    deadline = time.monotonic() + 10
    state = registry.handle("GET", "/api/state", {}, token)
    while state["to_move"] != 1 and time.monotonic() < deadline:
        time.sleep(0.02)
        state = registry.handle("GET", "/api/state", {}, token)
    assert state["to_move"] == 1 and state["legal_actions"]
    assert 0 in state["vertex_owner"], "the bot placed its first settlement"


def test_a_duel_comes_back_from_its_journal_as_a_duel(tmp_path):
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    data, table = deal(registry, game_type="duel-variant", board_mode="random")
    table.join("Bea")
    drive(table.session, 12, random.Random(4))
    header = journalled(tmp_path)[0]
    assert (header["game_type"], header["locked"], header["board_mode"]) == ("duel-variant", [2, 3], "random")

    # A fresh server, as after a restart.
    resumed = new_tables(games_dir=str(tmp_path), seed=99).get(data["code"]).session

    assert resumed.game_type is DUEL_VARIANT_GAME and resumed.seats_fixed
    assert resumed.game._state.rules == DUEL_VARIANT
    assert sorted(resumed.game.locked) == [2, 3]
    assert resumed.game._state.board == random_base_board(random.Random(99))
    assert resumed.game._state.vertex_owner == table.session.game._state.vertex_owner
    assert resumed.game._state.hands == table.session.game._state.hands
    assert from_events(journalled(tmp_path), partial=True).locked == (2, 3)


def test_a_journal_from_before_the_choices_resumes_as_a_standard_game_on_a_random_board(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99, board_mode="random")
    seats = [Seat(kind=SeatKind.PLAYER, name=name, token="t-" + name) for name in ("Ada", "Bea", "Cy", "Di")]
    session = build_session("abc234", seats, config, first=0)
    drive(session, 10, random.Random(4))
    path = next(Path(tmp_path).glob("*.jsonl"))
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    for key in ("board_mode", "game_type", "locked"):
        del header[key]
    path.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")

    resumed = new_tables(games_dir=str(tmp_path)).get("abc234").session

    assert resumed.board_mode == "random" and resumed.game_type is STANDARD_GAME
    assert resumed.game._state.board == random_base_board(random.Random(99))
    assert resumed.game._state.vertex_owner == session.game._state.vertex_owner
