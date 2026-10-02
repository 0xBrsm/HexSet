"""What `hexset.server.api` answers a token-free read with, the lock every
token-free view is built under, and the name an empty rename leaves.
"""

from __future__ import annotations

import random

import pytest

from hexset.game import is_over
from hexset.server import api
from hexset.server.api import (
    Config,
    Seat,
    SeatKind,
    build_session,
)

from conftest import new_tables

SOLO = ["test-trader", "test-trader", "test-trader"]


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


def _hidden(view: dict) -> set[int]:
    """The seats whose hand `view` does not spell out."""
    return {p["seat"] for p in view["players"] if "hand" not in p}


# --- a token-free read -------------------------------------------------------


def test_a_spectator_sees_every_hand_and_a_seat_still_sees_only_its_own():
    registry = new_tables()
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    mine = registry.by_token(token)[1]

    watched = registry.handle("GET", f"/api/table/{code}", {}, None)
    seated = registry.handle("GET", "/api/state", {}, token)

    assert _hidden(watched) == set()
    assert all("dev_cards" in p for p in watched["players"])
    assert _hidden(seated) == set(range(api.MAX_SEATS)) - {mine}


def test_a_seat_replaying_its_own_game_by_an_uppercase_code_is_still_that_seat(tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    mine = registry.by_token(token)[1]

    past = registry.handle("GET", f"/api/table/{code.upper()}/replay?round=0", {}, token)

    assert _hidden(past) == set(range(api.MAX_SEATS)) - {mine}


def _play_out(session, rng: random.Random) -> None:
    """Every seat's moves at random until the game ends."""
    from hexset.actions import Action, ActionType, legal_actions
    from hexset.game import to_move
    from hexset.server.wire import action_to_wire

    for _ in range(3000):
        if is_over(session.game):
            return
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))


def test_once_the_game_is_over_a_spectator_sees_every_hand(tmp_path):
    seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada")] + [
        Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader") for _ in range(3)
    ]
    session = build_session("abc234", seats, Config(games_dir=str(tmp_path), seed=99), first=0)
    _play_out(session, random.Random(4))
    assert is_over(session.game)

    watched = new_tables(games_dir=str(tmp_path)).handle("GET", "/api/table/abc234", {}, None)

    assert _hidden(watched) == set()


# --- locks ---------------------------------------------------------------------


def test_every_token_free_view_is_built_under_the_tables_lock(monkeypatch):
    """The spectator read, the create reply and the reclaim reply, each of
    which once built its view with the table's lock released."""
    held: list[bool] = []
    real = api.Table.view

    def view(self, *args, **kwargs):
        held.append(self.lock.locked())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(api.Table, "view", view)
    registry = new_tables()
    secret = "a secret of the creator's own"
    import hashlib

    client = {"id": hashlib.sha256(secret.encode()).hexdigest(), "kind": "api"}
    code = registry.handle("POST", "/api/games", {"bots": [], "client": client}, None)["code"]
    registry.handle("GET", f"/api/table/{code}", {}, None)
    registry.handle("POST", "/api/reclaim", {"code": code, "secret": secret}, None)

    assert held == [True, True, True]


# --- names ---------------------------------------------------------------------


def test_an_empty_name_falls_back_to_the_seats_default():
    registry = new_tables()
    dealt = registry.handle("POST", "/api/games", {"bots": [], "name": "Ada"}, None)
    token = dealt["token"]
    seat = registry.by_token(token)[1]

    view = registry.handle("POST", "/api/name", {"name": "   "}, token)

    assert view["seats"][seat]["name"] == api.default_seat_name("api")
