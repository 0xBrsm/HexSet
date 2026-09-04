"""HTTP-layer tests for `hexset.server.web`.

Only what the transport itself adds: status codes, the static file, the token
header, and that a refusal `api.py` raises arrives as the status it carries
rather than as a dropped connection. The rules those refusals come from are
`test_api.py`'s, tested there without a socket.

Torch-free on purpose: the opponents are named `search2` at every call, so this
suite runs anywhere the rest of the engine's tests do.
"""

from __future__ import annotations

import json

import random

import threading

import urllib.error

import urllib.request

import pytest

from hexset.actions import ActionType, legal_actions

from conftest import new_tables

from hexset.server.web import TOKEN_HEADER, HexSetServer

from hexset.server.webplay import action_to_wire

SOLO = ["search2", "search2", "search2"]


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    """Turn order is seat order from seat 0 (`Tables.create` always deals
    `first=0`, see `hexset.server.seating`'s module docstring). Without this,
    a randomly seated creator can leave a bot seat on move first, whose
    background runner thread starts playing setup concurrently with this
    test's own client requests -- pin the creator to seat 0 so the game sits
    idle until the client acts."""
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


@pytest.fixture
def live_server():
    server = HexSetServer(("127.0.0.1", 0), new_tables())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class Client:
    """One seat's worth of HTTP: whatever token the server last minted, sent
    on every request after, the same way a browser sends the one it kept in
    localStorage."""

    def __init__(self, base: str) -> None:
        self.base = base
        self.token: str | None = None

    def _send(self, request: urllib.request.Request):
        if self.token is not None:
            request.add_header(TOKEN_HEADER, self.token)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def get(self, path: str):
        return self._send(urllib.request.Request(self.base + path))

    def post(self, path: str, payload: dict):
        status, data = self._send(
            urllib.request.Request(
                self.base + path,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        if isinstance(data, dict) and data.get("token"):
            self.token = data["token"]
        return status, data


def seated(base: str, **kwargs) -> tuple[Client, dict]:
    """A client holding a token for a freshly dealt (and, since there is no
    lobby, already-playable) game."""
    client = Client(base)
    kwargs.setdefault("bots", SOLO)
    status, data = client.post("/api/games", kwargs)
    assert status == 200, data
    return client, data


def test_a_game_is_dealt_and_played_over_http(live_server):
    server, base = live_server
    client, table = seated(base)
    assert table["code"]
    assert table["phase"] == "SETUP_SETTLEMENT"  # already playable, no separate start

    session = server.tables.get(table["code"]).session
    settlement = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    )
    status, data = client.post("/api/action", {"action": settlement})
    assert status == 200
    assert data["phase"] == "SETUP_ROAD"
    assert len(data["log"]) >= 1
