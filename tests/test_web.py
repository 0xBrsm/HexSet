"""HTTP-layer tests for `hexset_ui.web`.

Only what the transport itself adds: status codes, the static file, the token
header, and that a refusal `api.py` raises arrives as the status it carries
rather than as a dropped connection. The rules those refusals come from are
`test_api.py`'s, tested there without a socket.

Torch-free on purpose: the opponents are named `search2` at every call, so this
suite runs anywhere the rest of the engine's tests do.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from hexset_ui.actions import ActionType, legal_actions
from hexset_ui.api import Config, Tables
from hexset_ui.web import TOKEN_HEADER, HexSetServer, is_code
from hexset_ui.webplay import action_to_wire

SOLO = ["search2", "search2", "search2"]


@pytest.fixture
def live_server():
    server = HexSetServer(("127.0.0.1", 0), Tables(Config(games_dir="")))
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
    """A client holding a token for a freshly dealt table."""
    client = Client(base)
    kwargs.setdefault("bots", SOLO[: 3 - kwargs.get("open_seats", 0)])
    status, data = client.post("/api/tables", kwargs)
    assert status == 200, data
    return client, data


# --- Static files and code URLs -------------------------------------------------


def test_index_serves_html(live_server):
    _, base = live_server
    with urllib.request.urlopen(base + "/", timeout=5) as response:
        assert response.status == 200
        assert "text/html" in response.headers.get("Content-Type", "")
        body = response.read()
    assert b"<html" in body.lower()


def test_a_code_url_serves_the_same_page_the_front_door_does(live_server):
    """The server does not resolve the code — a code that does not exist
    should say so in the page rather than as a raw 404."""
    _, base = live_server
    with urllib.request.urlopen(base + "/", timeout=5) as response:
        front = response.read()
    with urllib.request.urlopen(base + "/ZZZZZZ", timeout=5) as response:
        assert response.status == 200
        assert response.read() == front


def test_the_page_is_never_served_from_a_stale_cache(live_server):
    """A phone browser reopening a background tab is exactly the case that
    falls back to a stale copy, which is indistinguishable from a fix that
    did not work."""
    _, base = live_server
    with urllib.request.urlopen(base + "/", timeout=5) as response:
        assert response.headers.get("Cache-Control") == "no-store"


def test_only_something_shaped_like_a_code_is_treated_as_one():
    """Checked against the alphabet rather than just the length, so a missing
    asset still 404s as itself."""
    assert is_code("/ABC234")
    assert is_code("/abc234")
    assert not is_code("/favicon")  # right length, wrong alphabet
    assert not is_code("/ABC01I")  # 0, 1 and I are not in the alphabet at all
    assert not is_code("/ABC23")
    assert not is_code("/robots.txt")


def test_an_asset_that_is_not_there_is_a_404(live_server):
    _, base = live_server
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(base + "/favicon", timeout=5)
    assert caught.value.code == 404


def test_a_post_outside_the_api_is_a_404(live_server):
    """A code is a page to GET, not somewhere to post to."""
    _, base = live_server
    request = urllib.request.Request(base + "/ABC234", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 404


# --- Requests and refusals ------------------------------------------------------


def test_a_table_is_dealt_and_then_played_over_http(live_server):
    server, base = live_server
    client, table = seated(base)
    assert table["code"]

    status, data = client.post("/api/start", {})
    assert status == 200
    assert data["phase"] == "SETUP_SETTLEMENT"

    session = server.tables.get(table["code"]).session
    settlement = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    )
    status, data = client.post("/api/action", {"action": settlement})
    assert status == 200
    assert data["phase"] == "SETUP_ROAD"
    assert len(data["log"]) >= 1


def test_advance_plays_one_seat_at_a_time(live_server):
    """POST /api/advance moves the game by exactly one seat's turn, not the
    whole cascade back to the human in a single response."""
    server, base = live_server
    client, table = seated(base)
    client.post("/api/start", {})
    session = server.tables.get(table["code"]).session

    for kind in (ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD):
        wire = action_to_wire(next(a for a in legal_actions(session.game) if a.type is kind))
        status, data = client.post("/api/action", {"action": wire})
        assert status == 200

    assert data["awaiting_confirm"]  # the setup road just handed off to a bot
    assert data["to_move"] != 0

    status, data = client.post("/api/confirm", {})
    assert status == 200
    assert not data["awaiting_confirm"]

    steps = 0
    while data["to_move"] != 0 and steps < 20:
        status, data = client.post("/api/advance", {})
        assert status == 200
        steps += 1
    assert data["to_move"] == 0  # eventually comes all the way back
    assert steps >= 1  # at least one bot seat actually moved

    # A harmless no-op, not an error, once it is already this seat's turn.
    status, data = client.post("/api/advance", {})
    assert status == 200
    assert data["to_move"] == 0


def test_an_api_refusal_arrives_as_its_own_status_and_not_a_dropped_connection(live_server):
    """Every `ApiError` carries the status to answer with, so the browser gets
    a message rather than a network failure."""
    _, base = live_server
    client, _ = seated(base)

    assert client.post("/api/join", {"code": "ZZZZZZ"})[0] == 404
    assert client.post("/api/action", {"action": {"type": "ROLL"}})[0] == 409  # not started
    assert Client(base).get("/api/state")[0] == 401  # no token at all

    status, data = client.post("/api/tables", {"bots": ["nope"]})
    assert status == 400
    assert "error" in data


def test_an_unhandled_error_is_answered_rather_than_dropped(live_server):
    """http.server's own default is to close the connection mid-response,
    which reaches the browser as a network failure and says nothing at all."""
    server, base = live_server
    client, _ = seated(base)

    def boom(*args, **kwargs):
        raise RuntimeError("a checkpoint that will not load")

    server.tables.handle = boom
    status, data = client.get("/api/state")

    assert status == 500
    assert data["error"] == "RuntimeError: a checkpoint that will not load"


def test_malformed_json_body_is_rejected(live_server):
    _, base = live_server
    request = urllib.request.Request(
        base + "/api/tables",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 400


def test_a_body_that_is_not_an_object_is_rejected(live_server):
    _, base = live_server
    request = urllib.request.Request(
        base + "/api/tables",
        data=b"[1, 2, 3]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 400


# --- Two browsers ---------------------------------------------------------------


def test_two_browsers_at_one_table_play_the_same_game_from_two_seats(live_server):
    _, base = live_server
    ada, table = seated(base, bots=["search2"], open_seats=2)
    bea = Client(base)
    status, joined = bea.post("/api/join", {"code": table["code"], "name": "Bea"})
    assert status == 200
    assert joined["seat"] == 2

    ada.post("/api/start", {})

    _, ada_state = ada.get("/api/state")
    _, bea_state = bea.get("/api/state")
    assert ada_state["code"] == bea_state["code"] == table["code"]
    assert ada_state["seat"] == 0 and bea_state["seat"] == 2
    assert ada_state["human_seats"] == [0, 2]


def test_two_browsers_at_two_tables_never_see_each_others_game(live_server):
    _, base = live_server
    ada, ada_table = seated(base)
    bea, bea_table = seated(base)

    assert ada_table["code"] != bea_table["code"]
    # A token names one seat at one table, so neither request has to say which
    # game it means and neither can be answered about the other one.
    assert ada.get("/api/state")[1]["code"] == ada_table["code"]
    assert bea.get("/api/state")[1]["code"] == bea_table["code"]
