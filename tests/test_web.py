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
from hexset_ui.web import TOKEN_HEADER, HexSetServer, is_code, looks_like_a_code_attempt
from hexset_ui.webplay import action_to_wire

SOLO = ["search2", "search2", "search2"]


@pytest.fixture
def live_server():
    server = HexSetServer(("127.0.0.1", 0), Tables(Config(games_dir="", seat_grace=0.0)))
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


def test_a_malformed_code_shaped_path_still_gets_the_page():
    """Six characters is the length of a real code even if one of them isn't
    in CODE_ALPHABET — a mistyped 0/1/I/L, say — so it belongs on the same
    page a real code does, where the SPA can say "no such game" itself,
    rather than a bare 404 that reads as a missing asset."""
    assert looks_like_a_code_attempt("/ABC01I")  # confusable characters, still 6 long
    assert not looks_like_a_code_attempt("/favicon")  # the wrong length entirely
    assert not looks_like_a_code_attempt("/ABC23")  # too short


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


def test_an_api_refusal_arrives_as_its_own_status_and_not_a_dropped_connection(live_server):
    """Every `ApiError` carries the status to answer with, so the browser gets
    a message rather than a network failure."""
    _, base = live_server
    client, table = seated(base)  # a full game: creator + SOLO's three bots

    assert client.post("/api/join", {"code": "ZZZZZZ"})[0] == 404
    assert client.post("/api/join", {"code": table["code"]})[0] == 409  # no open seats left
    assert Client(base).get("/api/state")[0] == 401  # no token at all

    status, data = client.post("/api/games", {"bots": ["nope"]})
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
        base + "/api/games",
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
        base + "/api/games",
        data=b"[1, 2, 3]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 400


# --- Observers --------------------------------------------------------------


def test_an_observer_reads_the_game_over_http_without_ever_posting(live_server):
    _, base = live_server
    _, table = seated(base, bots=["search2"])  # two open seats left

    with urllib.request.urlopen(base + f"/api/table/{table['code']}", timeout=5) as response:
        data = json.loads(response.read())

    assert data["code"] == table["code"]
    assert data["seat"] is None
    assert data["legal_actions"] == []


# --- Two browsers ---------------------------------------------------------------


def test_two_browsers_at_one_game_play_from_two_different_seats(live_server):
    _, base = live_server
    ada, table = seated(base, bots=["search2"])
    bea = Client(base)
    status, joined = bea.post("/api/join", {"code": table["code"], "name": "Bea"})
    assert status == 200

    _, ada_state = ada.get("/api/state")
    _, bea_state = bea.get("/api/state")
    assert ada_state["code"] == bea_state["code"] == table["code"]
    assert ada_state["seat"] == table["seat"] and bea_state["seat"] == joined["seat"]
    assert ada_state["seat"] != bea_state["seat"]
    assert set(ada_state["claimed_seats"]) >= {ada_state["seat"], bea_state["seat"]}


def test_two_browsers_at_two_games_never_see_each_others_game(live_server):
    _, base = live_server
    ada, ada_table = seated(base)
    bea, bea_table = seated(base)

    assert ada_table["code"] != bea_table["code"]
    # A token names one seat at one game, so neither request has to say which
    # game it means and neither can be answered about the other one.
    assert ada.get("/api/state")[1]["code"] == ada_table["code"]
    assert bea.get("/api/state")[1]["code"] == bea_table["code"]
