"""HTTP-layer tests for `catan.webserver`.

Torch-free on purpose: the opponent here is `catan.bots.RandomBot`, not a
loaded checkpoint, so this suite runs anywhere the rest of the engine's tests
do. What it is pinning is the transport — status codes, JSON shape, and that
an action `legal_actions` did not offer is refused over HTTP the same way
`GameSession.apply_human_action` refuses it in-process (`test_webplay.py`
covers that half directly).
"""

from __future__ import annotations

import json
import random
import threading
import urllib.error
import urllib.request

import pytest

from catan.actions import Action, ActionType, legal_actions
from catan.board.board import random_base_board
from catan.bots import RandomBot
from catan.game import start, to_move
from catan.webplay import GameSession, action_to_wire, board_layout
from catan.webserver import CatanServer


def _new_session(seed: int) -> GameSession:
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    return GameSession(game=game, human_seat=to_move(game), bot=RandomBot(rng=random.Random(seed)))


@pytest.fixture
def live_server():
    session = _new_session(1)
    layout = board_layout(session.game.state.board)
    # `new_session` is always called with a `bots` argument in real use
    # (`_handle_new` passes through whatever `/api/new` resolved, `None` when
    # the request omitted it) — this fixture always seats the same fixed
    # RandomBot lineup regardless, so it just needs to accept and ignore it.
    server = CatanServer(("127.0.0.1", 0), session, layout, lambda bots=None: _new_session(2))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return json.loads(response.read())


def _post(base: str, path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_index_serves_html(live_server):
    _, base = live_server
    with urllib.request.urlopen(base + "/", timeout=5) as response:
        assert response.status == 200
        assert "text/html" in response.headers.get("Content-Type", "")
        body = response.read()
    assert b"<html" in body.lower()


def test_board_endpoint_matches_the_static_layout(live_server):
    server, base = live_server
    data = _get(base, "/api/board")
    assert data["hexes"] == server.layout["hexes"]
    assert len(data["vertices"]) == server.layout["vertices"].__len__()


def test_state_endpoint_reflects_the_live_session(live_server):
    server, base = live_server
    data = _get(base, "/api/state")
    assert data["human_seat"] == server.session.human_seat
    assert data["phase"] == "SETUP_SETTLEMENT"
    assert data["round"] == 0


def test_a_legal_action_is_accepted_and_advances_the_game(live_server):
    server, base = live_server
    human_seat = server.session.human_seat
    options = legal_actions(server.session.game)
    assert to_move(server.session.game) == human_seat  # fixture guarantees this
    wire = action_to_wire(options[0])

    status, data = _post(base, "/api/action", wire)
    assert status == 200
    assert "error" not in data
    # A setup settlement is immediately followed by that same player's road.
    assert data["phase"] == "SETUP_ROAD"
    assert len(data["log"]) >= 1


def test_an_action_absent_from_legal_actions_is_rejected_over_http(live_server):
    server, base = live_server
    # ROLL is never legal during setup placement.
    wire = action_to_wire(Action(ActionType.ROLL))
    status, data = _post(base, "/api/action", wire)
    assert status == 400
    assert "error" in data
    # The game must not have moved on.
    assert data["phase"] == "SETUP_SETTLEMENT"
    assert data["round"] == 0


def test_malformed_json_body_is_rejected(live_server):
    _, base = live_server
    request = urllib.request.Request(
        base + "/api/action", data=b"{not json", headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=5)
        assert False, "expected an HTTPError"
    except urllib.error.HTTPError as exc:
        assert exc.code == 400


def test_new_game_replaces_the_session_and_its_cached_board(live_server):
    server, base = live_server
    old_session = server.session
    status, data = _post(base, "/api/new", {})
    assert status == 200
    assert data["round"] == 0
    assert data["phase"] == "SETUP_SETTLEMENT"
    # A genuinely new session object, not the same one mutated in place.
    assert server.session is not old_session
    # The cached /api/board layout was rebuilt from the new session's board,
    # not left describing the board the old session started with.
    assert _get(base, "/api/board") == board_layout(server.session.game.state.board)


def test_unknown_paths_404(live_server):
    _, base = live_server
    try:
        urllib.request.urlopen(base + "/api/nope", timeout=5)
        assert False, "expected an HTTPError"
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
