"""HTTP-layer tests for `catan.webserver`.

Torch-free on purpose: the opponent here is `catan.bots.RandomBot`, not a
loaded checkpoint, so this suite runs anywhere the rest of the engine's tests
do. What it is pinning is the transport — status codes, JSON shape, that an
action `legal_actions` did not offer is refused over HTTP the same way
`GameSession.apply_human_action` refuses it in-process (`test_webplay.py`
covers that half directly), and — since `CatanServer` now keys games off an
identity cookie rather than holding one shared session — that two "browsers"
(two separate cookie jars, via `_client`) really do get two independent
games while one browser's own requests keep landing on the same one.
"""

from __future__ import annotations

import http.cookiejar
import json
import random
import threading
import time
import urllib.error
import urllib.request

import pytest

from catan import journal
from catan.actions import Action, ActionType, legal_actions
from catan.board.board import random_base_board
from catan.bots import RandomBot
from catan.game import is_over, start, to_move
from catan.webplay import GameSession, action_to_wire, board_layout
from catan.webserver import (
    COOKIE_NAME,
    CatanServer,
    _build_session,
    _Entry,
    _resume_session,
)


def _new_session(seed: int) -> GameSession:
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    return GameSession(game=game, human_seat=to_move(game), bot=RandomBot(rng=random.Random(seed)))


@pytest.fixture
def live_server():
    # Every identity's first request deals it a session via this same
    # callable — unlike the single eager session `main()` used to build
    # before the server even started, there is no longer one "the" session
    # to hand the fixture up front (see CatanServer.entry).
    # No resume_session is passed: these tests are about the HTTP surface,
    # and the default (never resume) keeps every first request a fresh deal.
    server = CatanServer(("127.0.0.1", 0), lambda bots, identity: _new_session(1))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class _Client:
    """One simulated browser: a private cookie jar, so its `catan_id` (once
    the server hands it one) is remembered across calls the same way a real
    browser remembers it, and never leaks to another `_Client` instance.
    """

    def __init__(self, base: str) -> None:
        self.base = base
        jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self._jar = jar

    @property
    def identity(self) -> str | None:
        for cookie in self._jar:
            if cookie.name == COOKIE_NAME:
                return cookie.value
        return None

    def get(self, path: str) -> dict:
        with self._opener.open(self.base + path, timeout=5) as response:
            return json.loads(response.read())

    def post(self, path: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with self._opener.open(request, timeout=5) as response:
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


def test_the_first_response_hands_out_an_identity_cookie(live_server):
    _, base = live_server
    client = _Client(base)
    assert client.identity is None
    client.get("/api/state")
    assert client.identity is not None


def test_a_returning_cookie_is_not_reissued(live_server):
    """A request that already carries `catan_id` gets the exact same value
    back, not a fresh one — otherwise every poll would silently start a new
    game."""
    _, base = live_server
    client = _Client(base)
    client.get("/api/state")
    first = client.identity
    client.get("/api/state")
    assert client.identity == first


def test_board_endpoint_matches_the_sessions_own_board(live_server):
    server, base = live_server
    client = _Client(base)
    data = client.get("/api/board")
    entry = server.sessions[client.identity]
    assert data["hexes"] == entry.layout["hexes"]
    assert len(data["vertices"]) == len(entry.layout["vertices"])


def test_state_endpoint_reflects_the_live_session(live_server):
    server, base = live_server
    client = _Client(base)
    data = client.get("/api/state")
    entry = server.sessions[client.identity]
    assert data["human_seat"] == entry.session.human_seat
    assert data["phase"] == "SETUP_SETTLEMENT"
    assert data["round"] == 0


def test_a_legal_action_is_accepted_and_advances_the_game(live_server):
    server, base = live_server
    client = _Client(base)
    client.get("/api/state")  # establishes the identity/session first
    session = server.sessions[client.identity].session
    human_seat = session.human_seat
    options = legal_actions(session.game)
    assert to_move(session.game) == human_seat  # a fresh session guarantees this
    wire = action_to_wire(options[0])

    status, data = client.post("/api/action", wire)
    assert status == 200
    assert "error" not in data
    # A setup settlement is immediately followed by that same player's road.
    assert data["phase"] == "SETUP_ROAD"
    assert len(data["log"]) >= 1


def test_an_action_absent_from_legal_actions_is_rejected_over_http(live_server):
    _, base = live_server
    client = _Client(base)
    # ROLL is never legal during setup placement.
    wire = action_to_wire(Action(ActionType.ROLL))
    status, data = client.post("/api/action", wire)
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


def test_new_game_replaces_the_sessions_own_board(live_server):
    server, base = live_server
    client = _Client(base)
    client.get("/api/state")
    old_session = server.sessions[client.identity].session

    status, data = client.post("/api/new", {})

    assert status == 200
    assert data["round"] == 0
    assert data["phase"] == "SETUP_SETTLEMENT"
    # A genuinely new session object, not the same one mutated in place.
    new_session = server.sessions[client.identity].session
    assert new_session is not old_session
    # The cached /api/board layout was rebuilt from the new session's board,
    # not left describing the board the old session started with.
    assert client.get("/api/board") == board_layout(new_session.game.state.board)


def test_unknown_paths_404(live_server):
    _, base = live_server
    try:
        urllib.request.urlopen(base + "/api/nope", timeout=5)
        assert False, "expected an HTTPError"
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_two_browsers_get_two_independent_games(live_server):
    """The point of keying sessions off an identity cookie rather than one
    shared server-wide session: two different "browsers" (two cookie jars)
    must not see, or be able to affect, each other's game."""
    server, base = live_server
    alice = _Client(base)
    bob = _Client(base)

    alice.get("/api/state")
    bob.get("/api/state")

    assert alice.identity is not None
    assert bob.identity is not None
    assert alice.identity != bob.identity
    assert len(server.sessions) == 2

    alice_session = server.sessions[alice.identity].session
    bob_session = server.sessions[bob.identity].session
    assert alice_session is not bob_session

    # Alice acting must not touch Bob's game at all.
    options = legal_actions(alice_session.game)
    alice.post("/api/action", action_to_wire(options[0]))
    assert server.sessions[bob.identity].session is bob_session
    assert bob_session.game.turns == 0
    assert to_move(bob_session.game) == bob_session.human_seat


def test_an_idle_identity_is_evicted_and_gets_a_fresh_game(live_server):
    """`SESSION_TTL_SECONDS` is what actually gates this in production; the
    fixture's callable doesn't know about wall-clock time, so this fakes
    `last_seen` directly rather than sleeping for hours in a test."""
    server, base = live_server
    client = _Client(base)
    client.get("/api/state")
    entry = server.sessions[client.identity]
    entry.last_seen -= 999_999  # far enough in the past to count as stale

    client.get("/api/state")

    assert server.sessions[client.identity] is not entry


# --- Resuming a game a restart or an eviction interrupted ----------------------


def _drive(session, moves: int, rng: random.Random) -> None:
    """Play `moves` human turns against the bots, leaving it the human's move."""
    for _ in range(moves):
        if is_over(session.game):
            break
        session.apply_human_action(action_to_wire(rng.choice(legal_actions(session.game))))
        if session.awaiting_confirm:
            session.confirm_setup_turn()
        else:
            session.advance_bots()


def _press_new_game(session, identity: str) -> None:
    """Drive the real `CatanServer.replace` over an existing session — what
    `POST /api/new` reaches. Called rather than `journal.abandoned()` directly
    because the decision under test, whether that line gets written at all,
    lives in `replace` and not in the journal."""
    server = CatanServer(("127.0.0.1", 0), lambda bots, identity: _new_session(1))
    try:
        layout = board_layout(session.game.state.board)
        server.sessions[identity] = _Entry(session, layout, time.monotonic())
        server.replace(identity, None)
    finally:
        server.server_close()


def test_an_unfinished_game_comes_back_where_it_was_left(tmp_path):
    """The whole point of journalling every action: a session lives in memory,
    so a deploy or a crash used to take every game in flight with it."""
    bots = [("search2", "search2")] * 3
    session = _build_session(bots, 0, 99, "cpu", 1, str(tmp_path), "cookie-abc")
    _drive(session, 12, random.Random(4))
    assert not is_over(session.game)

    resumed = _resume_session("cookie-abc", "cpu", 1, str(tmp_path))

    assert resumed is not None
    assert resumed.game.phase is session.game.phase
    assert resumed.game.state.hands == session.game.state.hands
    assert resumed.game.state.vertex_owner == session.game.state.vertex_owner
    assert resumed.game.state.edge_owner == session.game.state.edge_owner
    assert resumed.game.state.deck == session.game.state.deck
    assert resumed.game.state.robber == session.game.state.robber
    assert resumed.game.turns == session.game.turns
    # Rebuilt by replaying, not stored: same actions in, same account out.
    assert resumed.log == session.log
    assert (resumed.seed, resumed.human_seat) == (session.seed, session.human_seat)
    assert resumed.bot_names == session.bot_names


def test_resuming_appends_to_the_same_file_rather_than_starting_another(tmp_path):
    bots = [("search2", "search2")] * 3
    session = _build_session(bots, 0, 99, "cpu", 1, str(tmp_path), "cookie-abc")
    _drive(session, 8, random.Random(4))
    before = session.journal.path

    resumed = _resume_session("cookie-abc", "cpu", 1, str(tmp_path))

    assert [p.name for p in tmp_path.glob("*.jsonl")] == [before.name]
    assert resumed.journal.path == before
    events = journal.read(before)
    # The seam is written down, and the steps carry on from where they stopped
    # rather than restarting at zero and colliding with the lines above.
    seam = [e for e in events if e["kind"] == "reopened"]
    assert len(seam) == 1
    assert seam[0]["at_step"] == len(journal.replayable(events))


def test_a_game_played_out_is_not_handed_back(tmp_path):
    """Only a game still in progress is waiting for anyone. A finished one is
    a result, and the next visit is a new game."""
    bots = [("search2", "search2")] * 3
    session = _build_session(bots, 0, 99, "cpu", 1, str(tmp_path), "cookie-abc")
    _drive(session, 6, random.Random(4))
    session.journal.finish(session.game)

    assert _resume_session("cookie-abc", "cpu", 1, str(tmp_path)) is None


def test_pressing_new_game_ends_the_old_one_for_good(tmp_path):
    """Otherwise the game they walked away from has no closing line of its
    own, and the next cache miss would hand it straight back."""
    bots = [("search2", "search2")] * 3
    session = _build_session(bots, 0, 99, "cpu", 1, str(tmp_path), "cookie-abc")
    _drive(session, 6, random.Random(4))
    assert journal.resumable(str(tmp_path), "cookie-abc") == session.journal.path

    _press_new_game(session, "cookie-abc")

    assert journal.resumable(str(tmp_path), "cookie-abc") is None


def test_a_game_that_was_won_is_not_also_filed_as_abandoned(tmp_path):
    """A won game already closed itself with `result`. Adding `abandoned`
    after it on the way out files a game the human played to the end as one
    they walked away from, and played-out-or-not is the single distinction
    anything counting these files exists to make."""
    bots = [("search2", "search2")] * 3
    session = _build_session(bots, 0, 99, "cpu", 1, str(tmp_path), "cookie-abc")
    rng = random.Random(4)
    while not is_over(session.game):
        _drive(session, 1, rng)
    path = session.journal.path

    _press_new_game(session, "cookie-abc")

    kinds = [e["kind"] for e in journal.read(path)]
    assert kinds.count("result") == 1
    assert "abandoned" not in kinds


def test_one_browsers_game_is_never_handed_to_another(tmp_path):
    bots = [("search2", "search2")] * 3
    mine = _build_session(bots, 0, 99, "cpu", 1, str(tmp_path), "cookie-abc")
    _drive(mine, 6, random.Random(4))

    assert _resume_session("cookie-xyz", "cpu", 1, str(tmp_path)) is None


def test_an_undone_placement_is_not_replayed_back_onto_the_board(tmp_path):
    """Undone actions stay in the file by design (see Journal.undo), so a
    resume that read the lines straight through would rebuild the board the
    human explicitly rejected."""
    bots = [("search2", "search2")] * 3
    session = _build_session(bots, 0, 99, "cpu", 1, str(tmp_path), "cookie-abc")
    rng = random.Random(4)
    for _ in range(400):
        _drive(session, 1, rng)
        if session._undo is not None or is_over(session.game):
            break
    assert session._undo is not None, "no undoable build came up to test with"
    session.undo_last_build()

    resumed = _resume_session("cookie-abc", "cpu", 1, str(tmp_path))

    assert resumed.game.state.vertex_owner == session.game.state.vertex_owner
    assert resumed.game.state.edge_owner == session.game.state.edge_owner
