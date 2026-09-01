"""Tables, seats, codes and tokens — `hexset_ui.api` without a socket.

Everything here calls `Tables.handle` the way `web.py` would, so the routing
and the rules are pinned together; `test_web.py` covers only what the HTTP
transport adds on top. `search2` is named explicitly at every call rather than
left to `default_lineup()`, which would seat whatever `.onnx` files happen to
be in `models/` and drag onnxruntime into a suite that has no need of it.
"""

from __future__ import annotations

import random

import pytest

from hexset_ui import journal
from hexset_ui.actions import ActionType, legal_actions
from hexset_ui.api import (
    CODE_ALPHABET,
    CODE_LENGTH,
    ApiError,
    Config,
    Seat,
    SeatKind,
    Tables,
    build_session,
    new_code,
    resume_session,
)
from hexset_ui.game import is_over, to_move
from hexset_ui.webplay import action_to_wire

SOLO = ["search2", "search2", "search2"]


def tables(**config) -> Tables:
    # games_dir="" rather than the default None: None means "wherever
    # HEXSET_UI_GAMES_DIR points", and a test suite should not journal into a
    # real player's games directory.
    config.setdefault("games_dir", "")
    return Tables(Config(**config))


def deal(registry: Tables, **kwargs) -> tuple[str, str]:
    """A new table, returned as (code, the creator's token)."""
    kwargs.setdefault("bots", SOLO[: 3 - kwargs.get("open_seats", 0)])
    data = registry.handle("POST", "/api/tables", kwargs, None)
    return data["code"], data["token"]


# --- Codes --------------------------------------------------------------------


def test_a_code_is_six_unambiguous_characters():
    code = new_code(set())
    assert len(code) == CODE_LENGTH
    assert set(code) <= set(CODE_ALPHABET)
    # The pairs that get misread aloud or retyped wrong are not in the alphabet
    # at all, so a code can never contain one.
    assert not set("01OIL") & set(CODE_ALPHABET)


def test_new_code_never_returns_one_already_in_use():
    taken = {new_code(set()) for _ in range(50)}
    assert new_code(taken) not in taken


def test_two_tables_get_two_codes():
    registry = tables()
    first, _ = deal(registry)
    second, _ = deal(registry)
    assert first != second


# --- Dealing a table ----------------------------------------------------------


def test_dealing_a_table_seats_the_caller_first_and_mints_them_a_token():
    registry = tables()
    data = registry.handle("POST", "/api/tables", {"bots": SOLO, "name": "Ada"}, None)

    assert data["token"]
    assert data["seat"] == 0
    assert data["started"] is False
    assert [s["kind"] for s in data["seats"]] == ["player", "bot", "bot", "bot"]
    assert data["seats"][0]["name"] == "Ada"
    # The token is the one secret here and belongs to the response that mints
    # it, never to the seat list everyone at the table can read.
    assert all("token" not in s for s in data["seats"])


def test_open_seats_are_left_empty_for_other_people():
    registry = tables()
    data = registry.handle("POST", "/api/tables", {"bots": ["search2"], "open_seats": 2}, None)
    assert [s["kind"] for s in data["seats"]] == ["player", "bot", "empty", "empty"]


def test_the_default_lineup_gives_way_to_the_seats_asked_to_be_kept_open():
    """A caller who names no bots is asking for a full table, not for three
    bots specifically — so open seats shrink the lineup rather than
    overflowing it. A caller who *names* three still gets an error (below)."""
    registry = tables(default_bots=SOLO)
    data = registry.handle("POST", "/api/tables", {"open_seats": 2}, None)
    assert [s["kind"] for s in data["seats"]] == ["player", "bot", "empty", "empty"]


def test_naming_more_bots_than_fit_is_refused():
    registry = tables()
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/tables", {"bots": SOLO, "open_seats": 1}, None)
    assert "at most" in str(caught.value)


def test_an_unknown_model_is_refused_by_name():
    """Clients send display names and never specs, so this is also what stops
    a request pointing a bot at a file of its choosing."""
    registry = tables()
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/tables", {"bots": ["../../etc/passwd"]}, None)
    assert caught.value.status == 400
    assert "unknown model" in str(caught.value)


def test_models_lists_what_the_bots_argument_accepts():
    registry = tables()
    assert "search2" in registry.handle("GET", "/api/models", {}, None)["models"]


# --- Joining ------------------------------------------------------------------


def test_joining_by_code_takes_the_first_empty_seat():
    registry = tables()
    code, _ = deal(registry, bots=["search2"], open_seats=2)

    data = registry.handle("POST", "/api/join", {"code": code, "name": "Bea"}, None)

    assert data["seat"] == 2
    assert data["token"]
    assert [s["kind"] for s in data["seats"]] == ["player", "bot", "player", "empty"]
    assert data["seats"][2]["name"] == "Bea"


def test_a_code_is_matched_case_insensitively():
    registry = tables()
    code, _ = deal(registry, bots=["search2"], open_seats=2)
    assert registry.handle("POST", "/api/join", {"code": code.lower()}, None)["seat"] == 2


def test_joining_a_table_with_no_empty_seat_is_refused():
    registry = tables()
    code, _ = deal(registry)
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/join", {"code": code}, None)
    assert caught.value.status == 409
    assert "no empty seats" in str(caught.value)


def test_joining_a_game_already_in_progress_is_refused():
    """Not a policy that could be relaxed: starting drops the empty seats, so
    a game in progress has only the seats it was dealt with."""
    registry = tables()
    code, token = deal(registry, bots=["search2"], open_seats=2)
    registry.handle("POST", "/api/join", {"code": code}, None)
    registry.handle("POST", "/api/start", {}, token)

    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/join", {"code": code}, None)
    assert caught.value.status == 409
    assert "already started" in str(caught.value)


def test_an_unknown_code_is_a_404():
    registry = tables()
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/join", {"code": "ZZZZZZ"}, None)
    assert caught.value.status == 404


def test_the_lobby_is_readable_without_a_token():
    """What a browser opening /<code> shows someone who has not joined yet."""
    registry = tables()
    code, _ = deal(registry, bots=["search2"], open_seats=2)

    data = registry.handle("GET", f"/api/table/{code}", {}, None)

    assert data["code"] == code
    assert data["seat"] is None
    assert data["can_start"] is True


# --- Tokens -------------------------------------------------------------------


def test_acting_without_a_token_is_a_401():
    registry = tables()
    with pytest.raises(ApiError) as caught:
        registry.handle("GET", "/api/state", {}, None)
    assert caught.value.status == 401


def test_an_unknown_token_is_a_403():
    registry = tables()
    deal(registry)
    with pytest.raises(ApiError) as caught:
        registry.handle("GET", "/api/state", {}, "not-a-token")
    assert caught.value.status == 403


def test_a_token_names_one_seat_at_one_table():
    registry = tables()
    _, mine = deal(registry)
    _, theirs = deal(registry)

    mine_state = registry.handle("GET", "/api/state", {}, mine)
    theirs_state = registry.handle("GET", "/api/state", {}, theirs)

    assert mine_state["code"] != theirs_state["code"]


def test_an_unknown_endpoint_is_a_404_even_with_a_good_token():
    registry = tables()
    _, token = deal(registry)
    with pytest.raises(ApiError) as caught:
        registry.handle("GET", "/api/nope", {}, token)
    assert caught.value.status == 404


# --- Starting -----------------------------------------------------------------


def test_starting_drops_the_empty_seats_rather_than_dealing_them_in():
    """The engine never learns what "empty" means: a four-seat table two
    people took is a two-player game."""
    registry = tables()
    code, token = deal(registry, bots=[], open_seats=3)
    registry.handle("POST", "/api/join", {"code": code}, None)

    data = registry.handle("POST", "/api/start", {}, token)

    assert data["started"] is True
    assert [s["kind"] for s in data["seats"]] == ["player", "player"]
    assert len(data["players"]) == 2
    assert data["phase"] == "SETUP_SETTLEMENT"


def test_starting_renumbers_the_seats_that_are_left():
    """A seat number after the deal is an engine seat and nothing else, so a
    request that arrived naming seat 2 has to be answered about seat 1."""
    registry = tables()
    code, token = deal(registry, bots=["search2"], open_seats=2)
    joiner = registry.handle("POST", "/api/join", {"code": code}, None)
    assert joiner["seat"] == 2

    data = registry.handle("POST", "/api/start", {}, joiner["token"])

    assert [s["kind"] for s in data["seats"]] == ["player", "bot", "player"]
    assert data["seat"] == 2  # unchanged here: the empty seat was after it


def test_a_seat_the_deal_renumbered_is_answered_about_its_new_number():
    registry = tables()
    code, token = deal(registry, bots=["search2"], open_seats=1)
    # Forced rather than dealt this way: `create` always appends its empty
    # seats last, so nothing a client can ask for produces this layout today.
    # `start` renumbers regardless, and this is what that has to mean.
    table = registry.get(code)
    table.seats = [Seat(), table.seats[1], table.seats[0]]

    data = registry.handle("POST", "/api/start", {}, token)

    assert [s["kind"] for s in data["seats"]] == ["bot", "player"]
    assert data["seat"] == 1


def test_a_table_that_cannot_field_two_players_will_not_start():
    registry = tables()
    _, token = deal(registry, bots=[], open_seats=3)
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/start", {}, token)
    assert caught.value.status == 409
    assert "at least 2" in str(caught.value)


def test_starting_twice_is_refused():
    registry = tables()
    _, token = deal(registry)
    registry.handle("POST", "/api/start", {}, token)
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/start", {}, token)
    assert caught.value.status == 409


def test_the_board_is_not_there_to_read_until_the_game_is_dealt():
    registry = tables()
    _, token = deal(registry)
    with pytest.raises(ApiError) as caught:
        registry.handle("GET", "/api/board", {}, token)
    assert caught.value.status == 409

    registry.handle("POST", "/api/start", {}, token)
    assert len(registry.handle("GET", "/api/board", {}, token)["hexes"]) == 19


def test_state_answers_from_the_lobby_before_the_deal_and_the_game_after():
    """One endpoint either way, so a client polls the same place from the
    moment it joins to the moment the game ends."""
    registry = tables()
    _, token = deal(registry)

    lobby = registry.handle("GET", "/api/state", {}, token)
    assert lobby["started"] is False
    assert "legal_actions" not in lobby

    registry.handle("POST", "/api/start", {}, token)
    playing = registry.handle("GET", "/api/state", {}, token)
    assert playing["started"] is True
    assert playing["legal_actions"]


# --- Playing ------------------------------------------------------------------


def test_a_legal_action_is_accepted_and_an_illegal_one_is_a_400():
    registry = tables()
    _, token = deal(registry)
    registry.handle("POST", "/api/start", {}, token)
    session = registry.get(registry.handle("GET", "/api/state", {}, token)["code"]).session

    settlement = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    )
    data = registry.handle("POST", "/api/action", {"action": settlement}, token)
    assert data["phase"] == "SETUP_ROAD"
    assert data["log"]

    # ROLL is never legal during setup placement.
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/action", {"action": {"type": "ROLL"}}, token)
    assert caught.value.status == 400


def test_another_seats_turn_is_not_this_ones_to_play():
    registry = tables()
    code, token = deal(registry, bots=[], open_seats=3)
    other = registry.handle("POST", "/api/join", {"code": code}, None)["token"]
    registry.handle("POST", "/api/start", {}, token)

    session = registry.get(code).session
    waiting = other if to_move(session.game) == 0 else token
    settlement = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    )
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/action", {"action": settlement}, waiting)
    assert caught.value.status == 400
    assert "not your turn" in str(caught.value)


def test_two_people_at_one_table_are_shown_two_different_states():
    registry = tables()
    code, mine = deal(registry, bots=["search2"], open_seats=2)
    theirs = registry.handle("POST", "/api/join", {"code": code}, None)["token"]
    registry.handle("POST", "/api/start", {}, mine)

    ours = registry.handle("GET", "/api/state", {}, mine)
    hers = registry.handle("GET", "/api/state", {}, theirs)

    assert ours["seat"] == 0 and hers["seat"] == 2
    assert ours["human_seats"] == hers["human_seats"] == [0, 2]
    # Each is shown their own hand and nobody else's.
    assert "hand" in {p["seat"]: p for p in ours["players"]}[0]
    assert "hand" not in {p["seat"]: p for p in ours["players"]}[2]
    assert "hand" in {p["seat"]: p for p in hers["players"]}[2]
    assert "hand" not in {p["seat"]: p for p in hers["players"]}[0]


def test_a_bot_can_be_swapped_but_a_persons_seat_cannot():
    registry = tables()
    _, token = deal(registry)
    registry.handle("POST", "/api/start", {}, token)

    data = registry.handle("POST", "/api/bot", {"seat": 1, "model": "search2"}, token)
    assert data["seats"][1]["name"] == "search2"

    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/bot", {"seat": 0, "model": "search2"}, token)
    assert "no bot to swap" in str(caught.value)


def test_renaming_a_seat_reaches_the_log_as_well_as_the_seat_list():
    registry = tables()
    _, token = deal(registry)
    registry.handle("POST", "/api/start", {}, token)

    data = registry.handle("POST", "/api/name", {"name": "Ada"}, token)

    assert data["seats"][0]["name"] == "Ada"
    assert {p["seat"]: p for p in data["players"]}[0]["name"] == "Ada"


# --- Resuming a game a restart interrupted -------------------------------------


def player(name: str | None = None) -> Seat:
    return Seat(kind=SeatKind.PLAYER, name=name, token="t-" + (name or "x"))


def bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="search2", spec="search2")


def drive(session, moves: int, rng: random.Random) -> None:
    """Play `moves` human turns against the bots, leaving it a human's move."""
    for _ in range(moves):
        if is_over(session.game):
            break
        seat = to_move(session.game)
        session.apply_human_action(seat, action_to_wire(rng.choice(legal_actions(session.game))))
        if session.awaiting_confirm is not None:
            session.confirm_setup_turn(seat)
        session.advance_bots()


def test_an_unfinished_game_comes_back_where_it_was_left(tmp_path):
    """The whole point of journalling every action: a session lives in memory,
    so a deploy or a crash used to take every game in flight with it."""
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config)
    drive(session, 12, random.Random(4))
    assert not is_over(session.game)

    resumed = resume_session("ABC123", seats, config)

    assert resumed is not None
    assert resumed.game.phase is session.game.phase
    assert resumed.game.state.hands == session.game.state.hands
    assert resumed.game.state.vertex_owner == session.game.state.vertex_owner
    assert resumed.game.state.edge_owner == session.game.state.edge_owner
    assert resumed.game.state.deck == session.game.state.deck
    assert resumed.game.state.robber == session.game.state.robber
    assert resumed.game.turns == session.game.turns
    # Rebuilt by replaying, not stored: same actions in, same account out.
    assert resumed.log_for(0) == session.log_for(0)
    assert (resumed.seed, resumed.human_seats) == (session.seed, session.human_seats)
    assert resumed.player_names == session.player_names


def test_resuming_appends_to_the_same_file_rather_than_starting_another(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config)
    drive(session, 8, random.Random(4))
    before = session.journal.path

    resumed = resume_session("ABC123", seats, config)

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
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config)
    drive(session, 6, random.Random(4))
    session.journal.finish(session.game)

    assert resume_session("ABC123", seats, config) is None


def test_one_tables_game_is_never_handed_to_another(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    drive(build_session("ABC123", seats, config), 6, random.Random(4))

    assert resume_session("XYZ789", seats, config) is None


def test_an_undone_placement_is_not_replayed_back_onto_the_board(tmp_path):
    """Undone actions stay in the file by design (see Journal.undo), so a
    resume that read the lines straight through would rebuild the board the
    human explicitly rejected."""
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config)
    rng = random.Random(4)
    for _ in range(400):
        drive(session, 1, rng)
        if session._undo is not None or is_over(session.game):
            break
    assert session._undo is not None, "no undoable build came up to test with"
    session.undo_last_build(session._undo.actor)

    resumed = resume_session("ABC123", seats, config)

    assert resumed.game.state.vertex_owner == session.game.state.vertex_owner
    assert resumed.game.state.edge_owner == session.game.state.edge_owner


def test_a_resumed_game_keeps_the_seats_it_was_dealt_with(tmp_path):
    """A table dealt with empty seats deals a game with only the occupied
    ones, so the header's seat numbering is the engine's and resuming must not
    reintroduce the seats the deal dropped."""
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), player("Bea")]
    session = build_session("ABC123", seats, config)
    drive(session, 4, random.Random(4))

    resumed = resume_session("ABC123", seats, config)

    assert resumed.game.state.num_players == 2
    assert resumed.human_seats == frozenset({0, 1})
