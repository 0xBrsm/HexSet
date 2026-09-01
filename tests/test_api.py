"""Games, seats, codes and tokens — `hexset_ui.api` without a socket.

Everything here calls `Tables.handle` the way `web.py` would, so the routing
and the rules are pinned together; `test_web.py` covers only what the HTTP
transport adds on top. `search2` is named explicitly at every call rather than
left to `Config.default_bots`, which would seat whatever `.onnx` files happen
to be in `models/` and drag onnxruntime into a suite that has no need of it.
"""

from __future__ import annotations

import random
import time

import pytest

from hexset_ui import journal
from hexset_ui.actions import ActionType, legal_actions
from hexset_ui.api import (
    CODE_ALPHABET,
    CODE_LENGTH,
    MAX_SEATS,
    ApiError,
    Config,
    Seat,
    SeatKind,
    Tables,
    build_session,
    new_code,
    resume_session,
)
from hexset_ui.game import is_over, lock_seat, to_move
from hexset_ui.webplay import action_to_wire

SOLO = ["search2", "search2", "search2"]


def tables(**config) -> Tables:
    # games_dir="" rather than the default None: None means "wherever
    # HEXSET_UI_GAMES_DIR points", and a test suite should not journal into a
    # real player's games directory.
    config.setdefault("games_dir", "")
    config.setdefault("seat_grace", 0.0)  # deterministic locking by default
    return Tables(Config(**config))


def deal(registry: Tables, **kwargs) -> tuple[str, str]:
    """A new game, returned as (code, the creator's token)."""
    kwargs.setdefault("bots", SOLO)
    data = registry.handle("POST", "/api/games", kwargs, None)
    return data["code"], data["token"]


def empty_seats(table) -> list[int]:
    return [i for i, s in enumerate(table.seats) if s.kind is SeatKind.EMPTY]


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


def test_two_games_get_two_codes():
    registry = tables()
    first, _ = deal(registry)
    second, _ = deal(registry)
    assert first != second


# --- Dealing a game -------------------------------------------------------------


def test_dealing_seats_the_creator_at_a_random_seat_and_mints_a_token():
    registry = tables()
    data = registry.handle("POST", "/api/games", {"bots": SOLO, "name": "Ada"}, None)

    assert data["token"]
    assert 0 <= data["seat"] < MAX_SEATS
    assert len(data["seats"]) == MAX_SEATS
    kinds = [s["kind"] for s in data["seats"]]
    assert kinds.count("player") == 1
    assert kinds.count("bot") == 3
    assert data["seats"][data["seat"]]["kind"] == "player"
    assert data["seats"][data["seat"]]["name"] == "Ada"
    # The token is the one secret here and belongs to the response that mints
    # it, never to the seat list everyone at the table can read.
    assert all("token" not in s for s in data["seats"])
    # Playable immediately — no lobby, no separate start.
    assert data["phase"] == "SETUP_SETTLEMENT"
    assert data["to_move"] is not None


def test_seats_nobody_named_are_left_open():
    registry = tables()
    data = registry.handle("POST", "/api/games", {"bots": ["search2"]}, None)
    kinds = [s["kind"] for s in data["seats"]]
    assert kinds.count("player") == 1
    assert kinds.count("bot") == 1
    assert kinds.count("empty") == 2


def test_omitting_bots_seats_nobody_but_the_creator():
    """No automatic mixed lineup any more — filling the table is an explicit
    choice (see Config.default_bots's own docstring)."""
    registry = tables()
    data = registry.handle("POST", "/api/games", {}, None)
    kinds = [s["kind"] for s in data["seats"]]
    assert kinds.count("player") == 1
    assert kinds.count("empty") == MAX_SEATS - 1


def test_config_default_bots_fills_in_for_an_unnamed_lineup():
    """--checkpoint's own mechanism: a caller that names no bots at all
    falls back to the server's pinned default, not to an empty table."""
    registry = tables(default_bots=SOLO)
    data = registry.handle("POST", "/api/games", {}, None)
    kinds = [s["kind"] for s in data["seats"]]
    assert kinds.count("bot") == 3


def test_naming_more_bots_than_fit_is_refused():
    registry = tables()
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/games", {"bots": SOLO + ["search2"]}, None)
    assert "at most" in str(caught.value)


def test_an_unknown_model_is_refused_by_name():
    """Clients send display names and never specs, so this is also what stops
    a request pointing a bot at a file of its choosing."""
    registry = tables()
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/games", {"bots": ["../../etc/passwd"]}, None)
    assert caught.value.status == 400
    assert "unknown model" in str(caught.value)


def test_models_lists_what_the_bots_argument_accepts():
    registry = tables()
    assert "search2" in registry.handle("GET", "/api/models", {}, None)["models"]


# --- Joining ------------------------------------------------------------------


def test_joining_by_code_takes_a_random_open_seat():
    registry = tables()
    code, _ = deal(registry, bots=["search2"])
    table = registry.get(code)
    open_before = set(empty_seats(table))
    assert len(open_before) == 2

    data = registry.handle("POST", "/api/join", {"code": code, "name": "Bea"}, None)

    assert data["seat"] in open_before
    assert data["token"]
    assert data["seats"][data["seat"]]["kind"] == "player"
    assert data["seats"][data["seat"]]["name"] == "Bea"


def test_a_code_is_matched_case_insensitively():
    registry = tables()
    code, _ = deal(registry, bots=["search2"])
    data = registry.handle("POST", "/api/join", {"code": code.lower()}, None)
    assert 0 <= data["seat"] < MAX_SEATS


def test_joining_a_full_game_is_refused():
    registry = tables()
    code, _ = deal(registry, bots=SOLO)
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/join", {"code": code}, None)
    assert caught.value.status == 409
    assert "no open seats" in str(caught.value)


def test_join_never_offers_a_locked_seat():
    registry = tables()
    code, _ = deal(registry, bots=[])
    table = registry.get(code)
    open_seats = empty_seats(table)
    assert len(open_seats) == 3
    lock_seat(table.session.game, open_seats[0])
    lock_seat(table.session.game, open_seats[1])

    seat, _ = table.join(None)

    assert seat == open_seats[2]


def test_joining_a_game_where_every_open_seat_is_locked_is_refused():
    registry = tables()
    code, _ = deal(registry, bots=[])
    table = registry.get(code)
    for seat in empty_seats(table):
        lock_seat(table.session.game, seat)

    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/join", {"code": code}, None)
    assert caught.value.status == 409


def test_an_unknown_code_is_a_404():
    registry = tables()
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/join", {"code": "ZZZZZZ"}, None)
    assert caught.value.status == 404


def test_an_observer_can_read_a_game_without_a_token():
    """What a browser opening /<code> shows someone who hasn't (or can't)
    join it — a full state view, not a lobby-only shape, since there is no
    lobby any more."""
    registry = tables()
    code, _ = deal(registry, bots=["search2"])

    data = registry.handle("GET", f"/api/table/{code}", {}, None)

    assert data["code"] == code
    assert data["seat"] is None
    assert data["phase"]
    assert data["legal_actions"] == []  # nothing is an observer's to play


# --- The per-seat setup lock ---------------------------------------------------


def test_a_creator_seated_anywhere_but_the_snakes_first_slot_still_plays():
    """`first=` follows the creator's own (random) seat, not a hardcoded 0 —
    otherwise a creator seated anywhere else would find it seat 0's turn,
    seat 0 empty, and nothing able to ever advance."""
    registry = tables()
    code, token = deal(registry, bots=[])
    session = registry.get(code).session
    creator_seat = registry.handle("GET", "/api/state", {}, token)["seat"]

    assert to_move(session.game) == creator_seat
    options = legal_actions(session.game)
    assert options  # the creator really can act on their very first request


def test_settle_locks_needs_a_second_touch_before_it_locks():
    """The first time _settle_locks finds the snake waiting on an empty seat
    it only starts the window; a lock fires only once seat_grace has
    actually elapsed since then (see Table._settle_locks's own docstring)."""
    registry = tables(seat_grace=0.0)
    code, token = deal(registry, bots=[])
    table = registry.get(code)
    session = table.session
    creator_seat = to_move(session.game)

    settlement = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    )
    registry.handle("POST", "/api/action", {"action": settlement}, token)
    road = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_ROAD)
    )
    registry.handle("POST", "/api/action", {"action": road}, token)

    next_seat = to_move(session.game)
    assert next_seat != creator_seat
    assert next_seat not in session.game.locked  # the first touch only starts the window

    table._settle_locks(now=time.monotonic() + 1)  # a second touch, past the (zero) grace

    assert next_seat in session.game.locked


def test_a_locked_seat_stays_locked_but_the_rest_of_the_game_still_plays():
    registry = tables()
    code, token = deal(registry, bots=[])
    table = registry.get(code)
    session = table.session
    creator_seat = to_move(session.game)
    for seat in empty_seats(table):
        if seat != creator_seat:
            lock_seat(session.game, seat)

    data = registry.handle("GET", "/api/state", {}, token)
    assert set(data["locked"]) == set(empty_seats(table))
    assert data["to_move"] == creator_seat  # every other seat locked; back to the creator
    assert data["legal_actions"]  # and it's genuinely playable


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


def test_a_token_names_one_seat_at_one_game():
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


# --- Reading the board ----------------------------------------------------------


def test_the_board_is_readable_from_the_moment_the_game_exists():
    registry = tables()
    _, token = deal(registry)
    assert len(registry.handle("GET", "/api/board", {}, token)["hexes"]) == 19


def test_state_is_the_same_shape_from_the_first_request_onward():
    """One endpoint, one shape, from the moment a game exists to the moment
    it ends — there is no separate lobby shape to switch out of."""
    registry = tables()
    _, token = deal(registry)
    data = registry.handle("GET", "/api/state", {}, token)
    assert data["phase"] == "SETUP_SETTLEMENT"
    assert data["legal_actions"]


# --- Playing ------------------------------------------------------------------


def test_a_legal_action_is_accepted_and_an_illegal_one_is_a_400():
    registry = tables()
    _, token = deal(registry)
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
    code, token = deal(registry, bots=[])
    other = registry.handle("POST", "/api/join", {"code": code}, None)["token"]

    session = registry.get(code).session
    waiting = other if to_move(session.game) == registry.by_token(token)[1] else token
    settlement = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    )
    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/action", {"action": settlement}, waiting)
    assert caught.value.status == 400
    assert "not your turn" in str(caught.value)


def test_two_people_at_one_game_are_shown_two_different_states():
    registry = tables()
    code, mine = deal(registry, bots=[])
    theirs = registry.handle("POST", "/api/join", {"code": code}, None)["token"]

    mine_seat = registry.by_token(mine)[1]
    theirs_seat = registry.by_token(theirs)[1]
    ours = registry.handle("GET", "/api/state", {}, mine)
    hers = registry.handle("GET", "/api/state", {}, theirs)

    assert ours["seat"] == mine_seat and hers["seat"] == theirs_seat
    assert set(ours["claimed_seats"]) == set(hers["claimed_seats"]) == {mine_seat, theirs_seat}
    # Each is shown their own hand and nobody else's.
    assert "hand" in {p["seat"]: p for p in ours["players"]}[mine_seat]
    assert "hand" not in {p["seat"]: p for p in ours["players"]}[theirs_seat]
    assert "hand" in {p["seat"]: p for p in hers["players"]}[theirs_seat]
    assert "hand" not in {p["seat"]: p for p in hers["players"]}[mine_seat]


def test_a_bot_can_be_swapped_but_a_persons_seat_cannot():
    registry = tables()
    code, token = deal(registry, bots=["search2"])
    bot_seat = next(
        i for i, s in enumerate(registry.get(code).seats) if s.kind is SeatKind.BOT
    )
    mine_seat = registry.by_token(token)[1]

    data = registry.handle("POST", "/api/bot", {"seat": bot_seat, "model": "search2"}, token)
    assert data["seats"][bot_seat]["name"] == "search2"

    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/bot", {"seat": mine_seat, "model": "search2"}, token)
    assert "no bot to swap" in str(caught.value)


def test_renaming_a_seat_reaches_the_log_as_well_as_the_seat_list():
    registry = tables()
    _, token = deal(registry)
    mine_seat = registry.by_token(token)[1]

    data = registry.handle("POST", "/api/name", {"name": "Ada"}, token)

    assert data["seats"][mine_seat]["name"] == "Ada"
    assert {p["seat"]: p for p in data["players"]}[mine_seat]["name"] == "Ada"


def test_record_matches_the_seat_on_move():
    """A freshly-dealt game's setup snake starts at `first` — the creator's
    own (random) seat, see `hexset_ui.game.start` — so the creator's own
    token is always the mover's here."""
    registry = tables()
    code, token = deal(registry, bots=[])
    mover_seat = to_move(registry.get(code).session.game)
    assert registry.by_token(token)[1] == mover_seat

    data = registry.handle("GET", "/api/record", {}, token)
    assert data["perspective"] == mover_seat
    assert "action_mask" in data and "ledger_known" in data
    assert "options" in data and "offers_made" in data and "space" in data


def test_record_is_refused_when_it_is_not_your_turn():
    registry = tables()
    code, token = deal(registry, bots=[])
    other = registry.handle("POST", "/api/join", {"code": code}, None)["token"]
    session = registry.get(code).session
    mover = to_move(session.game)
    waiting = other if registry.by_token(token)[1] == mover else token

    with pytest.raises(ApiError) as caught:
        registry.handle("GET", "/api/record", {}, waiting)
    assert caught.value.status == 409


# --- A restart: the registry loses a game, its journal doesn't -----------------


def test_a_registry_miss_reopens_bot_seats_fresh_but_opens_the_humans(tmp_path):
    """A lost in-memory table is rebuilt from its journal: a bot seat's
    identity is just its spec, so it's re-tokened and reclaimed
    automatically; a human's old token cannot be recovered (it never
    touched disk — see api.py's module docstring), so their seat is simply
    open again, exactly like one nobody ever claimed."""
    registry = tables(games_dir=str(tmp_path))
    code, token = deal(registry, bots=SOLO)  # every seat filled, none open
    creator_seat = registry.by_token(token)[1]

    del registry._tables[code]  # simulate a restart

    reopened = registry.get(code)
    kinds = {i: s.kind for i, s in enumerate(reopened.seats)}
    assert kinds[creator_seat] is SeatKind.EMPTY
    for seat in range(MAX_SEATS):
        if seat != creator_seat:
            assert kinds[seat] is SeatKind.BOT
            assert reopened.seats[seat].token is not None
    assert reopened.session.claimed_seats == {s for s in range(MAX_SEATS) if s != creator_seat}
    # And the reopened game is still genuinely playable.
    assert reopened.session.state_view(None)["phase"]


def test_a_registry_miss_with_no_journal_is_still_just_a_404(tmp_path):
    registry = tables(games_dir=str(tmp_path))
    with pytest.raises(ApiError) as caught:
        registry.get("ZZZZZZ")
    assert caught.value.status == 404


# --- resume_session / build_session, one layer below Tables --------------------


def player(name: str | None = None) -> Seat:
    return Seat(kind=SeatKind.PLAYER, name=name, token="t-" + (name or "x"))


def bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="search2", spec="search2")


def drive(session, moves: int, rng: random.Random) -> None:
    """Play `moves` actions total, whoever's seat is up — there is no
    separate "human" driving here any more, every claimed seat submits the
    same way (see webplay.GameSession.submit)."""
    for _ in range(moves):
        if is_over(session.game):
            break
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))


def test_an_unfinished_game_comes_back_where_it_was_left(tmp_path):
    """The whole point of journalling every action: a session lives in memory,
    so a deploy or a crash used to take every game in flight with it."""
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
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
    assert (resumed.seed, resumed.claimed_seats) == (session.seed, session.claimed_seats)
    assert resumed.player_names == session.player_names


def test_resuming_appends_to_the_same_file_rather_than_starting_another(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
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
    session = build_session("ABC123", seats, config, first=0)
    drive(session, 6, random.Random(4))
    session.journal.finish(session.game)

    assert resume_session("ABC123", seats, config) is None


def test_one_games_journal_is_never_handed_to_another(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    drive(build_session("ABC123", seats, config, first=0), 6, random.Random(4))

    assert resume_session("XYZ789", seats, config) is None


def test_an_undone_placement_is_not_replayed_back_onto_the_board(tmp_path):
    """Undone actions stay in the file by design (see Journal.undo), so a
    resume that read the lines straight through would rebuild the board the
    player explicitly rejected."""
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
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


def test_a_resumed_game_always_deals_max_seats(tmp_path):
    """Every game deals MAX_SEATS from the start now, whether or not every
    seat is claimed (see api.py's module docstring) — resuming must not
    reintroduce the old "renumber down to just the occupied seats"
    behaviour, or a checkpoint trained for MAX_SEATS players would find a
    resumed 2-person game unplayable (see onnxbot.py's _check_players)."""
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), player("Bea"), Seat(), Seat()]
    session = build_session("ABC123", seats, config, first=0)
    drive(session, 4, random.Random(4))

    resumed = resume_session("ABC123", seats, config)

    assert resumed.game.state.num_players == MAX_SEATS
    assert resumed.claimed_seats == {0, 1}


def test_the_snake_starts_where_first_says_not_always_seat_zero(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [Seat(), Seat(), player("Ada"), bot_seat()]
    session = build_session("ABC123", seats, config, first=2)
    assert session.game.setup_queue[0] == 2
    drive(session, 2, random.Random(4))

    resumed = resume_session("ABC123", seats, config)

    assert resumed.game.setup_queue[0] == 2
