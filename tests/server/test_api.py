"""Games, seats, codes and tokens — `hexset.server.api` without a socket.

Everything here calls `Tables.handle` the way `web.py` would, so the routing
and the rules are pinned together; `test_web.py` covers only what the HTTP
transport adds on top. `search2` is named explicitly at every call rather than
left to `Config.default_bots`, which would seat whatever `.onnx` files happen
to be in `models/` and drag onnxruntime into a suite that has no need of it.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from hexset.server import journal
from hexset.actions import ActionType, legal_actions
from hexset.server.api import (
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
from hexset.game import is_over, to_move
from hexset.server.seating import lock_seat
from hexset.server.webplay import action_to_wire
from conftest import new_tables

SOLO = ["search2", "search2", "search2"]


def tables(**config) -> Tables:
    """`conftest.new_tables`: a registry whose bot runner threads are stopped
    when the test ends (see that fixture for why a test may not just build
    one)."""
    return new_tables(**config)


def deal(registry: Tables, **kwargs) -> tuple[str, str]:
    """A new game, returned as (code, the creator's token)."""
    kwargs.setdefault("bots", SOLO)
    data = registry.handle("POST", "/api/games", kwargs, None)
    return data["code"], data["token"]


def empty_seats(table) -> list[int]:
    return [i for i, s in enumerate(table.seats) if s.kind is SeatKind.EMPTY]


# --- Codes --------------------------------------------------------------------


def test_a_code_is_six_unambiguous_lowercase_characters():
    code = new_code(set())
    assert len(code) == CODE_LENGTH
    assert set(code) <= set(CODE_ALPHABET)
    # A code is only ever seen as a URL, so it is lowercase throughout.
    assert code == code.lower()
    # The pairs that get misread aloud or retyped wrong are not in the alphabet
    # at all, so a code can never contain one — in either case.
    assert not set("01oil") & set(CODE_ALPHABET)
    assert not set("01OIL") & set(CODE_ALPHABET.upper())


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
    """Codes are minted lowercase, so the case that has to keep working is
    the one somebody's phone capitalised on the way into the address bar."""
    registry = tables()
    code, _ = deal(registry, bots=["search2"])
    assert code == code.lower()
    data = registry.handle("POST", "/api/join", {"code": code.upper()}, None)
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
    # Nobody's hand, not even a bot's: `reveal` is the viewer's own seat or a
    # finished game, and an observer is neither.
    assert not any("hand" in p for p in data["players"])


def test_an_observer_can_read_the_board_without_a_token():
    """The layout a spectator's page is drawn on. Every game is public, so
    the board behind one is too — and it is the same board the seats get,
    since a board is public knowledge at the table anyway."""
    registry = tables()
    code, token = deal(registry, bots=["search2"])

    public = registry.handle("GET", f"/api/table/{code}/board", {}, None)

    assert public == registry.handle("GET", "/api/board", {}, token)
    assert public["hexes"] and public["vertices"] and public["ports"]


def test_the_public_routes_refuse_anything_but_a_game_and_its_board():
    registry = tables()
    code, _ = deal(registry)

    with pytest.raises(ApiError) as caught:
        registry.handle("GET", f"/api/table/{code}/record", {}, None)
    assert caught.value.status == 404


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


def test_an_empty_seat_the_snake_reaches_is_retired_on_sight():
    """No waiting window: the creator holds the table simply by not
    finishing their own placement, so the moment they do finish, an empty
    seat next in the snake is out (see Table._settle_locks's docstring)."""
    registry = tables()
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
    data = registry.handle("POST", "/api/action", {"action": road}, token)

    # Every other seat was empty, so finishing the first placement retires
    # all three at once and hands the table straight back to the creator.
    assert set(data["locked"]) == {s for s in range(MAX_SEATS) if s != creator_seat}
    assert to_move(session.game) == creator_seat


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
    # Answered as the seat that asked, never as the seat that was touched:
    # every response here is built for one viewer, so answering as the bot
    # would hand its whole hand back to whoever changed its picker.
    assert data["seat"] == mine_seat
    revealed = {p["seat"] for p in data["players"] if "hand" in p}
    assert revealed == {mine_seat}

    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/bot", {"seat": mine_seat, "model": "search2"}, token)
    assert "belongs to a player" in str(caught.value)


def test_an_open_seat_can_be_given_a_bot_from_the_table():
    """The lobby is gone, so the player list on the board is where a table
    decides who else is playing: the same request that swaps one bot for
    another fills a seat nobody has taken."""
    registry = tables()
    code, token = deal(registry, bots=[])
    table = registry.get(code)
    mine_seat = registry.by_token(token)[1]
    open_seat = next(i for i, s in enumerate(table.seats) if s.kind is SeatKind.EMPTY)

    data = registry.handle("POST", "/api/bot", {"seat": open_seat, "model": "search2"}, token)

    assert data["seats"][open_seat]["kind"] == "bot"
    assert data["seats"][open_seat]["name"] == "search2"
    # Claimed as far as the session is concerned, or the runner that was just
    # started could not play the seat it was given.
    assert set(data["claimed_seats"]) == {mine_seat, open_seat}
    assert data["seat"] == mine_seat
    assert table.seats[open_seat].token is not None
    assert any(runner.seat == open_seat for runner, _ in table.runners)


def test_a_retired_seat_cannot_be_given_a_bot():
    """A seat the setup snake waited out is out of the game for good — see
    `hexset.server.seating.lock_seat`. Nothing revives it, a bot included."""
    registry = tables()
    code, token = deal(registry, bots=[])
    table = registry.get(code)
    # Retired here rather than waited out — the grace window that does it for
    # real has its own test above.
    retired = empty_seats(table)[0]
    lock_seat(table.session.game, retired)

    with pytest.raises(ApiError) as caught:
        registry.handle("POST", "/api/bot", {"seat": retired, "model": "search2"}, token)
    assert "retired" in str(caught.value)


def test_renaming_a_seat_reaches_the_log_as_well_as_the_seat_list():
    registry = tables()
    _, token = deal(registry)
    mine_seat = registry.by_token(token)[1]

    data = registry.handle("POST", "/api/name", {"name": "Ada"}, token)

    assert data["seats"][mine_seat]["name"] == "Ada"
    assert {p["seat"]: p for p in data["players"]}[mine_seat]["name"] == "Ada"


def test_record_matches_the_seat_on_move():
    """A freshly-dealt game's setup snake starts at `first` — the creator's
    own (random) seat, see `hexset.server.seating.start_at` — so the creator's own
    token is always the mover's here."""
    registry = tables()
    code, token = deal(registry, bots=[])
    mover_seat = to_move(registry.get(code).session.game)
    assert registry.by_token(token)[1] == mover_seat

    data = registry.handle("GET", "/api/record", {}, token)
    assert data["perspective"] == mover_seat
    assert "action_mask" in data and "ledger_known" in data
    assert "options" in data and "space" in data
    assert "valuations" in data


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

    # Simulate a restart: the process going away takes its runner threads
    # with it. `stop_runners` rather than `close`, because a restart does not
    # mark the journal abandoned -- that is exactly what leaves the game
    # resumable, which is what this test is about. Dropping the entry alone
    # would leave three bots polling a table nothing can reach.
    registry._tables.pop(code).stop_runners()

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
    assert resumed.game._state.hands == session.game._state.hands
    assert resumed.game._state.vertex_owner == session.game._state.vertex_owner
    assert resumed.game._state.edge_owner == session.game._state.edge_owner
    assert resumed.game._state.deck == session.game._state.deck
    assert resumed.game._state.robber == session.game._state.robber
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


def test_a_game_journalled_under_a_capitalised_code_still_resumes(tmp_path):
    """Codes used to be minted in capitals. A game journalled back then is
    addressed by the lowercase code now, and `journal.resumable` matching
    without regard to case is what keeps it findable."""
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
    drive(session, 8, random.Random(4))

    assert journal.resumable(str(tmp_path), "abc123") == session.journal.path
    assert journal.resumable(str(tmp_path), "ABC123") == session.journal.path
    assert journal.resumable(str(tmp_path), "abc124") is None


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

    assert resumed.game._state.vertex_owner == session.game._state.vertex_owner
    assert resumed.game._state.edge_owner == session.game._state.edge_owner


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

    assert resumed.game._state.num_players == MAX_SEATS
    assert resumed.claimed_seats == {0, 1}


def test_the_snake_starts_where_first_says_not_always_seat_zero(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [Seat(), Seat(), player("Ada"), bot_seat()]
    session = build_session("ABC123", seats, config, first=2)
    assert session.game.setup_queue[0] == 2
    drive(session, 2, random.Random(4))

    resumed = resume_session("ABC123", seats, config)

    assert resumed.game.setup_queue[0] == 2


# --- One mask for every seat --------------------------------------------------

# There used to be two masks. The engine's own `legal_actions` filtered the
# `PROPOSE_TRADE` sample by opponents' true hands, so a served table had to
# build a second, honest one (`rules.fair_legal_actions`) -- and PR #2 defect
# 4 was an embedded bot searching the first while every other client got the
# second, which meant the same checkpoint played a different game depending
# on how it had been seated. Trading is no longer an action
# (`hexset.trading`), so no remaining action's legality depends on another
# seat's hand, there is one list, and the tests below pin that rather than
# the agreement of two.


def _a_position_where_no_opponent_holds_anything(mover: int = 0):
    """A `Game` in MAIN where the mover holds every resource and nobody else
    holds any -- the position that used to separate the two samples."""
    import random as _random

    from hexset.board.board import random_base_board
    from hexset.board.terrain import NUM_RESOURCES
    from hexset.game import Phase
    from hexset.server.seating import start_at

    game = start_at(random_base_board(_random.Random(0)), 4, _random.Random(1), first=0)
    game.phase = Phase.MAIN
    game.current_player = mover
    for hand in game._state.hands:
        hand[:] = [0] * NUM_RESOURCES
    game._state.hands[mover] = [1, 1, 1, 1, 1]
    return game


def test_an_embedded_bot_is_offered_the_same_list_the_wire_serves():
    from hexset.actions import legal_actions
    from hexset.clients.onnxbot import options_for as onnxbot_options_for

    game = _a_position_where_no_opponent_holds_anything()
    assert onnxbot_options_for(game) == legal_actions(game)


def test_record_matches_the_embedded_bots_options():
    """The claim at the level it was actually made, through the real route:
    the record `GET /api/record` serves and the record an in-process bot
    builds for itself (`onnxbot.V2Policy._run`) must agree field for field."""
    import numpy as np

    from hexset.actions import build_space

    from hexset.onnx_record import record_from_game
    from hexset.server.rules import options_for

    registry = tables()
    code, token = deal(registry, bots=[])
    table = registry.get(code)
    seat = registry.by_token(token)[1]

    table.session.game = _a_position_where_no_opponent_holds_anything(mover=seat)
    served = registry.record(table, seat)

    game = table.session.game
    topology = game._state.board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, game._state.num_players
    )
    in_process = record_from_game(game, seat, space, tuple(options_for(game)))

    for key, value in in_process.items():
        assert np.array_equal(np.asarray(served[key]), value), key


def test_the_option_list_does_not_move_when_opponents_hands_do():
    """The property the second enumeration existed to guarantee, asserted
    directly: nothing the mover may do depends on what anybody else holds."""
    from hexset.actions import legal_actions
    from hexset.board.terrain import NUM_RESOURCES

    game = _a_position_where_no_opponent_holds_anything()
    before = legal_actions(game)
    for seat in range(1, 4):
        game._state.hands[seat] = [2] * NUM_RESOURCES
    assert legal_actions(game) == before


# --- Runner lifecycle ---------------------------------------------------------


def test_a_bot_poll_does_not_keep_a_table_alive():
    """PR #2 defect 5. `by_token` refreshed `last_seen` for every request,
    including an embedded runner's once-a-second poll, and a runner lives
    until the game is over -- so `now - last_seen` could never reach
    `TABLE_TTL_SECONDS` while a bot sat at the table. A human who dealt
    against three bots and closed the tab left three threads polling a game
    that could never be evicted and a journal handle that was never closed.

    Liveness means a person or an external client is still here, which is
    exactly the seats that are not this server's own bots.
    """
    registry = tables()
    code, token = deal(registry, bots=SOLO)
    table = registry.get(code)
    bot_seat_index = next(
        i for i, s in enumerate(table.seats) if s.kind is SeatKind.BOT
    )
    bot_token = table.seats[bot_seat_index].token

    table.last_seen = 0.0
    registry.by_token(bot_token)
    assert table.last_seen == 0.0, "a bot's own poll refreshed the table"

    registry.by_token(token)
    assert table.last_seen > 0.0, "a person's request must refresh the table"


def test_a_table_only_bots_are_watching_is_evicted():
    """The consequence of the above, end to end: an abandoned bot game goes
    on somebody else's next request, runners and journal with it. PR #2 only
    ran eviction inside `create`, so a box that dealt one game and was then
    only read never reaped anything.

    The reaping request is a lookup of a *different* game, because `get`
    spares the code it was asked for (`keep`) -- a request is itself the
    liveness signal for the table it names.
    """
    import hexset.server.api as api

    registry = tables()
    abandoned, _ = deal(registry, bots=SOLO)
    watched, _ = deal(registry, bots=[])
    assert [t for t in threading.enumerate() if t.name.startswith(f"bot-{abandoned}")]

    original = api.TABLE_TTL_SECONDS
    api.TABLE_TTL_SECONDS = -1.0
    try:
        registry.get(watched)  # somebody else's poll does the reaping
    finally:
        api.TABLE_TTL_SECONDS = original

    assert abandoned not in registry._tables
    assert not [t for t in threading.enumerate() if t.name.startswith(f"bot-{abandoned}")]
    assert watched in registry._tables


def test_the_code_being_looked_up_is_never_evicted_out_from_under_the_request():
    """`get` runs eviction before the lookup, so without `keep` a request for
    a game that had just gone stale would 404 on the very table it came
    for -- a reopen from the journal at best, an error at worst."""
    import hexset.server.api as api

    registry = tables()
    code, _ = deal(registry, bots=[])
    registry.get(code).last_seen = 0.0

    original = api.TABLE_TTL_SECONDS
    api.TABLE_TTL_SECONDS = -1.0
    try:
        assert registry.get(code).code == code
    finally:
        api.TABLE_TTL_SECONDS = original


def test_closing_a_registry_stops_every_runner():
    """What the test suite's own fixture leans on, asserted directly."""
    registry = tables()
    code, _ = deal(registry, bots=SOLO)
    assert [t for t in threading.enumerate() if t.name.startswith(f"bot-{code}")]

    registry.close()

    assert not [t for t in threading.enumerate() if t.name.startswith(f"bot-{code}")]
    assert not registry._tables


# --- Trading (`hexset.trading`) -----------------------------------------------


def test_a_seat_publishes_its_valuation_and_every_viewer_sees_it():
    """`PUT /api/games/<CODE>/valuation` is the whole of the trading API
    surface: a seat sets its own vector, and the vector is public."""
    registry = tables()
    code, token = deal(registry)
    vector = [1.0, 0.0, 0.0, 0.0, -1.0]

    view = registry.handle("PUT", f"/api/games/{code}/valuation", {"valuation": vector}, token)
    assert view["valuations"][view["seat"]] == vector

    # Public: an observer with no seat at all reads the same block.
    watched = registry.handle("GET", f"/api/table/{code}", {}, None)
    assert watched["valuations"][view["seat"]] == vector


@pytest.mark.parametrize(
    "bad", [None, [1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 5.0], "wood"]
)
def test_a_malformed_valuation_is_refused(bad):
    registry = tables()
    code, token = deal(registry)
    with pytest.raises(ApiError):
        registry.handle("PUT", f"/api/games/{code}/valuation", {"valuation": bad}, token)


def test_a_published_valuation_clears_a_trade_and_it_shows_in_the_view():
    """End to end: a human seat publishes, the engine's trade event clears an
    exchange on the way into the main phase, and the game view reports it."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase, roll_dice

    registry = tables()
    code, token = deal(registry)
    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)
    other = next(s for s in range(game.num_players) if s != seat)

    # Park the game in ROLL with the human to move, holding one wood, and the
    # bot seat holding one ore. A non-seven roll opens the main phase, which
    # is where the trade event runs.
    game.phase = Phase.ROLL
    game.current_player = seat
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[other][Resource.ORE] = 1

    wants_ore = [0.0] * 5
    wants_ore[Resource.ORE] = 1.0
    wants_ore[Resource.WOOD] = -1.0
    registry.handle("PUT", f"/api/games/{code}/valuation", {"valuation": wants_ore}, token)
    # The bot seat wants the wood back; `PostedValuation` stands in for it so
    # the exchange has two willing sides without depending on what a
    # particular checkpoint would advertise.
    from hexset.server.webplay import PostedValuation

    table.session.set_trader(other, PostedValuation(tuple(-v for v in wants_ore)))

    roll_dice(game, 8)
    assert game.phase is Phase.MAIN

    view = table.view(seat)
    assert view["trades"], "the engine cleared nothing"
    trade = view["trades"][0]
    assert {trade["a"], trade["b"]} == {seat, other}
    assert state.hands[seat][Resource.ORE] == 1
    assert state.hands[other][Resource.WOOD] == 1
