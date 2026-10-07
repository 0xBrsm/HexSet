"""Games, seats, codes and tokens -- `hexset.server.api` without a socket.
`test-trader` (`tests/traders.py`) is named explicitly wherever a bot is seated.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, legal_actions

from hexset.server import _journal as journal
from hexset.server.api import (
    MAX_SEATS,
    ApiError,
    Config,
    Seat,
    SeatKind,
    Tables,
    build_session,
    reopen_session,
    reopened_seats,
    your_move,
)

from hexset.game import is_over, to_move

from hexset.server.wire import action_to_wire

from conftest import new_tables

SOLO = ["test-trader", "test-trader", "test-trader"]


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


def tables(**config) -> Tables:
    return new_tables(**config)


def deal(registry: Tables, **kwargs) -> tuple[str, str]:
    kwargs.setdefault("bots", SOLO)
    data = registry.handle("POST", "/api/games", kwargs, None)
    return data["code"], data["token"]


def player(name: str | None = None) -> Seat:
    return Seat(kind=SeatKind.PLAYER, name=name, token="t-" + (name or "x"))


def bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def reopened(code: str, directory, seats: list[Seat] | None = None):
    path = next(Path(directory).glob("*.jsonl"))
    events = journal.read(path)
    return reopen_session(code, reopened_seats(events) if seats is None else seats, path, events)


def drive(session, moves: int, rng: random.Random) -> None:
    for _ in range(moves):
        if is_over(session.game):
            break
        # A browser seat holding its setup turn open is the one time the seat
        # to play is not `to_move`'s: it owes an END_TURN
        # (`webplay.GameSession.awaiting_confirm`).
        if session.awaiting_confirm is not None:
            session.submit(
                session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN))
            )
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))


def test_a_spiral_table_journals_its_mode_and_comes_back_on_the_same_board(tmp_path):
    from hexset.board.board import spiral_base_board

    config = Config(games_dir=str(tmp_path), seed=99, board_mode="spiral")
    session = build_session("ABC123", [player("Ada"), bot_seat(), bot_seat(), bot_seat()], config, first=0)
    assert session.game._state.board == spiral_base_board(random.Random(99))
    header = journal.read(next(tmp_path.glob("*.jsonl")))[0]
    assert header["board_mode"] == "spiral"
    drive(session, 8, random.Random(4))

    resumed = reopened("ABC123", tmp_path)

    assert resumed.board_mode == "spiral"
    assert resumed.game._state.board == session.game._state.board
    assert resumed.game._state.vertex_owner == session.game._state.vertex_owner


def test_an_unknown_board_mode_is_refused_at_configuration():
    with pytest.raises(ValueError, match="unknown board mode"):
        Config(board_mode="hexagonal")


def test_an_unfinished_game_comes_back_where_it_was_left(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
    drive(session, 12, random.Random(4))
    assert not is_over(session.game)

    resumed = reopened("ABC123", tmp_path)

    assert resumed is not None
    assert resumed.game.phase is session.game.phase
    assert resumed.game._state.hands == session.game._state.hands
    assert resumed.game._state.vertex_owner == session.game._state.vertex_owner
    assert resumed.game._state.edge_owner == session.game._state.edge_owner
    assert resumed.game._state.deck == session.game._state.deck
    assert resumed.game._state.robber == session.game._state.robber
    assert resumed.game.turns == session.game.turns
    assert resumed.log_for(0) == session.log_for(0)
    assert (resumed.seed, resumed.claimed_seats) == (session.seed, session.claimed_seats)
    assert resumed.player_names == session.player_names


WEB_SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(WEB_SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


@pytest.fixture(scope="module")
def finished_game(tmp_path_factory):
    """Played out rather than forced, since an `is_over` set by hand leaves no
    trace in the actions a replay reconstructs from. Module-scoped and safe to
    share: a game replayed to `is_over` comes back with no journal attached.
    """
    directory = tmp_path_factory.mktemp("finished")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        bot_seat(),
        bot_seat(),
        bot_seat(),
    ]
    session = build_session("ABC123", seats, Config(games_dir=str(directory), seed=99), first=0)
    drive(session, 2000, random.Random(4))
    assert is_over(session.game)
    return session, directory


def test_a_finished_game_can_still_be_viewed_after_a_restart(finished_game):
    session, directory = finished_game

    replayed = reopened("ABC123", directory)

    assert replayed is not None
    assert is_over(replayed.game)
    assert replayed.game.won_by == session.game.won_by
    assert replayed.log_for(None, omniscient=True) == session.log_for(None, omniscient=True)
    assert replayed.journal is None
    assert replayed.player_names == session.player_names == {0: "Ada"}


def test_an_abandoned_unfinished_game_does_not_come_back(tmp_path):
    """Both `Journal.finish` and `Journal.abandoned` write a closing line, and
    which it was is the replayed game's to say: an unfinished game rebuilt as
    finished is a *mutable* table with no journal and no bot runners, so every
    action at it is dropped on the floor.
    """
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
    drive(session, 12, random.Random(4))
    assert not is_over(session.game)
    assert reopened("ABC123", tmp_path) is not None

    session.journal.abandoned()

    assert reopened("ABC123", tmp_path) is None
    fresh = new_tables(games_dir=str(tmp_path))
    with pytest.raises(ApiError) as gone:
        fresh.handle("GET", "/api/table/ABC123", {}, None)
    assert gone.value.status == 404


def test_a_journalled_round_trade_replays_into_the_record(tmp_path):
    from dataclasses import replace

    from hexset.game import Phase
    from hexset.record import from_journal, replay

    class _Wants:
        trade_floor = 0.0

        def __init__(self, resource: int):
            self.resource = resource

        def gains_many(self, view, received, counterparties):
            return [1.0 if r[self.resource] > 0 else -1.0 for r in received]

    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
    drive(session, 16, random.Random(4))
    game = session.game
    if game.phase is not Phase.MAIN:
        drive(session, 1, random.Random(5))
    while game.phase is not Phase.MAIN or game.current_player != 0:
        drive(session, 1, random.Random(6))
    state = game.state(0, hidden=False)
    for _ in range(40):
        gives = [r for r in range(5) if state.hands[0][r] > 0]
        gets = [r for r in range(5) if state.hands[1][r] > 0 and r not in gives]
        if gives and gets and game.phase is Phase.MAIN and game.current_player == 0:
            break
        drive(session, 1, random.Random(8))
    assert gives and gets, "no coverable exchange came up"
    session.confirm_mode(0)
    session.set_trader(1, _Wants(gives[0]))
    received = [0, 0, 0, 0, 0]
    received[gets[0]] = 1
    received[gives[0]] = -1
    session.open_round_for(0, tuple(received))
    session.execute_round_choice(0, 1, tuple(received))
    drive(session, 6, random.Random(7))
    session.journal.finish(game)

    path = next(tmp_path.glob("*.jsonl"))
    record = replace(from_journal(path), seed=None)
    assert len(record.trades) == 1
    replayed = replay(record)
    assert replayed.state(0, hidden=False).hands == game.state(0, hidden=False).hands


def test_trade_round_lines_survive_a_restart(tmp_path):
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
    session.game.phase = Phase.MAIN
    session.game.current_player = 0
    session.game._state.hands[0][Resource.WOOD] = 2
    received = [0, 0, 0, 0, 0]
    received[Resource.WOOD] = -2
    received[Resource.ORE] = 1
    session.open_round_for(0, tuple(received))
    session.decline_round(0)
    log = session.log_for(0)
    assert any("offers 2 Wood for 1 Ore." in line for line in log)

    resumed = reopened("ABC123", tmp_path)

    assert resumed is not None
    assert resumed.log_for(0) == log


def _a_position_where_no_opponent_holds_anything(mover: int = 0):
    import random as _random

    from hexset.board.board import random_base_board
    from hexset.board.terrain import NUM_RESOURCES
    from hexset.game import Phase, start

    game = start(random_base_board(_random.Random(0)), 4, _random.Random(1), first=0)
    game.phase = Phase.MAIN
    game.current_player = mover
    for hand in game._state.hands:
        hand[:] = [0] * NUM_RESOURCES
    game._state.hands[mover] = [1, 1, 1, 1, 1]
    return game


def test_record_matches_the_embedded_bots_options():
    import numpy as np

    from hexset.actions import build_space

    from hexset.onnx_record import record_from_game
    from hexset.actions import options_for

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


# --- a seven's discards are simultaneous, so no seat waits on another ---------


def _four_humans(registry: Tables) -> tuple[str, dict[int, str]]:
    code, token = deal(registry, bots=[])
    tokens = {registry.by_token(token)[1]: token}
    while len(tokens) < MAX_SEATS:
        data = registry.handle("POST", "/api/join", {"code": code}, None)
        tokens[registry.by_token(data["token"])[1]] = data["token"]
    return code, tokens


def _owing_seats_zero_and_three(registry: Tables, code: str) -> None:
    from hexset.board.terrain import NUM_RESOURCES, Resource
    from hexset.game import Phase

    game = registry.get(code).session.game
    game.phase = Phase.DISCARD
    game.current_player = 1
    for hand in game._state.hands:
        hand[:] = [0] * NUM_RESOURCES
    game._state.hands[0][Resource.WOOD] = 4
    game._state.hands[3][Resource.ORE] = 4
    game.discard_quota = [2, 0, 0, 2]


def test_a_seat_owing_nothing_is_still_refused_during_a_discard_round():
    from hexset.board.terrain import Resource

    registry = tables()
    code, tokens = _four_humans(registry)
    _owing_seats_zero_and_three(registry, code)

    with pytest.raises(ApiError) as excinfo:
        registry.handle(
            "POST",
            "/api/action",
            {"action": {"type": "DISCARD", "a": int(Resource.ORE)}},
            tokens[1],
        )
    assert "not your turn" in str(excinfo.value)


def test_every_owing_seat_is_offered_its_own_cards_by_state_and_record():
    from hexset.board.terrain import Resource

    registry = tables()
    code, tokens = _four_humans(registry)
    _owing_seats_zero_and_three(registry, code)

    theirs = registry.handle("GET", "/api/state", {}, tokens[3])["legal_actions"]
    lowest = registry.handle("GET", "/api/state", {}, tokens[0])["legal_actions"]
    assert [a["a"] for a in theirs] == [int(Resource.ORE)]
    assert [a["a"] for a in lowest] == [int(Resource.WOOD)]
    assert registry.handle("GET", "/api/state", {}, tokens[1])["legal_actions"] == []

    record = registry.handle("GET", "/api/record", {}, tokens[3])
    assert [a["a"] for a in record["options"]] == [int(Resource.ORE)]


# --- client identity + default seat names (`parse_client`, `default_seat_name`) -


def test_a_joined_seats_default_name_follows_its_clients_kind():
    registry = tables()
    code, _ = deal(registry, bots=[])

    api_join = registry.handle("POST", "/api/join", {"code": code}, None)
    table = registry.get(code)
    assert table.seats[api_join["seat"]].name == "api"

    web_join = registry.handle(
        "POST", "/api/join", {"code": code, "client": {"kind": "web"}}, None
    )
    assert table.seats[web_join["seat"]].name == "human"

    mcp_join = registry.handle(
        "POST", "/api/join", {"code": code, "client": {"kind": "mcp"}}, None
    )
    assert table.seats[mcp_join["seat"]].name == "mcp"


def test_join_refuses_an_unknown_client_kind_or_a_malformed_id():
    registry = tables()
    code, _ = deal(registry, bots=[])

    with pytest.raises(ApiError) as bad_kind:
        registry.handle("POST", "/api/join", {"code": code, "client": {"kind": "browser"}}, None)
    assert bad_kind.value.status == 400

    with pytest.raises(ApiError) as bad_id:
        registry.handle(
            "POST", "/api/join", {"code": code, "client": {"id": "not-hex", "kind": "web"}}, None
        )
    assert bad_id.value.status == 400


def test_the_journal_header_and_a_seated_event_carry_the_client(tmp_path):
    registry = tables(games_dir=str(tmp_path))
    creator_id = "a" * 64
    dealt = registry.handle(
        "POST",
        "/api/games",
        {"bots": [], "client": {"id": creator_id, "kind": "web"}},
        None,
    )
    creator_seat = registry.by_token(dealt["token"])[1]

    joiner_id = "b" * 64
    joined = registry.handle(
        "POST",
        "/api/join",
        {"code": dealt["code"], "client": {"id": joiner_id, "kind": "mcp"}},
        None,
    )
    joiner_seat = registry.by_token(joined["token"])[1]

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1, f"expected one game journal, found {files}"
    events = [json.loads(line) for line in files[0].read_text().splitlines()]

    header = events[0]
    assert header["clients"][str(creator_seat)] == {"id": creator_id, "kind": "web"}

    seated = next(e for e in events if e["kind"] == "seated")
    assert seated["seat"] == joiner_seat
    assert seated["client"] == {"id": joiner_id, "kind": "mcp"}


# --- POST /api/reclaim ---------------------------------------------------------


def test_reclaim_with_the_right_secret_mints_a_token_that_reads_state():
    registry = tables()
    secret = "a client's own secret"
    client_id = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    dealt = registry.handle(
        "POST", "/api/games", {"bots": [], "client": {"id": client_id, "kind": "api"}}, None
    )
    old_token = dealt["token"]

    reclaimed = registry.handle(
        "POST", "/api/reclaim", {"code": dealt["code"], "secret": secret}, None
    )
    new_token = reclaimed["token"]
    assert new_token != old_token
    assert reclaimed["code"] == dealt["code"]

    state = registry.handle("GET", "/api/state", {}, new_token)
    assert state["code"] == dealt["code"]

    with pytest.raises(ApiError) as expired:
        registry.handle("GET", "/api/state", {}, old_token)
    assert expired.value.status == 403


def test_reclaim_with_the_wrong_secret_403s():
    registry = tables()
    client_id = hashlib.sha256(b"the right secret").hexdigest()
    dealt = registry.handle(
        "POST", "/api/games", {"bots": [], "client": {"id": client_id, "kind": "api"}}, None
    )

    with pytest.raises(ApiError) as wrong:
        registry.handle(
            "POST", "/api/reclaim", {"code": dealt["code"], "secret": "not it"}, None
        )
    assert wrong.value.status == 403


def test_reclaim_gets_a_finished_game_s_seat_back_but_it_still_cannot_act(finished_game):
    _session, directory = finished_game
    fresh = new_tables(games_dir=str(directory))

    mine = fresh.handle("POST", "/api/reclaim", {"code": "ABC123", "secret": WEB_SECRET}, None)

    assert mine["seat"] == 0 and mine["game_over"] is True
    assert fresh.handle("GET", "/api/state", {}, mine["token"])["seat"] == 0
    for method, path, payload in [
        ("POST", "/api/name", {"name": "Grace"}),
        ("POST", "/api/action", {"action": {"type": "END_TURN"}}),
        ("POST", "/api/leave", {}),
        ("POST", "/api/bot", {"seat": 1, "model": "test-trader"}),
    ]:
        with pytest.raises(ApiError) as refused:
            fresh.handle(method, path, payload, mine["token"])
        assert "already over" in refused.value.args[0], f"{path} did not refuse"
        assert refused.value.status == 409, f"{path} refused with {refused.value.status}"


def test_reclaim_at_a_reopened_game_is_the_way_back_into_a_seat_no_one_else_gets(tmp_path):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        bot_seat(),
        bot_seat(),
        bot_seat(),
    ]
    session = build_session("ABC123", seats, config, first=0)
    drive(session, 12, random.Random(4))
    assert not is_over(session.game)

    fresh = new_tables(games_dir=str(tmp_path))
    with pytest.raises(ApiError) as refused:
        fresh.handle("POST", "/api/join", {"code": "ABC123"}, None)
    assert refused.value.status == 409

    mine = fresh.handle("POST", "/api/reclaim", {"code": "ABC123", "secret": WEB_SECRET}, None)

    assert mine["seat"] == 0
    assert fresh.handle("GET", "/api/state", {}, mine["token"])["seat"] == 0
    assert fresh.get("ABC123").session.is_manual(0)


# --- GET /api/version, and the optional `version` guard on acting routes ------


def test_version_route_answers_the_api_contract_not_the_package():
    import hexset
    from hexset.server.api import API_VERSION

    registry = tables()
    info = registry.handle("GET", "/api/version", {}, None)
    assert info == {"api": API_VERSION}
    assert isinstance(info["api"], int)
    assert "version" not in info and "git_commit" not in info
    assert not hasattr(hexset, "build_info")


def test_action_with_a_stale_version_409s_and_the_current_one_acts():
    registry = tables()
    code, token = deal(registry)
    state = registry.handle("GET", "/api/state", {}, token)
    current = state["version"]
    action = state["legal_actions"][0]

    with pytest.raises(ApiError) as stale:
        registry.handle(
            "POST", "/api/action", {"action": action, "version": current - 1}, token
        )
    assert stale.value.status == 409
    assert "version" in str(stale.value)

    acted = registry.handle("POST", "/api/action", {"action": action, "version": current}, token)
    assert acted["version"] > current




# --- your_move: what the table wants from a seat now ------------------------


def test_your_move_game_over_wins_over_everything():
    move, on = your_move({"game_over": True, "seat": 0, "to_move": 0, "legal_actions": [{"type": "END_TURN"}]})
    assert (move, on) == ("game_over", [])


def test_your_move_choose_trade_once_your_round_is_fully_answered():
    view = {
        "seat": 0,
        "to_move": 0,
        "phase": "MAIN",
        "legal_actions": [{"type": "END_TURN"}],  # still your turn -- but the round comes first
        "trade_round": {"responses": [{"seat": 1, "kind": "accept"}], "awaiting": []},
    }
    assert your_move(view) == ("choose_trade", [])


def test_your_move_answers_an_offer_while_the_bot_that_made_it_holds_the_table():
    view = {"seat": 2, "to_move": None, "phase": "MAIN", "legal_actions": [], "trade_wait": [2],
            "pending": [{"actor": 0, "bundle": [-1, 0, 0, 0, 1]}]}
    assert your_move(view) == ("answer_trade", [])


def test_your_move_acts_only_when_the_seat_is_to_move():
    """`legal_actions` lists a seat's moves while setup waits on an empty
    seat, or while the seat before it holds its setup turn open, and
    `POST /api/action` refuses them both times."""
    legal = [{"type": "SETUP_SETTLEMENT", "a": 3}]
    assert your_move({"seat": 0, "to_move": 0, "phase": "SETUP_SETTLEMENT", "legal_actions": legal}) == ("act", [])
    assert your_move({"seat": 0, "to_move": None, "phase": "SETUP_SETTLEMENT", "legal_actions": legal,
                      "waiting_for": [3]}) == ("wait", [3])
    assert your_move({"seat": 1, "to_move": 0, "phase": "SETUP_SETTLEMENT", "legal_actions": legal}) == ("wait", [0])


def test_your_move_discards_whoever_is_to_move():
    view = {"seat": 2, "to_move": 0, "phase": "DISCARD", "discard_quota": [2, 0, 2, 0],
            "legal_actions": [{"type": "DISCARD", "a": 4}]}
    assert your_move(view) == ("discard", [])


def test_your_move_wait_names_who_is_holding_things_up():
    assert your_move({"legal_actions": [], "trade_round": {"responses": [], "awaiting": [2, 3]}}) == (
        "wait",
        [2, 3],
    )
    assert your_move({"legal_actions": [], "trade_wait": [3]}) == ("wait", [3])
    assert your_move({"legal_actions": [], "waiting_for": [2]}) == ("wait", [2])
    assert your_move({"legal_actions": [], "phase": "DISCARD", "discard_quota": [0, 3, 0, 2]}) == (
        "wait",
        [1, 3],
    )
    assert your_move({"legal_actions": [], "to_move": 2}) == ("wait", [2])
    assert your_move({"legal_actions": [], "to_move": None}) == ("wait", [])


def test_every_seat_view_carries_your_move():
    tables = new_tables()
    created = tables.handle("POST", "/api/games", {}, None)
    view = tables.handle("GET", "/api/state", {}, created["token"])
    assert view["your_move"] == "wait" and view["waiting_on"]  # empty seats hold up setup
