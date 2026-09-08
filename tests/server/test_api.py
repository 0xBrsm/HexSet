"""Games, seats, codes and tokens — `hexset.server.api` without a socket.

Everything here calls `Tables.handle` the way `web.py` would, so the routing
and the rules are pinned together; `test_web.py` covers only what the HTTP
transport adds on top. `search2` is named explicitly at every call rather than
left to `Config.default_bots`, which would seat whatever `.onnx` files happen
to be in `models/` and drag onnxruntime into a suite that has no need of it.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import legal_actions

from hexset.server.api import (
    MAX_SEATS,
    ApiError,
    Config,
    Seat,
    SeatKind,
    Tables,
    build_session,
    resume_session,
)

from hexset.game import is_over, to_move

from hexset.server.webplay import action_to_wire

from conftest import new_tables

SOLO = ["search2", "search2", "search2"]


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    """Turn order is seat order from seat 0 (`Tables.create` always deals
    `first=0`, see `hexset.server.seating`'s module docstring); this file's
    tests treat the dealt token as the one that moves first, so pin the
    creator to seat 0 for that determinism."""
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)



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


def test_a_spectator_sees_every_hand_and_a_seat_still_sees_only_its_own():
    """Watching is omniscient on purpose: a spectator is outside the game and
    is shown all of it. A seat is inside it and is not — the same table, read
    two ways, and the difference is the whole of what the token buys."""
    registry = tables()
    code, token = deal(registry, bots=["search2", "search2", "search2"])
    mine = registry.by_token(token)[1]

    watched = registry.handle("GET", f"/api/table/{code}", {}, None)
    seated = registry.handle("GET", "/api/state", {}, token)

    assert {p["seat"] for p in watched["players"] if "hand" in p} == set(range(MAX_SEATS))
    assert {p["seat"] for p in watched["players"] if "dev_cards" in p} == set(range(MAX_SEATS))
    assert {p["seat"] for p in seated["players"] if "hand" in p} == {mine}


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


# --- a seven's discards are simultaneous, so no seat waits on another ---------


def _four_humans(registry: Tables) -> tuple[str, dict[int, str]]:
    """A table with a person on every seat, as `(code, {seat: token})`."""
    code, token = deal(registry, bots=[])
    tokens = {registry.by_token(token)[1]: token}
    while len(tokens) < MAX_SEATS:
        data = registry.handle("POST", "/api/join", {"code": code}, None)
        tokens[registry.by_token(data["token"])[1]] = data["token"]
    return code, tokens


def _owing_seats_zero_and_three(registry: Tables, code: str) -> None:
    """Park the table in `Phase.DISCARD` with seats 0 and 3 each owing two
    cards, and seat 1 -- who rolled the seven -- owing none."""
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


def test_a_higher_seat_discards_without_waiting_for_a_lower_one():
    """The bug this file's carve-out exists for: discarding on a seven is not
    a turn, but `POST /api/action` gated every submission on `to_move` -- the
    lowest-numbered owing seat -- so seat 3 was refused until seat 0 had
    finished, at a table where both were choosing at the same moment."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase, players_owing_discards

    registry = tables()
    code, tokens = _four_humans(registry)
    _owing_seats_zero_and_three(registry, code)
    game = registry.get(code).session.game
    assert players_owing_discards(game) == [0, 3]
    assert to_move(game) == 0  # unchanged: still one seat, for callers that want one

    registry.handle(
        "POST", "/api/action", {"action": {"type": "DISCARD", "a": int(Resource.ORE)}}, tokens[3]
    )

    assert game._state.hands[3][Resource.ORE] == 3
    assert game._state.hands[0][Resource.WOOD] == 4  # seat 0 has not moved
    assert game.discard_quota == [2, 0, 0, 1]
    assert game.phase is Phase.DISCARD


def test_a_seat_owing_nothing_is_still_refused_during_a_discard_round():
    """The carve-out is exactly the owing seats, not the whole table: seat 1
    rolled the seven and owes nothing, so it has nothing to play."""
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
    """`/api/state`'s option list and `/api/record` both answer per seat: seat
    3 is offered its own ore rather than seat 0's wood, and asking for the
    record no longer 409s a seat that a lower-numbered one has not yet let
    through."""
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


def test_a_discard_round_resolves_in_whatever_order_the_seats_answer_in():
    """Interleaved submissions from both owing seats close the round and hand
    the robber to whoever rolled -- not to whichever seat discarded last."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    registry = tables()
    code, tokens = _four_humans(registry)
    _owing_seats_zero_and_three(registry, code)
    game = registry.get(code).session.game

    for seat, resource in ((3, Resource.ORE), (0, Resource.WOOD)) * 2:
        registry.handle(
            "POST", "/api/action", {"action": {"type": "DISCARD", "a": int(resource)}}, tokens[seat]
        )

    assert game.discard_quota == [0, 0, 0, 0]
    assert game._state.hands[0][Resource.WOOD] == 2
    assert game._state.hands[3][Resource.ORE] == 2
    assert game.phase is Phase.ROBBER
    assert game.current_player == 1
