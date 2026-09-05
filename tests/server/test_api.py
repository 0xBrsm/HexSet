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

from hexset.server.webplay import PendingGate, action_to_wire

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


def test_a_human_seat_is_unconditionally_a_pending_gate_and_records_a_candidate():
    """`POST /api/games` installs a `PendingGate` on the creator's seat --
    there is no flag to ask for or opt out of any more (`agents/reference/
    trading-final.md` item 5: human and LLM seats are direct gates,
    unconditionally) -- so an otherwise-clearing exchange is recorded,
    unexecuted, rather than taken automatically.
    """
    from hexset.board.terrain import Resource
    from hexset.game import Phase, roll_dice

    registry = tables()
    code, token = deal(registry)  # no `confirm`: this is the default under test
    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)
    other = next(s for s in range(game.num_players) if s != seat)
    assert seat in table.session.confirm_seats
    assert isinstance(game.gates[seat], PendingGate)

    game.phase = Phase.ROLL
    game.current_player = seat
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[other][Resource.ORE] = 1

    roll_dice(game, 8)
    assert game.phase is Phase.MAIN

    view = table.view(seat)
    assert view["trades"] == [], "a PendingGate seat never auto-clears"
    assert view["pending"] == [{"counterparty": other, "gave": [1, 0, 0, 0, 0], "got": [0, 0, 0, 0, 1]}]
    assert state.hands[seat][Resource.WOOD] == 1, "a PendingGate must never itself move cards"
    assert state.hands[other][Resource.ORE] == 1


class _Wants:
    """A minimal gate: prices a candidate positively iff it hands this seat
    more of `resource` than it had. Mirrors `test_webplay.py`'s own `_Wants`,
    for this file's API-level trade tests."""

    def __init__(self, resource):
        self.resource = resource

    def gains_many(self, view, received, counterparties):
        return [1.0 if r[self.resource] > 0 else -1.0 for r in received]


class _NeverWants:
    """A gate that prices nothing above zero -- for pinning `POST .../trade`'s
    400 when the counterparty's own gate declines."""

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def _a_one_candidate_position(registry, code, token, *, actor_wants: int, actor_gives: int):
    """A table where exactly one bundle is coverable: `actor` (a bot seat)
    holds one `actor_gives` and wants `actor_wants`, the manual seat holds
    one `actor_wants`, and everyone else's hand is empty. `actor`'s own gate
    is overridden with `_Wants(actor_wants)`, deterministic regardless of
    which bot preset seated it. Returns `(game, seat, actor)`."""
    from hexset.game import Phase, roll_dice

    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)  # manual, PendingGate
    actor = next(s for s in range(game.num_players) if s != seat)
    table.session.set_trader(actor, _Wants(actor_wants))

    game.phase = Phase.ROLL
    game.current_player = actor
    state = game.state(actor, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[actor][actor_gives] = 1
    state.hands[seat][actor_wants] = 1

    roll_dice(game, 8)
    assert game.phase is Phase.MAIN
    return game, seat, actor


def test_confirm_route_executes_exactly_the_recorded_pending_offer():
    """A bot actor's own gate prices the exchange above zero first
    (`hexset.trading._best_clearing` asks the acting seat before the
    counterparty); the manual counterparty's `PendingGate` then records it
    rather than clearing it, and `POST .../trade/confirm` executes exactly
    that recorded `(a, b, received)` through `execute_trade`'s own
    re-validation."""
    from hexset.board.terrain import Resource

    registry = tables()
    code, token = deal(registry, bots=SOLO)
    game, seat, actor = _a_one_candidate_position(
        registry, code, token, actor_wants=Resource.ORE, actor_gives=Resource.WOOD
    )
    state = game._state

    assert game.trades == [], "the counterparty is a PendingGate -- nothing auto-clears"
    view = registry.get(code).view(seat)
    assert view["pending"] == [{"counterparty": actor, "gave": [0, 0, 0, 0, 1], "got": [1, 0, 0, 0, 0]}]

    data = registry.handle("POST", f"/api/games/{code}/trade/confirm", {"index": 0}, token)

    assert data["pending"] == []
    assert state.hands[seat][Resource.WOOD] == 1 and state.hands[seat][Resource.ORE] == 0
    assert state.hands[actor][Resource.ORE] == 1 and state.hands[actor][Resource.WOOD] == 0
    # Confirming executes it exactly the way a proposal does -- including
    # showing up in the sidebar log, not just `state.trades`
    # (`GameSession.execute_manual_trade`).
    assert any("traded" in line for line in data["log"])


def test_decline_route_drops_the_offer_and_moves_nothing():
    """Declining is final -- the bot actor already moved on the instant its
    trade event ran, there is no re-offering it -- and no cards move either
    way."""
    from hexset.board.terrain import Resource

    registry = tables()
    code, token = deal(registry, bots=SOLO)
    game, seat, actor = _a_one_candidate_position(
        registry, code, token, actor_wants=Resource.ORE, actor_gives=Resource.WOOD
    )
    state = game._state

    data = registry.handle("POST", f"/api/games/{code}/trade/decline", {"index": 0}, token)

    assert data["pending"] == []
    assert game.pending == []
    assert state.hands[seat][Resource.ORE] == 1 and state.hands[seat][Resource.WOOD] == 0
    assert state.hands[actor][Resource.WOOD] == 1 and state.hands[actor][Resource.ORE] == 0


def test_trade_acceptable_lists_only_bot_accepted_bundles_and_never_mutates():
    """`GET .../trade/acceptable` is the actor's own preview: every bundle a
    bot counterparty's gate already prices above zero, grouped by
    counterparty -- and computing it must not touch the game at all."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    registry = tables()
    code, token = deal(registry, bots=SOLO)
    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)
    others = [s for s in range(game.num_players) if s != seat]
    # `seat` holds wood and wants ore; a candidate against either bot hands
    # them wood in exchange for the ore they hold, so it is *their own*
    # gain from *receiving wood* that decides whether they accept.
    accepts, declines = others[0], others[1]
    table.session.set_trader(accepts, _Wants(Resource.WOOD))
    table.session.set_trader(declines, _NeverWants())

    game.phase = Phase.MAIN
    game.current_player = seat
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[accepts][Resource.ORE] = 1
    state.hands[declines][Resource.ORE] = 1
    before_hands = [hand[:] for hand in state.hands]
    before_trades = len(game.trades)

    data = registry.handle("GET", f"/api/games/{code}/trade/acceptable", {}, token)

    assert [o["counterparty"] for o in data["offers"]] == [accepts]
    deals = data["offers"][0]["deals"]
    assert deals == [{"gave": [1, 0, 0, 0, 0], "got": [0, 0, 0, 0, 1], "gain": 1.0}]
    # No engine mutation: same hands, no trade recorded, nothing pending.
    assert state.hands == before_hands
    assert len(game.trades) == before_trades
    assert game.pending == []


def test_trade_route_rejects_a_bundle_the_bots_gate_declines_with_400():
    """`POST .../trade` re-validates through `execute_trade`: a bundle the
    named counterparty's own gate prices at zero or less is refused with a
    400 naming the reason, and nothing moves."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    registry = tables()
    code, token = deal(registry, bots=SOLO)
    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)
    other = next(s for s in range(game.num_players) if s != seat)
    table.session.set_trader(other, _NeverWants())

    game.phase = Phase.MAIN
    game.current_player = seat
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[other][Resource.ORE] = 1

    with pytest.raises(ApiError) as excinfo:
        registry.handle(
            "POST",
            f"/api/games/{code}/trade",
            {"counterparty": other, "give": {"Wood": 1}, "receive": {"Ore": 1}},
            token,
        )
    assert excinfo.value.status == 400
    assert "does not want" in str(excinfo.value)
    assert state.hands[seat][Resource.WOOD] == 1 and state.hands[other][Resource.ORE] == 1


def test_a_proposed_trade_appears_in_the_log_and_the_journal(tmp_path):
    """`POST .../trade` moves cards via `hexset.game.Game.execute_trade`, not
    through `GameSession._apply` -- no board action happened, so it needs
    its own way onto the sidebar log and into the journal
    (`GameSession.execute_manual_trade`). Both are what a browser refresh
    and a server restart actually read, not `state.trades` alone."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    registry = tables(games_dir=str(tmp_path))
    code, token = deal(registry, bots=SOLO)
    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)
    other = next(s for s in range(game.num_players) if s != seat)
    table.session.set_trader(other, _Wants(Resource.WOOD))

    game.phase = Phase.MAIN
    game.current_player = seat
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[other][Resource.ORE] = 1

    data = registry.handle(
        "POST",
        f"/api/games/{code}/trade",
        {"counterparty": other, "give": {"Wood": 1}, "receive": {"Ore": 1}},
        token,
    )

    assert state.hands[seat][Resource.ORE] == 1 and state.hands[seat][Resource.WOOD] == 0
    assert any("traded" in line for line in data["log"])
    assert any(t["a"] == seat and t["b"] == other for t in data["trades"])

    journal_files = list(tmp_path.glob("*.jsonl"))
    assert len(journal_files) == 1
    import json

    lines = [json.loads(line) for line in journal_files[0].read_text().splitlines()]
    assert any(e.get("kind") == "trade" for e in lines), "a manual trade must not vanish on resume"
