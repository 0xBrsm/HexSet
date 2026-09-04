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


def test_a_published_valuation_clears_a_trade_and_it_shows_in_the_view():
    """End to end, `confirm=False` (the opt-out, `PostedValuation`): a human
    seat publishes, the engine's trade event clears an exchange on the way
    into the main phase, and the game view reports it. The default --
    `confirm` omitted entirely -- is covered by
    `test_a_human_seat_defaults_to_confirm_mode_and_records_a_pending_candidate`
    below, where the identical setup records a pending candidate instead."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase, roll_dice

    registry = tables()
    code, token = deal(registry, confirm=False)
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
    # The bot seat wants the wood back; publishing on its behalf stands in
    # for it so the exchange has two willing sides without depending on what
    # a particular checkpoint would advertise.
    table.session.publish(other, tuple(-v for v in wants_ore))

    roll_dice(game, 8)
    assert game.phase is Phase.MAIN

    view = table.view(seat)
    assert view["trades"], "the engine cleared nothing"
    trade = view["trades"][0]
    assert {trade["a"], trade["b"]} == {seat, other}
    assert state.hands[seat][Resource.ORE] == 1
    assert state.hands[other][Resource.WOOD] == 1


def test_a_human_seat_defaults_to_confirm_mode_and_records_a_pending_candidate():
    """The bug fix: `POST /api/games` with no `confirm` key at all -- exactly
    what the web page's own seat-up sends -- must default a human seat to
    `PendingGate`, not `PostedValuation`. Same setup as the opt-out test
    above; the only difference is the missing `confirm` kwarg, and the
    outcome flips from an executed trade to a recorded, unexecuted one."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase, roll_dice

    registry = tables()
    code, token = deal(registry)  # no `confirm`: this is the default under test
    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)
    other = next(s for s in range(game.num_players) if s != seat)
    assert seat in table.session.confirm_seats

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
    table.session.publish(other, tuple(-v for v in wants_ore))

    roll_dice(game, 8)
    assert game.phase is Phase.MAIN

    view = table.view(seat)
    assert view["trades"] == [], "nothing may auto-clear against a human without confirm=false"
    assert view["pending"] == [{"counterparty": other, "gave": [1, 0, 0, 0, 0], "got": [0, 0, 0, 0, 1]}]
    assert state.hands[seat][Resource.WOOD] == 1, "a PendingGate must never itself move cards"
    assert state.hands[other][Resource.ORE] == 1


class _StubBot:
    """A minimal seated bot for `game.gates` -- `valuation`/`accepts` answer
    a fixed vector, same shape as any real checkpoint `spawn_bot` would
    build. Swapped in for `other`'s real embedded checkpoint so the test
    below doesn't depend on what a particular one would advertise."""

    def __init__(self, vec):
        self.vec = vec

    def valuation(self, view):
        return self.vec

    def accepts(self, view, received, counterparty):
        return True


def test_a_spectator_poll_before_the_bots_publish_does_not_spend_the_event():
    """The regression a deploy report pinned down ("the served game never
    trades"): `GameSession.state_view` -- behind both `GET /api/state` and
    `GET /api/table/<CODE>` -- used to read the current player's own hidden
    view on *every* call, for *any* viewer, which fired this turn's first
    trade event before a bot seat ever got to publish -- and back then,
    `publish_due` was defined off whether that event had run, so the
    publish that should have followed the poll never happened, for that
    turn or any after it. `publish_due` is now keyed off whether the seat
    itself has published (`Game.publish_due`'s docstring): a spectator poll
    can still fire the first event early, on the seat's own standing vector
    (unaffected by this fix, `run_pending_event`'s own docstring), but the
    seat's publish moments later still works, and reaches the turn's next
    event -- every interleaved one after a MAIN action -- rather than being
    silently dropped.

    `confirm=False`: this test is about the publish/event race, not the
    negotiation interface's confirm mode, so the human seat is opted out of
    `PendingGate` the same way `test_a_published_valuation_clears_a_trade_...`
    is -- otherwise the human's own gate would record a pending candidate
    instead of clearing, and the assertion below would be testing the wrong
    thing."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase, roll_dice, run_trade_event

    registry = tables()
    code, token = deal(registry, confirm=False)
    table = registry.get(code)
    game = table.session.game
    seat = table.seat_of(token)
    other = next(s for s in range(game.num_players) if s != seat)

    # Park the game in ROLL with the bot seat `other` to move next, holding
    # one ore, and the human seat holding one wood.
    game.phase = Phase.ROLL
    game.current_player = other
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[other][Resource.ORE] = 1

    wants_ore = [0.0] * 5
    wants_ore[Resource.ORE] = 1.0
    wants_ore[Resource.WOOD] = -1.0
    registry.handle("PUT", f"/api/games/{code}/valuation", {"valuation": wants_ore}, token)
    # `other`'s embedded runner thread holds its own checkpoint's gate;
    # swapped for a controllable stub so this test doesn't depend on what a
    # particular checkpoint would advertise.
    table.session.set_trader(other, _StubBot(tuple(-v for v in wants_ore)))

    roll_dice(game, 8)  # a non-seven roll opens MAIN for `other`
    assert game.phase is Phase.MAIN

    # A spectator polls before `other` has published anything this turn --
    # exactly the race the regression reproduced. `other` has never
    # published before, so its standing vector is all-zero and nothing
    # clears -- but the poll must not spend `other`'s own publish.
    watched = registry.handle("GET", f"/api/table/{code}", {}, None)
    assert watched["trades"] == []
    assert game.publish_due(other) is True, "the poll must not spend the seat's own publish"

    # `other` now publishes, exactly as `LocalSearchBrain.decide` does before
    # `choose`. The first event already ran (on the stale vector, above), so
    # this alone does not yet trade -- but the fresh vector reaches the next
    # event, e.g. after `other`'s next MAIN action.
    table.session.publish(other, tuple(-v for v in wants_ore))
    run_trade_event(game)

    view = table.view(seat)
    assert view["trades"], "the engine cleared nothing"
    trade = view["trades"][0]
    assert {trade["a"], trade["b"]} == {seat, other}
