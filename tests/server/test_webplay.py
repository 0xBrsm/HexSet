from __future__ import annotations

import json

import random

from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions

from hexset.board.board import random_base_board

from hexset.cards import DevCard

from conftest import RandomBot

from hexset.game import Phase, is_over, to_move

from hexset.server.seating import start_at

from hexset.server.journal import open_journal, replayable

from hexset.trading import Trade

from hexset.server.webplay import (
    RESOURCE_NAMES,
    GameSession,
    PendingGate,
    action_to_wire,
    bundle_from_wire,
    wire_to_action,
)


class _Wants:
    """A gate that prices a candidate positively iff it hands this seat more
    of `resource` than it had -- enough to clear a clean swap without also
    pricing the reverse of it positively (see `tests/test_trading.py`'s
    `wants` helper, which this mirrors for the server-side suite)."""

    def __init__(self, resource: int):
        self.resource = resource

    def gains_many(self, view, received, counterparties):
        return [1.0 if r[self.resource] > 0 else -1.0 for r in received]


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start_at(random_base_board(rng), players, rng, first=0)


def a_session(game, claimed, **kwargs) -> GameSession:
    """A `GameSession` over `claimed` seats — every claimed seat submits its
    own actions through `submit` the same way now, human or "bot" (there is
    no `bot:` field any more; a seat played by a bot in these tests is just
    driven directly, via `_apply` or `submit`, exactly like any other seat —
    see `botclient.py` for how a real bot runner does the same from outside
    a session entirely)."""
    return GameSession(game=game, claimed_seats=set(claimed), **kwargs)


def test_wire_round_trips_across_a_played_out_game():
    game = a_game(seed=11)
    rng = random.Random(99)
    steps = 0
    while not is_over(game) and steps < 500:
        options = legal_actions(game)
        for action in options:
            assert wire_to_action(action_to_wire(action)) == action
        apply(game, rng.choice(options))
        steps += 1
    assert steps > 50  # sanity: the loop actually exercised many phases


def test_wire_to_action_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        wire_to_action({"type": "TELEPORT", "a": 0, "b": 0})


def test_session_rejects_an_action_not_currently_legal():
    game = a_game(seed=2)
    seat = to_move(game)
    session = a_session(game, {seat})

    # ROLL is never legal during setup placement.
    forged = action_to_wire(Action(ActionType.ROLL))
    with pytest.raises(ValueError):
        session.submit(seat, forged)
    assert game.phase is Phase.SETUP_SETTLEMENT
    assert all(owner == -1 for owner in game._state.vertex_owner)


def test_session_rejects_an_action_from_a_seat_that_has_not_claimed_it():
    game = a_game(seed=4)
    mover = to_move(game)
    other = (mover + 1) % game._state.num_players
    session = a_session(game, {other})

    # A perfectly legal action for whoever is actually on the move.
    legal_for_mover = action_to_wire(legal_actions(game)[0])
    with pytest.raises(ValueError):
        session.submit(other, legal_for_mover)


def test_legal_wire_actions_never_depend_on_an_opponents_hand():
    """HexSet hands are private: no client -- human, LLM, or bot -- must be
    able to learn what an opponent holds from what it is offered. The one
    place that could was the engine's `PROPOSE_TRADE` sample, which filtered
    to pairs some opponent could cover; trading is no longer an action, so
    this holds by construction and is checked by emptying every other hand
    and finding the option list unmoved."""
    game = a_game(seed=19)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game._state
    session = a_session(game, {0})
    before = session.legal_wire_actions(0)

    for seat in range(1, state.num_players):
        for r in range(len(state.hands[seat])):
            state.hands[seat][r] = 0

    assert session.legal_wire_actions(0) == before


def test_a_knight_resolves_through_the_session_like_a_seven():
    """The knight two-step fix: the page now sends a bare `PLAY_KNIGHT` and
    handles `Phase.ROBBER` exactly as it does after a seven, rather than
    picking a target client-side first. Played through a session, `PLAY_KNIGHT`
    carries no operand, enters `Phase.ROBBER`, and a `MOVE_ROBBER` there
    resumes `MAIN` -- the browser pass (owed) should target this same flow."""
    game = a_game(seed=7)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    session = a_session(game, {0})

    knight = next(a for a in legal_actions(game) if a.type is ActionType.PLAY_KNIGHT)
    assert knight == Action(ActionType.PLAY_KNIGHT)
    session.submit(0, action_to_wire(knight))
    assert game.phase is Phase.ROBBER

    move = next(a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER)
    session.submit(0, action_to_wire(move))
    assert game.phase is Phase.MAIN


def test_only_bank_trading_exists_and_only_in_the_main_phase():
    """Trading with the bank is a Main-phase act; trading with a player is
    not an act at all any more (`hexset.trading`)."""
    from hexset.board.terrain import Resource

    game = a_game(seed=19)
    game.phase = Phase.ROLL
    game.current_player = 0
    game._state.hands[0][Resource.WHEAT] += 6

    session = a_session(game, {0})
    kinds = {a["type"] for a in session.legal_wire_actions(0)}

    assert "PROPOSE_TRADE" not in kinds
    assert "BANK_TRADE" not in kinds
    assert "ROLL" in kinds


def _discard_all(session: GameSession, seat: int) -> None:
    """Run every DISCARD the engine asks `seat` for, one at a time."""
    while (
        session.game.phase is Phase.DISCARD
        and to_move(session.game) == seat
    ):
        action = next(a for a in legal_actions(session.game) if a.type is ActionType.DISCARD)
        session._apply(seat, action)


def _owing_game(seed: int, seat: int, hand: list[int]) -> GameSession:
    """A game parked in Phase.DISCARD with `seat` owing half of `hand`."""
    game = a_game(seed=seed)
    game.phase = Phase.DISCARD
    game.current_player = seat
    game._state.hands[seat] = list(hand)
    game.discard_quota = [0] * game._state.num_players
    game.discard_quota[seat] = sum(hand) // 2
    return game


def test_a_spectators_log_redacts_nothing_a_seats_log_would():
    """The same discard, read three ways. A seat sees its own cards named; a
    seat across the table sees a bare count; somebody watching from outside
    the game sees the cards, because they are outside it (see `render_log`'s
    `omniscient`)."""
    game = _owing_game(seed=22, seat=1, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0, 1})

    _discard_all(session, 1)

    theirs = session.log_for(1)[0]
    across = session.log_for(0)[0]
    watching = session.log_for(None, omniscient=True)[0]

    assert any(r in theirs for r in RESOURCE_NAMES)
    assert "discarded 4 cards" in across
    assert not any(r in across for r in RESOURCE_NAMES)
    assert watching == theirs


def test_state_view_hides_opponent_hands_but_reveals_the_viewers():
    game = a_game(seed=8)
    seat = to_move(game)
    other = (seat + 1) % game._state.num_players
    session = a_session(game, {seat})

    game._state.hands[seat][0] = 3
    game._state.hands[other][0] = 5

    view = session.state_view(seat)
    players = {p["seat"]: p for p in view["players"]}
    assert "hand" in players[seat]
    assert players[seat]["hand"]["Wood"] == 3
    assert "hand" not in players[other]
    assert players[other]["hand_size"] == 5


def test_state_view_reveals_every_hand_once_the_game_is_over():
    game = a_game(seed=9)
    seat = to_move(game)
    session = a_session(game, {seat})
    game.won_by = (seat + 1) % game._state.num_players
    game.phase = Phase.GAME_OVER

    view = session.state_view(seat)
    assert all("hand" in p for p in view["players"])
    assert view["legal_actions"] == []


def test_state_view_carries_the_public_ledger_for_every_seat():
    """Resource *counting* is public knowledge in this game — only a
    steal's identity and dev-card types are hidden (see `hexset.ledger`)
    — so `known`/`unknown` show up for every seat, reveal or not, unlike
    `hand`."""
    game = a_game(seed=8)
    seat = to_move(game)
    other = (seat + 1) % game._state.num_players
    session = a_session(game, {seat})

    game.ledger.receive(other, 0, 2)

    view = session.state_view(seat)
    players = {p["seat"]: p for p in view["players"]}
    assert players[other]["known"]["Wood"] == 2
    assert players[other]["unknown"] == 0
    assert "hand" not in players[other]


def test_state_view_reports_locked_seats():
    from hexset.server.seating import lock_seat

    game = a_game(seed=8)
    lock_seat(game, 2)
    session = a_session(game, {0})

    assert session.state_view(0)["locked"] == [2]


SEED = 42


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """One game played out in full, journalled to its own directory — every
    seat driven the same way, through `submit`, since there is no more
    distinction between "the human seat" and "the bot's seat" at this
    layer (see `botclient.py` for where that distinction now lives, one
    layer up).

    Module-scoped because playing a whole game is by far the slowest thing in
    this file: every test below reads the same finished game rather than
    dealing another one of its own.
    """
    directory = tmp_path_factory.mktemp("games")
    # Two independent random.Random(SEED) instances, matching what
    # `api.build_session` does: the board spends one stream and `start` gets
    # a fresh one, so the game's own rng must begin from the same untouched
    # state here too.
    board = random_base_board(random.Random(SEED))
    game = start_at(board, 4, random.Random(SEED), first=0)
    session = GameSession(
        game=game,
        claimed_seats={0, 1, 2, 3},
        seed=SEED,
        journal=open_journal(SEED, str(directory)),
    )

    driver = RandomBot(rng=random.Random(2))
    steps = 0
    while not is_over(session.game) and steps < 4000:
        seat = to_move(session.game)
        action = driver.choose(session.game)
        session.submit(seat, action_to_wire(action))
        steps += 1
    assert is_over(session.game)
    return session, directory


def journal_events(directory) -> list[dict]:
    """The one per-game journal in `directory`, parsed."""
    files = list(Path(directory).glob("*.jsonl"))
    assert len(files) == 1, f"expected one game journal, found {files}"
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_a_journalled_game_replays_clean(played):
    """The strongest check there is on the journal: fed back through the
    engine, its actions have to be legal in order and end the same game.

    Deliberately goes through `replayable` and `restore` — the same two calls
    `api.resume_session` makes — rather than a replay written for the
    test. A journal that replays clean here is one a returning player would
    actually get their game back from.
    """
    session, directory = played
    events = journal_events(directory)
    header = events[0]
    assert header["seed"] == SEED
    assert header["first"] == 0

    board = random_base_board(random.Random(SEED))
    resumed = GameSession(
        game=start_at(board, header["num_players"], random.Random(SEED), first=header["first"]),
        claimed_seats=set(header["human_seats"]),
        seed=SEED,
    )
    resumed.restore(replayable(events))  # raises ResumeError if it doesn't

    assert resumed.game.won_by == session.game.won_by
    assert resumed.game.turns == session.game.turns


def test_an_undone_placement_is_written_down_not_erased(tmp_path):
    """The journal is append-only and read forwards, so a step number that
    quietly came round twice would leave a reader unable to say which of the
    two actions counted."""
    game = a_game(seed=5)
    seat = to_move(game)
    session = a_session(game, {seat}, journal=open_journal(5, str(tmp_path)))
    settlement = next(
        a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT
    )
    session.submit(seat, action_to_wire(settlement))
    session.undo_last_build(seat)

    events = journal_events(tmp_path)
    assert [e["kind"] for e in events] == ["game", "action", "undo"]
    assert events[1]["type"] == "SETUP_SETTLEMENT"
    assert events[2]["back_to"] == 0  # everything from step 0 did not happen


def test_the_trade_log_rides_in_the_state_view():
    """What the engine cleared this turn (`hexset.trading`) is public and
    not filtered per viewer."""
    from hexset.board.terrain import Resource
    from hexset.game import roll_dice

    game = a_game(seed=13)
    game.phase = Phase.ROLL
    game.current_player = 0
    state = game._state
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[0][Resource.WOOD] = 1
    state.hands[1][Resource.ORE] = 1

    session = a_session(game, {0, 1})
    session.set_trader(0, _Wants(Resource.ORE))
    session.set_trader(1, _Wants(Resource.WOOD))

    roll_dice(game, 8)

    for viewer in (None, 0, 1, 2, 3):
        view = session.state_view(viewer)
        assert len(view["trades"]) == 1
        assert view["trades"][0]["a"] == 0 and view["trades"][0]["b"] == 1
        assert view["trades"][0]["got"][Resource.ORE] == 1
        assert view["trades"][0]["gave"][Resource.WOOD] == 1


def test_bundle_from_wire_is_signed_towards_the_proposer():
    from hexset.board.terrain import Resource

    b = bundle_from_wire({"Wood": 1}, {"Ore": 2})
    assert b[Resource.WOOD] == -1
    assert b[Resource.ORE] == 2


def test_confirm_mode_installs_a_pending_gate():
    game = a_game(seed=3)
    session = a_session(game, {0})
    session.confirm_mode(0)
    assert isinstance(game.gates[0], PendingGate)


def test_pending_gate_always_declines_and_records_the_candidate():
    game = a_game(seed=3)
    gate = PendingGate(game, seat=1)
    gains = gate.gains_many(None, [(1, 0, 0, 0, -1)], [0])
    assert gains == [-1.0]
    assert game.pending == [Trade(1, 0, (1, 0, 0, 0, -1))]


def test_pending_gate_batches_every_candidate_the_actor_priced_above_zero():
    """`_best_clearing` asks the acting seat's own gate once, over every
    coverable candidate, and only asks a counterparty's gate over the
    subset the actor priced above zero -- so a `PendingGate` sitting as the
    counterparty can record several candidates from one event, not only
    the first one another gate happened to accept."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase
    from hexset.trading import trade_event

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[0][Resource.BRICK] = 1
    game._state.hands[1][Resource.ORE] = 1

    class WantsOre:
        """Prices every candidate that hands back some ore above zero --
        several distinct bundles here, since giving wood, brick, or both
        for ore all qualify."""

        def gains_many(self, view, received, counterparties):
            return [1.0 if r[Resource.ORE] > 0 else -1.0 for r in received]

    pending_gate = PendingGate(game, seat=1)
    game.gates = (WantsOre(), pending_gate, None, None)

    trade_event(game, lambda seat, view, received, other: -1.0)
    assert len(game.pending) > 1
    # Recorded from seat 1's own side (`PendingGate.seat`): it is always the
    # one giving up its one ore here.
    assert all(t.a == 1 and t.b == 0 and t.received[Resource.ORE] < 0 for t in game.pending)


def test_pending_gate_records_the_acting_seats_own_gain():
    """A recorded `Trade`'s `gain_a` is the acting seat's own gain,
    recomputed by asking its gate again over the mirrored bundle -- what
    `GameSession.pending_for` sorts by. Skipped (defaults to `0.0`) when no
    `gates` are seated at all -- see `test_pending_gate_always_declines_
    and_records_the_candidate`, unaffected by this."""

    class GivesGain:
        def __init__(self, gain):
            self.gain = gain

        def gains_many(self, view, received, counterparties):
            return [self.gain] * len(received)

    game = a_game(seed=3)
    game.gates = (GivesGain(0.7), None, None, None)
    pending_gate = PendingGate(game, seat=1)

    pending_gate.gains_many(None, [(1, 0, 0, 0, -1)], [0])

    assert game.pending[-1].gain_a == 0.7


def test_pending_gate_never_asks_a_pendinggate_actor_for_its_own_gain():
    """An acting seat whose own gate is *also* a `PendingGate` is never
    asked to estimate its gain -- that would append its own entries to
    `game.pending` as a side effect of merely sorting. (Unreachable through
    a real trade event, since a `PendingGate` actor's own gain is always
    negative and `_best_clearing` never reaches a counterparty for it --
    this pins the guard directly, not by relying on that.)"""
    game = a_game(seed=3)
    actor_gate = PendingGate(game, seat=0)
    game.gates = (actor_gate, None, None, None)
    counterparty_gate = PendingGate(game, seat=1)

    before = len(game.pending)
    counterparty_gate.gains_many(None, [(1, 0, 0, 0, -1)], [0])

    # Exactly one entry recorded (the counterparty's own), the actor's
    # `PendingGate` never itself invoked.
    assert len(game.pending) == before + 1
    assert game.pending[-1].gain_a == 0.0


def test_pending_for_sorts_by_gain_descending_and_caps_at_five():
    """`GameSession.pending_for` is the one place the top-5-by-gain cap
    lives -- both `state_view`'s `pending` block and `api.Tables.
    _pending_of` read it, so a confirm/decline's `index` always counts into
    the same list a viewer was just shown."""
    game = a_game(seed=3)
    session = a_session(game, {1})
    for i in range(7):
        game.pending.append(Trade(1, 0, (i, 0, 0, 0, 0), gain_a=float(i)))
    # A different seat's own entries never leak into this seat's list.
    game.pending.append(Trade(2, 0, (9, 0, 0, 0, 0), gain_a=99.0))

    top = session.pending_for(1)

    assert len(top) == 5
    assert [t.gain_a for t in top] == [6.0, 5.0, 4.0, 3.0, 2.0]


def test_execute_trade_reaches_the_session_and_moves_cards():
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    session = a_session(game, {0, 1})
    session.set_trader(1, _Wants(Resource.WOOD))  # seat 1 wants wood, gives ore
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1
    trade = game.execute_trade(0, 1, tuple(received))
    assert trade.received == tuple(received)
    assert game._state.hands[0][Resource.ORE] == 1
    assert game._state.hands[1][Resource.WOOD] == 1


def test_state_view_pending_is_filtered_per_viewer():
    game = a_game(seed=3)
    game.pending.append(Trade(1, 0, (1, 0, 0, 0, -1)))
    session = a_session(game, {0, 1})
    assert session.state_view(1)["pending"] == [
        {"counterparty": 0, "gave": [0, 0, 0, 0, 1], "got": [1, 0, 0, 0, 0]}
    ]
    assert session.state_view(0)["pending"] == []
    assert session.state_view(None)["pending"] == []


def test_a_trade_the_roll_cleared_is_told_in_the_log_immediately():
    """The turn's first trade event fires eagerly, inside the ROLL action's
    own `apply` (`enter_main`) -- so `_apply`'s own `self.game.trades[
    trades_before:]` bookkeeping already attributes it to that action's
    `_Event`, and the transcript has it without anything else polling the
    table first.
    """
    from hexset.board.terrain import Resource

    # This seed's roll is an 8, so the turn reaches MAIN rather than
    # stopping on the robber -- asserted below rather than left to the
    # seed, so a change to the deal fails here instead of quietly testing
    # nothing.
    game = a_game(seed=2)
    game.phase = Phase.ROLL
    game.current_player = 0
    for hand in game._state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1

    session = a_session(game, {0, 1})
    session.set_trader(0, _Wants(Resource.ORE))
    session.set_trader(1, _Wants(Resource.WOOD))

    session._apply(0, Action(ActionType.ROLL))
    assert game.phase is Phase.MAIN
    assert len(session.events[-1].trades) == 1

    line = next(line for line in session.log_for(None) if " to Player " in line)
    assert "traded" in line and "Ore" in line and "Wood" in line
    # Told once, however many times the table is polled after it.
    session.state_view(None)
    session.state_view(0)
    assert sum(" to Player " in line for line in session.log_for(None)) == 1


def test_a_manually_executed_trade_appears_in_the_log():
    """`POST .../trade`/`.../trade/confirm` (`GameSession.execute_manual_trade`)
    bypasses the automatic event entirely, so there is no board action for
    the trade to ride along with -- it gets its own `_Event` (`action is
    None`) instead, and `render_log` still writes the same `_trade_lines`
    sentence for it."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    session = a_session(game, {0, 1})
    session.set_trader(1, _Wants(Resource.WOOD))
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    trade = session.execute_manual_trade(0, 1, tuple(received))

    assert trade.received == tuple(received)
    assert game._state.hands[0][Resource.ORE] == 1
    assert game._state.hands[1][Resource.WOOD] == 1
    line = next(line for line in session.log_for(None) if " to Player " in line)
    assert "traded" in line and "Wood" in line and "Ore" in line
    # Not folded into any build/discard/bank-trade run, and it clears
    # `can_undo` -- a manual trade moves cards same as a build does, and a
    # stale undo point must not silently erase it too.
    assert session.state_view(0)["can_undo"] is False


def test_a_manually_executed_trade_survives_journal_and_resume(tmp_path):
    """A manual trade (`POST .../trade`, `.../trade/confirm`) is not folded
    into any action's own journal line -- without its own line
    (`Journal.manual_trade`, replayed by `replayable`/`GameSession.restore`
    as an action-less step) a server restart would rebuild hands purely
    from recorded actions and silently forget the cards it moved."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase
    from hexset.server.journal import replayable

    game = a_game(seed=3)
    session = a_session(game, {0, 1}, journal=open_journal(3, str(tmp_path)))
    session.set_trader(1, _Wants(Resource.WOOD))
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    session.execute_manual_trade(0, 1, tuple(received))

    events = journal_events(tmp_path)
    assert any(e.get("kind") == "trade" for e in events)

    resumed_game = a_game(seed=3)
    resumed = a_session(resumed_game, {0, 1})
    resumed_game.phase = Phase.MAIN
    resumed_game.current_player = 0
    resumed_game._state.hands[0][Resource.WOOD] = 1
    resumed_game._state.hands[1][Resource.ORE] = 1

    resumed.restore(replayable(events))

    assert resumed_game._state.hands[0][Resource.ORE] == 1
    assert resumed_game._state.hands[0][Resource.WOOD] == 0
    assert resumed_game._state.hands[1][Resource.WOOD] == 1
    assert resumed_game._state.hands[1][Resource.ORE] == 0
    assert resumed._steps == session._steps == 1
    line = next(line for line in resumed.log_for(None) if " to Player " in line)
    assert "traded" in line
