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

from hexset.trading import RESPONSE_ACCEPT, RESPONSE_PASS, Trade

from hexset.server.webplay import (
    RESOURCE_NAMES,
    GameSession,
    PendingGate,
    action_to_wire,
    round_bundle_from_wire,
    signed_bundle_from_wire,
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


def test_round_bundle_from_wire_is_signed_towards_the_proposer():
    from hexset.board.terrain import Resource

    b = round_bundle_from_wire([1, 0, 0, 0, 0], [0, 0, 0, 0, 2])
    assert b[Resource.WOOD] == -1
    assert b[Resource.ORE] == 2


def test_round_bundle_from_wire_refuses_a_shared_resource():
    with pytest.raises(ValueError):
        round_bundle_from_wire([1, 0, 0, 0, 0], [1, 0, 0, 0, 0])


def test_round_bundle_from_wire_refuses_more_than_the_cap():
    with pytest.raises(ValueError):
        round_bundle_from_wire([1, 1, 1, 1, 0], [0, 0, 0, 0, 1])


def test_signed_bundle_from_wire_round_trips():
    assert signed_bundle_from_wire([-1, 0, 0, 0, 2]) == (-1, 0, 0, 0, 2)


def test_confirm_mode_installs_a_pending_gate():
    game = a_game(seed=3)
    session = a_session(game, {0})
    session.confirm_mode(0)
    assert isinstance(game.gates[0], PendingGate)


def test_the_trade_round_is_in_the_log_and_a_pass_is_an_answer():
    """A person's broadcast, one bot accepting and one passing: the offer
    and both answers are log lines, the pass stays in the round's
    `responses` (bundle `None`) so the pane can say "Passed", declining
    what was on the table is a line, and the trade the actor finally takes
    is the usual "traded ... to ... for ..." line."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    game._state.hands[2][Resource.SHEEP] = 1
    session = a_session(game, {0}, player_names={0: "Ada"})
    session.confirm_mode(0)
    session.set_trader(1, _Wants(Resource.WOOD))  # accepts: it gets wood
    session.set_trader(2, _Wants(Resource.BRICK))  # nothing in the offer for it
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1
    received = tuple(received)

    session.open_round_for(0, received)

    view = session.state_view(0)["trade_round"]
    assert view["offer"] == {"actor": 0, "bundle": list(received)}
    assert {r["seat"]: r["kind"] for r in view["responses"]} == {1: RESPONSE_ACCEPT, 2: RESPONSE_PASS}
    assert next(r for r in view["responses"] if r["seat"] == 2)["bundle"] is None
    # One line for the round, rewritten as it goes: the offer, then the
    # accept; the pass is in the record and on the pane, not in the line.
    line = session.log_for(None)[-1]
    assert line.endswith("offers 1 Wood for 1 Ore. Player 2 (bot) accepts.")
    assert "passes" not in line and len(session.events) == 3

    session.decline_round(0)
    assert session.log_for(None)[-1].endswith("offers 1 Wood for 1 Ore. Player 2 (bot) accepts. Player 1 (Ada) declines.")
    assert session.open_round is None

    session.open_round_for(0, received)
    session.execute_round_choice(0, 1, received)
    line = session.log_for(None)[-1]
    assert line.endswith("accepts. Player 1 (Ada) traded 1 Wood to Player 2 (bot) for 1 Ore.")
    assert game._state.hands[0][Resource.ORE] == 1 and game._state.hands[1][Resource.WOOD] == 1


def test_a_round_everyone_passes_on_reads_everyone_declines_at_once():
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    session = a_session(game, {0}, player_names={0: "Ada"})
    session.confirm_mode(0)
    for seat in (1, 2, 3):
        session.set_trader(seat, _Wants(Resource.BRICK))
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    session.open_round_for(0, tuple(received))

    assert session.log_for(None) == ["1\tPlayer 1 (Ada) offers 1 Wood for 1 Ore. Everyone declines."]
    assert [e.note.kind for e in session.events] == ["offer", "pass", "pass", "pass", "nobody"]
    session.decline_round(0)  # nothing was on the table: nothing more to say
    assert len(session.log_for(None)) == 1


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
    game.pending.append(Trade(0, 1, (1, 0, 0, 0, -1)))  # seat 0 broadcast, standing against seat 1
    session = a_session(game, {0, 1})
    assert session.state_view(1)["pending"] == [{"actor": 0, "bundle": [1, 0, 0, 0, -1]}]
    assert session.state_view(0)["pending"] == []
    assert session.state_view(None)["pending"] == []


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
    session.confirm_mode(0)  # a person: its pick is its consent, no gate asked
    session.set_trader(1, _Wants(Resource.WOOD))
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    trade = session._execute_round_trade(0, 1, tuple(received))

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
    session.confirm_mode(0)
    session.set_trader(1, _Wants(Resource.WOOD))
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    session._execute_round_trade(0, 1, tuple(received))

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


# --- a seven's discards are simultaneous, and the log says so ------------------


def _two_owing(seed: int = 11) -> GameSession:
    """A session parked in `Phase.DISCARD` with seats 0 and 3 each owing two
    cards and seat 1, who rolled the seven, owing none."""
    from hexset.board.terrain import NUM_RESOURCES, Resource

    game = a_game(seed=seed)
    game.phase = Phase.DISCARD
    game.current_player = 1
    for hand in game._state.hands:
        hand[:] = [0] * NUM_RESOURCES
    game._state.hands[0][Resource.WOOD] = 4
    game._state.hands[3][Resource.ORE] = 4
    game.discard_quota = [2, 0, 0, 2]
    return a_session(game, {0, 1, 2, 3})


def _discards(session: GameSession, seat: int | None, **kwargs) -> list[str]:
    return [line for line in session.log_for(seat, **kwargs) if "discard" in line]


def _discard_wire(resource) -> dict:
    return {"type": "DISCARD", "a": int(resource)}


def test_no_discard_is_logged_until_the_whole_round_has_resolved():
    """A seven's discards happen at once, so reporting seat 3's the moment it
    lands both tells a sequence that never happened and shows the table half a
    round while the other half is still choosing. Nothing is said -- to a
    seat, to the seat itself, or to a spectator -- until nobody still owes."""
    from hexset.board.terrain import Resource

    session = _two_owing()

    session.submit(3, _discard_wire(Resource.ORE))
    session.submit(3, _discard_wire(Resource.ORE))
    # Seat 3 is done and owes nothing; seat 0 has not started. The round is
    # still open, so it is still nobody's business.
    assert session.game.discard_quota == [2, 0, 0, 0]
    assert _discards(session, 3) == []
    assert _discards(session, 0) == []
    assert _discards(session, None) == []
    assert _discards(session, None, omniscient=True) == []

    session.submit(0, _discard_wire(Resource.WOOD))
    assert _discards(session, 0) == []  # one card short, still open

    session.submit(0, _discard_wire(Resource.WOOD))

    lines = _discards(session, None, omniscient=True)
    assert len(lines) == 2
    # Seat order, not submission order: seat 3 finished first and is second.
    assert "Player 1 " in lines[0] and "Player 4 " in lines[1]


def test_the_revealed_round_is_still_redacted_per_reader():
    """Holding the round back changes when the lines appear, not what each
    reader is allowed to see in them (see `render_log`'s `omniscient`)."""
    from hexset.board.terrain import Resource

    session = _two_owing()
    for seat, resource in ((3, Resource.ORE), (0, Resource.WOOD)) * 2:
        session.submit(seat, _discard_wire(resource))

    mine = next(line for line in _discards(session, 3) if "Player 4" in line)
    across = next(line for line in _discards(session, 0) if "Player 4" in line)

    assert "Ore" in mine
    assert "discarded 2 cards" in across and not any(r in across for r in RESOURCE_NAMES)


def test_a_round_closed_by_a_locked_seat_still_reveals_the_rest():
    """A round can end without a discard: `lock_seat` zeroes a retired seat's
    quota. The reveal follows the round, so what was already given up is
    reported rather than sitting unwritten until the next action."""
    from hexset.board.terrain import Resource
    from hexset.game import lock_seat

    session = _two_owing()
    session.submit(3, _discard_wire(Resource.ORE))
    session.submit(3, _discard_wire(Resource.ORE))
    assert _discards(session, None, omniscient=True) == []

    lock_seat(session.game, 0)

    # Seat 3's line only: seat 0 retired owing two and never gave up a card.
    # (`_who` numbers among the seats still in the game, so seat 3 reads as
    # "Player 3" once seat 0 is gone -- that renumbering is `SeatLabels`' and
    # is not what this test is about.)
    lines = _discards(session, None, omniscient=True)
    assert len(lines) == 1 and "2 Ore" in lines[0]


def test_a_second_seven_starts_a_fresh_discard_line():
    """The run this replaced could not reach back across an intervening line,
    and neither can the held-back round: the robber move between two sevens
    closes the first, so the second is its own set of lines."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    session = _two_owing()
    for seat, resource in ((3, Resource.ORE), (0, Resource.WOOD)) * 2:
        session.submit(seat, _discard_wire(resource))
    assert session.game.phase is Phase.ROBBER
    assert len(_discards(session, None, omniscient=True)) == 2

    robber = next(a for a in legal_actions(session.game) if a.type is ActionType.MOVE_ROBBER)
    session._apply(1, robber)

    session.game.phase = Phase.DISCARD
    session.game.discard_quota = [0, 0, 0, 2]
    session.submit(3, _discard_wire(Resource.ORE))
    session.submit(3, _discard_wire(Resource.ORE))

    lines = _discards(session, None, omniscient=True)
    assert len(lines) == 3  # two from the first seven, one from the second
    assert "discarded 2" in lines[2]
