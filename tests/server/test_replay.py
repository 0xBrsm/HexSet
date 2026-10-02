# SPDX-License-Identifier: GPL-3.0-only
"""Stepping back through a game -- `journal.replayable_rounds`,
`api.replay_session` and `/api/table/<code>/replay`.

Round N is the position at the **end** of round N, so a round reads as what
happened *in* it. And reading cannot write: a reader builds a throwaway
session off the journal.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, legal_actions

from hexset.server import _journal as journal
from hexset.server.api import (
    ApiError,
    Config,
    Seat,
    SeatKind,
    build_session,
    replay_session,
    reopened_seats,
    round_query,
)

from hexset.game import is_over, to_move

from hexset.server.wire import action_to_wire

from conftest import new_tables

SOLO = ["test-trader", "test-trader", "test-trader"]


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


def player(name: str | None = None) -> Seat:
    return Seat(kind=SeatKind.PLAYER, name=name, token="t-" + (name or "x"))


def bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def drive(session, moves: int, rng: random.Random) -> None:
    for _ in range(moves):
        if is_over(session.game):
            break
        if session.awaiting_confirm is not None:
            session.submit(
                session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN))
            )
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))


def journalled(tmp_path, moves: int = 60, seed: int = 4):
    config = Config(games_dir=str(tmp_path), seed=99)
    seats = [player("Ada"), bot_seat(), bot_seat(), bot_seat()]
    session = build_session("ABC123", seats, config, first=0)
    drive(session, moves, random.Random(seed))
    events = journal.read(next(Path(tmp_path).glob("*.jsonl")))
    return session, events


def at_round(events, wanted: int):
    steps, rounds = journal.replayable_rounds(events)
    upto = next((i for i, played in enumerate(rounds) if played > wanted), len(steps))
    return replay_session("ABC123", reopened_seats(events), events, upto)


def test_an_undo_takes_back_the_round_with_the_step(tmp_path):
    _, events = journalled(tmp_path, moves=40)
    steps, rounds = journal.replayable_rounds(events)

    undone = events + [{"kind": "undo", "back_to": 5}]
    cut_steps, cut_rounds = journal.replayable_rounds(undone)

    assert len(cut_steps) == len(cut_rounds) == 5
    assert cut_steps == steps[:5]
    assert cut_rounds == rounds[:5]


def test_a_replay_session_is_never_given_a_journal(tmp_path):
    _, events = journalled(tmp_path)

    replayed = at_round(events, 1)

    assert getattr(replayed, "journal", None) is None


def test_the_route_serves_a_past_round_and_says_where_it_is(tmp_path):
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code = dealt["code"]
    table = registry.get(code)
    drive(table.session, 40, random.Random(4))

    view = registry.handle("GET", f"/api/table/{code}/replay?round=1", {}, None)

    assert view["replay"]["round"] == 1
    assert view["replay"]["last_round"] >= 1
    assert all("hand" in p for p in view["players"])
    live = registry.handle("GET", f"/api/table/{code}", {}, None)
    assert set(live) - set(view) == set(), "replay is missing a field the live view has"
    assert view["code"] == code
    assert len(view["seats"]) == len(live["seats"])


def test_a_seat_reading_a_past_round_still_sees_only_its_own_hand(tmp_path):
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    mine = registry.by_token(token)[1]
    drive(registry.get(code).session, 40, random.Random(4))

    live = registry.handle("GET", "/api/state", {}, token)
    past = registry.handle("GET", f"/api/table/{code}/replay?round=1", {}, token)

    assert {p["seat"] for p in live["players"] if "hand" in p} == {mine}, "live: own hand only"
    assert {p["seat"] for p in past["players"] if "hand" in p} == {mine}, "replayed: the same"
    assert {p["seat"] for p in past["players"] if "dev_cards" in p} == {mine}
    assert past["seat"] == mine, "and it is still read as that seat's view"


def test_every_round_of_a_finished_game_is_read_in_full(tmp_path):
    """Disclosure follows the *game*, not the round -- and the round cannot tell,
    since a stand-in stopped mid-game reads `is_over` false however the game
    ended, so the journal's `result` line decides it. That line is written
    here after a short game rather than played out to, since it is the whole
    of what the read consults.
    """
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    mine = registry.by_token(token)[1]
    session = registry.get(code).session
    drive(session, 60, random.Random(4))
    session.journal.finish(session.game)

    last = registry.handle(
        "GET", f"/api/table/{code}/replay?round=99999", {}, token
    )["replay"]["last_round"]
    for wanted in (0, last // 2, last):
        past = registry.handle("GET", f"/api/table/{code}/replay?round={wanted}", {}, token)
        seats = {p["seat"] for p in past["players"] if "hand" in p}
        assert seats == {0, 1, 2, 3}, f"round {wanted} held a hand back: {seats}"
        assert past["replay"]["finished"] is True, f"round {wanted} did not say so"
        assert past["seat"] == mine


def test_a_replayed_round_offers_no_move(tmp_path):
    """The page wires a click to every `legal_actions` entry, which POSTs to the
    live table at a later position. The second half stops this passing
    vacuously: the stand-in genuinely would have offered something.
    """
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    mine = registry.by_token(token)[1]
    drive(registry.get(code).session, 40, random.Random(4))

    last = registry.handle(
        "GET", f"/api/table/{code}/replay?round=99999", {}, token
    )["replay"]["last_round"]
    for wanted in range(last + 1):
        past = registry.handle("GET", f"/api/table/{code}/replay?round={wanted}", {}, token)
        assert past["legal_actions"] == [], f"round {wanted} offered a move"
        assert past["can_undo"] is False, f"round {wanted} offered an undo"

    events = journal.read(next(Path(tmp_path).glob("*.jsonl")))
    seats = reopened_seats(events)
    steps, rounds = journal.replayable_rounds(events)
    would_have = False
    for wanted in range(last + 1):
        upto = next((i for i, played in enumerate(rounds) if played > wanted), len(steps))
        standin = replay_session(code, seats, events, upto)
        if standin.legal_wire_actions(mine):
            would_have = True
            break
    assert would_have, "no round this seat held: the emptying above proves nothing"


def test_a_replayed_round_is_tagged_the_round_its_own_log_lines_are(tmp_path):
    """The current-round pane filters `log` on `round`. The trap: round N's cut
    falls after its last END_TURN, which ticks the session into N+1.
    """
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    drive(registry.get(code).session, 150, random.Random(4))

    last = registry.handle(
        "GET", f"/api/table/{code}/replay?round=99999", {}, token
    )["replay"]["last_round"]
    assert last >= 2, "fixture played too few rounds to be worth walking"
    for wanted in range(last + 1):
        past = registry.handle("GET", f"/api/table/{code}/replay?round={wanted}", {}, token)
        assert past["round"] == wanted, f"round {wanted} reported as {past['round']}"
        tagged = [line for line in past["log"] if line.split("\t")[0] == str(wanted)]
        assert tagged, f"round {wanted} would render an empty current-round pane"


def test_a_token_for_another_table_reads_as_a_spectator(tmp_path):
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    mine = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    theirs = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    drive(registry.get(theirs["code"]).session, 30, random.Random(4))

    past = registry.handle(
        "GET", f"/api/table/{theirs['code']}/replay?round=1", {}, mine["token"]
    )

    assert all("hand" in p for p in past["players"]), "a stranger's token is no seat here"
    assert past.get("seat") is None


def test_a_negative_round_is_the_beginning(tmp_path):
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    code = registry.handle("POST", "/api/games", {"bots": SOLO}, None)["code"]
    drive(registry.get(code).session, 20, random.Random(4))

    view = registry.handle("GET", f"/api/table/{code}/replay?round=-5", {}, None)

    assert view["replay"]["round"] == 0


def test_a_game_whose_first_seat_closed_still_replays(tmp_path, monkeypatch):
    """`first` is recorded as seat 0 when a game is dealt, and closing seat 0
    retires it: a rebuild leaving `current_player` there has seat 0 to move
    while the record's first step belongs to seat 1.
    """
    # Seat 0 is then an empty seat the creator may close, while the header
    # still names it as the seat to move first.
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 1)
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    # No bots: seating one spawns a runner thread that races `drive` below.
    dealt = registry.handle("POST", "/api/games", {"bots": []}, None)
    code, token = dealt["code"], dealt["token"]
    for seat in (0, 2, 3):
        registry.handle("POST", "/api/close", {"seat": seat}, token)
    drive(registry.get(code).session, 30, random.Random(4))

    events = journal.read(next(Path(tmp_path).glob("*.jsonl")))
    assert events[0].get("first", 0) == 0, "the header still names seat 0 first"
    assert 0 in journal.locked_seats(events), "and seat 0 is the one that closed"
    steps, _ = journal.replayable_rounds(events)
    assert steps[0][0] != 0, "so the seat that actually moved first is not 0"

    replayed = replay_session(code, reopened_seats(events), events, len(steps))
    assert replayed is not None
    assert replayed.game.turns == registry.get(code).session.game.turns

    view = registry.handle("GET", f"/api/table/{code}/replay?round=1", {}, None)
    assert view["replay"]["round"] == 1


def test_a_replay_read_has_to_name_a_round():
    with pytest.raises(ApiError):
        round_query("")
    with pytest.raises(ApiError):
        round_query("round=soon")
    assert round_query("round=3") == 3


# --- a record this server did not deal (`journal.journal_of`) -----------------


def _recorded(num_players: int = 4, seed: int = 4, **options):
    from hexset.board.board import random_base_board
    from hexset.bots import RandomBot
    from hexset.game import UNSTRUCTURED_TURN_CAP
    from hexset.record import record_game

    board = random_base_board(random.Random(seed))
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(num_players)]
    return record_game(bots, board, seed, turn_cap=UNSTRUCTURED_TURN_CAP,
                       trade_mode="external", **options)


def test_a_journalled_record_reads_back_as_the_same_record(tmp_path):
    from dataclasses import replace

    from hexset.record import from_journal, replay

    record = _recorded()
    path = journal.journal_of(record, str(tmp_path), code="abc234", names={0: "Ada"})

    header = journal.read(path)[0]
    assert header["seed"] is None and header["code"] == "abc234"
    back = from_journal(path)
    # A journal names every discard's seat and says nothing of how the dice
    # were drawn, both of which a record may leave to their defaults.
    assert replace(back, actors=(), balanced_dice=record.balanced_dice) == replace(
        record, seed=None, actors=()
    )
    assert (replay(back).won_by, replay(back).turns) == (record.winner, record.turns)


def test_the_route_steps_through_a_journalled_record_to_its_end(tmp_path):
    from hexset.record import replay
    from hexset.server.wire import RESOURCE_NAMES

    record = _recorded()
    journal.journal_of(record, str(tmp_path), code="abc234")
    registry = new_tables(games_dir=str(tmp_path))

    first = registry.handle("GET", "/api/table/abc234/replay?round=0", {}, None)
    last = registry.handle("GET", "/api/table/abc234/replay?round=9999", {}, None)

    assert first["replay"]["round"] == 0 and last["replay"]["finished"]
    ended = replay(record).state(0, hidden=False)
    hands = [[p["hand"][name] for name in RESOURCE_NAMES] for p in last["players"]]
    assert hands == ended.hands
    # Opened from the journal as a table, the game is over and nobody moves.
    table = registry.handle("GET", "/api/table/abc234", {}, None)
    assert table["to_move"] is None and table["legal_actions"] == []


def test_a_two_seat_record_under_other_rules_keeps_both(tmp_path):
    from hexset.rules import DUEL_VARIANT, DUEL_VARIANT_GAME

    record = _recorded(num_players=2, seed=7, game_type=DUEL_VARIANT_GAME)
    journal.journal_of(record, str(tmp_path), code="abc234")
    events = journal.read(next(Path(tmp_path).glob("*.jsonl")))

    steps, _ = journal.replayable_rounds(events)
    replayed = replay_session("abc234", reopened_seats(events), events, len(steps))

    assert replayed.game.num_players == 2
    assert replayed.game.state(0, hidden=False).rules == DUEL_VARIANT
    assert (replayed.game.won_by, replayed.game.turns) == (record.winner, record.turns)


def test_a_record_with_an_unnamed_card_is_refused(tmp_path):
    """A replay never reads a deck slot below the last card bought, but the
    header writes every one."""
    from dataclasses import replace

    record = _recorded()
    unnamed = replace(record, chance=(("deck", -1),) + record.chance[1:])

    with pytest.raises(ValueError, match="not one"):
        journal.journal_of(unnamed, str(tmp_path), code="abc234")


def _with_a_round(record):
    """`record` with one exchange and its reverse cleared after a roll, so
    every later action is as legal as before, and the round around them:
    `(record, notes, a, b, c)`."""
    from dataclasses import replace

    from hexset.game import Phase
    from hexset.record import advance, moves, open_record
    from hexset.server._webplay import RoundNote

    game = open_record(record)
    for step, (actor, action, trades) in enumerate(moves(record)):
        seat = to_move(game) if actor is None else actor
        advance(game, action, trades, seat)
        if action.type is not ActionType.ROLL or game.phase is not Phase.MAIN:
            continue
        hands = game.state(0, hidden=False).hands
        for b in range(record.num_players):
            give = next((r for r in range(5) if hands[seat][r]), None)
            get = next((r for r in range(5) if hands[b][r] and r != give), None)
            if b != seat and give is not None and get is not None:
                bundle = tuple((r == get) - (r == give) for r in range(5))
                back = tuple(-n for n in bundle)
                c = next(s for s in range(record.num_players) if s not in (seat, b))
                notes = [
                    (step, 0, RoundNote("offer", seat, seat, bundle)),
                    (step, 0, RoundNote("pass", c, seat, None)),
                    (step, 0, RoundNote("accept", b, seat, bundle)),
                    (step, 1, RoundNote("offer", seat, seat, back)),
                    (step, 1, RoundNote("accept", b, seat, back)),
                ]
                traded = replace(record, trades=((step, seat, b, bundle), (step, seat, b, back)))
                return traded, notes, seat, b, c
    raise AssertionError("no roll left two seats able to swap a card")


def test_a_journalled_round_reads_offer_answers_and_trade_in_order(tmp_path):
    """A record's trade rounds go in where they happened: after the roll, the
    offer, a seat accepting it, the exchange -- a pass unwritten, as at a
    table here -- and a second offer on a line of its own."""
    from dataclasses import replace

    from hexset.record import from_journal

    record, notes, a, b, c = _with_a_round(_recorded())
    path = journal.journal_of(record, str(tmp_path), code="abc234", notes=notes)
    registry = new_tables(games_dir=str(tmp_path))
    log = [line.split("\t", 1)[1] for line in
           registry.handle("GET", "/api/table/abc234/replay?round=9999", {}, None)["log"]]

    first = next(i for i, line in enumerate(log) if " offers " in line)
    # The round is one line, and the exchange that closes it ends it.
    line, again = log[first], log[first + 1]
    assert (line.index(" offers ") < line.index(" accepts.")
            < line.index(f" Traded with Player {b + 1} ")) and "declines" not in line
    assert again.index(" offers ") < again.index(" accepts.") < again.index(" Traded with ")
    assert replace(from_journal(path), actors=()) == replace(record, seed=None, actors=())

