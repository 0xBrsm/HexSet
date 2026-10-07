# SPDX-License-Identifier: GPL-3.0-only
"""A dealt game's dice and steals are fixed by its seed, whatever is played.

Before 1.10.0 one generator served the deck, the dice and the steals, so a
single different move -- a steal taken or not -- shifted every roll after it,
and two players compared on one seed met different dice from that move on.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import pytest

import hexset.game
import hexset.record
from hexset.actions import Action, ActionType, legal_actions
from hexset.board.board import random_base_board
from hexset.bots import RandomBot
from hexset.chance import Balanced, Live, Recording, for_rules
from hexset.game import imagine, is_over, start, to_move
from hexset.record import ReplayError, from_json, record_game, replay, to_json
from hexset.rules import DUEL_VARIANT, STANDARD_GAME

ACTION_CAP = 600


def rolls(record):
    return [value for kind, value in record.chance if kind == "roll"]


def standard_record(seed: int, bot_seed: int):
    board = random_base_board(random.Random(seed))
    bots = [RandomBot(random.Random(bot_seed * 10 + s)) for s in range(4)]
    # RandomBot declines every offer: skipping the trade rounds plays the same game.
    return record_game(bots, board, seed, game_type=STANDARD_GAME,
                       action_cap=ACTION_CAP, trade_mode="external")


def test_a_seed_rolls_the_same_dice_turn_for_turn_however_it_is_played():
    one, other = standard_record(11, bot_seed=1), standard_record(11, bot_seed=2)
    assert one.actions[:16] != other.actions[:16], "the two games must be played differently"
    a, b = rolls(one), rolls(other)
    n = min(len(a), len(b))
    assert n > 30
    assert a[:n] == b[:n]


def test_steals_do_not_move_the_dice():
    """Unsplit, a steal between two rolls shifts the second; split, it does not."""

    def dice(split: bool, steal_between: bool) -> list[int]:
        source = Live(random.Random(3), split=split)
        out = []
        for _ in range(40):
            out.append(source.roll())
            if steal_between:
                source.steal([2, 0, 5, 1, 0])
        return out

    assert dice(True, False) == dice(True, True)
    assert dice(False, False) != dice(False, True)


def test_the_kth_steal_reads_the_kth_draw_whatever_the_hands_before_it():
    a, b = Live(random.Random(5), split=True), Live(random.Random(5), split=True)
    for hand_a, hand_b in [([3, 0, 0, 0, 0], [0, 9, 9, 9, 9]), ([1, 1, 1, 1, 1], [40, 0, 0, 0, 2])]:
        a.steal(hand_a)
        b.steal(hand_b)
    hand = [1, 2, 3, 4, 5]
    assert [a.steal(hand) for _ in range(20)] == [b.steal(hand) for _ in range(20)]


def test_a_split_steal_takes_in_proportion_to_the_hand():
    source = Live(random.Random(9), split=True)
    taken = [source.steal([1, 0, 3, 0, 0]) for _ in range(20_000)]
    assert set(taken) == {0, 2}
    assert taken.count(2) / len(taken) == pytest.approx(0.75, abs=0.015)


def test_a_balanced_deck_is_fixed_by_the_seed_too():
    def dice(steal_between: bool) -> list[int]:
        source = Balanced(random.Random(4), split=True)
        out = []
        for turn in range(80):
            out.append(source.roll_by(turn % 2))
            if steal_between:
                source.steal([1, 1, 1, 1, 1])
        return out

    assert dice(False) == dice(True)


def test_sources_built_in_turn_on_one_generator_roll_different_dice():
    rng = random.Random(0)
    one, other = Live(rng, split=True), Live(rng, split=True)
    assert [one.roll() for _ in range(30)] != [other.roll() for _ in range(30)]


def test_split_dice_are_two_fair_dice():
    source = Live(random.Random(2), split=True)
    n = 72_000
    counts = [0] * 13
    for _ in range(n):
        counts[source.roll()] += 1
    for total in range(2, 13):
        expected = (6 - abs(total - 7)) / 36
        assert counts[total] / n == pytest.approx(expected, abs=0.006), total


def test_every_dealt_game_is_split_and_a_search_copy_is_not():
    game = start(random_base_board(random.Random(1)), 4, random.Random(1))
    assert game.chance.split
    assert type(for_rules(DUEL_VARIANT, random.Random(0))) is Balanced
    assert for_rules(DUEL_VARIANT, random.Random(0)).split
    assert not imagine(game, random.Random(2)).chance.split


def test_a_split_record_says_so_and_replays_under_its_seed():
    record = standard_record(12, bot_seed=1)
    assert record.split_streams
    game = replay(from_json(to_json(record)))
    assert (game.won_by, game.turns) == (record.winner, record.turns)


def test_the_seed_check_reads_the_split_flag():
    record = standard_record(12, bot_seed=1)
    with pytest.raises(ReplayError):
        replay(dataclasses.replace(record, split_streams=False))


def test_a_record_without_the_flag_reads_as_one_shared_stream():
    """Every record written before the flag drew from one shared stream."""
    raw = json.loads(to_json(standard_record(12, bot_seed=1)))
    del raw["split_streams"]
    assert from_json(json.dumps(raw)).split_streams is False


def test_a_shared_stream_record_still_replays_under_its_seed(monkeypatch):
    monkeypatch.setattr(
        hexset.record, "recording",
        lambda rng, rules: Recording(for_rules(rules, rng, split=False)),
    )
    record = standard_record(13, bot_seed=1)
    assert not record.split_streams
    game = replay(from_json(to_json(record)))
    assert (game.won_by, game.turns) == (record.winner, record.turns)


def test_a_journal_from_before_the_split_reopens_on_its_own_dice(tmp_path, monkeypatch):
    """A served game dealt on the shared stream, journalled with no flag, is
    rebuilt from its seed on the same dice it was played on."""
    from hexset.server import _journal as journal
    from hexset.server.api import Config, Seat, SeatKind, build_session, reopened_seats, replay_session
    from hexset.server.wire import action_to_wire

    with monkeypatch.context() as patch:
        patch.setattr(hexset.game, "chance_for", lambda rules, rng: for_rules(rules, rng, split=False))
        seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada")] + [
            Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader") for _ in range(3)
        ]
        session = build_session("ABC123", seats, Config(games_dir=str(tmp_path), seed=99), first=0)
        rng = random.Random(4)
        for _ in range(120):
            if is_over(session.game):
                break
            if session.awaiting_confirm is not None:
                session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
                continue
            seat = to_move(session.game)
            session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))

    path = next(Path(tmp_path).glob("*.jsonl"))
    events = journal.read(path)
    assert events[0]["split_streams"] is False
    del events[0]["split_streams"]
    steps, _ = journal.replayable_rounds(events)
    assert session.game.turns > 3, "the game must have rolled"

    reopened = replay_session("ABC123", reopened_seats(events), events, len(steps))

    live, back = session.game.state(0, hidden=False), reopened.game.state(0, hidden=False)
    assert (reopened.game.turns, reopened.game.last_roll) == (session.game.turns, session.game.last_roll)
    assert back.hands == live.hands
