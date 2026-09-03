# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType
from hexset.board.board import random_base_board
from hexset.bots import RandomBot, greedy
from hexset.bots.evaluate import Evaluator
from hexset.game import is_over
from hexset.record import (
    Record,
    ReplayError,
    board_of,
    from_json,
    read,
    record_game,
    replay,
    to_json,
    write,
)


def a_record(seed: int = 0, bot: str = "random") -> Record:
    rng = random.Random(seed)
    board = random_base_board(rng)
    if bot == "greedy":
        bots = [greedy(Evaluator(board), random.Random(seed * 10 + s)) for s in range(4)]
    else:
        bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    return record_game(bots, board, seed)


def test_a_recorded_game_finished_and_kept_its_actions():
    record = a_record()
    assert record.decided
    assert record.turns > 0
    assert len(record.actions) > record.turns
    assert record.num_players == 4


def test_the_board_survives_the_round_trip_through_a_record():
    rng = random.Random(3)
    board = random_base_board(rng)
    rebuilt = board_of(record_game([RandomBot(random.Random(1))] * 4, board, 3))

    assert rebuilt.terrain == board.terrain
    assert rebuilt.tokens == board.tokens
    assert rebuilt.topology == board.topology
    assert rebuilt.hexes_by_roll == board.hexes_by_roll
    assert rebuilt.ports == board.ports


def test_replaying_reproduces_the_game_that_was_recorded():
    record = a_record(seed=5)
    game = replay(record)
    assert is_over(game)
    assert game.won_by == record.winner
    assert game.turns == record.turns


@pytest.mark.parametrize("bot", ["random", "greedy"])
def test_replay_holds_for_either_bot(bot):
    replay(a_record(seed=7, bot=bot))


def test_a_record_survives_json():
    record = a_record(seed=2)
    assert from_json(to_json(record)) == record


def test_json_records_replay_as_well_as_the_originals():
    record = a_record(seed=4)
    replay(from_json(to_json(record)))


def test_a_tampered_action_is_caught_rather_than_replayed():
    record = a_record(seed=6)
    actions = list(record.actions)
    # An end-of-turn where setup expects a settlement can never be legal.
    actions[0] = (int(ActionType.END_TURN), 0, 0)
    with pytest.raises(ReplayError, match="not legal"):
        replay(Record(**{**record.__dict__, "actions": tuple(actions)}))


def test_a_tampered_outcome_is_caught():
    record = a_record(seed=6)
    with pytest.raises(ReplayError, match="record says"):
        replay(Record(**{**record.__dict__, "turns": record.turns + 1}))


def test_the_action_cap_bounds_a_record():
    rng = random.Random(0)
    board = random_base_board(rng)
    record = record_game([RandomBot(random.Random(1))] * 4, board, 0, action_cap=10)
    assert len(record.actions) == 10
    assert record.winner is None
    assert not record.decided


def test_records_round_trip_through_a_file(tmp_path):
    records = [a_record(seed=s) for s in (1, 2)]
    path = str(tmp_path / "games.jsonl")

    assert write(path, records) == 2
    assert list(read(path)) == records
    # Appending, not truncating: a generator run should be resumable.
    assert write(path, records[:1]) == 1
    assert len(list(read(path))) == 3


def test_actions_are_stored_as_plain_triples_not_enums():
    """So a record does not depend on the action space's current numbering."""
    record = a_record()
    first = record.actions[0]
    assert isinstance(first, tuple) and len(first) == 3
    assert all(type(part) is int for part in first)
    assert Action(ActionType(first[0]), first[1], first[2]).type is ActionType.SETUP_SETTLEMENT
