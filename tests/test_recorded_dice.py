# SPDX-License-Identifier: GPL-3.0-only
"""A recorded game rolls the dice its rules ask for.

`record.recording` used to wrap independent dice whatever the rules said, so
every recorded `balanced_dice` game -- `bench.duel --records`,
`bench.generate` -- rolled a different game from the unrecorded one, under a
record that named the dice deck.
"""
from __future__ import annotations

import dataclasses
import json
import random

import pytest

from hexset.arena import _play_one, lineup_from_names
from hexset.board.board import random_base_board
from hexset.bots import RandomBot
from hexset.chance import Balanced, Live
from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.record import ReplayError, from_json, record_game, recording, replay, to_json
from hexset.rules import DUEL_VARIANT, DUEL_VARIANT_GAME, STANDARD

# Enough actions for many rolls of the deck; a game need not finish to show
# which dice it rolled.
ACTION_CAP = 150


def test_recording_wraps_the_source_the_rules_ask_for():
    assert isinstance(recording(random.Random(0), DUEL_VARIANT).inner, Balanced)
    assert type(recording(random.Random(0), STANDARD).inner) is Live


def test_a_recorded_game_is_the_unrecorded_game():
    """The arena's recorded path plays exactly what `play_game` plays: the
    balanced dice are the ruleset whose recorded path once rolled Live ones."""
    entrants = tuple(lineup_from_names(["random", "retired", "random", "retired"]))

    def outcome(records: bool):
        o = _play_one((entrants, 1, 5100, ACTION_CAP, True, records, "round",
                       DUEL_VARIANT_GAME, UNSTRUCTURED_TURN_CAP))
        return o.winner, o.turns, o.points, o.seating

    assert outcome(records=True) == outcome(records=False)


@pytest.fixture(scope="module")
def duel_record():
    seed = 7
    board = random_base_board(random.Random(seed))
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(2)]
    # RandomBot declines every offer: skipping the trade rounds plays the same game.
    return record_game(bots, board, seed, game_type=DUEL_VARIANT_GAME,
                       action_cap=ACTION_CAP, trade_mode="external")


def test_a_balanced_record_says_so_and_replays_under_its_seed(duel_record):
    record = duel_record
    assert record.rules.balanced_dice
    assert record.balanced_dice
    game = replay(from_json(to_json(record)))
    assert game.won_by == record.winner
    assert game.turns == record.turns


def test_the_seed_check_reads_the_stream_flag_not_the_rules(duel_record):
    """A record claiming independent dice for a deck stream fails the seed
    cross-check: the flag is what regenerates the draws."""
    with pytest.raises(ReplayError):
        replay(dataclasses.replace(duel_record, balanced_dice=False))


def test_a_record_without_the_flag_reads_as_independent_dice(duel_record):
    """Every record written before the flag held independent dice."""
    raw = json.loads(to_json(duel_record))
    del raw["balanced_dice"]
    assert from_json(json.dumps(raw)).balanced_dice is False
