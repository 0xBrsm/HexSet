# SPDX-License-Identifier: GPL-3.0-only
"""A game that cannot finish is a defect, and is not scored.

A game that reaches `MAX_TURNS` ends without a winner, and `compete` stops
on the first one rather than score it for either side.
"""
import random

import pytest

from hexset.actions import ActionType, options_for
from hexset.arena import Exhausted, compete, lineup_from_names, register_entrant_kind
from hexset.game import MAX_TURNS, start
from hexset.rules import GameType, Rules

from helpers import mini_board


class Idler:
    """Rolls and ends its turn, and does the least it can elsewhere. A table
    of these places its two settlements and then never builds again, so it
    can never reach the win threshold and the game runs to the cap."""

    def choose(self, game):
        options = options_for(game)
        for wanted in (ActionType.ROLL, ActionType.END_TURN):
            for option in options:
                if option.type is wanted:
                    return option
        return options[0]


register_entrant_kind("idler", lambda entrant, board, rng: Idler())


def test_the_cap_is_a_flat_total_of_turns():
    """Summed over every seat: the size of the table does not change it."""
    assert MAX_TURNS == 300
    caps = {start(mini_board(), n, random.Random(0)).turn_cap for n in (2, 3, 4)}
    assert caps == {MAX_TURNS}


def test_a_stuck_run_aborts_rather_than_scoring():
    from hexset.arena import Entrant, register_preset

    register_preset("idler", Entrant("idler", kind="idler"))
    with pytest.raises(Exhausted) as caught:
        compete(lineup_from_names(["idler", "idler"]), 2, seed=11, workers=1)
    assert caught.value.turns >= MAX_TURNS


def test_the_abort_carries_what_it_takes_to_replay():
    """At a run's own, shorter `turn_cap`, which is also what a run measuring
    unstructured play raises it with."""
    from hexset.arena import Entrant, register_preset

    register_preset("idler", Entrant("idler", kind="idler"))
    with pytest.raises(Exhausted) as caught:
        compete(lineup_from_names(["idler", "idler"]), 2, seed=11, workers=1, turn_cap=30)
    failure = caught.value
    assert 30 <= failure.turns < MAX_TURNS
    assert failure.seed == 11
    assert failure.index == 0
    assert len(failure.seating) == 2
    message = str(failure)
    assert "deal_game(11, 0, 2)" in message
    assert "is a defect" in message


def test_a_run_that_finishes_is_untouched():
    """The abort is a guard, not a behaviour change: a run where every game
    ends still returns its tournament, at the default cap. Random bots take
    about four times the turns real play does, so they play to three points
    here rather than to a raised cap."""
    sprint = GameType("sprint", Rules(winning_points=3), (2,))
    tournament = compete(
        lineup_from_names(["random"] * 2), 2, seed=3, workers=1, game_type=sprint
    )
    assert tournament.games == 2
    assert all(winner is not None for winner in tournament.winners)
