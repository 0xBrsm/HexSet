# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from .actions import Action, apply, options_for
from .board.board import Board, random_base_board
from .game import UNSTRUCTURED_TURN_CAP, Game, is_over, start

__all__ = [
    "step_randomly",
    "play_random_game",
]


def step_randomly(game: Game, rng: random.Random) -> Action:
    """Apply one legal action chosen uniformly by `rng` and return it. Raises
    `Stuck` when the seat to move has none."""
    options = options_for(game)
    action = rng.choice(options)
    apply(game, action)
    return action


def play_random_game(
    board: Board | None = None,
    num_players: int = 4,
    rng: random.Random | None = None,
    turn_cap: int = UNSTRUCTURED_TURN_CAP,
) -> Game:
    """A game played by nobody in particular, to a finish.

    `turn_cap` is `UNSTRUCTURED_TURN_CAP` rather than the engine's default,
    because that default is read off agents that are trying to win and random
    play takes about four times as long. This function is unstructured play by
    definition, so it asks for the horizon unstructured play needs.
    """
    rng = rng or random.Random()
    board = board if board is not None else random_base_board(rng)
    game = start(board, num_players, rng, turn_cap=turn_cap)
    while not is_over(game):
        step_randomly(game, rng)
    return game
