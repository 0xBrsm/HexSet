# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from .actions import Action, apply, options_for
from .board.board import Board, random_base_board
from .game import Game, Phase, is_over, start


def step_randomly(game: Game, rng: random.Random) -> Action:
    options = options_for(game)
    action = rng.choice(options)
    apply(game, action)
    return action


def play_random_game(
    board: Board | None = None,
    num_players: int = 4,
    rng: random.Random | None = None,
) -> Game:
    rng = rng or random.Random()
    board = board if board is not None else random_base_board(rng)
    game = start(board, num_players, rng)
    while not is_over(game):
        step_randomly(game, rng)
    return game


def summarise(game: Game) -> dict[str, object]:
    from .victory import victory_points

    # true state: the summary's victory points include hidden VP dev cards.
    state = game.state(0, hidden=False)
    return {
        "winner": game.won_by,
        "turns": game.turns,
        "points": [victory_points(state, p) for p in range(state.num_players)],
        "exhausted": game.won_by is None and game.phase is Phase.GAME_OVER,
    }
