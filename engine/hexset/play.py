# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from .actions import Action, apply, legal_actions
from .board.board import Board, random_base_board
from .game import Game, Phase, is_over, start


class Stuck(RuntimeError):
    """Raised when a live game offers no legal action, which is always a bug."""


def step_randomly(game: Game, rng: random.Random) -> Action:
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {game.current_player}")
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

    return {
        "winner": game.won_by,
        "turns": game.turns,
        "points": [victory_points(game.state, p) for p in range(game.state.num_players)],
        "exhausted": game.won_by is None and game.phase is Phase.GAME_OVER,
    }
