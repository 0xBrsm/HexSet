# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import os
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
    *,
    journal: str | os.PathLike,
    seed: int | str | None = None,
) -> Game:
    """A game played by nobody in particular, to a finish, recorded and kept:
    its `hexset.record.Record` is appended to `journal` (`hexset.gamelog`,
    headed on first use) before the game is returned. `seed` is filed with the
    record to name the game.

    `turn_cap` is `UNSTRUCTURED_TURN_CAP` rather than the engine's default,
    because that default is read off agents that are trying to win and random
    play takes about four times as long. This function is unstructured play by
    definition, so it asks for the horizon unstructured play needs.
    """
    import json

    from . import __version__, gamelog
    from .record import Tape, recording, to_json
    from .rules import STANDARD

    if not journal:
        raise ValueError("a game is played into a journal: name one")
    rng = rng or random.Random()
    board = board if board is not None else random_base_board(rng)
    game = start(board, num_players, rng, turn_cap=turn_cap, chance=recording(rng, STANDARD))
    tape = Tape()
    while not is_over(game):
        before, shown = len(game.trades), len(game.shown)
        action = step_randomly(game, rng)
        tape.step(action, game.trades[before:], game.shown[shown:], before=before)
    record = tape.sealed(game, seed=seed)
    gamelog.append_after_header(
        journal,
        {"kind": "header", "journal": gamelog.JOURNAL_VERSION, "hexset": __version__, "games": "random"},
        {"kind": "game", "seed": seed, "winner": game.won_by, "turns": game.turns,
         "record": json.loads(to_json(record))},
    )
    return game
