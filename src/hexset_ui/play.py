from __future__ import annotations

import random

from .actions import Action, apply, legal_actions
from .game import Game


class Stuck(RuntimeError):
    """Raised when a live game offers no legal action, which is always a bug."""


def step_randomly(game: Game, rng: random.Random) -> Action:
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {game.current_player}")
    action = rng.choice(options)
    apply(game, action)
    return action
