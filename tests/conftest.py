"""Test-only ways to drive a game forward.

Neither of these is an opponent anyone plays: the server deals `search2` or a
checkpoint from `models/` and nothing else. They live here rather than in
`src/` because a random mover is a fixture, and the package should not ship one
to make its own tests convenient.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from hexset_ui.actions import Action, apply, options_for
from hexset_ui.game import Game


@dataclass
class RandomBot:
    """Satisfies what a `GameSession` needs of an opponent: `choose(game)`."""

    rng: random.Random = field(default_factory=random.Random)

    def choose(self, game: Game) -> Action:
        return self.rng.choice(options_for(game))


def step_randomly(game: Game, rng: random.Random) -> Action:
    """Pick a legal action at random and apply it, returning what was played."""
    action = rng.choice(options_for(game))
    apply(game, action)
    return action
