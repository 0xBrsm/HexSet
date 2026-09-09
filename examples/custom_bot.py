"""Minimal bot integration and bounded replay smoke check.

Run after installing HexSet: ``python examples/custom_bot.py``.
The uniform policy demonstrates the interface; it is not a strength baseline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from hexset.actions import Action, options_for
from hexset.arena import Entrant, compete, register_entrant_kind
from hexset.game import Game
from hexset.record import replay


@dataclass
class ExampleBot:
    rng: random.Random

    def choose(self, game: Game) -> Action:
        # Evaluate game.state(to_move(game)) when adding a decision rule.
        # The Game is live: do not mutate it while choosing an action.
        return self.rng.choice(options_for(game))


def make_bot(entrant, board, rng):
    """Use the runner's RNG; global or unseeded randomness breaks repeatability."""
    return ExampleBot(rng)


def register_example() -> None:
    """Module-level initializers also work in spawned worker processes."""
    register_entrant_kind("example", make_bot)


def main() -> None:
    tournament = compete(
        [Entrant("example", kind="example"), Entrant("random", kind="random")],
        games=2,
        seed=0,
        workers=1,
        worker_initializer=register_example,
        action_cap=40,
        records=True,
    )
    for record in tournament.records:
        replay(record)
    print(f"Replayed {len(tournament.records)} bounded games; this is a smoke check, not a strength measurement.")


if __name__ == "__main__":
    main()
