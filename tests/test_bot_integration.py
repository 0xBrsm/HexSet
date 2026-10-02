# SPDX-License-Identifier: GPL-3.0-only
"""The documented extension path end to end, and the install smoke check.
`docs/guide.md#implement-a-bot` carries this code as prose; keep the two in
step. The uniform policy demonstrates the interface, not a strength baseline.
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
        # The Game is live: do not mutate it while choosing an action.
        return self.rng.choice(options_for(game))


def make_bot(entrant, board, rng):
    """Use the runner's RNG; global randomness breaks repeatability."""
    return ExampleBot(rng)


def register_example() -> None:
    """Module level, so a spawned worker can import and call it."""
    register_entrant_kind("example", make_bot)


def test_a_registered_bot_plays_and_its_records_replay():
    tournament = compete(
        [Entrant("example", kind="example"), Entrant("random", kind="random")],
        games=2,
        seed=0,
        workers=1,
        worker_initializer=register_example,
        action_cap=40,
        records=True,
    )
    assert len(tournament.records) == 2
    for record in tournament.records:
        replay(record)
