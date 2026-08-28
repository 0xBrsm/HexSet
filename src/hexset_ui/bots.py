"""The small vocabulary every bot shares, and nothing else.

Deliberately thin. `hexset_ui.search2` (handcrafted) and `hexset_ui.onnxbot`
(learned) both depend on this and neither depends on the other, so this module
is the whole of what they have in common: what a bot is, how to ask the engine
for its options, and how a seat reads a per-seat vector.

`RandomBot` and `step_randomly` are here too — not opponents anyone plays, but
the cheapest thing that drives a game forward, which is what the tests need.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .actions import Action, apply, legal_actions
from .game import Game, to_move


class Stuck(RuntimeError):
    """Raised when a live game offers no legal action, which is always a bug."""


class Bot(Protocol):
    """Anything that can pick an action, handcrafted or learned alike.

    The entire interface between the game and a bot: a position in, an action
    out. Nothing about how the action was arrived at crosses this line.
    """

    def choose(self, game: Game) -> Action: ...


def own(vector: Sequence[float], seat: int) -> float:
    """Plain max^n: each seat wants its own score high and ignores the rest."""
    return vector[seat]


def relative(vector: Sequence[float], seat: int) -> float:
    """Own score less the average of everyone else's.

    A constant-sum reading of the vector. HexSet has exactly one winner, so a
    position is only worth what it is worth *compared to* the table, and an
    action that lifts everyone equally has achieved nothing.
    """
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - sum(others) / len(others)


def paranoid(vector: Sequence[float], seat: int) -> float:
    """Own score less the best opponent's. The leader is the only rival."""
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - max(others)


# How a seat turns the per-seat vector into the one number it maximises. The
# evaluation is unchanged; only the reading of it differs.
STANCES = {"own": own, "relative": relative, "paranoid": paranoid}


def options_for(game: Game) -> list[Action]:
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {to_move(game)}")
    return options


@dataclass
class RandomBot:
    rng: random.Random = field(default_factory=random.Random)

    def choose(self, game: Game) -> Action:
        return self.rng.choice(options_for(game))


def step_randomly(game: Game, rng: random.Random) -> Action:
    """Pick a legal action at random and apply it, returning what was played."""
    action = rng.choice(options_for(game))
    apply(game, action)
    return action
