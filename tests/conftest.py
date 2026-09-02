"""Test-only ways to drive a game forward, and the fixture that stops the
bots afterwards.

The two movers are not opponents anyone plays: the server deals `heximax`,
`search2` or a checkpoint from `models/` and nothing else. They live here
rather than in `src/` because a random mover is a fixture, and the package
should not ship one to make its own tests convenient.

`registry` is the one way a test should build a `Tables`. Every bot seat at
every table starts a runner thread that polls once a second until its game
ends, and a test that deals three bots and then asserts one thing leaves three
of them running: PR #2's suite left 67 live `bot-*` threads behind
(`docs/engine-divergence-2026-09-02.md`, defect 5). Going through the fixture
means teardown stops them.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field

import pytest

from hexset.actions import Action, apply
from hexset.game import Game

from hexset_ui.api import Config, Tables
from hexset_ui.rules import options_for


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


# Every `Tables` a test builds, so teardown can stop its bot runners. A test
# module registers through `track` rather than constructing a `Tables` and
# forgetting about it.
_TRACKED: list[Tables] = []


def track(tables: Tables) -> Tables:
    """Remember `tables` so `stop_bot_runners` closes it after the test."""
    _TRACKED.append(tables)
    return tables


def new_tables(**config) -> Tables:
    """A tracked `Tables` for a test.

    `games_dir=""` rather than the default `None`: `None` means "wherever
    `HEXSET_UI_GAMES_DIR` points", and a test suite must not journal into a
    real player's games directory. `seat_grace=0.0` makes the setup lock
    deterministic (two touches, no wall clock).
    """
    config.setdefault("games_dir", "")
    config.setdefault("seat_grace", 0.0)
    return track(Tables(Config(**config)))


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Closes every tracked registry when the test ends, then fails the test
    if a runner thread is somehow still alive. Both halves matter: the close
    is what stops them, the assertion is what stops the suite quietly growing
    a new leak later."""
    _TRACKED.clear()
    yield
    while _TRACKED:
        _TRACKED.pop().close()
    live = [t.name for t in threading.enumerate() if t.name.startswith("bot-")]
    assert not live, f"bot runner threads still alive after this test: {live}"
