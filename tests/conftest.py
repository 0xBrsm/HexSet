"""Shared table lifecycle fixtures; teardown closes every bot runner."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import shutil
import tempfile

import pytest

from hexset import seeds

# A suite's games are fixtures, not runs: no seed policy holds them
# (`hexset.seeds`), here or in any process a test starts.
seeds.suspend()

from hexset.bots import RandomBot
from hexset.play import step_randomly

import traders  # noqa: F401 -- registers `test-trader` with hexset.arena

if TYPE_CHECKING:
    from hexset.server.api import Tables


# Every `Tables` a test builds, so teardown can stop its bot runners.
_TRACKED: list[Tables] = []


def track(tables: Tables) -> Tables:
    _TRACKED.append(tables)
    return tables


# The temporary games directories those tables journal into, removed at teardown.
_GAMES_DIRS: list[str] = []


def new_tables(**config) -> Tables:
    """A `Tables` journalling into a fresh temporary directory unless the test
    names one: every game is journalled, and tests must not journal into a real
    games directory."""
    from hexset.server.api import Config, Tables

    if not config.get("games_dir"):
        config["games_dir"] = tempfile.mkdtemp(prefix="hexset-test-games-")
        _GAMES_DIRS.append(config["games_dir"])
    return track(Tables(Config(**config)))


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Close every tracked registry, then fail if a runner thread outlived it."""
    _TRACKED.clear()
    yield
    while _TRACKED:
        _TRACKED.pop().close()
    while _GAMES_DIRS:
        shutil.rmtree(_GAMES_DIRS.pop(), ignore_errors=True)
    live = [t.name for t in threading.enumerate() if t.name.startswith("bot-")]
    assert not live, f"bot runner threads still alive after this test: {live}"
