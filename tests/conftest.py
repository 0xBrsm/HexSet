"""Shared table lifecycle fixtures; teardown closes every bot runner."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from hexset.bots import RandomBot
from hexset.play import step_randomly

if TYPE_CHECKING:
    from hexset.server.api import Tables


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
    real player's games directory.
    """
    from hexset.server.api import Config, Tables

    config.setdefault("games_dir", "")
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
