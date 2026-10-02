"""Shared table lifecycle fixtures; teardown closes every bot runner."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

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


def new_tables(**config) -> Tables:
    """`games_dir=""` rather than the default `None` ("wherever
    `HEXSET_UI_GAMES_DIR` points"): tests must not journal into a real
    games directory.
    """
    from hexset.server.api import Config, Tables

    config.setdefault("games_dir", "")
    return track(Tables(Config(**config)))


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Close every tracked registry, then fail if a runner thread outlived it."""
    _TRACKED.clear()
    yield
    while _TRACKED:
        _TRACKED.pop().close()
    live = [t.name for t in threading.enumerate() if t.name.startswith("bot-")]
    assert not live, f"bot runner threads still alive after this test: {live}"
