# SPDX-License-Identifier: GPL-3.0-only
"""Game g of a catanatron duel must play the same game at any --workers.

`run_duel` shards games across worker processes; the seed each game starts
from must be a function of the duel seed and the game index only, never of
how the games were sharded. Otherwise --workers silently reselects the games
being measured, and two runs at the same --seed are not comparable.
"""

from __future__ import annotations

import random

import pytest

# Same import-skip dance as the sibling catanatron tests: this directory is
# itself named `catanatron`, so a bare `import catanatron` can resolve to it.
pytest.importorskip("catanatron.game")

from hexset.catanatron import duel
from hexset.catanatron.duel import run_duel


class _InlinePool:
    """Stand-in for multiprocessing.Pool that runs the map inline."""

    def __init__(self, _n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, args):
        return [fn(a) for a in args]


@pytest.fixture
def first_draws(monkeypatch):
    """Replace the game loop: record each game's first draw from the global
    `random` stream -- the stream catanatron's `Game` draws its seed from, so
    the first draw identifies the game exactly."""
    draws = []

    def fake_play_batch(num_games, players, quiet=False):
        assert num_games == 1
        draws.append(random.random())
        return {}, {}, 1

    monkeypatch.setattr(duel, "play_batch", fake_play_batch)
    monkeypatch.setattr(duel, "parse_cli_string", lambda spec: [])
    monkeypatch.setattr(duel, "Pool", _InlinePool)
    return draws


@pytest.mark.parametrize("workers", [1, 2, 3, 7])
def test_game_seeds_depend_only_on_duel_seed_and_game_index(first_draws, workers):
    """10 games at 1, 2, 3 and 7 workers: every game starts from the same
    global-random state, and that state is `seed + g`, whatever the sharding."""
    run_duel("DC:heximax,AB:2", 10, workers, seed=42)

    assert len(first_draws) == 10  # every game played exactly once
    assert first_draws == [random.Random(42 + g).random() for g in range(10)]


def test_uneven_sharding_still_covers_every_game_exactly_once(first_draws):
    """7 games over 3 workers shards 3/3/1; no game dropped, none repeated."""
    run_duel("DC:heximax,AB:2", 7, 3, seed=0)

    assert first_draws == [random.Random(g).random() for g in range(7)]
