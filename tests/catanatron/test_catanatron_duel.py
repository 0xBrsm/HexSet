# SPDX-License-Identifier: GPL-3.0-only
"""The duel CLI plays every game under the requested rules variant.

`--game-type` selects the catanatron `GameConfigOptions` for the whole duel;
these tests pin the mapping and the plumbing without playing real games.
"""

from __future__ import annotations

import random

import pytest

# Same import-skip dance as the sibling catanatron tests: this directory is
# itself named `catanatron`, so a bare `import catanatron` can resolve to it.
pytest.importorskip("catanatron.game")

from hexset.catanatron import duel
from hexset.catanatron.duel import game_config_for, run_duel
from hexset.rules import COLONIST_1V1


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


def test_game_config_for_maps_game_types_to_catanatron_options():
    standard = game_config_for("standard")
    assert (standard.vps_to_win, standard.discard_limit) == (10, 7)
    colonist = game_config_for("colonist-1v1")
    assert (colonist.vps_to_win, colonist.discard_limit) == (15, 9)


def test_game_config_for_rejects_unknown_game_types():
    with pytest.raises(ValueError, match="unknown game type"):
        game_config_for("no-such-game")


def test_colonist_duel_games_are_built_with_colonist_rules(monkeypatch):
    """The game type reaches catanatron's `Game` constructor: every game in
    the duel is built 15VP/9-discard, not just labelled that way."""
    configs = []

    def fake_play_batch(num_games, players, game_config=None, quiet=False):
        configs.append(game_config)
        return {}, {}, 1

    monkeypatch.setattr(duel, "play_batch", fake_play_batch)
    monkeypatch.setattr(duel, "parse_cli_string", lambda spec: [])
    monkeypatch.setattr(duel, "Pool", _InlinePool)

    run_duel("DC:heximax,AB:2", 3, 2, seed=0, game_type="colonist-1v1")

    assert len(configs) == 3
    assert all(c.vps_to_win == 15 and c.discard_limit == 9 for c in configs)


def test_run_duel_rejects_an_unknown_game_type_before_playing(monkeypatch):
    """Fail fast: a bad `--game-type` raises before any shard plays a game."""
    monkeypatch.setattr(duel, "Pool", _InlinePool)
    with pytest.raises(ValueError, match="unknown game type"):
        run_duel("DC:heximax,AB:2", 2, 1, seed=0, game_type="no-such-game")


def test_translate_reads_the_catanatron_games_rules():
    """The HexSet bot reasons about the position with the table's own rules:
    a 15VP/9-discard catanatron game translates to colonist rules, so
    Heximax's win bonus and discard model match the game being played."""
    from catanatron.game import Game as CatanatronGame
    from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
    from catanatron.models.player import Color, RandomPlayer

    from hexset.catanatron.board import translate_board
    from hexset.catanatron.state import to_catanatron, translate

    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE)]
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    game = CatanatronGame(
        players, catan_map=catan_map, vps_to_win=15, discard_limit=9
    )
    mapping = translate_board(catan_map)
    our_game, seats = translate(game, mapping, random.Random(0))
    assert our_game._state.rules == COLONIST_1V1
    mirror = to_catanatron(our_game, mapping, seats)
    assert (mirror.vps_to_win, mirror.state.discard_limit) == (15, 9)


@pytest.fixture
def first_draws(monkeypatch):
    """Replace the game loop: record each game's first draw from the global
    `random` stream -- the stream catanatron's `Game` draws its seed from, so
    the first draw identifies the game exactly."""
    draws = []

    def fake_play_batch(num_games, players, game_config=None, quiet=False):
        assert num_games == 1
        draws.append(random.random())
        return {}, {}, 1

    monkeypatch.setattr(duel, "play_batch", fake_play_batch)
    monkeypatch.setattr(duel, "parse_cli_string", lambda spec: [])
    monkeypatch.setattr(duel, "Pool", _InlinePool)
    return draws


@pytest.mark.parametrize("game_type", ["standard", "colonist-1v1"])
@pytest.mark.parametrize("workers", [1, 2, 3, 7])
def test_game_seeds_depend_only_on_duel_seed_and_game_index(first_draws, workers, game_type):
    """10 games at 1, 2, 3 and 7 workers: every game starts from the same
    global-random state, and that state is `seed + g`, whatever the sharding."""
    run_duel("DC:heximax,AB:2", 10, workers, seed=42, game_type=game_type)

    assert len(first_draws) == 10  # every game played exactly once
    assert first_draws == [random.Random(42 + g).random() for g in range(10)]


def test_uneven_sharding_still_covers_every_game_exactly_once(first_draws):
    """7 games over 3 workers shards 3/3/1; no game dropped, none repeated."""
    run_duel("DC:heximax,AB:2", 7, 3, seed=0)

    assert first_draws == [random.Random(g).random() for g in range(7)]
