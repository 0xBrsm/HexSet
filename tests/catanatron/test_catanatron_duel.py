# SPDX-License-Identifier: GPL-3.0-only
"""The duel CLI plays every game under the requested rules variant.

`--game-type` selects the catanatron `GameConfigOptions` for the whole duel;
these tests pin the mapping and the plumbing without playing real games.
They also pin `build_players`, which routes player specs through catanatron's
shared player registry (`parse_cli_string` is gone upstream).
"""

from __future__ import annotations

import random

import pytest

# Same import-skip dance as the sibling catanatron tests: this directory is
# itself named `catanatron`, so a bare `import catanatron` can resolve to it.
pytest.importorskip("catanatron.game")

from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.registry import REGISTRY, SpecError

from hexset.catanatron import duel
from hexset.catanatron.duel import build_players, game_config_for, run_duel
from hexset.catanatron.player import DevCatanPlayer
from hexset.rules import COLONIST_1V1


class _InlinePool:
    """Stand-in for multiprocessing.Pool that runs the map inline."""

    def __init__(self, _n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, args, chunksize=1):
        return iter(fn(a) for a in args)


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
    monkeypatch.setattr(duel, "build_players", lambda spec: [])
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
    monkeypatch.setattr(duel, "build_players", lambda spec: [])
    monkeypatch.setattr(duel, "Pool", _InlinePool)
    return draws


@pytest.mark.parametrize("scheduling", ["dynamic", "static"])
@pytest.mark.parametrize("game_type", ["standard", "colonist-1v1"])
@pytest.mark.parametrize("workers", [1, 2, 3, 7])
def test_game_seeds_depend_only_on_duel_seed_and_game_index(first_draws, workers, game_type, scheduling):
    """10 games at 1, 2, 3 and 7 workers: every game starts from the same
    global-random state, and that state is `seed + g`, whatever the sharding."""
    run_duel("DC:heximax,AB:2", 10, workers, seed=42, game_type=game_type, scheduling=scheduling)

    assert len(first_draws) == 10  # every game played exactly once
    assert first_draws == [random.Random(42 + g).random() for g in range(10)]


def test_uneven_sharding_still_covers_every_game_exactly_once(first_draws):
    """7 games over 3 workers shards 3/3/1; no game dropped, none repeated."""
    run_duel("DC:heximax,AB:2", 7, 3, seed=0)

    assert first_draws == [random.Random(g).random() for g in range(7)]


@pytest.mark.parametrize('speedups', ['off', 'basic', 'cached', 'fast'])
def test_speedups_are_scoped_to_worker_and_reported(monkeypatch, speedups):
    from catanatron.models.board import Board
    from hexset.catanatron.speedups import clone_board_mutable_structures
    observed = []

    def fake_play_batch(num_games, players, game_config=None, quiet=False):
        observed.append(Board.copy is clone_board_mutable_structures)
        return {}, {}, 1

    before = Board.copy
    monkeypatch.setattr(duel, 'play_batch', fake_play_batch)
    monkeypatch.setattr(duel, "build_players", lambda spec: [])
    monkeypatch.setattr(duel, 'Pool', _InlinePool)
    result = run_duel('DC:heximax,AB:2', 3, 2, speedups=speedups)
    assert observed == [speedups != 'off'] * 3
    assert Board.copy is before
    assert result.speedups == speedups
    assert f'Catanatron speedups: {speedups}' in result.report()


def test_dynamic_results_are_ordered_by_game_index_not_completion(monkeypatch):
    from catanatron.models.player import Color
    class ReversePool(_InlinePool):
        def imap_unordered(self, fn, args, chunksize=1):
            jobs = list(args)
            assert len(jobs) == 7
            assert all(job[2] == 1 for job in jobs)
            assert chunksize == 1
            return iter(fn(a) for a in reversed(jobs))
    def fake_job(args):
        index = args[1]
        return index, 1, {Color.RED: 1}, {Color.RED: [index]}, 1.0
    monkeypatch.setattr(duel, 'Pool', ReversePool)
    monkeypatch.setattr(duel, '_play_job', fake_job)
    result = run_duel('F,F', 7, 3)
    assert result.wins[Color.RED] == 7
    assert result.points[Color.RED] == list(range(7))
    assert result.worker_seconds == 7
    assert 'dynamically assigned' in result.report()


@pytest.mark.parametrize('games,workers', [(0, 1), (1, 0), (-1, 2)])
def test_invalid_game_or_worker_count_fails_before_pool(games, workers):
    with pytest.raises(ValueError, match='positive'):
        run_duel('F,F', games, workers)


# `build_players` and the registry-era `DevCatanPlayer`: the seams the
# catanatron ecf93118 upgrade touched.


def test_build_players_routes_builtins_through_the_registry():
    players = build_players("AB:2,R")
    assert isinstance(players[0], AlphaBetaPlayer)
    assert players[0].params.depth == 2
    assert players[0].color is Color.RED
    assert isinstance(players[1], RandomPlayer)
    assert players[1].color is Color.BLUE


def test_build_players_keeps_the_whole_dc_tail():
    """An entrant spec may itself contain colons (`network:<path>`); the
    registry's own colon-splitting cannot round-trip those, so `DC` is built
    directly from the unsplit tail."""
    players = build_players("DC:network:/tmp/x.pt,AB:2")
    assert isinstance(players[0], DevCatanPlayer)
    assert players[0].params.entrant == "network:/tmp/x.pt"


def test_build_players_dc_defaults_to_heximax_notrade():
    (player, _) = build_players("DC,AB:2")
    assert isinstance(player, DevCatanPlayer)
    assert player.params.entrant == "heximax"


def test_dc_is_also_buildable_through_the_registry():
    """The `catanatron-play --bot` path (`REGISTRY.build`), where the spec
    has no structural colons to preserve."""
    player = REGISTRY.build("DC:heximax", Color.RED)
    assert isinstance(player, DevCatanPlayer)
    assert player.params.entrant == "heximax"
    assert player.color is Color.RED


def test_direct_construction_rejoins_colon_split_parts():
    """The pre-registry calling convention -- one positional piece per
    colon-split part -- still works (the tests and duel shards use it)."""
    assert DevCatanPlayer(Color.RED, "heximax").params.entrant == (
        "heximax"
    )
    assert DevCatanPlayer(Color.RED, "network", "/tmp/x.pt").params.entrant == (
        "network:/tmp/x.pt"
    )


def test_build_players_rejects_wrong_seat_counts():
    with pytest.raises(SpecError):
        build_players("AB:2")
    with pytest.raises(SpecError):
        build_players("AB:2,R,AB:2,R,AB:2")


def test_before_resets_the_per_game_state():
    """`play_batch` no longer calls `reset_state()` between games; the
    `before` observer hook is its replacement, fired once per game from
    `Game.__init__`."""
    player = DevCatanPlayer(Color.RED, "heximax")
    player._mapping = object()
    player._bot = object()
    player._rng = object()
    player.before(None)
    assert player._mapping is None
    assert player._bot is None
    assert player._rng is None


@pytest.mark.parametrize("spec", ["heximax", "network:/tmp/x.pt", "heximax:pin-weights=1:trading=off"])
def test_build_players_accepts_named_dc_entrant(spec):
    player = build_players(f"DC:entrant={spec},R")[0]
    assert player.params.entrant == spec


def test_game_construction_resets_reused_dc_player():
    from catanatron.game import Game
    player = DevCatanPlayer(Color.RED)
    for seed in (1451, 1452):
        player._mapping = player._bot = player._rng = object()
        Game([player, RandomPlayer(Color.BLUE)], seed=seed)
        assert player._mapping is player._bot is player._rng is None
