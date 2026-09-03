# SPDX-License-Identifier: GPL-3.0-only
"""End-to-end: DevCatanPlayer playing real, complete catanatron games.

This is the test the earlier ones were building towards -- `search2-notrade`
(no torch, so it runs anywhere) driven entirely through the bridge, against
catanatron's own bots, for whole games rather than isolated decisions. It is
also what actually exercises PLAY_KNIGHT's two-step resolution, which none of
the earlier tests could: that path only exists once a real bot, not a random
one probing every legal option, chooses to play a knight.
"""

import random

import pytest

# A submodule, not bare "catanatron": this directory is itself named
# `catanatron`, and once pytest's default import mode puts `tests/` on
# sys.path (for the sibling top-level test modules), a bare `catanatron`
# import can resolve to *this directory* as an empty namespace package
# instead of failing -- silently skipping nothing and then blowing up on
# the first real submodule access. `catanatron.game` only exists in the
# real distribution.
pytest.importorskip("catanatron.game")

from catanatron.game import Game as CatanatronGame
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.search import VictoryPointPlayer

from hexset.catanatron.player import DevCatanPlayer


def test_entrant_spec_survives_catanatrons_own_colon_split():
    # catanatron's CLI splits `--players` on every `:` and passes each piece
    # as a separate positional argument, so a spec that itself contains a
    # colon -- `network:<path>`, `mcts:<path>@N` -- arrives here in parts.
    # Rejoining them is this constructor's job.
    bridge = DevCatanPlayer(Color.RED, "network", "/some/path/to/checkpoint.pt")
    assert bridge.entrant_spec == "network:/some/path/to/checkpoint.pt"


def test_entrant_spec_defaults_when_no_parts_given():
    bridge = DevCatanPlayer(Color.RED)
    assert bridge.entrant_spec == "search2-notrade"


@pytest.mark.parametrize("seed", range(8))
def test_full_games_complete_against_random(seed):
    random.seed(seed)
    bridge = DevCatanPlayer(Color.RED, "search2-notrade")
    players = [bridge, RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE), RandomPlayer(Color.ORANGE)]
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    game = CatanatronGame(players, catan_map=catan_map)

    winner = game.play()

    assert bridge.decisions > 0
    # A handful of engine-difference fallbacks (see actions.py/player.py) are
    # expected and fine, and cluster late in long games (the piece cap in
    # particular needs a game to run long to matter), so one game's rate is
    # noisy -- this only needs to catch "the adapter is broken and resolving
    # most decisions by chance". Measured aggregate over 10 games against
    # random opponents (about as adversarial to this as it gets): 1.5-4.6%.
    assert bridge.fallbacks / bridge.decisions < 0.2


def test_rng_is_seeded_from_the_game_seed_and_seat_not_from_process_state():
    """The bug this guards against: `self._rng = random.Random()` with no
    argument draws its initial state from OS randomness, independent of
    catanatron's own `--seed` -- so two runs of the same `--seed` produced
    different belief-sampling/steal/draw decisions in the DC seat even
    though catanatron's own dice and deck (driven by the global `random`
    module, which duel.py's shard-seeding *does* reach) reproduced exactly.
    A correctly seeded bridge must draw the same rng stream for the same
    game seed and seat, and a different one for a different seed or seat.
    """

    def first_draws(seed, color, n=5):
        catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
        players = [
            DevCatanPlayer(color, "search2-notrade") if c is color else RandomPlayer(c)
            for c in Color
        ]
        game = CatanatronGame(players, catan_map=catan_map, seed=seed)
        bridge = next(p for p in players if isinstance(p, DevCatanPlayer))
        bridge.decide(game, game.playable_actions)
        return [bridge._rng.random() for _ in range(n)]

    same_a = first_draws(12345, Color.RED)
    same_b = first_draws(12345, Color.RED)
    assert same_a == same_b, "same game seed and seat must reproduce the same rng stream"

    different_seed = first_draws(99999, Color.RED)
    assert same_a != different_seed, "a different game seed must not collide"

    different_seat = first_draws(12345, Color.BLUE)
    assert same_a != different_seat, "a different seat must not collide"


def test_a_search_bot_beats_random_and_victory_point_most_of_the_time():
    wins = 0
    games = 20
    for seed in range(games):
        random.seed(seed)
        bridge = DevCatanPlayer(Color.RED, "search2-notrade")
        players = [
            bridge,
            RandomPlayer(Color.BLUE),
            VictoryPointPlayer(Color.WHITE),
            RandomPlayer(Color.ORANGE),
        ]
        catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
        game = CatanatronGame(players, catan_map=catan_map)
        winner = game.play()
        if winner is Color.RED:
            wins += 1

    # This is a smoke test, not a strength claim -- 20 games is nowhere near
    # enough for a real number (that is `arena.py`'s job once this is wired
    # up for real duels). It only has to catch "the adapter is broken and
    # search2 is actually losing to weaker opponents most of the time".
    assert wins >= games * 0.5
