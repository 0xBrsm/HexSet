# SPDX-License-Identifier: GPL-3.0-only
"""End-to-end: DevCatanPlayer playing real, complete catanatron games.

This is the test the earlier ones were building towards -- `search2-notrade`
(no torch, so it runs anywhere) driven entirely through the bridge, against
catanatron's own bots, for whole games rather than isolated decisions. It is
also what actually exercises a knight played for real: `PLAY_KNIGHT` maps
one-to-one onto catanatron's `PLAY_KNIGHT_CARD` (`actions.py`), and the
`MOVE_ROBBER` decision that follows resolves on the bridge's own next
`decide()` call, translated fresh from catanatron's own post-knight state --
that path only exists once a real bot, not a random one probing every legal
option, chooses to play a knight.
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
from hexset.arena import PRESETS
from hexset.catanatron.player import DevCatanPlayer


def test_importing_the_bridge_registers_the_heximax_presets():
    """`hexset.catanatron.player` (and `.duel`, which imports it) used to
    import neither `hexset.bots` nor anything that does, so a worker process
    asking for `DC:heximax-notrade` raised a bare `KeyError` on the name --
    `PRESETS` only gains "heximax"/"heximax-omni"/"heximax-notrade" as an
    import-time side effect of importing `hexset.bots.heximax`. Importing
    this module (done above, at collection time) is what this test is
    actually checking survived; the assertion below just makes that explicit
    rather than relying on the import above not raising.
    """
    assert "heximax-notrade" in PRESETS
    assert "heximax" in PRESETS


@pytest.mark.parametrize("seed", range(1))
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
